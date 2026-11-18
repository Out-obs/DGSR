# Credible dynamics set, conformal calibration and the worst-case margin rho.

import os

import numpy as np

IGNORE_C1 = os.environ.get("DSH_IGNORE_C1", "0") == "1"
import os

import torch
import torch.nn.functional as F

def sample_vectors(state, action, diffusion, num_samples=32):
    return diffusion.sample_vectors(state, action, num_samples=num_samples)

def barrier_values_and_gradients(cbf, state, obstacle):
    state_for_grad = state.detach().clone().requires_grad_(True)
    obstacle_for_grad = obstacle.detach()
    h = cbf(state_for_grad, obstacle_for_grad)
    gradients = []
    for obstacle_index in range(h.shape[1]):
        grad = torch.autograd.grad(
            h[:, obstacle_index].sum(),
            state_for_grad,
            retain_graph=True,
            create_graph=False,
        )[0]
        gradients.append(grad)
    gradients = torch.stack(gradients, dim=1)
    return h.detach(), gradients.detach()

def align_obstacle_slots(obstacle, obstacle_next, n_pos=3):
    if obstacle.ndim != 3 or obstacle_next.shape != obstacle.shape:
        raise ValueError("obstacle tensors must both have shape (B,K,D)")
    batch, count, _ = obstacle.shape
    if count == 0:
        return obstacle_next.clone(), obstacle.new_zeros((batch, 0))
    position = obstacle[:, :, :int(n_pos)]
    next_position = obstacle_next[:, :, :int(n_pos)]
    cost = torch.cdist(position, next_position, p=2)
    aligned = torch.empty_like(obstacle_next)
    distances = []
    try:
        from scipy.optimize import linear_sum_assignment
        for index in range(batch):
            rows, cols = linear_sum_assignment(cost[index].detach().cpu().numpy())
            order = torch.as_tensor(cols, device=obstacle.device, dtype=torch.long)
            aligned[index] = obstacle_next[index].index_select(0, order)
            distances.append(cost[index, rows, cols].mean())
    except Exception:

        for index in range(batch):
            available = list(range(count))
            selected = []
            for row in range(count):
                col = min(available, key=lambda candidate: float(cost[index, row, candidate]))
                selected.append(col)
                available.remove(col)
            order = torch.as_tensor(selected, device=obstacle.device, dtype=torch.long)
            aligned[index] = obstacle_next[index].index_select(0, order)
            distances.append(cost[index, torch.arange(count, device=obstacle.device), order].mean())
    return aligned, torch.stack(distances)

def _sample_covariance(centered, ridge=1e-5, method="oas"):
    if centered.ndim != 3:
        raise ValueError("centered samples must have shape (batch, samples, dim)")
    batch, count, dim = centered.shape
    cov = centered.transpose(1, 2).bmm(centered) / max(count - 1, 1)
    if method == "oas":
        trace = torch.diagonal(cov, dim1=-2, dim2=-1).sum(-1)
        trace_sq = (cov * cov).sum(dim=(-2, -1))
        trace_square = trace * trace
        numerator = (1.0 - 2.0 / dim) * trace_sq + trace_square
        denominator = (count + 1.0 - 2.0 / dim) * (
            trace_sq - trace_square / dim).clamp_min(1e-12)
        shrink = torch.clamp(numerator / denominator, min=0.0, max=1.0)
        target = (trace / dim).view(batch, 1, 1) * torch.eye(
            dim, device=centered.device, dtype=centered.dtype).unsqueeze(0)
        cov = (1.0 - shrink).view(batch, 1, 1) * cov + shrink.view(batch, 1, 1) * target
    elif method != "none":
        raise ValueError("covariance method must be 'oas' or 'none'")
    scale = torch.diagonal(cov, dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12)
    eye = torch.eye(dim, device=centered.device, dtype=centered.dtype).unsqueeze(0)
    cov = cov + (float(ridge) * scale).view(batch, 1, 1) * eye
    return cov

def _mahalanobis_distances(samples, mean, cov, ridge=1e-4):
    centered = samples - mean.unsqueeze(-2) if samples.ndim == 3 else samples - mean

    eye = torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype)
    if cov.ndim == 3:
        eye = eye.unsqueeze(0)
    for attempt in range(5):
        try:
            cov_reg = cov + (ridge * (10 ** attempt)) * eye
            if cov_reg.ndim == 3:
                L = torch.linalg.cholesky(cov_reg)
                if centered.ndim == 3:
                    solved = torch.linalg.solve_triangular(
                        L, centered.transpose(1, 2), upper=False)
                    distances = (solved ** 2).sum(dim=1)
                else:
                    solved = torch.linalg.solve_triangular(
                        L, centered.unsqueeze(-1), upper=False)
                    distances = (solved.squeeze(-1) ** 2).sum(dim=1)
            else:
                L = torch.linalg.cholesky(cov_reg)
                solved = torch.linalg.solve_triangular(
                    L, centered.T, upper=False)
                distances = (solved ** 2).sum(dim=0)
            return distances
        except RuntimeError:
            continue

    distances = ((centered) ** 2).sum(dim=-1)
    return distances

def _fit_mvee(points, ridge=1e-5, tolerance=1e-3, max_iter=64):
    point_count, dimension = points.shape
    q = torch.cat([
        points.transpose(0, 1),
        torch.ones(1, point_count, device=points.device, dtype=points.dtype),
    ], dim=0)
    weights = torch.full(
        (point_count,), 1.0 / point_count,
        device=points.device, dtype=points.dtype)

    for _ in range(max_iter):
        x_matrix = (q * weights.unsqueeze(0)) @ q.transpose(0, 1)
        inverse = torch.linalg.pinv(x_matrix)
        leverage = torch.sum(q * (inverse @ q), dim=0)
        max_value, max_index = torch.max(leverage, dim=0)
        denominator = (dimension + 1.0) * (max_value - 1.0)
        step = (max_value - dimension - 1.0) / denominator.clamp_min(1e-12)
        step = torch.clamp(step, min=0.0, max=1.0)
        vertex = F.one_hot(max_index, num_classes=point_count).to(
            device=points.device, dtype=points.dtype)
        new_weights = (1.0 - step) * weights + step * vertex
        if torch.norm(new_weights - weights).detach().item() <= tolerance:
            weights = new_weights
            break
        weights = new_weights

    center = torch.sum(points * weights.unsqueeze(1), dim=0)
    second_moment = points.transpose(0, 1) @ (points * weights.unsqueeze(1))
    covariance = second_moment - torch.outer(center, center)
    shape = dimension * covariance
    shape = shape + ridge * torch.eye(
        dimension, device=points.device, dtype=points.dtype)
    distances = _mahalanobis_distances(points, center, shape)
    max_distance = torch.maximum(
        distances.max(),
        torch.tensor(1.0, device=points.device, dtype=points.dtype),
    )
    return center, shape * max_distance, max_distance

def construct_credible_dynamics_set(
        diffusion,
        state,
        action,
        num_samples=32,
        q_alpha=None,
        q_alpha_scale=1.0,
        vector_samples=None,
        ridge=1e-5):
    if q_alpha is None:
        q_alpha = diffusion.q_alpha if diffusion.q_alpha is not None else 1.0

        _q = float(q_alpha)
        _ref = float(os.environ.get("DSH_Q_ALPHA_CLAMP", "0"))
        if _ref > 0.0 and _q > _ref:
            print("[robust_verify] q_alpha %.3f exceeds clamp %.3f; using the clamp" % (_q, _ref))
            q_alpha = _ref
    if not torch.is_tensor(q_alpha):
        q_alpha = torch.tensor(float(q_alpha), dtype=state.dtype, device=state.device)
    q_alpha_val = q_alpha * float(q_alpha_scale)

    if vector_samples is None:

        with torch.no_grad():
            vector_samples = sample_vectors(state, action, diffusion,
                                            num_samples=num_samples)

    batch_size = vector_samples.shape[0]
    sample_dim = vector_samples.shape[2]
    device = vector_samples.device

    _fit_device = device
    if os.environ.get('DSH_MVEE_FIT_CPU', '0') == '1' and device.type == 'cuda':
        vector_samples = vector_samples.cpu()
        if torch.is_tensor(q_alpha_val):
            q_alpha_val = q_alpha_val.cpu()
        else:
            q_alpha_val = torch.tensor(float(q_alpha_val))
        device = vector_samples.device

    c_alpha_list = []
    P_alpha_list = []
    retained_counts = []
    max_distances = []

    for b in range(batch_size):

        vs = vector_samples[b]

        mu = vs.mean(dim=0)
        centered_all = vs - mu.unsqueeze(0)
        cov = _sample_covariance(centered_all.unsqueeze(0), ridge=1e-4)[0]

        distances = _mahalanobis_distances(vs, mu, cov)

        retained_mask = distances <= q_alpha_val
        retained = vs[retained_mask]
        retained_count = retained.shape[0]
        retained_counts.append(retained_count)

        if retained_count < sample_dim + 1:

            retained = vs
            retained_count = len(vs)

        try:
            c, P, max_dist = _fit_mvee(retained, ridge=ridge)
        except Exception:

            c = retained.mean(dim=0)
            centered = retained - c.unsqueeze(0)
            cov = _sample_covariance(
                centered.unsqueeze(0), ridge=ridge, method="oas")[0]
            max_dist = torch.maximum(
                _mahalanobis_distances(retained, c, cov).max(),
                torch.tensor(1.0, device=cov.device, dtype=cov.dtype),
            )
            P = cov * max_dist
        P = P * float(getattr(diffusion, "credible_scale", 1.0))

        c_alpha_list.append(c)
        P_alpha_list.append(P)
        max_distances.append(max_dist.detach())

    c_alpha = torch.stack(c_alpha_list, dim=0)
    P_alpha = torch.stack(P_alpha_list, dim=0)
    mean_vector = vector_samples.mean(dim=1)
    if _fit_device != device:
        c_alpha = c_alpha.to(_fit_device)
        P_alpha = P_alpha.to(_fit_device)
        mean_vector = mean_vector.to(_fit_device)

    return {
        "c_alpha": c_alpha,
        "P_alpha": P_alpha,
        "mean_vector": mean_vector,
        "retained_count": retained_counts,
        "max_distance": torch.stack(max_distances, dim=0),
        "vector_samples": vector_samples,
    }

def construct_calibrated_ellipsoid_set(
        diffusion,
        state,
        action,
        num_samples=32,
        q_alpha=None,
        q_alpha_scale=1.0,
        vector_samples=None,
        ridge=1e-5):
    if q_alpha is None:
        q_alpha = diffusion.q_alpha if diffusion.q_alpha is not None else 1.0
    if not torch.is_tensor(q_alpha):
        q_alpha = torch.tensor(float(q_alpha), dtype=state.dtype, device=state.device)
    q_alpha_val = q_alpha * float(q_alpha_scale)

    if hasattr(diffusion, "conditional_moments"):

        mean_vector, covariance = diffusion.conditional_moments(state, action)
        sample_count = int(num_samples)
        if vector_samples is None:
            vector_samples = sample_vectors(state, action, diffusion, num_samples=num_samples)
    else:
        if vector_samples is None:
            vector_samples = sample_vectors(state, action, diffusion, num_samples=num_samples)
        sample_count = vector_samples.shape[1]
        mean_vector = vector_samples.mean(dim=1)
        centered = vector_samples - mean_vector.unsqueeze(1)
        covariance = _sample_covariance(centered, ridge=ridge, method="oas")
    P_alpha = covariance * q_alpha_val

    return {
        "c_alpha": mean_vector,
        "P_alpha": P_alpha,
        "mean_vector": mean_vector,
        "retained_count": [sample_count] * vector_samples.shape[0],
        "max_distance": torch.full(
            (vector_samples.shape[0],),
            float(q_alpha_val.detach().cpu().item()) if q_alpha_val.ndim == 0 else float("nan"),
            device=vector_samples.device,
            dtype=vector_samples.dtype,
        ),
        "vector_samples": vector_samples,
    }

_MEAN_FIELD_MODEL = None

def set_mean_field_model(model):
    global _MEAN_FIELD_MODEL
    _MEAN_FIELD_MODEL = model

def meanfield_barrier_margin(cbf, state, obstacle, action, q_alpha, q_alpha_scale=1.0):
    h, gradients = barrier_values_and_gradients(cbf, state, obstacle)
    if _MEAN_FIELD_MODEL is None:
        raise RuntimeError("meanfield credible mode requires set_mean_field_model(...) first")

    _clamp = float(os.environ.get("DSH_Q_ALPHA_CLAMP", "0") or 0)
    if _clamp > 0.0 and (q_alpha is None or float(q_alpha) > _clamp):
        q_alpha = _clamp
    with torch.no_grad():
        f_mean = _MEAN_FIELD_MODEL(state.detach(), action.detach())
    lie_mean = (gradients * f_mean.unsqueeze(1)).sum(dim=2)

    if q_alpha is None:
        q_alpha = 0.3646
    radius = float(q_alpha) * float(q_alpha_scale)
    norm_grad = gradients.norm(dim=2)
    rho = lie_mean + h - radius * norm_grad
    return {
        "rho": rho,
        "h": h,
        "worst_lie": lie_mean - radius * norm_grad,
        "c_alpha": f_mean.unsqueeze(1).expand_as(gradients),
        "P_alpha": torch.zeros_like(gradients),
        "mean_vector": f_mean,
        "covariance_quad": (radius * norm_grad),
        "vector_samples": f_mean.unsqueeze(1),
        "grad_b": gradients,
        "lie_mean": lie_mean,
    }

def gradient_lipschitz_constant(cbf):
    bound = 1.0
    for module in cbf.modules():
        weight = getattr(module, "weight", None)
        if weight is None or weight.dim() < 2:
            continue
        w = weight.detach().reshape(weight.shape[0], -1)
        bound *= float(torch.linalg.matrix_norm(w, ord=2))
    return bound

def lipschitz_barrier_margin(cbf, state, obstacle, action, q_alpha, q_alpha_scale=1.0,
                             lipschitz=None, floor=1e-3):
    if _MEAN_FIELD_MODEL is None:
        raise RuntimeError("lipschitz credible mode requires set_mean_field_model(...) first")
    clamp = float(os.environ.get("DSH_Q_ALPHA_CLAMP", "0") or 0)
    if clamp > 0.0 and (q_alpha is None or float(q_alpha) > clamp):
        q_alpha = clamp
    if q_alpha is None:
        q_alpha = 0.3646
    radius = float(q_alpha) * float(q_alpha_scale)

    h, gradients = barrier_values_and_gradients(cbf, state, obstacle)
    with torch.no_grad():
        f_mean = _MEAN_FIELD_MODEL(state.detach(), action.detach())
    lie_mean = (gradients * f_mean.unsqueeze(1)).sum(dim=2)
    norm_grad = gradients.norm(dim=2)
    base = lie_mean + h

    L = float(lipschitz) if lipschitz is not None else gradient_lipschitz_constant(cbf)
    if L <= floor:
        q_star = torch.full_like(base, radius)
    else:
        disc = torch.clamp(norm_grad ** 2 + 4.0 * L * base, min=0.0)
        q_star = (-norm_grad + torch.sqrt(disc)) / (2.0 * L)
        q_star = torch.where(base < 0.0, torch.zeros_like(q_star), q_star)
        q_star = torch.clamp(q_star, min=0.0, max=radius)
    q_used = q_star.min(dim=1, keepdim=True).values.expand_as(norm_grad)
    rho = lie_mean + h - q_used * (norm_grad + L * q_used)
    return {
        "rho": rho,
        "h": h,
        "worst_lie": lie_mean - q_used * (norm_grad + L * q_used),
        "c_alpha": f_mean.unsqueeze(1).expand_as(gradients),
        "P_alpha": torch.zeros_like(gradients),
        "mean_vector": f_mean,
        "covariance_quad": q_used * (norm_grad + L * q_used),
        "vector_samples": f_mean.unsqueeze(1),
        "grad_b": gradients,
        "lie_mean": lie_mean,
        "q_used": q_used,
        "q_star": q_star,
        "lipschitz": torch.full_like(norm_grad, L),
    }

def enclosure_radius(state, ref_states, lip_const, eps, q_alpha):
    if ref_states is None or lip_const is None:
        return torch.full((state.shape[0], 1), float(q_alpha), dtype=state.dtype)
    d = torch.cdist(state.detach(), ref_states.detach()).min(dim=1, keepdim=True).values
    q_eff = lip_const * d + float(eps)
    return torch.clamp(q_eff, min=float(q_alpha))

def enclosure_barrier_margin(cbf, state, obstacle, action, q_alpha, q_alpha_scale=1.0,
                             lip_const=None, eps=None):
    if _MEAN_FIELD_MODEL is None:
        raise RuntimeError("enclosure credible mode requires set_mean_field_model(...) first")
    clamp = float(os.environ.get("DSH_Q_ALPHA_CLAMP", "0") or 0)
    if clamp > 0.0 and (q_alpha is None or float(q_alpha) > clamp):
        q_alpha = clamp
    if q_alpha is None:
        q_alpha = 0.3646
    refs = _ENCLOSURE_REFERENCES
    lip = float(os.environ.get("DSH_LIPSCHITZ_F", "0") or 0) if lip_const is None else float(lip_const)
    err = float(os.environ.get("DSH_OBS_EPS", "0") or 0) if eps is None else float(eps)

    h, gradients = barrier_values_and_gradients(cbf, state, obstacle)
    with torch.no_grad():
        f_mean = _MEAN_FIELD_MODEL(state.detach(), action.detach())
    lie_mean = (gradients * f_mean.unsqueeze(1)).sum(dim=2)
    norm_grad = gradients.norm(dim=2)
    radius = enclosure_radius(state, refs, lip, err, float(q_alpha) * float(q_alpha_scale))
    rho = lie_mean + h - radius * norm_grad
    return {
        "rho": rho,
        "h": h,
        "worst_lie": lie_mean - radius * norm_grad,
        "c_alpha": f_mean.unsqueeze(1).expand_as(gradients),
        "P_alpha": torch.zeros_like(gradients),
        "mean_vector": f_mean,
        "covariance_quad": radius * norm_grad,
        "vector_samples": f_mean.unsqueeze(1),
        "grad_b": gradients,
        "lie_mean": lie_mean,
        "q_used": radius.expand_as(norm_grad),
    }

def set_enclosure_references(states):
    global _ENCLOSURE_REFERENCES
    _ENCLOSURE_REFERENCES = None if states is None else torch.as_tensor(states, dtype=torch.float32)

def robust_barrier_margin(
        cbf,
        state,
        obstacle,
        action,
        diffusion,
        num_samples=32,
        q_alpha=None,
        q_alpha_scale=1.0,
        vector_samples=None,
        credible_set=None,
        credible_mode="filtered_envelope",
        offset=None,
        eps=1e-8):
    if credible_mode == "meanfield":

        return meanfield_barrier_margin(cbf, state, obstacle, action, q_alpha,
                                        q_alpha_scale)

    if credible_mode == "lipschitz":

        return enclosure_barrier_margin(cbf, state, obstacle, action, q_alpha, q_alpha_scale)

    if q_alpha is None:
        q_alpha = diffusion.q_alpha if diffusion.q_alpha is not None else 1.0
    if not torch.is_tensor(q_alpha):
        q_alpha = torch.tensor(float(q_alpha), dtype=state.dtype, device=state.device)
    q_alpha_val = q_alpha * float(q_alpha_scale)

    h, grad_b = barrier_values_and_gradients(cbf, state, obstacle)

    if credible_set is not None:
        c_alpha = credible_set["c_alpha"]
        P_alpha = credible_set["P_alpha"]
        mean_vector = credible_set.get("mean_vector", c_alpha)
        vector_samples_out = credible_set.get("vector_samples", None)
    else:
        if vector_samples is None:
            with torch.no_grad():
                vector_samples = sample_vectors(state, action, diffusion,
                                                num_samples=num_samples)
        if credible_mode in ("filtered_envelope", "filtered", "mvee"):
            credible_set = construct_credible_dynamics_set(
                diffusion, state, action, num_samples=num_samples,
                q_alpha=q_alpha, q_alpha_scale=q_alpha_scale,
                vector_samples=vector_samples)
        elif credible_mode in ("calibrated_ellipsoid", "calibrated", "gaussian"):
            credible_set = construct_calibrated_ellipsoid_set(
                diffusion, state, action, num_samples=num_samples,
                q_alpha=q_alpha, q_alpha_scale=q_alpha_scale,
                vector_samples=vector_samples)
        else:
            raise ValueError("unknown credible_mode: {}".format(credible_mode))
        c_alpha = credible_set["c_alpha"]
        P_alpha = credible_set["P_alpha"]
        mean_vector = credible_set["mean_vector"]
        vector_samples_out = vector_samples

    if offset is not None:

        c_alpha = c_alpha + offset.to(dtype=c_alpha.dtype)

    lie_mean = torch.einsum("bkn,bn->bk", grad_b, c_alpha)

    P_grad = torch.einsum("bij,bkj->bki", P_alpha, grad_b)
    cov_quad = torch.einsum("bki,bki->bk", grad_b, P_grad)

    worst_lie = lie_mean - torch.sqrt(torch.clamp(cov_quad, min=eps))
    rho = worst_lie + h

    return {
        "rho": rho,
        "h": h,
        "worst_lie": worst_lie,
        "c_alpha": c_alpha,
        "P_alpha": P_alpha,
        "mean_vector": mean_vector,
        "covariance_quad": cov_quad,
        "vector_samples": vector_samples_out,
        "grad_b": grad_b,
        "lie_mean": lie_mean,
        "credible_mode": credible_mode,
    }

def robust_barrier_margin_legacy(
        cbf,
        state,
        obstacle,
        action,
        diffusion,
        num_samples=32,
        q_alpha=None,
        q_alpha_scale=1.0,
        vector_samples=None,
        credible_mode=None,
        eps=1e-8):
    if q_alpha is None:
        q_alpha = diffusion.q_alpha if diffusion.q_alpha is not None else 1.0
    if not torch.is_tensor(q_alpha):
        q_alpha = torch.tensor(float(q_alpha), dtype=state.dtype, device=state.device)
    q_alpha = q_alpha * float(q_alpha_scale)

    h, grad_b = barrier_values_and_gradients(cbf, state, obstacle)
    if vector_samples is None:
        vector_samples = sample_vectors(state, action, diffusion, num_samples=num_samples)

    sample_count = vector_samples.shape[1]
    mean_vector = vector_samples.mean(dim=1)
    centered = vector_samples - mean_vector.unsqueeze(1)

    lie_mean = torch.einsum("bkn,bn->bk", grad_b, mean_vector)
    projected = torch.einsum("bmn,bkn->bkm", centered, grad_b)
    denom = max(sample_count - 1, 1)
    covariance_quad = torch.sum(projected ** 2, dim=2) / denom

    worst_lie = lie_mean - torch.sqrt(torch.clamp(q_alpha * covariance_quad, min=eps))
    rho = worst_lie + h

    return {
        "rho": rho,
        "h": h,
        "worst_lie": worst_lie,
        "mean_vector": mean_vector,
        "covariance_quad": covariance_quad,
        "vector_samples": vector_samples,
        "grad_b": grad_b,
        "lie_mean": lie_mean,
    }

def certificate_condition_masks(
        state,
        obstacle,
        safe_dist=3.0,
        dang_dist=1.0,
        mask_type="ship",
        n_pos=2):
    if mask_type == "ship":
        yaw = state[:, 2:3].unsqueeze(-1)
        yaw_vect = torch.cat([torch.cos(yaw), torch.sin(yaw)], dim=1)
        yaw_vect_perp = torch.cat([-torch.sin(yaw), torch.cos(yaw)], dim=1)

        state_pos = torch.unsqueeze(state, 2)[:, :2]
        obstacle_pos = obstacle.permute(0, 2, 1)[:, :2]
        diff = obstacle_pos - state_pos

        front_dist = torch.abs(torch.sum(diff * yaw_vect, dim=1))
        side_dist = torch.abs(torch.sum(diff * yaw_vect_perp, dim=1))
        dang_mask = torch.logical_and(
            front_dist <= dang_dist * 1.2,
            side_dist <= dang_dist * 0.8,
        )
        safe_mask = torch.logical_or(
            front_dist >= safe_dist * 1.2,
            side_dist >= safe_dist * 0.8,
        )
        return safe_mask, dang_mask

    if mask_type == "distance":
        state_pos = torch.unsqueeze(state, 2)[:, :n_pos]
        obstacle_pos = obstacle.permute(0, 2, 1)[:, :n_pos]
        dist = torch.norm(state_pos - obstacle_pos, dim=1)
        return dist >= safe_dist, dist <= dang_dist

    raise ValueError("unknown mask_type: {}".format(mask_type))

def _masked_min(value, mask):
    filled = value.masked_fill(~mask, float("inf"))
    result = torch.min(filled, dim=1)[0]
    return torch.where(torch.isinf(result), torch.full_like(result, float("nan")), result)

def _masked_max(value, mask):
    filled = value.masked_fill(~mask, -float("inf"))
    result = torch.max(filled, dim=1)[0]
    return torch.where(torch.isinf(result), torch.full_like(result, float("nan")), result)

def finite_sample_conformal_quantile(scores, coverage=0.95):
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("conformal calibration requires at least one score")
    if not 0.0 < float(coverage) < 1.0:
        raise ValueError("coverage must lie strictly between 0 and 1")
    if not np.all(np.isfinite(values)):
        raise ValueError("conformal scores must all be finite")
    rank = int(np.ceil((values.size + 1) * float(coverage)))
    if rank > values.size:
        return float("inf")
    return float(np.partition(values, rank - 1)[rank - 1])

def calibrate_q_alpha(
        diffusion,
        dataset,
        indices,
        coverage=0.95,
        batch_size=256,
        num_samples=32,
        device=None,
        ridge=1e-5,
        group_aggregation="episode_random",
        aggregation_seed=0):
    arrays = dataset.arrays()
    scores = []
    device = torch.device("cpu") if device is None else device

    diffusion.eval()
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        state = torch.from_numpy(arrays["state"][batch_indices]).float().to(device)
        action = torch.from_numpy(arrays["action"][batch_indices]).float().to(device)
        vector = torch.from_numpy(arrays["vector"][batch_indices]).float().to(device)
        with torch.no_grad():
            if hasattr(diffusion, "conditional_moments"):
                mean_vector, covariance = diffusion.conditional_moments(state, action)
            else:
                samples = diffusion.sample_vectors(state, action, num_samples=num_samples)
                mean_vector = samples.mean(dim=1)
                centered = samples - mean_vector.unsqueeze(1)
                covariance = _sample_covariance(centered, ridge=ridge, method="oas")
            residual = (vector - mean_vector).unsqueeze(2)
            solved = torch.linalg.solve(covariance, residual)
            batch_scores = torch.bmm(residual.transpose(1, 2), solved).squeeze()
            scores.append(batch_scores.detach().cpu().numpy().reshape(-1))

    scores = np.concatenate(scores, axis=0).reshape(-1)
    episode_ids = arrays.get("episode_id")
    has_groups = episode_ids is not None and np.all(episode_ids[indices] >= 0)
    if group_aggregation not in ("episode_random", "max", "none"):
        raise ValueError("group_aggregation must be episode_random, max, or none")
    q_episode_max = float("nan")
    if has_groups and group_aggregation in ("episode_random", "max"):
        groups = episode_ids[indices]
        unique_groups = np.unique(groups)
        per_episode_max = np.asarray([
            np.max(scores[groups == group]) for group in unique_groups
        ], dtype=np.float64)
        q_episode_max = finite_sample_conformal_quantile(per_episode_max, coverage=coverage)
        if group_aggregation == "episode_random":
            rng = np.random.RandomState(int(aggregation_seed))
            random_scores = np.asarray([
                scores[np.flatnonzero(groups == group)[rng.randint(np.sum(groups == group))]]
                for group in unique_groups
            ], dtype=np.float64)
            q_alpha = finite_sample_conformal_quantile(random_scores, coverage=coverage)
            score_count = int(len(random_scores))
        else:
            q_alpha = q_episode_max
            score_count = int(len(per_episode_max))
    elif group_aggregation == "none":
        q_alpha = finite_sample_conformal_quantile(scores, coverage=coverage)
        score_count = int(len(scores))
    else:
        raise ValueError("episode-level split conformal requires valid episode IDs")
    diffusion.q_alpha = q_alpha
    diffusion.q_alpha_episode_max = q_episode_max
    n_units = int(len(np.unique(episode_ids[indices]))) if has_groups else int(len(scores))
    diffusion.calibration_metadata = {
        "method": "split_conformal_order_statistic",
        "covariance_estimator": "oracle_approximating_shrinkage_plus_scaled_ridge",
        "coverage_target": float(coverage),
        "aggregation_unit": ("one_random_transition_per_episode" if group_aggregation == "episode_random"
                             else "episode_max" if group_aggregation == "max" else "transition"),
        "calibration_transitions": int(len(indices)),
        "calibration_units": int(n_units),
        "calibration_score_count": int(score_count),
        "finite_sample_q_alpha": float(q_alpha),
        "finite_sample_q_alpha_episode_max": float(q_episode_max),
        "aggregation_seed": int(aggregation_seed),
    }
    return q_alpha

def classify_violation_type(condition1_fail, condition2_fail, condition3_fail):
    violation_type = []
    recommended_action = []
    condition_failure = []
    for c1, c2, c3 in zip(condition1_fail, condition2_fail, condition3_fail):
        failures = []
        if c1:
            failures.append("condition1")
        if c2:
            failures.append("condition2")
        if c3:
            failures.append("condition3")
        condition_failure.append(";".join(failures))

        if c1 or c2:
            violation_type.append("certificate_retrain")
            recommended_action.append("retrain_cbf_controller")
        else:
            violation_type.append("controller_fixable")
            recommended_action.append("repair_residual_controller")

    return (
        np.array(violation_type, dtype=str),
        np.array(recommended_action, dtype=str),
        np.array(condition_failure, dtype=str),
    )

def _empty_violation_output(n_state, k_obstacle, m_control):
    return {
        'state_next': np.zeros((0, n_state), dtype=np.float32),
        'dt': np.zeros((0,), dtype=np.float32),
        "state": np.zeros((0, n_state), dtype=np.float32),
        "obstacle": np.zeros((0, k_obstacle, n_state), dtype=np.float32),
        "action": np.zeros((0, m_control), dtype=np.float32),
        "u_nominal": np.zeros((0, m_control), dtype=np.float32),
        "state_error": np.zeros((0, n_state), dtype=np.float32),
        "rho_min": np.zeros((0,), dtype=np.float32),
        "h_min": np.zeros((0,), dtype=np.float32),
        "safe_h_min": np.zeros((0,), dtype=np.float32),
        "dang_h_max": np.zeros((0,), dtype=np.float32),
        "grad_norm": np.zeros((0,), dtype=np.float32),
        "condition1_fail": np.zeros((0,), dtype=np.float32),
        "condition2_fail": np.zeros((0,), dtype=np.float32),
        "condition3_fail": np.zeros((0,), dtype=np.float32),
        "violation_type": np.array([], dtype=str),
        "recommended_action": np.array([], dtype=str),
        "condition_failure": np.array([], dtype=str),
    }

def find_violations(
        cbf,
        controller,
        diffusion,
        dataset,
        indices=None,
        batch_size=256,
        num_samples=32,
        q_alpha=None,
        device=None,
        q_alpha_scale=1.0,
        credible_mode="filtered_envelope",
        offset_fn=None,
        safe_dist=3.0,
        dang_dist=1.0,
        condition_margin=0.0,
        rho_margin=0.0,
        mask_type="ship",
        n_pos=2):
    arrays = dataset.arrays()
    if indices is None:
        indices = np.arange(arrays["state"].shape[0])
    device = torch.device("cpu") if device is None else device

    violation_batches = {
        'state_next': [],
        'dt': [],
        "state": [],
        "obstacle": [],
        "action": [],
        "u_nominal": [],
        "state_error": [],
        "rho_min": [],
        "h_min": [],
        "safe_h_min": [],
        "dang_h_max": [],
        "grad_norm": [],
        "condition1_fail": [],
        "condition2_fail": [],
        "condition3_fail": [],
        "violation_type": [],
        "recommended_action": [],
        "condition_failure": [],
    }
    rho_values = []
    h_values = []
    condition1_count = 0
    condition2_count = 0
    condition3_count = 0
    condition1_pair_count = 0
    condition1_pair_fail_count = 0
    condition2_pair_count = 0
    condition2_pair_false_safe_count = 0
    retrain_count = 0
    residual_repair_count = 0

    cbf.eval()
    controller.eval()
    diffusion.eval()

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        state = torch.from_numpy(arrays["state"][batch_indices]).float().to(device)
        obstacle = torch.from_numpy(arrays["obstacle"][batch_indices]).float().to(device)
        u_nominal = torch.from_numpy(arrays["u_nominal"][batch_indices]).float().to(device)
        state_error = torch.from_numpy(arrays["state_error"][batch_indices]).float().to(device)
        with torch.no_grad():
            action = controller(state, obstacle, u_nominal, state_error)
        result = robust_barrier_margin(
            cbf,
            state,
            obstacle,
            action,
            diffusion,
            offset=(offset_fn(state, action) if offset_fn is not None else None),
            num_samples=num_samples,
            q_alpha=q_alpha,
            q_alpha_scale=q_alpha_scale,
            credible_mode=credible_mode,
        )
        rho = result["rho"].detach()
        h = result["h"].detach()
        grad_b = result["grad_b"].detach()
        safe_mask, dang_mask = certificate_condition_masks(
            state,
            obstacle,
            safe_dist=safe_dist,
            dang_dist=dang_dist,
            mask_type=mask_type,
            n_pos=n_pos,
        )

        rho_values.append(rho.cpu().numpy())
        h_values.append(h.cpu().numpy())

        condition1_pair = torch.logical_and(safe_mask, h < condition_margin)
        condition2_pair = torch.logical_and(dang_mask, h >= -condition_margin)
        condition3_pair = torch.logical_and(h >= condition_margin, rho < -rho_margin)

        condition1_state = condition1_pair.any(dim=1)
        condition2_state = condition2_pair.any(dim=1)
        condition3_state = condition3_pair.any(dim=1)
        if IGNORE_C1:

            retrain_state = condition2_state
        else:
            retrain_state = torch.logical_or(condition1_state, condition2_state)
        residual_repair_state = torch.logical_and(condition3_state, ~retrain_state)

        condition1_count += int(condition1_state.sum().detach().cpu().item())
        condition2_count += int(condition2_state.sum().detach().cpu().item())
        condition3_count += int(condition3_state.sum().detach().cpu().item())
        condition1_pair_count += int(safe_mask.sum().detach().cpu().item())
        condition1_pair_fail_count += int(condition1_pair.sum().detach().cpu().item())
        condition2_pair_count += int(dang_mask.sum().detach().cpu().item())
        condition2_pair_false_safe_count += int(condition2_pair.sum().detach().cpu().item())
        retrain_count += int(retrain_state.sum().detach().cpu().item())
        residual_repair_count += int(residual_repair_state.sum().detach().cpu().item())

        violation_mask = torch.logical_or(retrain_state, residual_repair_state)
        if torch.any(violation_mask):
            mask_np = violation_mask.cpu().numpy()
            condition1_np = condition1_state.detach().cpu().numpy()[mask_np]
            condition2_np = condition2_state.detach().cpu().numpy()[mask_np]
            condition3_np = condition3_state.detach().cpu().numpy()[mask_np]
            types, actions, failures = classify_violation_type(
                condition1_np, condition2_np, condition3_np)
            grad_norms = torch.norm(grad_b.reshape(grad_b.shape[0], -1), dim=1)

            violation_batches['state_next'].append(
                arrays['state_next'][batch_indices][mask_np])
            violation_batches['dt'].append(arrays['dt'][batch_indices][mask_np])

            violation_batches["state"].append(arrays["state"][batch_indices][mask_np])
            violation_batches["obstacle"].append(arrays["obstacle"][batch_indices][mask_np])
            violation_batches["u_nominal"].append(arrays["u_nominal"][batch_indices][mask_np])
            violation_batches["state_error"].append(arrays["state_error"][batch_indices][mask_np])
            violation_batches["action"].append(action.detach().cpu().numpy()[mask_np])
            violation_batches["rho_min"].append(rho.min(dim=1)[0].cpu().numpy()[mask_np])
            violation_batches["h_min"].append(h.min(dim=1)[0].cpu().numpy()[mask_np])
            violation_batches["safe_h_min"].append(_masked_min(h, safe_mask).cpu().numpy()[mask_np])
            violation_batches["dang_h_max"].append(_masked_max(h, dang_mask).cpu().numpy()[mask_np])
            violation_batches["grad_norm"].append(grad_norms.cpu().numpy()[mask_np])
            violation_batches["condition1_fail"].append(condition1_np.astype(np.float32))
            violation_batches["condition2_fail"].append(condition2_np.astype(np.float32))
            violation_batches["condition3_fail"].append(condition3_np.astype(np.float32))
            violation_batches["violation_type"].append(types)
            violation_batches["recommended_action"].append(actions)
            violation_batches["condition_failure"].append(failures)

    output = _empty_violation_output(
        arrays["state"].shape[1],
        arrays["obstacle"].shape[1],
        arrays["action"].shape[1],
    )
    for key, values in violation_batches.items():
        if len(values) == 0:
            continue
        if key in ("violation_type", "recommended_action", "condition_failure"):
            output[key] = np.concatenate(values, axis=0)
        else:
            output[key] = np.concatenate(values, axis=0).astype(np.float32)

    rho_values = np.concatenate(rho_values, axis=0)
    h_values = np.concatenate(h_values, axis=0)
    num_violations = int(output["state"].shape[0])
    num_fixable = int(np.sum(output["violation_type"] == "controller_fixable"))
    num_cbf_problem = int(np.sum(output["violation_type"] == "certificate_retrain"))
    stats = {
        "num_checked": int(len(indices)),
        "num_violations": num_violations,
        "num_controller_fixable": num_fixable,
        "num_cbf_problem": num_cbf_problem,
        "num_condition1_fail": condition1_count,
        "num_condition2_fail": condition2_count,
        "num_condition3_fail": condition3_count,
        "c1_safe_pair_count": condition1_pair_count,
        "c1_pair_failure_count": condition1_pair_fail_count,
        "c1_pair_failure_rate": (
            float(condition1_pair_fail_count / condition1_pair_count)
            if condition1_pair_count else float("nan")),
        "c2_danger_pair_count": condition2_pair_count,
        "c2_pair_false_safe_count": condition2_pair_false_safe_count,
        "c2_pair_false_safe_rate": (
            float(condition2_pair_false_safe_count / condition2_pair_count)
            if condition2_pair_count else float("nan")),
        "num_retrain_cbf_controller": retrain_count,
        "num_repair_residual_controller": residual_repair_count,
        "rho_mean": float(rho_values.mean()),
        "rho_min": float(rho_values.min()),
        "h_mean": float(h_values.mean()),
        "h_min": float(h_values.min()),
        "barrier_rate": float(np.mean(np.min(h_values, axis=1) >= 0.0)),
        "robust_rate": float(np.mean(np.min(rho_values, axis=1) >= 0.0)),
    }
    return output, stats
