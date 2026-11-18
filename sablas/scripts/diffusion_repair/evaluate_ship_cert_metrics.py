# Metric evaluator on ship trajectories (SABLAS official benchmarks only).

import argparse
import os
import random
import sys
sys.path.insert(1, os.path.abspath('.'))

import numpy as np
import torch

from modules.diffusion_repair.ship_env import (
    add_ship_env_arguments,
    is_safe_ship,
    make_ship_env,
)
from modules import utils
from modules.diffusion_repair.diffusion import load_uncertainty_model
from modules.network import CBF, NNController
from modules.diffusion_repair.repair import ResidualController
from modules.diffusion_repair.robust_verify import (
    robust_barrier_margin,
    certificate_condition_masks,
    align_obstacle_slots,
)
from modules.diffusion_repair.experiment_protocol import (
    certificate_metrics,
    episode_safety_metrics,
    wilson_interval,
)
from modules.utils import angle_to_sin_cos_torch

def load_ship_models(args, device):
    preprocess_func = lambda x: angle_to_sin_cos_torch(x, [2])
    controller = NNController(
        n_state=7,
        k_obstacle=8,
        m_control=2,
        preprocess_func=preprocess_func,
        output_scale=1.1,
    ).to(device)
    cbf = CBF(
        n_state=7,
        k_obstacle=8,
        m_control=2,
        preprocess_func=preprocess_func,
    ).to(device)
    residual = ResidualController(
        n_state=7,
        k_obstacle=8,
        m_control=2,
        preprocess_func=preprocess_func,
        output_scale=args.residual_scale,
    ).to(device)
    controller.load_state_dict(torch.load(args.controller, map_location=device))
    reference_controller = NNController(
        n_state=7, k_obstacle=8, m_control=2,
        preprocess_func=preprocess_func, output_scale=1.1).to(device)
    reference_path = getattr(args, "reference_controller", None)
    if reference_path and os.path.exists(reference_path):
        reference_controller.load_state_dict(torch.load(reference_path, map_location=device))
    else:
        reference_controller.load_state_dict(torch.load(args.controller, map_location=device))
    reference_controller.eval()
    cbf.load_state_dict(torch.load(args.cbf, map_location=device))
    use_residual = args.residual is not None and os.path.exists(args.residual)
    if use_residual:
        residual.load_state_dict(torch.load(args.residual, map_location=device))
    diffusion = None
    if args.diffusion is not None and os.path.exists(args.diffusion):
        diffusion = load_uncertainty_model(args.diffusion, map_location=device).to(device)
    if getattr(args, "mean_field", None) is not None and os.path.exists(args.mean_field):

        import sys as _sys
        _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        if _root not in _sys.path:
            _sys.path.insert(0, _root)
        from experiments.tools.distill_mean_field import MeanFieldMLP
        from modules.diffusion_repair.robust_verify import set_mean_field_model
        _ck = torch.load(args.mean_field, map_location=device)
        _in = _ck["state_dict"]["net.0.weight"].shape[1]
        _ns = _ck["state_dict"]["net.6.weight"].shape[0]
        _mf = MeanFieldMLP(n_state=_ns, hidden=_ck.get("hidden", 512), m_control=_in - _ns).to(device)
        _mf.load_state_dict(_ck["state_dict"]); _mf.eval()
        for _p in _mf.parameters(): _p.requires_grad_(False)
        set_mean_field_model(_ShipMeanFieldBridge(_mf))
        diffusion.eval()
    controller.eval()
    cbf.eval()
    residual.eval()
    return cbf, controller, residual, reference_controller, use_residual, diffusion

def tensorize(state, obstacle, u_nominal, state_error, device):
    return (
        torch.from_numpy(state.reshape(1, 6).astype(np.float32)).to(device),
        torch.from_numpy(obstacle.reshape(1, 8, 6).astype(np.float32)).to(device),
        torch.from_numpy(u_nominal.reshape(1, 2).astype(np.float32)).to(device),
        torch.from_numpy(state_error.reshape(1, 6).astype(np.float32)).to(device),
    )

def cbf_values(cbf, state, obstacle):
    with torch.no_grad():
        return cbf(state, obstacle).detach()

class _ShipMeanFieldBridge(torch.nn.Module):

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, state, action):
        if state.shape[1] == 6:

            return self.inner(state, action)

        cos_t, sin_t = state[:, 2], state[:, 3]
        theta = torch.atan2(sin_t, cos_t)
        raw = torch.stack([state[:, 0], state[:, 1], theta,
                           state[:, 4], state[:, 5], state[:, 6]], dim=1)
        v = self.inner(raw, action)
        d_theta = v[:, 2]
        out = torch.stack([v[:, 0], v[:, 1],
                           -sin_t * d_theta, cos_t * d_theta,
                           v[:, 3], v[:, 4], v[:, 5]], dim=1)
        return out

def main():
    parser = argparse.ArgumentParser()
    add_ship_env_arguments(parser, default_env='moving_ship')
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--episodes", type=int, default=-1)
    parser.add_argument("--fixed_scene", action="store_true",
                        help="same obstacle layout and start every episode: isolates trajectory variation")
    parser.add_argument("--single_episode", action="store_true",
                        help="Stop at the first terminal event; use one common-random-number scenario per seed.")
    parser.add_argument("--cbf", type=str, default="./data/ship_cbf_weights.pth")
    parser.add_argument("--controller", type=str, default="./data/ship_controller_weights.pth")
    parser.add_argument("--reference_controller", type=str, default=None)
    parser.add_argument("--residual", type=str, default=None)
    parser.add_argument("--estimated_param", type=str,
                        default="./data/estimated_model_ship.pth",
                        help="estimated dynamics model used for the nominal path")
    parser.add_argument("--diffusion", type=str, default=None)
    parser.add_argument("--out", type=str, default="./data/ship_cert_metrics.npz")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--robust_batch_size", type=int, default=128)
    parser.add_argument("--residual_scale", type=float, default=0.2)
    parser.add_argument("--action_limit", type=float, default=1.1)
    parser.add_argument("--q_alpha_scale", type=float, default=1.0)
    parser.add_argument("--mean_field", type=str, default=None,
                        help="distilled mean-field net; required for --credible_mode meanfield")
    parser.add_argument("--credible_mode", type=str, default="filtered_envelope",
                        choices=["filtered_envelope", "calibrated_ellipsoid", "meanfield"])
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_gpu = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.gpu) if use_gpu else "cpu")
    estimated_path = getattr(args, "estimated_param", "./data/estimated_model_ship.pth")
    estimated_param = estimated_path if os.path.exists(estimated_path) else None
    env = make_ship_env(args, gpu_id=args.gpu if use_gpu else -1,
                        estimated_param=estimated_param)
    cbf, controller, residual, reference_controller, use_residual, diffusion = load_ship_models(args, device)

    states, obstacles, actions, vectors = [], [], [], []
    h_rows, derivative_rows, aligned_derivative_rows, safe_masks, danger_masks, physical_safe = [], [], [], [], [], []
    obstacle_assignment_distance = []
    rho_episode_max_rows = []
    action_delta_norm, policy_change_norm, control_energy, action_smoothness = [], [], [], []

    residual_norm, action_clamp_norm = [], []
    episode_lengths, completed_goal, per_step_episode_id = [], [], []
    current_episode_steps = 0
    episode_index = 0
    previous_action = None
    state, obstacle, goal = env.reset()

    _scene = (state.copy(), obstacle.copy(), goal.copy()) if args.fixed_scene else None
    state_error = np.zeros((6,), dtype=np.float32)

    for step in range(args.steps):
        u_nominal = env.nominal_controller(state, goal).astype(np.float32)
        state_t, obstacle_t, u_nominal_t, state_error_t = tensorize(
            state, obstacle, u_nominal, state_error, device)
        with torch.no_grad():
            base_action = controller(state_t, obstacle_t, u_nominal_t, state_error_t)
            initial_action = reference_controller(state_t, obstacle_t, u_nominal_t, state_error_t)
            delta = residual(state_t, obstacle_t, u_nominal_t, state_error_t) if use_residual else torch.zeros_like(base_action)
            action_t = base_action + delta
            if use_residual and args.action_limit > 0:
                before_clamp = action_t
                action_t = torch.clamp(action_t, min=-args.action_limit, max=args.action_limit)
                action_clamp_norm.append(float(
                    torch.max(torch.abs(action_t - before_clamp)).item()))
            if use_residual:
                residual_norm.append(float(torch.norm(delta).item()))
            h = cbf(state_t, obstacle_t)

        h_np = h.detach().cpu().numpy().reshape(8)
        safe_mask, danger_mask = certificate_condition_masks(
            state_t, obstacle_t, safe_dist=3.0, dang_dist=0.7,
            mask_type="ship", n_pos=2)
        safe_masks.append(safe_mask.detach().cpu().numpy().reshape(8))
        danger_masks.append(danger_mask.detach().cpu().numpy().reshape(8))
        h_rows.append(h_np)
        action = action_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
        states.append(state.copy().astype(np.float32))
        obstacles.append(obstacle.copy().astype(np.float32))
        per_step_episode_id.append(episode_index)
        actions.append(action)
        physical_safe.append(is_safe_ship(state, obstacle))
        action_delta_norm.append(float(np.linalg.norm(action - u_nominal)))
        policy_change_norm.append(float(np.linalg.norm(action - initial_action.detach().cpu().numpy().reshape(-1))))
        control_energy.append(float(np.sum(action ** 2)))
        if previous_action is not None:
            action_smoothness.append(float(np.sum((action - previous_action) ** 2)))
        previous_action = action.copy()

        state_next, state_nominal_next, obstacle_next, goal_next, done = env.step(action)
        vector = ((state_next - state) / max(float(env.dt), 1e-8)).astype(np.float32)
        vectors.append(vector)
        state_next_t = torch.from_numpy(state_next.reshape(1, 6).astype(np.float32)).to(device)
        obstacle_next_t = torch.from_numpy(obstacle_next.reshape(1, 8, 6).astype(np.float32)).to(device)
        h_next = cbf_values(cbf, state_next_t, obstacle_next_t)
        deriv_cond = (h_next - h.detach()) / env.dt + h.detach()
        derivative_rows.append(deriv_cond.detach().cpu().numpy().reshape(8))
        aligned_obstacle_next_t, assignment_distance = align_obstacle_slots(
            obstacle_t, obstacle_next_t, n_pos=2)
        h_next_aligned = cbf_values(cbf, state_next_t, aligned_obstacle_next_t)
        aligned_derivative_rows.append(
            ((h_next_aligned - h.detach()) / env.dt + h.detach()).detach().cpu().numpy().reshape(8))
        obstacle_assignment_distance.append(float(assignment_distance.mean().cpu().item()))

        current_episode_steps += 1
        state_error = ((state_next - state_nominal_next) / env.dt).astype(np.float32)
        state, obstacle, goal = state_next, obstacle_next, goal_next
        if done:
            episode_lengths.append(current_episode_steps)
            success = bool(np.linalg.norm(state[:2] - goal[:2]) < float(env.safe_dist))
            completed_goal.append(success)
            episode_index += 1
            current_episode_steps = 0
            previous_action = None
            if args.single_episode or (args.episodes > 0 and len(episode_lengths) >= args.episodes):
                break
            state, obstacle, goal = (env.reset() if not args.fixed_scene else
                                     (_scene[0].copy(), _scene[1].copy(), _scene[2].copy()))
            state_error = np.zeros((6,), dtype=np.float32)

    total = len(h_rows)
    h_array = np.asarray(h_rows, dtype=np.float32)
    derivative_array = np.asarray(derivative_rows, dtype=np.float32)
    aligned_derivative_array = np.asarray(aligned_derivative_rows, dtype=np.float32)
    safe_array = np.asarray(safe_masks, dtype=bool)
    danger_array = np.asarray(danger_masks, dtype=bool)
    vector_array = np.asarray(vectors, dtype=np.float32)
    rho_array = np.full_like(h_array, np.nan)
    coverage = np.full((total,), np.nan, dtype=np.float32)
    coverage_episode_max = np.full((total,), np.nan, dtype=np.float32)
    log_volume = np.full((total,), np.nan, dtype=np.float32)
    if diffusion is not None and total:
        state_array = np.asarray(states, dtype=np.float32)
        obstacle_array = np.asarray(obstacles, dtype=np.float32)
        action_array = np.asarray(actions, dtype=np.float32)
        for start_idx in range(0, total, args.robust_batch_size):
            end_idx = min(start_idx + args.robust_batch_size, total)
            state_batch = torch.from_numpy(state_array[start_idx:end_idx]).float().to(device)
            obstacle_batch = torch.from_numpy(obstacle_array[start_idx:end_idx]).float().to(device)
            action_batch = torch.from_numpy(action_array[start_idx:end_idx]).float().to(device)
            vector_batch = torch.from_numpy(vector_array[start_idx:end_idx]).float().to(device)
            with torch.no_grad():
                vector_samples = diffusion.sample_vectors(
                    state_batch, action_batch, num_samples=args.num_samples)
            result = robust_barrier_margin(
                cbf, state_batch, obstacle_batch, action_batch, diffusion,
                num_samples=args.num_samples, vector_samples=vector_samples.detach(),
                q_alpha_scale=args.q_alpha_scale, credible_mode=args.credible_mode)
            rho_array[start_idx:end_idx] = result["rho"].detach().cpu().numpy()
            if args.credible_mode == "meanfield":

                coverage[start_idx:end_idx] = np.nan
                coverage_episode_max[start_idx:end_idx] = np.nan
                log_volume[start_idx:end_idx] = np.nan
                rho_episode_max_rows.append(result["rho"].detach().cpu().numpy())
            else:
                q_random = float(getattr(diffusion, "q_alpha", 1.0) or 1.0)
                q_episode_max = getattr(diffusion, "q_alpha_episode_max", None)
                ratio = (float(q_episode_max) / q_random
                         if q_episode_max is not None and np.isfinite(float(q_episode_max))
                         and q_random > 0 else 1.0)
                rho_simultaneous = (
                    result["h"] + result["lie_mean"]
                    - torch.sqrt(torch.clamp(result["covariance_quad"] * ratio, min=1e-8))
                )
                rho_episode_max_rows.append(rho_simultaneous.detach().cpu().numpy())
                center = result["c_alpha"].detach()
                shape = result["P_alpha"].detach()
                diff = vector_batch - center
                solved = torch.linalg.solve(shape, diff.unsqueeze(-1)).squeeze(-1)
                mahal = torch.sum(diff * solved, dim=1)
                coverage[start_idx:end_idx] = (mahal <= 1.0).float().cpu().numpy()
                coverage_episode_max[start_idx:end_idx] = (mahal <= ratio).float().cpu().numpy()
                sign, logdet = torch.linalg.slogdet(shape)
                log_volume[start_idx:end_idx] = torch.where(
                    sign > 0, 0.5 * logdet, torch.full_like(logdet, float("nan"))
                ).cpu().numpy()

    cert = certificate_metrics(h_array, derivative_array, rho_array,
                               safe_array, danger_array)
    aligned_cert = certificate_metrics(h_array, aligned_derivative_array, rho_array,
                                       safe_array, danger_array)
    rho_episode_max_array = (np.concatenate(rho_episode_max_rows, axis=0)
                             if rho_episode_max_rows else np.full_like(rho_array, np.nan))
    episode_max_cert = certificate_metrics(h_array, derivative_array, rho_episode_max_array,
                                           safe_array, danger_array)
    safe_count = int(np.sum(physical_safe))
    robust_state = np.all(rho_array >= 0.0, axis=1) if diffusion is not None else np.zeros(total, dtype=bool)
    active = h_array >= 0.0
    completed = len(episode_lengths)
    coverage_count = int(np.sum(np.isfinite(coverage)))
    coverage_success = int(np.nansum(coverage)) if coverage_count else 0
    cov_low, cov_high = wilson_interval(coverage_success, coverage_count)
    episode_id_array = np.asarray(per_step_episode_id, dtype=np.int64)
    episode_joint_coverage = []
    episode_max_joint_coverage = []
    if coverage_count and len(episode_id_array) == total:
        for episode_id in np.unique(episode_id_array):
            mask = episode_id_array == episode_id
            episode_joint_coverage.append(bool(np.all(coverage[mask] >= 1.0)))
            episode_max_joint_coverage.append(bool(np.all(coverage_episode_max[mask] >= 1.0)))
    completed_episode_ids = np.arange(len(episode_lengths), dtype=np.int64)
    safety_summary = episode_safety_metrics(
        physical_safe, episode_id_array, completed_episode_ids=completed_episode_ids)
    metrics = {
        **cert,
        "nondecreasing_rate_slot_legacy": cert["nondecreasing_rate"],
        "active_nondecreasing_rate_slot_legacy": cert["active_nondecreasing_rate"],
        "nondecreasing_rate_obstacle_assignment": aligned_cert["nondecreasing_rate"],
        "active_nondecreasing_rate_obstacle_assignment": aligned_cert["active_nondecreasing_rate"],
        "obstacle_assignment_distance_mean": float(np.mean(obstacle_assignment_distance)) if obstacle_assignment_distance else float("nan"),
        "obstacle_derivative_semantics": "nearest_position_assignment_diagnostic_not_ground_truth_identity",
        **safety_summary,
        "steps": int(total),
        "episodes": int(completed),
        "completed_episode_count": int(completed),
        "censored_episode_steps": int(current_episode_steps),
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else float("nan"),
        "goal_rate": float(np.mean(completed_goal)) if completed_goal else float("nan"),
        "safety_rate": float(safe_count / total) if total else float("nan"),
        "safety_rate_definition": "pre_action_physical_safe_step_fraction",
        "collision_count": int(total - safe_count),
        "collision_count_definition": "unsafe_recorded_pre_action_steps_not_collision_events",
        "robust_rate": float(np.mean(robust_state)) if diffusion is not None and total else float("nan"),
        "robust_episode_max_rate": float(episode_max_cert["robust_rate"]) if diffusion is not None and total else float("nan"),
        "active_robust_episode_max_rate": float(episode_max_cert["active_robust_rate"]) if diffusion is not None and total else float("nan"),
        "rho_episode_max_min": float(np.nanmin(rho_episode_max_array)) if diffusion is not None and total else float("nan"),
        "alert_rate": float(1.0 - np.mean(robust_state)) if diffusion is not None and total else float("nan"),
        "rho_mean": float(np.nanmean(rho_array)) if diffusion is not None and total else float("nan"),
        "rho_min": float(np.nanmin(rho_array)) if diffusion is not None and total else float("nan"),
        "credible_empirical_coverage": float(coverage_success / coverage_count) if coverage_count else float("nan"),
        "credible_episode_joint_coverage": float(np.mean(episode_joint_coverage)) if episode_joint_coverage else float("nan"),
        "credible_episode_joint_coverage_count": int(len(episode_joint_coverage)),
        "credible_episode_joint_coverage_at_episode_max_q": float(np.mean(episode_max_joint_coverage)) if episode_max_joint_coverage else float("nan"),
        "credible_transition_coverage_wilson_low_diagnostic": cov_low,
        "credible_transition_coverage_wilson_high_diagnostic": cov_high,
        "credible_coverage_count": coverage_count,
        "credible_episode_coverage_lower_bound_count": int(coverage_success == coverage_count) if coverage_count else 0,
        "credible_mean_log_volume_half": float(np.nanmean(log_volume)) if coverage_count else float("nan"),
        "mean_control_delta_norm": float(np.mean(action_delta_norm)) if action_delta_norm else float("nan"),
        "mean_initial_policy_change_norm": float(np.mean(policy_change_norm)) if policy_change_norm else float("nan"),
        "mean_control_energy": float(np.mean(control_energy)) if control_energy else float("nan"),
        "mean_action_smoothness": float(np.mean(action_smoothness)) if action_smoothness else float("nan"),
        "mean_residual_norm": float(np.mean(residual_norm)) if residual_norm else 0.0,
        "max_residual_norm": float(np.max(residual_norm)) if residual_norm else 0.0,
        "action_clamp_fraction": (float(np.mean(np.asarray(action_clamp_norm) > 1e-9))
                                  if action_clamp_norm else 0.0),
        "mean_action_clamp_norm": (float(np.mean(action_clamp_norm))
                                   if action_clamp_norm else 0.0),
        "use_residual": int(use_residual),
        "use_diffusion": int(diffusion is not None),
        "seed": int(args.seed),
    }
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(args.out, **metrics,
             per_step_h=h_array, per_step_derivative=derivative_array,
             per_step_derivative_obstacle_assignment=aligned_derivative_array,
             per_step_obstacle_assignment_distance=np.asarray(obstacle_assignment_distance, dtype=np.float32),
             per_step_state=np.asarray(states, dtype=np.float32),
             per_step_obstacle=np.asarray(obstacles, dtype=np.float32),
             per_step_rho=rho_array, per_step_rho_episode_max=rho_episode_max_array,
             per_step_safe_mask=safe_array,
             per_step_danger_mask=danger_array,
             per_step_physical_safe=np.asarray(physical_safe, dtype=bool),
             per_step_episode_id=episode_id_array,
             per_step_residual_norm=np.asarray(residual_norm, dtype=np.float32),
             per_step_action_clamp_norm=np.asarray(action_clamp_norm, dtype=np.float32),
             per_step_vector=vector_array, per_step_credible_covered=coverage,
             per_step_credible_covered_episode_max_q=coverage_episode_max,
             episode_lengths=np.asarray(episode_lengths, dtype=np.int32),
             episode_goal_success=np.asarray(completed_goal, dtype=bool))
    print("SR {safety_rate:.4f}, BR {barrier_rate:.4f}, NDR {nondecreasing_rate:.4f}, "
          "active-NDR {active_nondecreasing_rate:.4f}, goal {goal_rate:.4f}, "
          "robust {robust_rate:.4f}, active-robust {active_robust_rate:.4f}, "
          "C1/C2 {condition12_state_violation_rate:.4f}, coverage {credible_empirical_coverage:.4f}".format(**metrics))
    print("Saved metrics to {}".format(args.out))

if __name__ == "__main__":
    main()
