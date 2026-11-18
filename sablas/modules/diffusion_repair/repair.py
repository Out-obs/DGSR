# Residual-controller repair utilities.

import warnings

import torch
from torch import nn
import torch.nn.functional as F

from modules.diffusion_repair.robust_verify import robust_barrier_margin

_FILTERED_GRAD_WARNED = False

class ResidualController(nn.Module):

    def __init__(self, n_state, k_obstacle, m_control, preprocess_func=None, output_scale=0.2):
        super().__init__()
        self.n_state = n_state
        self.k_obstacle = k_obstacle
        self.m_control = m_control
        self.preprocess_func = preprocess_func

        self.conv0 = nn.Conv1d(n_state, 64, 1)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.fc0 = nn.Linear(128 + m_control + n_state, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, m_control)
        self.activation = nn.ReLU()
        self.output_activation = nn.Tanh()
        self.output_scale = output_scale
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, state, obstacle, u_nominal, state_error):
        state_conv = torch.unsqueeze(state, 2)
        obstacle_conv = obstacle.permute(0, 2, 1)
        state_diff = state_conv - obstacle_conv

        if self.preprocess_func is not None:
            state_diff = self.preprocess_func(state_diff)
            state_error = self.preprocess_func(state_error)

        x = self.activation(self.conv0(state_diff))
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x, _ = torch.max(x, dim=2)
        x = torch.cat([x, u_nominal, state_error], dim=1)
        x = self.activation(self.fc0(x))
        x = self.activation(self.fc1(x))
        return self.output_activation(self.fc2(x)) * self.output_scale

class RepairedController(nn.Module):

    def __init__(self, base_controller, residual_controller, action_limit=None):
        super().__init__()
        self.base_controller = base_controller
        self.residual_controller = residual_controller
        self.action_limit = action_limit

    def forward(self, state, obstacle, u_nominal, state_error):
        with torch.no_grad():
            base_action = self.base_controller(state, obstacle, u_nominal, state_error)
        action = base_action + self.residual_controller(
            state, obstacle, u_nominal, state_error)
        if self.action_limit is not None:
            action = torch.clamp(
                action, min=-float(self.action_limit), max=float(self.action_limit))
        return action

def freeze_module(module):
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)

_OPT_B_CACHE = [None]
_OPT_PRINTED = [False]

def repair_loss(
        cbf,
        base_controller,
        residual_controller,
        diffusion,
        state,
        obstacle,
        u_nominal,
        state_error,
        num_samples=32,
        q_alpha=None,
        q_alpha_scale=1.0,
        credible_mode="calibrated_ellipsoid",
        ctrl_weight=1e-2,
        rho_target=0.01,
        sample_seed=None,
        action_limit=None,
        geometry_weight=0.0,
        geometry_margin=1.05,
        geometry_dang_dist=0.7,
        geometry_dt=0.1,
        geometry_horizon=1,
        margin_fn=None,
        offset_fn=None):
    global _FILTERED_GRAD_WARNED
    if credible_mode in ("filtered_envelope", "filtered", "mvee") and not _FILTERED_GRAD_WARNED:
        _FILTERED_GRAD_WARNED = True
        warnings.warn(
            "credible_mode='filtered_envelope' backpropagates through a hard "
            "Mahalanobis mask; its straight-through gradient is discontinuous at "
            "the threshold (see scripts/diffusion_repair/_archive/grad_probe.py). For "
            "repair, prefer credible_mode='calibrated_ellipsoid' (smooth gradient).",
            stacklevel=2,
        )
    with torch.no_grad():
        base_action = base_controller(state, obstacle, u_nominal, state_error)
    delta_action = residual_controller(state, obstacle, u_nominal, state_error)
    repaired_action = base_action + delta_action
    if action_limit is not None:
        repaired_action = torch.clamp(
            repaired_action, min=-float(action_limit), max=float(action_limit))

    if margin_fn is None:
        from modules.diffusion_repair.robust_verify import robust_barrier_margin as margin_fn

    def evaluate_margin():
        return margin_fn(
            cbf,
            state,
            obstacle,
            repaired_action,
            diffusion,
            offset=(offset_fn(state, repaired_action)
                    if offset_fn is not None else None),
            num_samples=num_samples,
            q_alpha=q_alpha,
            q_alpha_scale=q_alpha_scale,
            credible_mode=credible_mode,
        )

    if sample_seed is None:
        result = evaluate_margin()
    else:
        cuda_devices = []
        if state.device.type == "cuda":
            cuda_devices = [state.device.index]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(int(sample_seed))
            if state.device.type == "cuda":
                torch.cuda.manual_seed_all(int(sample_seed))
            result = evaluate_margin()
    geometry_loss = result["rho"].new_zeros(())
    if geometry_weight > 0.0 and state.shape[1] >= 3:
        predicted_state = (
            state + result["mean_vector"]
            * float(geometry_dt) * int(geometry_horizon)
        )
        yaw = predicted_state[:, 2]
        yaw_vector = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=1)
        yaw_perp = torch.stack([-torch.sin(yaw), torch.cos(yaw)], dim=1)
        position_difference = obstacle[:, :, :2] - predicted_state[:, None, :2]
        front_distance = torch.abs(torch.sum(
            position_difference * yaw_vector[:, None, :], dim=2))
        side_distance = torch.abs(torch.sum(
            position_difference * yaw_perp[:, None, :], dim=2))
        normalized_clearance = torch.maximum(
            front_distance / max(float(geometry_dang_dist) * 1.2, 1e-6),
            side_distance / max(float(geometry_dang_dist) * 0.8, 1e-6),
        )
        worst_clearance = torch.min(normalized_clearance, dim=1)[0]
        geometry_loss = torch.mean(F.relu(
            float(geometry_margin) - worst_clearance))

    active_mask = result["h"] >= 0.0
    masked_rho = result["rho"].masked_fill(~active_mask, float("inf"))
    worst_active_rho = torch.min(masked_rho, dim=1)[0]
    has_active = torch.any(active_mask, dim=1)
    worst_state_rho = torch.where(has_active, worst_active_rho, torch.zeros_like(worst_active_rho))
    safety_loss = (
        F.relu(float(rho_target) - worst_state_rho[has_active]).mean()
        if torch.any(has_active) else repaired_action.sum() * 0.0
    )
    control_loss = torch.mean(delta_action ** 2)

    barrier_membership_weight = float(getattr(
        repair_loss, '_barrier_membership_weight',
        float(__import__('os').environ.get('DSH_MEMBERSHIP_WEIGHT', 0.0))))
    barrier_membership_dt = float(getattr(repair_loss, '_barrier_membership_dt', 0.1))

    barrier_membership_margin = float(__import__('os').environ.get('DSH_MEMBERSHIP_MARGIN', 0.0))
    membership_loss = delta_action.new_zeros(())
    if barrier_membership_weight > 0.0:
        mean_v = result["mean_vector"]
        predicted_next = state + mean_v * barrier_membership_dt
        h_pred = cbf(predicted_next, obstacle)
        membership_loss = torch.mean(torch.relu(barrier_membership_margin - h_pred))

    safety_weight = float(__import__('os').environ.get('DSH_SAFETY_WEIGHT', 1.0))

    opt_weight = float(__import__('os').environ.get("DSH_OPT_WEIGHT", 0.0))
    opt_loss = delta_action.new_zeros(())
    if opt_weight > 0.0:
        _param_path = __import__('os').environ.get(
            "DSH_OPT_PARAM", "data/estimated_model_drone.npz")
        if _OPT_B_CACHE[0] is None:
            import numpy as _np
            _OPT_B_CACHE[0] = _np.asarray(_np.load(_param_path)["B"], dtype=_np.float32)
        _B = torch.as_tensor(_OPT_B_CACHE[0], dtype=state.dtype, device=state.device)
        _x = state.detach().clone().requires_grad_(True)
        _h = cbf(_x, obstacle)
        _g = torch.autograd.grad(_h.sum(), _x)[0].detach()
        _dir = torch.sign(_g @ _B)
        _u_star = float(action_limit) * _dir
        _base = (repaired_action - delta_action).detach()
        opt_loss = torch.mean((delta_action - (_u_star - _base)) ** 2)
        if not _OPT_PRINTED[0]:
            _OPT_PRINTED[0] = True
            print("OPT-AUX loss at first call: %.6f (must be > 0)" % float(opt_loss))
    total_loss = (
        safety_weight * safety_loss + ctrl_weight * control_loss
        + opt_weight * opt_loss
        + float(geometry_weight) * geometry_loss
        + barrier_membership_weight * membership_loss
    )
    stats = {
        "loss": float(total_loss.detach().cpu().item()),
        "safety_loss": float(safety_loss.detach().cpu().item()),
        "control_loss": float(control_loss.detach().cpu().item()),
        "geometry_loss": float(geometry_loss.detach().cpu().item()),
        "rho_mean": float(result["rho"].detach().cpu().mean().item()),
        "rho_min": float(result["rho"].detach().cpu().min().item()),
        "worst_state_rho_mean": float(
            worst_state_rho.detach().cpu().mean().item()),
        "delta_norm": float(torch.norm(delta_action, dim=1).detach().cpu().mean().item()),
        "membership_loss": float(membership_loss.detach().cpu().item()),
        "opt_loss": float(opt_loss.detach().cpu().item()),
    }
    return total_loss, stats
