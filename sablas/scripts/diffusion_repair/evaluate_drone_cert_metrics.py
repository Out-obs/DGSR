# Metric evaluator on drone trajectories (SABLAS official benchmarks only).

import argparse
import os
import sys
sys.path.insert(1, os.path.abspath('.'))

import numpy as np
import torch

from envs.env_drone import Drone, MovingDrone
from modules import utils
from modules.network import CBF, NNController
from modules.diffusion_repair.diffusion import load_uncertainty_model
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

def load_drone_models(args, device):
    controller = NNController(n_state=8, k_obstacle=8, m_control=3).to(device)
    cbf = CBF(n_state=8, k_obstacle=8, m_control=3).to(device)
    residual = ResidualController(
        n_state=8, k_obstacle=8, m_control=3,
        output_scale=args.residual_scale,
    ).to(device)
    controller.load_state_dict(torch.load(args.controller, map_location=device))
    reference_controller = NNController(n_state=8, k_obstacle=8, m_control=3).to(device)
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
        diffusion.eval()
    if args.mean_field is not None and os.path.exists(args.mean_field):

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
        _mf.load_state_dict(_ck["state_dict"])
        _mf.eval()
        for _p in _mf.parameters():
            _p.requires_grad_(False)
        set_mean_field_model(_mf)
    controller.eval()
    cbf.eval()
    residual.eval()
    return cbf, controller, residual, reference_controller, use_residual, diffusion

def tensorize(state, obstacle, u_nominal, state_error, device):
    return (
        torch.from_numpy(state.reshape(1, 8).astype(np.float32)).to(device),
        torch.from_numpy(obstacle.reshape(1, 8, 8).astype(np.float32)).to(device),
        torch.from_numpy(u_nominal.reshape(1, 3).astype(np.float32)).to(device),
        torch.from_numpy(state_error.reshape(1, 8).astype(np.float32)).to(device),
    )

def push_into_danger(env, state, target):
    pos = np.asarray(state[:3], dtype=float)
    dist = np.linalg.norm(env.obstacle - pos, axis=1)
    i = int(np.argmin(dist))
    direction = pos - env.obstacle[i]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        direction = np.array([1.0, 0.0, 0.0])
        norm = 1.0
    moved = state.copy()
    moved[:3] = env.obstacle[i] + direction / norm * float(target)
    env.state = moved
    return moved, env.get_obstacle(moved)

_FIXED_LAYOUT = [None]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--episodes", type=int, default=-1)
    parser.add_argument("--max_speed", type=float, default=None,
                        help="override the velocity cap (default keeps env_drone's 0.5)")
    parser.add_argument("--fixed_scene", action="store_true",
                        help="keep the first episode obstacle layout, vary only the trajectory")
    parser.add_argument("--single_episode", action="store_true",
                        help="Stop at the first terminal event; use one common-random-number scenario per seed.")
    parser.add_argument("--cbf", type=str, default="./data/drone_cbf_weights.pth")
    parser.add_argument("--controller", type=str, default="./data/drone_controller_weights.pth")
    parser.add_argument("--reference_controller", type=str, default=None)
    parser.add_argument("--residual", type=str, default=None)
    parser.add_argument("--diffusion", type=str, default=None)
    parser.add_argument("--mean_field", type=str, default=None,
                        help="distilled mean-field net; required for --credible_mode meanfield")
    parser.add_argument("--param", type=str, default="./data/estimated_model_drone.npz")
    parser.add_argument("--out", type=str, default="./data/drone_cert_metrics_5k.npz")
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--residual_scale", type=float, default=0.2)
    parser.add_argument("--action_limit", type=float, default=None,
                        help="Per-dimension clamp applied to the repaired action, "
                             "matching training (train_dgsr_repair.py default 1.1). "
                             "Default None reproduces the historical evaluator, which "
                             "applied base+delta unclamped.")
    parser.add_argument("--q_alpha_scale", type=float, default=1.0)
    parser.add_argument("--credible_mode", type=str, default="filtered_envelope",
                        choices=["filtered_envelope", "calibrated_ellipsoid", "meanfield"])
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nominal_only", action="store_true",
                        help="drive with the env nominal controller to obtain alpha_2, "
                             "the denominator of SABLAS's relative safety rate beta")
    parser.add_argument("--env", type=str, default="drone",
                        choices=["drone", "city"],
                        help="SABLAS official drone benchmarks only: "
                             "'drone' (open field) and 'city' (CityEnv, 1024 NPC). "
                             "The 'moving_drone' subclass was a local addition for "
                             "emulating Yu et al. and is intentionally not offered.")
    parser.add_argument("--npc_speed", type=float, default=0.05)
    parser.add_argument("--total_obstacle", type=int, default=1024)
    parser.add_argument("--safety_filter", type=float, default=None,
                        help="clearance margin enforced at run time by blending the "
                             "commanded action towards the nominal one")
    parser.add_argument("--init_danger", type=float, default=None,
                        help="push each episode start to this distance from the nearest obstacle")
    parser.add_argument("--noise_std", type=float, default=0.1,
                        help="true-dynamics process noise (train-time value is 0.1)")
    parser.add_argument("--preplanned_traj", type=str, default=None,
                        help="cached City trajectories; skips planning")
    parser.add_argument("--env_max_steps", type=int, default=500)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_gpu = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device("cuda:{}".format(args.gpu) if use_gpu else "cpu")
    estimated_param = np.load(open(args.param, "rb"), allow_pickle=True) if os.path.exists(args.param) else None
    if args.env == "city":

        import json as _json
        import tempfile as _tempfile

        from envs.env_drone import City as _City

        _raw = np.array(_json.load(open("data/city_buildings.json")), dtype=float)
        if _raw.ndim == 2 and _raw.shape[1] == 6:
            _raw = _raw.reshape(-1, 2, 3)
        _tf = _tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8")
        _json.dump(_raw.tolist(), _tf)
        _tf.close()
        env = _City(static_obstacle=_tf.name,
                    preplanned_traj=getattr(args, 'preplanned_traj', None),
                    num_npc=args.total_obstacle,
                    npc_speed=args.npc_speed,
                    max_steps=args.env_max_steps,
                    estimated_param=estimated_param)
    elif args.env == "moving_drone":
        env = MovingDrone(npc_speed=args.npc_speed,
                          total_obstacle=args.total_obstacle,
                          max_steps=args.env_max_steps,
                          estimated_param=estimated_param)
    else:
        env = Drone(total_obstacle=args.total_obstacle,
                    max_steps=args.env_max_steps,
                    noise_std=args.noise_std,
                    estimated_param=estimated_param)
    cbf, controller, residual, reference_controller, use_residual, diffusion = load_drone_models(args, device)

    recovery_flags, recovery_steps_list, init_clearance_list = [], [], []
    episode_recovered, episode_recover_step = False, None
    state, obstacle, goal = env.reset()
    if args.init_danger is not None:
        state, obstacle = push_into_danger(env, state, args.init_danger)
        init_clearance_list.append(float(np.min(np.linalg.norm(
            env.obstacle - state[:3], axis=1))))
    state_error = np.zeros((8,), dtype=np.float32)
    states, obstacles, actions, vectors = [], [], [], []
    h_rows, derivative_rows, aligned_derivative_rows, rho_rows, rho_episode_max_rows = [], [], [], [], []
    action_rows = []

    u_nominal_rows, state_error_rows = [], []
    obstacle_assignment_distance = []
    safe_masks, danger_masks, physical_safe = [], [], []
    coverage, coverage_episode_max, log_volume = [], [], []
    action_delta_norm, policy_change_norm, control_energy, action_smoothness = [], [], [], []
    episode_lengths, completed_goal, per_step_episode_id = [], [], []
    current_episode_steps = 0
    episode_index = 0
    previous_action = None

    for step in range(args.steps):
        u_nominal = env.nominal_controller(state, goal).astype(np.float32)
        state_t, obstacle_t, u_nominal_t, state_error_t = tensorize(
            state, obstacle, u_nominal, state_error, device)
        with torch.no_grad():
            base_action = controller(state_t, obstacle_t, u_nominal_t, state_error_t)
            initial_action = reference_controller(state_t, obstacle_t, u_nominal_t, state_error_t)
            delta = residual(state_t, obstacle_t, u_nominal_t, state_error_t) if use_residual else torch.zeros_like(base_action)
            action_t = base_action + delta
            if getattr(args, "nominal_only", False):

                action_t = u_nominal_t
            if args.safety_filter is not None:

                margin = float(args.safety_filter)
                best = None
                for alpha in (1.0, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, 0.0):
                    cand = u_nominal_t + float(alpha) * (action_t - u_nominal_t)
                    s_np = state_t.detach().cpu().numpy().reshape(-1)
                    a_np = cand.detach().cpu().numpy().reshape(-1)
                    dsdt = env.nominal_dynamics(s_np, a_np).reshape(-1)
                    nxt = s_np + dsdt * env.dt
                    clear = float(np.min(np.linalg.norm(
                        env.obstacle[:, :3] - nxt[:3][None, :], axis=1)))
                    if clear >= margin:
                        best = cand
                        break
                if best is not None:
                    action_t = best
            if args.action_limit is not None and float(args.action_limit) > 0:

                action_t = torch.clamp(
                    action_t, min=-float(args.action_limit), max=float(args.action_limit))
            h = cbf(state_t, obstacle_t)
        safe_mask, danger_mask = certificate_condition_masks(
            state_t, obstacle_t, safe_dist=1.0, dang_dist=0.6,
            mask_type="distance", n_pos=3)
        h_rows.append(h.detach().cpu().numpy().reshape(8))
        if args.init_danger is not None and not episode_recovered:
            if bool(np.all(h.detach().cpu().numpy().reshape(8) >= 0.0)):
                episode_recovered = True
                episode_recover_step = current_episode_steps + 1
        safe_masks.append(safe_mask.detach().cpu().numpy().reshape(8))
        danger_masks.append(danger_mask.detach().cpu().numpy().reshape(8))
        states.append(state.copy().astype(np.float32))
        obstacles.append(obstacle.copy().astype(np.float32))
        per_step_episode_id.append(episode_index)
        physical_safe.append(utils.is_safe(state, obstacle, n_pos=3, dang_dist=0.6))
        action = action_t.detach().cpu().numpy().reshape(-1).astype(np.float32)
        actions.append(action)
        action_delta_norm.append(float(np.linalg.norm(action - u_nominal)))
        policy_change_norm.append(float(np.linalg.norm(action - initial_action.detach().cpu().numpy().reshape(-1))))
        control_energy.append(float(np.sum(action ** 2)))
        if previous_action is not None:
            action_smoothness.append(float(np.sum((action - previous_action) ** 2)))
        previous_action = action.copy()

        margin_result = None
        if diffusion is not None:
            if args.credible_mode == "meanfield":

                vector_samples = None
            else:
                with torch.no_grad():
                    vector_samples = diffusion.sample_vectors(
                        state_t, action_t, num_samples=args.num_samples)
            robust_result = robust_barrier_margin(
                cbf, state_t, obstacle_t, action_t.detach(), diffusion,
                num_samples=args.num_samples,
                vector_samples=(None if vector_samples is None else vector_samples.detach()),
                q_alpha_scale=args.q_alpha_scale,
                credible_mode=args.credible_mode)
            rho_rows.append(robust_result["rho"].detach().cpu().numpy().reshape(8))
            q_random = float(getattr(diffusion, "q_alpha", 1.0) or 1.0)
            q_episode_max = getattr(diffusion, "q_alpha_episode_max", None)
            ratio = (float(q_episode_max) / q_random
                     if q_episode_max is not None and np.isfinite(float(q_episode_max))
                     and q_random > 0 else 1.0)
            rho_simultaneous = (
                robust_result["h"] + robust_result["lie_mean"]
                - torch.sqrt(torch.clamp(robust_result["covariance_quad"] * ratio, min=1e-8))
            )
            rho_episode_max_rows.append(rho_simultaneous.detach().cpu().numpy().reshape(8))
            margin_result = robust_result

        state_next, state_nominal_next, obstacle_next, goal_next, done = env.step(action)
        vector = ((state_next - state) / max(float(env.dt), 1e-8)).astype(np.float32)
        vectors.append(vector)
        if margin_result is not None:
            vector_t = torch.from_numpy(vector.reshape(1, 8)).float().to(device)
            if args.credible_mode == "meanfield":

                coverage.append(float("nan"))
            else:
                diff_v = vector_t - margin_result["c_alpha"].detach()
                solved_v = torch.linalg.solve(
                    margin_result["P_alpha"].detach(), diff_v.unsqueeze(-1)).squeeze(-1)
                score_v = torch.sum(diff_v * solved_v, dim=1)
                coverage.append(float((score_v <= 1.0).float().item()))
            q_random = float(getattr(diffusion, "q_alpha", 1.0) or 1.0)
            q_episode_max = getattr(diffusion, "q_alpha_episode_max", None)
            ratio_cov = (float(q_episode_max) / q_random
                         if q_episode_max is not None and np.isfinite(float(q_episode_max))
                         and q_random > 0 else 1.0)
            coverage_episode_max.append(float((score_v <= ratio_cov).float().item()) if args.credible_mode != "meanfield" else float("nan"))
            sign_v, logdet_v = torch.linalg.slogdet(margin_result["P_alpha"].detach())
            log_volume.append(float(0.5 * logdet_v.item())
                              if (args.credible_mode != "meanfield" and sign_v.item() > 0)
                              else float("nan"))

        state_next_t = torch.from_numpy(state_next.reshape(1, 8).astype(np.float32)).to(device)
        obstacle_next_t = torch.from_numpy(obstacle_next.reshape(1, 8, 8).astype(np.float32)).to(device)
        with torch.no_grad():
            h_next = cbf(state_next_t, obstacle_next_t)
        action_rows.append(np.asarray(action, dtype=np.float64).reshape(-1))
        u_nominal_rows.append(np.asarray(u_nominal, dtype=np.float32).reshape(-1))
        state_error_rows.append(
            np.asarray(state_error, dtype=np.float32).reshape(-1))
        derivative_rows.append(((h_next - h.detach()) / env.dt + h.detach()).cpu().numpy().reshape(8))
        aligned_obstacle_next_t, assignment_distance = align_obstacle_slots(
            obstacle_t, obstacle_next_t, n_pos=3)
        with torch.no_grad():
            h_next_aligned = cbf(state_next_t, aligned_obstacle_next_t)
        aligned_derivative_rows.append(
            ((h_next_aligned - h.detach()) / env.dt + h.detach()).cpu().numpy().reshape(8))
        obstacle_assignment_distance.append(float(assignment_distance.mean().cpu().item()))

        current_episode_steps += 1
        state_error = ((state_next - state_nominal_next) / env.dt).astype(np.float32)
        state, obstacle, goal = state_next, obstacle_next, goal_next
        if done:
            episode_lengths.append(current_episode_steps)
            if args.init_danger is not None:
                recovery_flags.append(bool(episode_recovered))
                recovery_steps_list.append(
                    float(episode_recover_step) if episode_recover_step is not None
                    else float("nan"))
                episode_recovered, episode_recover_step = False, None
            completed_goal.append(bool(np.linalg.norm(state[:3] - goal[:3]) < float(env.safe_dist)))
            episode_index += 1
            current_episode_steps = 0
            previous_action = None
            if args.single_episode or (args.episodes > 0 and len(episode_lengths) >= args.episodes):
                break
            state, obstacle, goal = env.reset()
        if getattr(args, 'max_speed', None):
            env.max_speed = float(args.max_speed)
        if args.fixed_scene:
            if _FIXED_LAYOUT[0] is None:
                _FIXED_LAYOUT[0] = env.obstacle.copy()
            else:
                env.obstacle = _FIXED_LAYOUT[0].copy()
                obstacle = env.get_obstacle(state)
            if args.init_danger is not None:
                state, obstacle = push_into_danger(env, state, args.init_danger)
                init_clearance_list.append(float(np.min(np.linalg.norm(
                    env.obstacle - state[:3], axis=1))))
            state_error = np.zeros((8,), dtype=np.float32)

    total = len(h_rows)
    h_array = np.asarray(h_rows, dtype=np.float32)
    derivative_array = np.asarray(derivative_rows, dtype=np.float32)
    aligned_derivative_array = np.asarray(aligned_derivative_rows, dtype=np.float32)
    safe_array = np.asarray(safe_masks, dtype=bool)
    danger_array = np.asarray(danger_masks, dtype=bool)
    rho_array = np.asarray(rho_rows, dtype=np.float32) if diffusion is not None else np.full_like(h_array, np.nan)
    rho_episode_max_array = np.asarray(rho_episode_max_rows, dtype=np.float32) if diffusion is not None else np.full_like(h_array, np.nan)
    vector_array = np.asarray(vectors, dtype=np.float32)
    coverage_array = np.asarray(coverage, dtype=np.float32) if diffusion is not None else np.full((total,), np.nan, dtype=np.float32)
    log_volume_array = np.asarray(log_volume, dtype=np.float32) if diffusion is not None else np.full((total,), np.nan, dtype=np.float32)
    coverage_episode_max_array = np.asarray(coverage_episode_max, dtype=np.float32) if diffusion is not None else np.full((total,), np.nan, dtype=np.float32)

    cert = certificate_metrics(h_array, derivative_array, rho_array,
                               safe_array, danger_array)
    aligned_cert = certificate_metrics(h_array, aligned_derivative_array, rho_array,
                                       safe_array, danger_array)
    episode_max_cert = certificate_metrics(h_array, derivative_array, rho_episode_max_array,
                                           safe_array, danger_array)
    total_safe = int(np.sum(physical_safe))
    state_robust = np.all(rho_array >= 0.0, axis=1) if diffusion is not None else np.zeros(total, dtype=bool)
    coverage_count = int(np.isfinite(coverage_array).sum())
    coverage_success = int(np.nansum(coverage_array)) if coverage_count else 0
    cov_low, cov_high = wilson_interval(coverage_success, coverage_count)
    episode_id_array = np.asarray(per_step_episode_id, dtype=np.int64)
    episode_joint_coverage = []
    episode_max_joint_coverage = []
    if coverage_count and len(episode_id_array) == total:
        for episode_id in np.unique(episode_id_array):
            mask = episode_id_array == episode_id
            episode_joint_coverage.append(bool(np.all(coverage_array[mask] >= 1.0)))
            episode_max_joint_coverage.append(bool(np.all(coverage_episode_max_array[mask] >= 1.0)))
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
        "episodes": int(len(episode_lengths)),
        "completed_episode_count": int(len(episode_lengths)),
        "censored_episode_steps": int(current_episode_steps),
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else float("nan"),
        "goal_rate": float(np.mean(completed_goal)) if completed_goal else float("nan"),
        "recovery_success_rate": (float(np.mean(recovery_flags))
                                  if recovery_flags else float("nan")),
        "recovery_steps_mean": (float(np.nanmean(recovery_steps_list))
                                if recovery_steps_list and
                                np.any(np.isfinite(recovery_steps_list))
                                else float("nan")),
        "init_clearance_mean": (float(np.mean(init_clearance_list))
                                if init_clearance_list else float("nan")),
        "safety_rate": float(total_safe / total) if total else float("nan"),
        "safety_rate_definition": "pre_action_physical_safe_step_fraction",
        "collision_count": int(total - total_safe),
        "collision_count_definition": "unsafe_recorded_pre_action_steps_not_collision_events",
        "robust_rate": float(np.mean(state_robust)) if diffusion is not None and total else float("nan"),
        "robust_episode_max_rate": float(episode_max_cert["robust_rate"]) if diffusion is not None and total else float("nan"),
        "active_robust_episode_max_rate": float(episode_max_cert["active_robust_rate"]) if diffusion is not None and total else float("nan"),
        "rho_episode_max_min": float(np.nanmin(rho_episode_max_array)) if diffusion is not None and total else float("nan"),
        "alert_rate": float(1.0 - np.mean(state_robust)) if diffusion is not None and total else float("nan"),
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
        "credible_mean_log_volume_half": float(np.nanmean(log_volume_array)) if coverage_count else float("nan"),
        "mean_control_delta_norm": float(np.mean(action_delta_norm)) if action_delta_norm else float("nan"),
        "mean_initial_policy_change_norm": float(np.mean(policy_change_norm)) if policy_change_norm else float("nan"),
        "mean_control_energy": float(np.mean(control_energy)) if control_energy else float("nan"),
        "mean_action_smoothness": float(np.mean(action_smoothness)) if action_smoothness else float("nan"),
        "use_residual": int(use_residual),
        "use_diffusion": int(diffusion is not None),
        "seed": int(args.seed),
    }
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    metrics['action_limit'] = np.float32(-1.0 if args.action_limit is None else args.action_limit)
    metrics['action_clamped'] = np.int8(0 if args.action_limit is None else 1)
    np.savez(args.out, **metrics,
             per_step_h=h_array, per_step_derivative=derivative_array,
        per_step_action=np.asarray(action_rows).reshape(-1, 3),
             per_step_u_nominal=np.asarray(u_nominal_rows).reshape(-1, 3),
             per_step_state_error=np.asarray(state_error_rows).reshape(-1, 8),
             per_step_derivative_obstacle_assignment=aligned_derivative_array,
             per_step_obstacle_assignment_distance=np.asarray(obstacle_assignment_distance, dtype=np.float32),
             per_step_state=np.asarray(states, dtype=np.float32),
             per_step_obstacle=np.asarray(obstacles, dtype=np.float32),
             per_step_rho=rho_array, per_step_rho_episode_max=rho_episode_max_array,
             per_step_safe_mask=safe_array,
             per_step_danger_mask=danger_array,
             per_step_physical_safe=np.asarray(physical_safe, dtype=bool),
             per_step_episode_id=episode_id_array,
             per_step_vector=vector_array, per_step_credible_covered=coverage_array,
             per_step_credible_covered_episode_max_q=coverage_episode_max_array,
             episode_lengths=np.asarray(episode_lengths, dtype=np.int32),
             episode_goal_success=np.asarray(completed_goal, dtype=bool))

    _hs = np.asarray(h_rows, dtype=np.float64).min(axis=1)
    _ep = np.asarray(per_step_episode_id)
    _d = np.zeros_like(_hs)
    _d[:-1] = np.diff(_hs)
    _same = _ep[1:] == _ep[:-1]
    _d[1:][~_same] = 0.0
    _viol = (((_d / 0.02) + _hs)[:-1] < 0)
    agent_ndr = float(1.0 - _viol.mean()) if len(_viol) else float("nan")
    metrics["agent_ndr"] = agent_ndr
    print("SR {safety_rate:.4f}, BR {barrier_rate:.4f}, NDR_agent {agent_ndr:.4f}, NDR_all_slot {nondecreasing_rate:.4f}, "
          "active-NDR {active_nondecreasing_rate:.4f}, goal {goal_rate:.4f}, "
          "robust {robust_rate:.4f}, active-robust {active_robust_rate:.4f}, "
          "C1/C2 {condition12_state_violation_rate:.4f}, coverage {credible_empirical_coverage:.4f}".format(**metrics))
    print("Saved metrics to {}".format(args.out))

if __name__ == "__main__":
    main()
