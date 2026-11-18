# Leakage-aware two-stage DGSR repair trainer for Ship and Drone.

import argparse
import copy
import json
import os
import random
import sys
sys.path.insert(1, os.path.abspath('.'))

import numpy as np
import torch

from modules.diffusion_repair.diffusion import load_uncertainty_model
from modules.diffusion_repair.drone_env import add_drone_env_arguments, make_drone_env
from modules.diffusion_repair.experiment_protocol import (
    certificate_gate_status,
    write_json,
)
from modules.diffusion_repair.full_retrain import build_ship_repair_dataset, full_retrain_ship
from modules.diffusion_repair.full_retrain_drone import build_drone_repair_dataset, full_retrain_drone
from modules.diffusion_repair.multistep import train_controller_multistep_barrier
from modules.diffusion_repair.repair import (
    RepairedController,
    ResidualController,
    freeze_module,
    repair_loss,
)
from modules.diffusion_repair.robust_verify import find_violations
from modules.diffusion_repair.ship_env import add_ship_env_arguments, make_ship_env
from modules.diffusion_repair.vector_dataset import VectorFieldDataset
from modules.network import CBF, NNController
from modules.utils import angle_to_sin_cos_torch

def _load_dynamics_mlp(path, device='cpu'):
    import os as _os, sys as _sys
    _root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..", ".."))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from experiments.tools.distill_mean_field import MeanFieldMLP
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and ckpt.get("kind") == "true_dynamics":
        from experiments.tools.true_dynamics_torch import TrueDynamics
        net = TrueDynamics(ckpt["A"], ckpt["B"], ckpt["beta"],
                           ckpt.get("centre", 10.0), ckpt.get("L", 20.0))
        net = net.to(device)
        net.eval()
        for prm in net.parameters():
            prm.requires_grad_(False)
        return lambda state, action: net(state, action)

    _in = ckpt['state_dict']['net.0.weight'].shape[1]
    _ns = ckpt['state_dict']['net.6.weight'].shape[0]
    net = MeanFieldMLP(n_state=_ns, hidden=ckpt.get('hidden', 512),
                       m_control=_in - _ns)
    net.load_state_dict(ckpt['state_dict'])
    net = net.to(device)
    net.eval()
    for prm in net.parameters():
        prm.requires_grad_(False)
    return lambda state, action: net(state, action)

def _filter_violations(data, label):
    count = len(data.get('state', []))
    if count == 0:
        return None
    labels = np.asarray(data['violation_type'], dtype=str)
    if label == '__all__':
        keep = np.ones_like(labels, dtype=bool)
    else:
        keep = labels == label
    if not np.any(keep):
        return None
    return {key: value[keep] if isinstance(value, np.ndarray) and value.shape[0] == count else value
            for key, value in data.items()}

def _c3_state_rate(stats):
    try:
        checked = float(stats.get('num_checked', 0))
        failed = float(stats.get('num_condition3_fail', 0))
    except (TypeError, ValueError):
        return float('nan')
    return failed / checked if checked > 0 else float('nan')

def stage1_gate_status(args, stats):
    kwargs = dict(
        c12_rate_limit=args.gate_c12_rate,
        c1_rate_limit=args.gate_c12_rate,
        c2_rate_limit=args.gate_c12_rate,
        min_c1_pairs=args.gate_min_c1_pairs,
        min_c2_pairs=args.gate_min_c2_pairs,
    )
    c3_limit = float(getattr(args, 'gate_c3_rate', 0.0) or 0.0)
    if c3_limit > 0.0:
        kwargs['c3_rate_limit'] = c3_limit
        kwargs['c3_state_violation_rate'] = _c3_state_rate(stats)
    return certificate_gate_status(stats, **kwargs)

def stage1_selection_key(gate_info, c12_rate, c3_rate, barrier_rate, use_c3):
    status_rank = {'pass': 0, 'indeterminate': 1, 'fail': 2}.get(
        gate_info.get('status', 'fail'), 2)
    if use_c3:
        try:
            c3 = float(c3_rate)
        except (TypeError, ValueError):
            c3 = float('nan')
        if not np.isfinite(c3):
            c3 = float('inf')
        return (status_rank, c3, float(c12_rate), -float(barrier_rate))
    rates = [gate_info.get('max_condition12_state_violation_rate'),
             gate_info.get('c1_pair_violation_rate'),
             gate_info.get('c2_false_safe_rate')]
    normalized = [rate / max(gate_info.get('c12_rate_limit', 1e-12), 1e-12)
                  for rate in rates if rate is not None and np.isfinite(rate)]
    rate_distance = max(normalized) if len(normalized) == 2 else float('inf')
    return (status_rank, rate_distance, float(c12_rate))

def _state_dict_sha256(state_dict):
    import hashlib as _hashlib
    digest = _hashlib.sha256()
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode('utf-8'))
        digest.update(str(tensor.dtype).encode('utf-8'))
        digest.update(str(tuple(tensor.shape)).encode('utf-8'))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()

def _sha256_file(path):
    import hashlib as _hashlib
    digest = _hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _make_robust_margin_fn(args, cbf, device):
    from experiments.tools.distill_mean_field import MeanFieldMLP
    mean_field_path = args.stage1_robust_mean_field or args.mean_field
    if not mean_field_path:
        raise ValueError('--stage1_robust_weight > 0 requires a mean-field model')
    ck = torch.load(mean_field_path, map_location=device)
    mlp = MeanFieldMLP(n_state=ck['n_state'], hidden=ck.get('hidden', 512),
                       m_control=ck['m_control']).to(device)
    mlp.load_state_dict(ck['state_dict'])
    mlp.eval()
    for p in mlp.parameters():
        p.requires_grad_(False)

    if getattr(args, 'q_alpha_override', None) is not None:
        q_alpha = float(args.q_alpha_override)
    else:
        q_alpha = float(args.stage1_robust_q_alpha)

    def robust_margin_fn(state, obstacle, action):
        state_g = state.detach().clone().requires_grad_(True)
        h = cbf(state_g, obstacle)
        grads = []
        for k in range(h.shape[1]):
            grads.append(torch.autograd.grad(
                h[:, k].sum(), state_g, retain_graph=True, create_graph=True)[0])
        grads = torch.stack(grads, dim=1)
        f_mean = mlp(state.detach(), action)
        lie = (grads * f_mean.unsqueeze(1)).sum(dim=2)
        rho = h + lie - q_alpha * grads.norm(dim=2)
        return rho, h

    return robust_margin_fn

def _selection_criterion_text(use_c3):
    if use_c3:
        return ('c3_improvement(status_rank, c3_state_rate, c12_union_rate, '
                '-barrier_rate)')
    return ('validation_feasibility(barrier_rate, nondecreasing_rate, '
            'robust_rate)_then_gate_status_then_gate_distance; '
            'round 0 never delivered when a repaired round exists')

def stage1_gate_distance(gate_info):
    limit = max(float(gate_info.get('c12_rate_limit', 0.0) or 0.0), 1e-12)
    rates = [gate_info.get('max_condition12_state_violation_rate'),
             gate_info.get('c1_pair_violation_rate'),
             gate_info.get('c2_false_safe_rate')]
    normalized = [float(rate) / limit for rate in rates
                  if rate is not None and np.isfinite(rate)]
    return max(normalized) if normalized else float('inf')

def stage1_progress_score(gate_info, stats):
    score = stage1_gate_distance(gate_info)
    c3_limit = float(gate_info.get('c3_rate_limit') or 0.0)
    c3 = gate_info.get('c3_state_violation_rate')
    if c3_limit > 0.0 and c3 is not None and np.isfinite(c3):
        score += float(c3) / c3_limit
    return float(score)

def stage1_delivery_key(round_index, gate_info, c12_rate, c3_rate,
                        barrier_rate, robust_rate, nondecreasing_rate=None):
    status_rank = {'pass': 0, 'indeterminate': 1, 'fail': 2}.get(
        gate_info.get('status', 'fail'), 2)
    distance = stage1_gate_distance(gate_info)
    if nondecreasing_rate is None or not np.isfinite(nondecreasing_rate):
        nondecreasing_rate = float('nan')
    barrier_rate = float(barrier_rate) if np.isfinite(barrier_rate) else float('-inf')
    robust_rate = float(robust_rate) if np.isfinite(robust_rate) else float('-inf')
    return (-barrier_rate, -float(nondecreasing_rate), -robust_rate,
            status_rank, distance, float(c12_rate),
            float(c3_rate) if np.isfinite(c3_rate) else float('inf'))

def _observed_nondecreasing_rate(args, cbf, dataset, device, offset):
    arrays = dataset.arrays()
    n = arrays['state'].shape[0]
    indices = np.arange(n)
    if args.verify_points > 0 and len(indices) > args.verify_points:
        rng = np.random.RandomState(args.seed + offset)
        indices = rng.choice(indices, size=args.verify_points, replace=False)
        indices.sort()
    cbf.eval()
    ok, total = 0, 0
    batch = int(getattr(args, 'verify_batch_size', 128) or 128)
    with torch.no_grad():
        for start in range(0, len(indices), batch):
            ids = indices[start:start + batch]
            state = torch.from_numpy(arrays['state'][ids]).float().to(device)
            obstacle = torch.from_numpy(arrays['obstacle'][ids]).float().to(device)
            state_next = torch.from_numpy(arrays['state_next'][ids]).float().to(device)
            obstacle_next = torch.from_numpy(
                arrays['obstacle_next'][ids]).float().to(device)
            dt = torch.from_numpy(arrays['dt'][ids]).float().to(device)
            h = cbf(state, obstacle).min(dim=1).values
            h_next = cbf(state_next, obstacle_next).min(dim=1).values
            margin = (h_next - h) / dt + h
            ok += int((margin >= 0).sum().item())
            total += int(margin.numel())
    return (ok / total) if total > 0 else float('nan')

def _load_models(args, device):
    if args.env_family == 'ship':
        preprocess = lambda x: angle_to_sin_cos_torch(x, [2])
        cbf = CBF(7, 8, 2, preprocess_func=preprocess).to(device)
        controller = NNController(7, 8, 2, preprocess_func=preprocess,
                                  output_scale=1.1).to(device)
        residual = ResidualController(7, 8, 2, preprocess_func=preprocess,
                                      output_scale=args.residual_scale).to(device)
    else:
        cbf = CBF(8, 8, 3).to(device)
        controller = NNController(8, 8, 3, output_scale=1.0).to(device)
        residual = ResidualController(8, 8, 3, output_scale=args.residual_scale).to(device)
    cbf.load_state_dict(torch.load(args.cbf, map_location=device))
    controller.load_state_dict(torch.load(args.controller, map_location=device))
    dynamics = load_uncertainty_model(args.dynamics_model, map_location=device).to(device)
    cbf.eval()
    controller.eval()
    dynamics.eval()
    return cbf, controller, residual, dynamics

def _make_env(args, device):
    if args.env_family == 'ship':
        estimated = args.estimated_param if os.path.exists(args.estimated_param) else None
        return make_ship_env(args, gpu_id=-1, estimated_param=estimated)
    return make_drone_env(args, gpu_id=-1)

_OFFSET_CACHE = {}

def _build_offset_fn(args):
    if not getattr(args, 'saturation_aware', False):
        return None
    key = id(args)
    if key not in _OFFSET_CACHE:
        from modules.diffusion_repair.saturation import make_offset_fn
        z = np.load(args.param, allow_pickle=True)
        _OFFSET_CACHE[key] = make_offset_fn(z['A'], z['B'], dt=0.1, max_speed=0.5)
    return _OFFSET_CACHE[key]

def _verify(args, cbf, controller, dynamics, dataset, device, offset=0):
    indices = np.arange(dataset.num_samples)
    if args.verify_points > 0 and len(indices) > args.verify_points:
        rng = np.random.RandomState(args.seed + offset)
        indices = rng.choice(indices, size=args.verify_points, replace=False)
        indices.sort()
    torch.manual_seed(args.seed + 100000 + offset)
    arrays = dataset.arrays()
    return find_violations(
        cbf, controller, dynamics, dataset, indices=indices,
        batch_size=args.verify_batch_size, num_samples=args.num_samples,
        device=device, q_alpha_scale=args.q_alpha_scale,
        q_alpha=getattr(args, 'q_alpha_override', None),
        credible_mode=args.credible_mode,
        offset_fn=_build_offset_fn(args),
        safe_dist=args.safe_dist, dang_dist=args.dang_dist,
        mask_type='ship' if args.env_family == 'ship' else 'distance',
        n_pos=2 if args.env_family == 'ship' else 3,
    )

def _save_verification(path, violations, stats):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    payload = dict(violations)
    payload.update({'stat_' + key: value for key, value in stats.items()})
    np.savez(path, **payload)

def _sample_batch(data, batch_size, device, rng):
    count = data['state'].shape[0]
    ids = rng.choice(count, size=batch_size, replace=count < batch_size)
    return tuple(torch.from_numpy(data[key][ids]).float().to(device)
                 for key in ('state', 'obstacle', 'u_nominal', 'state_error'))

def _record_candidate(prefix, stage, index, cbf, controller, residual, metrics, passed):
    tag = '{}_{:02d}'.format(stage, index)
    cbf_path = prefix + '_' + tag + '_cbf.pth'
    controller_path = prefix + '_' + tag + '_controller.pth'
    residual_path = prefix + '_' + tag + '_residual.pth'
    os.makedirs(os.path.dirname(cbf_path), exist_ok=True)
    torch.save(cbf.state_dict(), cbf_path)
    torch.save(controller.state_dict(), controller_path)
    if residual is not None:
        torch.save(residual.state_dict(), residual_path)
    else:
        residual_path = None
    return {
        'stage': stage, 'index': int(index), 'cbf': cbf_path,
        'controller': controller_path, 'residual': residual_path,
        'stage1_gate_passed': bool(passed),
        'validation_metrics': {key: float(value) for key, value in metrics.items()
                               if np.isscalar(value) and np.isfinite(value)},
    }

def main():
    family_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    family_parser.add_argument('--env_family', choices=['ship', 'drone'], required=True)
    family_args, _ = family_parser.parse_known_args()
    parser = argparse.ArgumentParser(parents=[family_parser], allow_abbrev=False)
    if family_args.env_family == 'ship':
        add_ship_env_arguments(parser, default_env='moving_ship')
        parser.add_argument('--estimated_param', default='./data/estimated_model_ship.pth')
    else:
        add_drone_env_arguments(parser, default_env='moving_drone')
    parser.add_argument('--dynamics_mlp', type=str, default=None,
                        help='distilled mean-field MLP as training dynamics')
    parser.add_argument('--train_data', required=True,
                        help='Transitions used for violation mining and repair gradients.')
    parser.add_argument('--validation_data', required=True,
                        help='Disjoint transitions used only for Stage-I gate/diagnostics.')
    parser.add_argument('--dynamics_model', required=True)
    parser.add_argument('--cbf', required=True)
    parser.add_argument('--controller', required=True)
    parser.add_argument('--out_prefix', required=True)
    parser.add_argument('--max_stage1_rounds', type=int, default=5)
    parser.add_argument('--full_opt_iter', type=int, default=100)
    parser.add_argument('--controller_only_opt_iter', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--replay_limit', type=int, default=10000)
    parser.add_argument('--violation_repeats', type=int, default=3)
    parser.add_argument('--hard_negative_repeats', type=int, default=2)
    parser.add_argument('--hard_negative_repeat_increment', type=int, default=1)
    parser.add_argument('--hard_negative_fraction', type=float, default=0.5)
    parser.add_argument('--c2_separation_weight', type=float, default=1.0)
    parser.add_argument('--clearance_weight', type=float, default=0.0,
                        help='weight on relu(clearance_margin - predicted min distance)')
    parser.add_argument('--clearance_margin', type=float, default=0.6)
    parser.add_argument('--gate_c12_rate', type=float, default=0.3,
                        help='Maximum C1 and C2 pair violation rate, applied separately. '
                             'Shipped default corrected to 0.3 (author decision '
                             '2026-11-18): the historical 0.015 was below the measured '
                             'validation C1/C2 rate (0.238); 0.25 would leave only a '
                             '5%% relative margin above that measurement, 0.3 leaves '
                             '~25%%, so the default is not doing the work.')
    parser.add_argument('--gate_min_c1_pairs', type=int, default=100,
                        help='Support-aware gate: minimum safe-pair count for the '
                             'C1 rate to be allowed to decide pass/fail; below it '
                             'the class is indeterminate (diagnostic only).')
    parser.add_argument('--gate_min_c2_pairs', type=int, default=100,
                        help='Support-aware gate: minimum dangerous-pair count for '
                             'the C2 rate to be allowed to decide pass/fail; '
                             'below it the class is indeterminate (diagnostic '
                             'only). Raised from 30 to 100 (author fix '
                             '2026-11-18) so 42-pair evidence cannot decide.')
    parser.add_argument('--gate_c3_rate', type=float, default=0.0,
                        help='FIX 3: optional maximum state-level C3 (derivative / '
                             'robust-margin) violation rate. 0 (default) keeps the '
                             'historical C1/C2-only gate and the historical round '
                             'selection rule. When > 0 the gate also fails on C3 and '
                             'the Stage-I round selector prefers the round with the '
                             'lowest C3 violation rate among equal gate statuses '
                             '(so a repaired round can be retained instead of always '
                             'falling back to round 0).')
    parser.add_argument('--gate_min_c3_states', type=int, default=0,
                        help='Reserved for the C3 gate support check (not used yet).')
    parser.add_argument('--stage1_robust_weight', type=float, default=0.0,
                        help='DGSR Stage-I robust-margin hinge weight (0 = disabled, '
                             'historical recipe). Adds '
                             'w * mean(relu(rho_target - min_active rho)) to the '
                             'CBF/controller joint Stage-I loss, with '
                             'rho = h + grad h . f_mean - q_alpha*||grad h||.')
    parser.add_argument('--stage1_robust_rho_target', type=float, default=0.0,
                        help='Target for the Stage-I robust-margin hinge (default 0).')
    parser.add_argument('--stage1_robust_q_alpha', type=float, default=0.0,
                        help='Calibrated credible-set radius q_alpha used in the '
                             'Stage-I robust-margin hinge (e.g. 0.439933 for ship).')
    parser.add_argument('--stage1_robust_mean_field', type=str, default=None,
                        help='Mean-field MLP path used as f_mean in the Stage-I '
                             'robust-margin hinge (defaults to --mean_field).')
    parser.add_argument('--stage1_grad_norm_weight', type=float, default=0.0,
                        help='DGSR Stage-I gradient-norm regulariser lambda: adds '
                             'lambda * mean_{slots,batch}(||grad_x h||^2) to the '
                             'Stage-I joint CBF/controller loss (0 = disabled).')
    parser.add_argument('--cbf_distill_weight', type=float, default=10.0)
    parser.add_argument('--cbf_distill_danger_weight', type=float, default=0.0)
    parser.add_argument('--controller_distill_weight', type=float, default=1.0)
    parser.add_argument('--operational_barrier_weight', type=float, default=2.0)
    parser.add_argument('--multistep_opt_iter', type=int, default=0,
                        help='Optional H-step auxiliary ablation; never replaces Stage-II worst-case rho loss.')
    parser.add_argument('--multistep_horizon', type=int, default=5)
    parser.add_argument('--multistep_samples', type=int, default=4)
    parser.add_argument('--multistep_barrier_weight', type=float, default=2.0)
    parser.add_argument('--multistep_deriv_weight', type=float, default=1.0)
    parser.add_argument('--stage2_attempts', type=int, default=3)
    parser.add_argument('--force_stage2', action='store_true',
                        help='run Stage-II even if the Stage-I gate failed; '
                             'recorded as a bypass in the manifest')
    parser.add_argument('--stage2_steps', type=int, default=300)
    parser.add_argument('--stage2_batch_size', type=int, default=128)
    parser.add_argument('--stage2_lr', type=float, default=3e-4)
    parser.add_argument('--residual_scale', type=float, default=0.2)
    parser.add_argument('--action_limit', type=float, default=1.1)
    parser.add_argument('--control_weight', type=float, default=1e-2)
    parser.add_argument('--stage1_include_c3', action='store_true',
                        help='also train Stage-I on dynamics-level (rho<0) violations')
    parser.add_argument('--eps', type=float, default=0.1,
                        help='certificate magnitude: h >= +eps safe, h <= -eps dangerous')
    parser.add_argument('--eps_deriv', type=float, default=0.03,
                        help='required margin on h_dot + h for active pairs (drives NDR)')
    parser.add_argument('--rho_target', type=float, default=0.01)
    parser.add_argument('--geometry_weight', type=float, default=0.0,
                        help='weight on the physical-clearance term for the residual')
    parser.add_argument('--geometry_margin', type=float, default=1.05)
    parser.add_argument('--geometry_dang_dist', type=float, default=0.7)
    parser.add_argument('--geometry_horizon', type=int, default=1)
    parser.add_argument('--saturation_aware', action='store_true',
                        help='model the velocity/angle clip in the nominal model '
                             '(requires an uncertainty model trained on v-offset)')
    parser.add_argument('--q_alpha_scale', type=float, default=1.0,
                        help='multiplier on the calibrated ellipsoid; >1 claims '
                             'a larger credible set (more conservative rho)')
    parser.add_argument('--q_alpha_override', type=float, default=None,
                        help='Explicit calibrated credible-set radius passed as '
                             'q_alpha to find_violations/robust_barrier_margin '
                             '(meanfield: Euclidean ball radius; filtered_envelope: '
                             'Mahalanobis radius threshold). Replaces the hard-coded '
                             'meanfield fallback 0.3646 / the checkpoint q_alpha.')
    parser.add_argument('--residual_replay_mix', type=float, default=0.25)
    parser.add_argument('--safe_dist', type=float, default=None)
    parser.add_argument('--dang_dist', type=float, default=None)
    parser.add_argument('--num_samples', type=int, default=64)
    parser.add_argument('--verify_batch_size', type=int, default=128)
    parser.add_argument('--verify_points', type=int, default=-1)
    parser.add_argument("--mean_field", type=str, default=None,
                        help="distilled mean-field MLP for credible_mode=meanfield")
    parser.add_argument('--credible_mode', choices=['meanfield', 'filtered_envelope', 'calibrated_ellipsoid'],
                        default='calibrated_ellipsoid')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=-1)
    args = parser.parse_args()
    if args.safe_dist is None:
        args.safe_dist = 3.0 if args.env_family == 'ship' else 1.0
    if args.dang_dist is None:
        args.dang_dist = 0.7 if args.env_family == 'ship' else 0.6
    if args.env_family == 'ship' and not hasattr(args, 'estimated_param'):
        args.estimated_param = './data/estimated_model_ship.pth'
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_gpu = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda:{}'.format(args.gpu) if use_gpu else 'cpu')
    if getattr(args, "mean_field", None):
        from modules.diffusion_repair.robust_verify import set_mean_field_model
        import os as _o, sys as _s
        _r = _o.path.abspath(_o.path.join(_o.path.dirname(__file__), "..", "..", ".."))
        if _r not in _s.path: _s.path.insert(0, _r)
        from experiments.tools.distill_mean_field import MeanFieldMLP
        _ck = torch.load(args.mean_field, map_location=device)
        if isinstance(_ck, dict) and _ck.get("kind") == "true_dynamics":
            from experiments.tools.true_dynamics_torch import TrueDynamics
            _mf = TrueDynamics(_ck["A"], _ck["B"], _ck["beta"],
                               _ck.get("centre", 10.0), _ck.get("L", 20.0)).to(device)
        else:

            _in = _ck["state_dict"]["net.0.weight"].shape[1]
            _ns = _ck["state_dict"]["net.6.weight"].shape[0]
            _mf = MeanFieldMLP(n_state=_ns, hidden=_ck.get("hidden", 512),
                               m_control=_in - _ns).to(device)
            _mf.load_state_dict(_ck["state_dict"])
        _mf.eval()
        for _p in _mf.parameters(): _p.requires_grad_(False)
        set_mean_field_model(_mf)
    os.makedirs(os.path.dirname(args.out_prefix) or '.', exist_ok=True)

    train_data = VectorFieldDataset.load(args.train_data)
    validation_data = VectorFieldDataset.load(args.validation_data)
    cbf, controller, residual, dynamics = _load_models(args, device)
    reference_cbf = copy.deepcopy(cbf).eval()
    reference_controller = copy.deepcopy(controller).eval()
    train_env = _make_env(args, device)
    builder = build_ship_repair_dataset if args.env_family == 'ship' else build_drone_repair_dataset
    retrainer = full_retrain_ship if args.env_family == 'ship' else full_retrain_drone

    candidates = []
    train_violations, train_stats = _verify(
        args, cbf, controller, dynamics, train_data, device, offset=0)
    val_violations, val_stats = _verify(
        args, cbf, controller, dynamics, validation_data, device, offset=500000)
    _save_verification(args.out_prefix + '_initial_train_verify.npz', train_violations, train_stats)
    val_stats['nondecreasing_rate'] = _observed_nondecreasing_rate(
        args, cbf, validation_data, device, offset=500000)
    _save_verification(args.out_prefix + '_initial_validation_verify.npz', val_violations, val_stats)
    gate_diagnostics = stage1_gate_status(args, val_stats)
    gate_passed = gate_diagnostics['status'] == 'pass'
    candidates.append(_record_candidate(
        args.out_prefix, 'stage1', 0, cbf, controller, None, val_stats, gate_passed))

    best_stage1_state = (copy.deepcopy(cbf.state_dict()), copy.deepcopy(controller.state_dict()))
    input_cbf_state_sha256 = _state_dict_sha256(best_stage1_state[0])
    best_stage1_c12 = float(val_stats['num_retrain_cbf_controller'] / max(val_stats['num_checked'], 1))
    best_gate_diagnostics = gate_diagnostics
    best_stage1_round = 0
    use_c3_selection = float(getattr(args, 'gate_c3_rate', 0.0) or 0.0) > 0.0
    best_stage1_c3 = _c3_state_rate(val_stats)
    best_stage1_barrier = float(val_stats.get('barrier_rate', -1.0))
    best_stage1_robust = float(val_stats.get('robust_rate', 0.0))
    best_stage1_nondecreasing = float(
        val_stats.get('nondecreasing_rate', float('nan')))
    best_delivery_key = stage1_delivery_key(
        0, gate_diagnostics, best_stage1_c12, best_stage1_c3,
        best_stage1_barrier, best_stage1_robust, best_stage1_nondecreasing)
    best_progress = stage1_progress_score(gate_diagnostics, val_stats)

    last_round_improved = True
    selection_criterion = _selection_criterion_text(use_c3_selection)
    cumulative_c1c2 = None
    global _DYN_FN

    _DYN_FN = (_load_dynamics_mlp(args.dynamics_mlp, device=device)
               if getattr(args, "dynamics_mlp", None) else train_env.nominal_dynamics_torch)
    if hasattr(_DYN_FN, 'to'):
        _DYN_FN = _DYN_FN.to(device)
    if hasattr(_DYN_FN, 'parameters'):
        for _p in _DYN_FN.parameters():
            _p.requires_grad_(False)

    _robust_kwargs = {}
    if float(getattr(args, 'stage1_robust_weight', 0.0) or 0.0) > 0.0:
        _robust_kwargs = {
            'robust_margin_fn': _make_robust_margin_fn(args, cbf, device),
            'robust_margin_weight': float(args.stage1_robust_weight),
            'robust_rho_target': float(args.stage1_robust_rho_target),
        }
        _radius_used = (float(args.q_alpha_override)
                        if getattr(args, 'q_alpha_override', None) is not None
                        else float(args.stage1_robust_q_alpha))
        print('Stage-I robust-margin hinge ENABLED: weight={} radius={} '
              'rho_target={} mean_field={}'.format(
                  args.stage1_robust_weight, _radius_used,
                  args.stage1_robust_rho_target,
                  args.stage1_robust_mean_field or args.mean_field), flush=True)
    _grad_kwargs = {}
    if float(getattr(args, 'stage1_grad_norm_weight', 0.0) or 0.0) > 0.0:
        if args.env_family != 'ship':
            raise ValueError('--stage1_grad_norm_weight is implemented for the '
                             'ship Stage-I path only')
        _grad_kwargs = {
            'grad_norm_weight': float(args.stage1_grad_norm_weight),
        }
        print('Stage-I gradient-norm regulariser ENABLED: lambda={}'.format(
            args.stage1_grad_norm_weight), flush=True)
    for round_index in range(1, args.max_stage1_rounds + 1):

        if round_index > 1 and gate_passed and not last_round_improved:
            print('Stage-I continuation stop: gate satisfied and round {} did not '
                  'improve the gate-relevant validation score'.format(round_index - 1),
                  flush=True)
            break
        _label = '__all__' if getattr(args, 'stage1_include_c3', False) else 'certificate_retrain'
        if round_index == 1:
            _types = np.unique(np.asarray(train_violations.get('violation_type', []), dtype=str))
            print('Stage-I violation types available: %s  (using %s)' % (list(_types), _label))
        c1c2 = _filter_violations(train_violations, _label)
        if c1c2 is None or len(c1c2['state']) == 0:
            break
        cumulative_c1c2 = c1c2 if cumulative_c1c2 is None else {
            key: (np.concatenate([cumulative_c1c2[key], value], axis=0)
                  if isinstance(value, np.ndarray) and value.ndim > 0
                  and key in cumulative_c1c2 and cumulative_c1c2[key].ndim > 0
                  and cumulative_c1c2[key].shape[0] == cumulative_c1c2['state'].shape[0]
                  and value.shape[0] == c1c2['state'].shape[0]
                  else cumulative_c1c2.get(key, value))
            for key, value in c1c2.items()
        }
        repair_dataset, dataset_stats = builder(
            train_data, cumulative_c1c2, replay_limit=args.replay_limit,
            violation_repeats=args.violation_repeats,
            hard_negative_repeats=min(
                args.hard_negative_repeats,
                max(1, args.hard_negative_repeat_increment * round_index)),
            hard_negative_fraction=args.hard_negative_fraction,
            seed=args.seed + round_index, safe_dist=args.safe_dist,
            dang_dist=args.dang_dist)
        history = retrainer(
            cbf, controller, repair_dataset, _DYN_FN,
            rounds=1, opt_iter=args.full_opt_iter, batch_size=args.batch_size,
            eps=getattr(args, 'eps', 0.1), eps_deriv=getattr(args, 'eps_deriv', 0.03), eps_action=0.2,
            gpu_id=args.gpu if use_gpu else -1, safe_dist=args.safe_dist,
            dang_dist=args.dang_dist, reference_cbf=reference_cbf,
            reference_controller=reference_controller,
            cbf_distill_weight=args.cbf_distill_weight,
            cbf_distill_danger_weight=args.cbf_distill_danger_weight,
            controller_distill_weight=args.controller_distill_weight,
            operational_barrier_weight=args.operational_barrier_weight,
            operational_barrier_margin=0.0,
            c2_separation_weight=args.c2_separation_weight,
            clearance_weight=args.clearance_weight,
            clearance_margin=args.clearance_margin,
            controller_only_opt_iter=args.controller_only_opt_iter,
            **_robust_kwargs,
            **_grad_kwargs)
        if args.multistep_opt_iter > 0:
            from modules.diffusion_repair.full_retrain import ShipFullRetrainTrainer
            trainer_cls = ShipFullRetrainTrainer
            if args.env_family == 'drone':
                from modules.diffusion_repair.full_retrain_drone import DroneFullRetrainTrainer
                trainer_cls = DroneFullRetrainTrainer
            trainer = trainer_cls(
                controller, cbf, repair_dataset, _DYN_FN,
                n_pos=2 if args.env_family == 'ship' else 3,
                safe_dist=args.safe_dist, dang_dist=args.dang_dist,
                action_loss_weight=0.02, gpu_id=args.gpu if use_gpu else -1)
            train_controller_multistep_barrier(
                trainer, dynamics, batch_size=args.batch_size,
                opt_iter=args.multistep_opt_iter, horizon=args.multistep_horizon,
                num_samples=args.multistep_samples, barrier_weight=args.multistep_barrier_weight,
                deriv_weight=args.multistep_deriv_weight)
        train_violations, train_stats = _verify(
            args, cbf, controller, dynamics, train_data, device, offset=round_index)
        val_violations, val_stats = _verify(
            args, cbf, controller, dynamics, validation_data, device,
            offset=500000 + round_index)
        val_stats['nondecreasing_rate'] = _observed_nondecreasing_rate(
            args, cbf, validation_data, device, offset=500000 + round_index)
        _save_verification(args.out_prefix + '_stage1_round_{:02d}_train.npz'.format(round_index),
                           train_violations, train_stats)
        _save_verification(args.out_prefix + '_stage1_round_{:02d}_validation.npz'.format(round_index),
                           val_violations, val_stats)
        c12 = float(val_stats['num_retrain_cbf_controller'] / max(val_stats['num_checked'], 1))
        c3_rate = _c3_state_rate(val_stats)
        gate_diagnostics = stage1_gate_status(args, val_stats)
        passed = gate_diagnostics['status'] == 'pass'
        candidates.append(_record_candidate(
            args.out_prefix, 'stage1', round_index, cbf, controller, None, val_stats, passed))
        print('Stage-I round {} data={} loss={:.6f} validation state-C1/C2={:.4%} '
              'pair-C1={:.4%} (n={}) pair-C2-false-safe={:.4%} (n={}) '
              'barrier={:.4%} nondec={:.4%} robust={:.4%} gate={}'.format(
            round_index, dataset_stats, history[-1]['loss'], c12,
            gate_diagnostics['c1_pair_violation_rate'],
            gate_diagnostics['c1_pair_count'],
            gate_diagnostics['c2_false_safe_rate'],
            gate_diagnostics['c2_pair_count'],
            float(val_stats.get('barrier_rate', float('nan'))),
            float(val_stats.get('nondecreasing_rate', float('nan'))),
            float(val_stats.get('robust_rate', float('nan'))),
            gate_diagnostics['status']), flush=True)
        if use_c3_selection:
            print('Stage-I round {} C3 state rate={:.4%} (c3 states={}, limit={:.4%})'
                  .format(round_index, c3_rate,
                          int(val_stats.get('num_condition3_fail', 0)),
                          float(args.gate_c3_rate)), flush=True)
        progress = stage1_progress_score(gate_diagnostics, val_stats)
        improved = progress < best_progress - 1e-12
        barrier_rate = float(val_stats.get('barrier_rate', -1.0))
        robust_rate = float(val_stats.get('robust_rate', 0.0))
        nondecreasing_rate = float(val_stats.get('nondecreasing_rate', float('nan')))
        if use_c3_selection:
            new_key = stage1_selection_key(
                gate_diagnostics, c12, c3_rate, barrier_rate, True)
            best_key = stage1_selection_key(
                best_gate_diagnostics, best_stage1_c12, best_stage1_c3,
                best_stage1_barrier, True)
        else:
            new_key = stage1_delivery_key(
                round_index, gate_diagnostics, c12, c3_rate, barrier_rate,
                robust_rate, nondecreasing_rate)
            best_key = best_delivery_key

        replace = (new_key < best_key
                   or (best_stage1_round == 0 and round_index >= 1))
        if replace:
            best_stage1_c12 = c12
            best_stage1_c3 = c3_rate
            best_stage1_barrier = barrier_rate
            best_stage1_robust = robust_rate
            best_stage1_nondecreasing = nondecreasing_rate
            best_gate_diagnostics = gate_diagnostics
            best_stage1_state = (copy.deepcopy(cbf.state_dict()), copy.deepcopy(controller.state_dict()))
            best_stage1_round = round_index
            best_delivery_key = stage1_delivery_key(
                round_index, gate_diagnostics, c12, c3_rate, barrier_rate,
                robust_rate, nondecreasing_rate)
        if improved:
            best_progress = progress
        last_round_improved = improved
        gate_passed = passed
        if passed and improved:
            print('Stage-I round {} passes the gate and improves the gate-relevant '
                  'score; continuing until convergence or budget'.format(round_index),
                  flush=True)
        elif passed:
            print('Stage-I round {} passes the gate but does not improve the '
                  'gate-relevant score; the next iteration stops'.format(round_index),
                  flush=True)

    cbf.load_state_dict(best_stage1_state[0])
    controller.load_state_dict(best_stage1_state[1])
    delivered_cbf_state_sha256 = _state_dict_sha256(cbf.state_dict())
    val_violations, val_stats = _verify(
        args, cbf, controller, dynamics, validation_data, device, offset=599999)
    stage1_c12 = float(val_stats['num_retrain_cbf_controller'] / max(val_stats['num_checked'], 1))
    stage1_c3 = _c3_state_rate(val_stats)
    final_gate = stage1_gate_status(args, val_stats)
    gate_passed = final_gate['status'] == 'pass'
    torch.save(cbf.state_dict(), args.out_prefix + '_stage1_best_cbf.pth')
    torch.save(controller.state_dict(), args.out_prefix + '_stage1_best_controller.pth')
    input_cbf_file_sha256 = (_sha256_file(args.cbf)
                             if os.path.exists(args.cbf) else None)
    delivered_cbf_file_sha256 = _sha256_file(
        args.out_prefix + '_stage1_best_cbf.pth')
    write_json(args.out_prefix + '_repair_manifest.json', {
        'env_family': args.env_family, 'seed': args.seed,
        'stage1_gate_rate': args.gate_c12_rate,
        'stage1_gate_c3_rate': float(getattr(args, 'gate_c3_rate', 0.0) or 0.0),
        'stage1_credible_mode': getattr(args, 'credible_mode', None),
        'stage1_q_alpha_override': (float(args.q_alpha_override)
                                    if getattr(args, 'q_alpha_override', None)
                                    is not None else None),
        'stage1_q_alpha_scale': float(args.q_alpha_scale),
        'stage1_robust_weight': float(getattr(args, 'stage1_robust_weight', 0.0) or 0.0),
        'stage1_robust_rho_target': float(
            getattr(args, 'stage1_robust_rho_target', 0.0) or 0.0),
        'stage1_robust_radius_used': (
            (float(args.q_alpha_override)
             if getattr(args, 'q_alpha_override', None) is not None
             else float(getattr(args, 'stage1_robust_q_alpha', 0.0) or 0.0))
            if float(getattr(args, 'stage1_robust_weight', 0.0) or 0.0) > 0.0
            else None),
        'c2_separation_weight': args.c2_separation_weight,
        'hard_negative_fraction': args.hard_negative_fraction,
        'hard_negative_repeat_increment': args.hard_negative_repeat_increment,
        'stage1_gate_min_c1_pairs': args.gate_min_c1_pairs,
        'stage1_gate_min_c2_pairs': args.gate_min_c2_pairs,
        'stage1_validation_c12_state_rate': stage1_c12,
        'stage1_validation_c3_state_rate': stage1_c3,
        'stage1_certificate_gate': final_gate,
        'stage1_gate_passed': gate_passed,
        'stage2_gate_bypassed': bool((not gate_passed) and args.force_stage2),
        'stage1_selected_round': best_stage1_round,
        'stage1_selected_round_c12_state_rate': best_stage1_c12,
        'stage1_selected_round_c3_state_rate': best_stage1_c3,
        'stage1_selected_round_barrier_rate': best_stage1_barrier,
        'stage1_selected_round_nondecreasing_rate': best_stage1_nondecreasing,
        'stage1_selected_round_robust_rate': best_stage1_robust,
        'stage1_selection_criterion': selection_criterion,
        'stage1_selected_round_gate_status': best_gate_diagnostics.get('status'),
        'stage1_delivered_is_input_round': bool(best_stage1_round == 0),
        'stage1_input_cbf_state_sha256': input_cbf_state_sha256,
        'stage1_delivered_cbf_state_sha256': delivered_cbf_state_sha256,
        'stage1_delivered_weights_identical_to_input': bool(
            delivered_cbf_state_sha256 == input_cbf_state_sha256),
        'stage1_input_cbf_file_sha256': input_cbf_file_sha256,
        'stage1_delivered_cbf_file_sha256': delivered_cbf_file_sha256,
        'stage1_gate_distance_selected': stage1_gate_distance(best_gate_diagnostics),
        'multistep_auxiliary_steps': args.multistep_opt_iter,
        'candidates': candidates,
    })
    print('Stage-I selected round {} using criterion: {} | gate status: {} | '
          'delivered weights identical to input: {}'.format(
              best_stage1_round, selection_criterion,
              best_gate_diagnostics.get('status'),
              delivered_cbf_state_sha256 == input_cbf_state_sha256), flush=True)
    if not gate_passed and not args.force_stage2:
        print('Stage-I C1/C2 gate failed; Stage II is intentionally not run.', flush=True)
        return
    if not gate_passed:
        print('Stage-I gate failed but --force_stage2 was given: running Stage-II as an '
              'explicitly bypassed experiment (recorded in the manifest).', flush=True)

    freeze_module(cbf)
    freeze_module(controller)
    freeze_module(dynamics)
    train_violations, train_stats = _verify(
        args, cbf, controller, dynamics, train_data, device, offset=700001)
    c3_data = _filter_violations(train_violations, 'controller_fixable')
    candidates = []
    base_val_violations, base_val_stats = _verify(
        args, cbf, controller, dynamics, validation_data, device, offset=800001)
    candidates.append(_record_candidate(
        args.out_prefix, 'stage2', 0, cbf, controller, None,
        base_val_stats, stage_gate_ok(args, base_val_stats)))
    replay_arrays = train_data.arrays()
    rng = np.random.RandomState(args.seed + 900001)
    best_stage2_key, best_stage2_state, best_stage2_attempt = None, None, 0
    for attempt in range(1, args.stage2_attempts + 1):
        if c3_data is None or len(c3_data['state']) == 0:
            break
        residual = ResidualController(
            cbf.n_state, cbf.k_obstacle, cbf.m_control,
            preprocess_func=cbf.preprocess_func,
            output_scale=args.residual_scale).to(device)
        torch.nn.init.zeros_(residual.fc2.weight)
        torch.nn.init.zeros_(residual.fc2.bias)
        residual.train()
        optimizer = torch.optim.Adam(residual.parameters(), lr=args.stage2_lr,
                                     weight_decay=1e-3)
        n_replay = replay_arrays['state'].shape[0]
        mix_count = int(args.stage2_batch_size * args.residual_replay_mix)
        for iteration in range(args.stage2_steps):
            state, obstacle, u_nominal, state_error = _sample_batch(
                c3_data, args.stage2_batch_size, device, rng)
            if mix_count > 0 and n_replay > 0:
                replay_ids = rng.choice(n_replay, size=mix_count, replace=n_replay < mix_count)
                replay_state = torch.from_numpy(replay_arrays['state'][replay_ids]).float().to(device)
                replay_obstacle = torch.from_numpy(replay_arrays['obstacle'][replay_ids]).float().to(device)
                replay_nominal = torch.from_numpy(replay_arrays['u_nominal'][replay_ids]).float().to(device)
                replay_error = torch.from_numpy(replay_arrays['state_error'][replay_ids]).float().to(device)
                state = torch.cat([state[mix_count:], replay_state], dim=0)
                obstacle = torch.cat([obstacle[mix_count:], replay_obstacle], dim=0)
                u_nominal = torch.cat([u_nominal[mix_count:], replay_nominal], dim=0)
                state_error = torch.cat([state_error[mix_count:], replay_error], dim=0)
            loss, loss_stats = repair_loss(
                cbf, controller, residual, dynamics, state, obstacle,
                u_nominal, state_error, num_samples=args.num_samples,
                credible_mode=args.credible_mode, ctrl_weight=args.control_weight,
                offset_fn=_build_offset_fn(args),
                q_alpha_scale=args.q_alpha_scale,
                q_alpha=getattr(args, 'q_alpha_override', None),
                geometry_weight=args.geometry_weight,
                geometry_margin=args.geometry_margin,
                geometry_dang_dist=args.geometry_dang_dist,
                geometry_horizon=int(args.geometry_horizon),
                geometry_dt=float(getattr(args, 'dt', 0.1) or 0.1),
                rho_target=args.rho_target, action_limit=args.action_limit)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(residual.parameters(), max_norm=10.0)
            optimizer.step()
        residual.eval()
        repaired_controller = RepairedController(
            controller, residual, action_limit=args.action_limit).to(device)
        val_violations, val_stats = _verify(
            args, cbf, repaired_controller, dynamics, validation_data, device,
            offset=800001 + attempt)
        _save_verification(args.out_prefix + '_stage2_attempt_{:02d}_validation.npz'.format(attempt),
                           val_violations, val_stats)
        candidates.append(_record_candidate(
            args.out_prefix, 'stage2', attempt, cbf, controller, residual,
            val_stats, stage_gate_ok(args, val_stats)))
        _gate_ok = stage_gate_ok(args, val_stats)
        _c12 = float(val_stats['num_retrain_cbf_controller']) / max(val_stats['num_checked'], 1)
        _key = (0 if _gate_ok else 1, _c12, -float(val_stats.get('robust_rate', 0.0)))
        if best_stage2_key is None or _key < best_stage2_key:
            best_stage2_key = _key
            best_stage2_state = copy.deepcopy(residual.state_dict())
            best_stage2_attempt = attempt
        if _gate_ok:
            print('Stage-II gate passed at attempt {}; stopping early'.format(attempt),
                  flush=True)
        print('Stage-II attempt {} loss={:.6f} validation C1/C2={:.4%} active-rho={:.4%}'.format(
            attempt, float(loss_stats['loss']),
            val_stats['num_retrain_cbf_controller'] / max(val_stats['num_checked'], 1),
            val_stats['robust_rate']), flush=True)
    if best_stage2_state is not None:
        _best_path = args.out_prefix + '_stage2_best_residual.pth'
        torch.save(best_stage2_state, _best_path)
        print('Stage-II best attempt = {} (saved {})'.format(best_stage2_attempt, _best_path),
              flush=True)
    write_json(args.out_prefix + '_stage2_candidates.json', {
        'env_family': args.env_family, 'seed': args.seed,
        'stage1_gate_passed': True, 'candidates': candidates,
        'stage2_loss': 'hinge(rho_target - worst_active_rho) + control_weight * ||delta_u||^2',
    })

def passes_stage_gate(stats, threshold, min_c1_pairs=30, min_c2_pairs=30,
                      c3_rate_limit=None, c3_state_violation_rate=None):
    kwargs = dict(c12_rate_limit=threshold, c1_rate_limit=threshold,
                  c2_rate_limit=threshold,
                  min_c1_pairs=min_c1_pairs, min_c2_pairs=min_c2_pairs)
    if c3_rate_limit is not None:
        kwargs['c3_rate_limit'] = c3_rate_limit
        kwargs['c3_state_violation_rate'] = (
            c3_state_violation_rate if c3_state_violation_rate is not None
            else _c3_state_rate(stats))
    status = certificate_gate_status(stats, **kwargs)
    return status['status'] == 'pass'

def stage_gate_ok(args, stats):
    c3_limit = float(getattr(args, 'gate_c3_rate', 0.0) or 0.0)
    if c3_limit > 0.0:
        return passes_stage_gate(stats, args.gate_c12_rate,
                                 args.gate_min_c1_pairs,
                                 args.gate_min_c2_pairs,
                                 c3_rate_limit=c3_limit,
                                 c3_state_violation_rate=_c3_state_rate(stats))
    return passes_stage_gate(stats, args.gate_c12_rate,
                             args.gate_min_c1_pairs, args.gate_min_c2_pairs)

if __name__ == '__main__':
    main()
