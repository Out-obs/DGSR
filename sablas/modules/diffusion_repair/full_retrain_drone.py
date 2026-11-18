# Shared full-retraining utilities for Drone monitor baselines.

import copy
import numpy as np
import torch as _torch

from modules.dataset import Dataset
from modules.diffusion_repair.full_retrain import (
    ShipFullRetrainTrainer,
    load_transition_arrays,
    add_transition_arrays,
    validate_ship_dataset as validate_dataset,
)

class DroneFullRetrainTrainer(ShipFullRetrainTrainer):

    def get_mask(self, state, obstacle):

        state_pos = _torch.unsqueeze(state, 2)[:, :self.n_pos]
        obstacle_pos = obstacle.permute(0, 2, 1)[:, :self.n_pos]
        dist = _torch.norm(state_pos - obstacle_pos, dim=1)
        safe_mask = (dist >= self.safe_dist).float()
        dang_mask = (dist <= self.dang_dist).float()
        mid_mask = (1.0 - safe_mask) * (1.0 - dang_mask)
        return safe_mask, dang_mask, mid_mask

def build_drone_repair_dataset(
        replay, violations=None, replay_limit=10000,
        violation_repeats=3, hard_negative_repeats=0,
        hard_negative_fraction=0.5,
        seed=0, buffer_size=None, safe_dist=1.0, dang_dist=0.6):
    replay_arrays = load_transition_arrays(replay)
    replay_count = replay_arrays['state'].shape[0]
    replay_indices = np.arange(replay_count)
    rng = np.random.RandomState(seed)
    rng.shuffle(replay_indices)
    if replay_limit > 0:
        replay_indices = replay_indices[:min(replay_limit, replay_count)]

    violation_count = 0
    if violations is not None:
        violations = load_transition_arrays(violations)
        violation_count = violations['state'].shape[0]
    hard_negative_count = 0
    if violation_count > 0 and hard_negative_repeats > 0:
        c2 = np.asarray(violations.get('condition2_fail', np.zeros(violation_count))) > 0
        distance = np.linalg.norm(
            violations['obstacle'][:, :, :3] - violations['state'][:, None, :3], axis=2)
        boundary_score = np.max(
            np.where(distance <= dang_dist, distance / max(float(dang_dist), 1e-8), 0.0),
            axis=1)
        hard_indices = np.flatnonzero(c2)
        if hard_indices.size:
            fraction = float(np.clip(hard_negative_fraction, 0.0, 1.0))
            keep = max(1, int(np.ceil(hard_indices.size * fraction)))
            hard_indices = hard_indices[np.argsort(boundary_score[hard_indices])[-keep:]]
            hard_negative_count = int(hard_indices.size)
    else:
        hard_indices = np.zeros((0,), dtype=np.int64)
    if buffer_size is None:
        buffer_size = max(
            len(replay_indices) + violation_count * violation_repeats
            + hard_negative_count * hard_negative_repeats,
            10000,
        )

    dataset = Dataset(
        n_state=8, k_obstacle=8, m_control=3, n_pos=3,
        buffer_size=buffer_size, safe_dist=safe_dist, dang_dist=dang_dist,
    )
    add_transition_arrays(dataset, replay_arrays, replay_indices)
    if violation_count > 0:
        add_transition_arrays(
            dataset, violations, np.arange(violation_count),
            repeats=violation_repeats,
        )
    if hard_negative_count > 0:
        add_transition_arrays(dataset, violations, hard_indices,
                             repeats=hard_negative_repeats)
    sizes = validate_dataset(dataset)
    return dataset, {
        'replay_count': int(len(replay_indices)),
        'violation_count': int(violation_count),
        'violation_repeats': int(violation_repeats),
        'hard_negative_count': int(hard_negative_count),
        'hard_negative_repeats': int(hard_negative_repeats),
        'hard_negative_fraction': float(hard_negative_fraction),
        'buffer_safe': sizes['safe'],
        'buffer_dangerous': sizes['dangerous'],
        'buffer_middle': sizes['middle'],
    }

def full_retrain_drone(
        cbf, controller, dataset, nominal_dynamics,
        rounds=3,
    monitor_fn=None, opt_iter=100, batch_size=1024,
        eps=0.1, eps_deriv=0.03, eps_action=0.2,
        gpu_id=-1, action_loss_weight=0.02,
        safe_dist=1.0, dang_dist=0.6,
        reference_cbf=None, reference_controller=None,
        cbf_distill_weight=0.0, cbf_distill_danger_weight=1.0,
        controller_distill_weight=0.0,
        controller_only_opt_iter=0, controller_barrier_margin=0.0,
        operational_barrier_weight=0.0, operational_barrier_margin=0.0,
        c2_separation_weight=1.0,
        clearance_weight=0.0, clearance_margin=0.6,
        controller_barrier_weight=2.0, controller_deriv_weight=1.0,
        checkpoint_prefix=None, **kwargs):

    if reference_cbf is None:
        reference_cbf = copy.deepcopy(cbf).eval()
    if reference_controller is None:
        reference_controller = copy.deepcopy(controller).eval()
    trainer = DroneFullRetrainTrainer(
        controller, cbf, dataset, nominal_dynamics,
        n_pos=3, safe_dist=safe_dist, dang_dist=dang_dist,
        action_loss_weight=action_loss_weight, gpu_id=gpu_id,
        reference_cbf=reference_cbf,
        reference_controller=reference_controller,
        cbf_distill_weight=cbf_distill_weight,
        cbf_distill_danger_weight=cbf_distill_danger_weight,
        controller_distill_weight=controller_distill_weight,
        operational_barrier_weight=operational_barrier_weight,
        operational_barrier_margin=operational_barrier_margin,
        c2_separation_weight=c2_separation_weight,
        clearance_weight=clearance_weight,
        clearance_margin=clearance_margin,
    )

    trainer.robust_margin_fn = kwargs.get('robust_margin_fn')
    trainer.robust_margin_weight = float(kwargs.get('robust_margin_weight', 0.0) or 0.0)
    trainer.robust_rho_target = float(kwargs.get('robust_rho_target', 0.0) or 0.0)
    trainer.grad_norm_weight = float(kwargs.get('grad_norm_weight', 0.0) or 0.0)
    controller.train()
    cbf.train()
    history = []
    stop_reason = 'budget'
    for repair_round in range(rounds):

        violations_before = None
        if monitor_fn is not None:
            cbf.eval()
            controller.eval()
            violations_before = int(monitor_fn())
            cbf.train()
            controller.train()
            print('  monitor before round {}: {} violations'.format(
                repair_round + 1, violations_before), flush=True)
            if violations_before == 0:
                stop_reason = 'no_violations'
                print('  stopping: monitor found no violations', flush=True)
                break
        if opt_iter > 0:
            loss, accuracy = trainer.train_cbf_and_controller(
                batch_size=batch_size,
                opt_iter=opt_iter,
                eps=eps,
                eps_deriv=eps_deriv,
                eps_action=eps_action,
            )
        else:

            loss, accuracy = float('nan'), float('nan')
        controller_loss = None
        controller_accuracy = None
        if controller_only_opt_iter > 0:
            controller_loss, controller_accuracy = \
                trainer.train_controller_for_barrier_recovery(
                    batch_size=batch_size,
                    opt_iter=controller_only_opt_iter,
                    eps_deriv=eps_deriv,
                    barrier_margin=controller_barrier_margin,
                    barrier_weight=controller_barrier_weight,
                    deriv_weight=controller_deriv_weight,
                )
        history.append({
            'round': repair_round + 1,
            'violations_before_round': violations_before,
            'loss': float(loss),
            'accuracy': np.asarray(accuracy, dtype=np.float32),
            'controller_loss': (
                None if controller_loss is None else float(controller_loss)),
            'controller_accuracy': (
                None if controller_accuracy is None
                else np.asarray(controller_accuracy, dtype=np.float32)),
        })
        if checkpoint_prefix is not None:
            round_idx = repair_round + 1
            _torch.save(cbf.state_dict(),
                        '{}_round_{:02d}_cbf.pth'.format(checkpoint_prefix, round_idx))
            _torch.save(controller.state_dict(),
                        '{}_round_{:02d}_controller.pth'.format(checkpoint_prefix, round_idx))
    controller.eval()
    cbf.eval()
    if history:
        history[-1]['stop_reason'] = stop_reason
    print('  retrain finished: reason={} rounds_used={}'.format(
        stop_reason, len(history)), flush=True)
    return history
