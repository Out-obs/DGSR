# Shared full-retraining utilities for Ship monitor baselines and our method.

import copy
import os

import numpy as np
import torch

from modules.dataset import Dataset
from modules.trainer import Trainer

class ShipFullRetrainTrainer(Trainer):

    def __init__(
            self, *args, reference_cbf=None, reference_controller=None,
            cbf_distill_weight=0.0, cbf_distill_danger_weight=1.0,
            controller_distill_weight=0.0,
            operational_barrier_weight=0.0,
            operational_barrier_margin=0.0,
            c2_separation_weight=1.0,
            clearance_weight=0.0, clearance_margin=0.6,
            use_euclidean=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference_cbf = reference_cbf
        self.reference_controller = reference_controller
        self.cbf_distill_weight = float(cbf_distill_weight)
        self.cbf_distill_danger_weight = float(cbf_distill_danger_weight)
        self.controller_distill_weight = float(controller_distill_weight)
        self.operational_barrier_weight = float(operational_barrier_weight)
        self.operational_barrier_margin = float(operational_barrier_margin)
        self.c2_separation_weight = float(c2_separation_weight)
        self.clearance_weight = float(clearance_weight)
        self.clearance_margin = float(clearance_margin)
        self.use_euclidean = use_euclidean
        for module in [self.reference_cbf, self.reference_controller]:
            if module is not None:
                module.eval()
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    def controller_regularization_loss(
            self, state, obstacle, u_nominal, state_error, u):
        if self.reference_controller is None or self.controller_distill_weight <= 0.0:
            return u.new_zeros(())
        with torch.no_grad():
            reference_u = self.reference_controller(
                state, obstacle, u_nominal, state_error)
        return self.controller_distill_weight * torch.mean(
            (u - reference_u) ** 2)
    def joint_regularization_loss(
            self, state, obstacle, u_nominal, state_error, h, u):
        loss = h.new_zeros(())
        with torch.no_grad():
            if self.reference_cbf is not None:
                reference_h = self.reference_cbf(state, obstacle)
            else:
                reference_h = None
            if self.reference_controller is not None:
                reference_u = self.reference_controller(
                    state, obstacle, u_nominal, state_error)
            else:
                reference_u = None
        if reference_h is not None and self.cbf_distill_weight > 0.0:
            safe_mask, dangerous_mask, middle_mask = self.get_mask(state, obstacle)
            distill_weight = safe_mask + middle_mask + (
                self.cbf_distill_danger_weight * dangerous_mask)
            denominator = distill_weight.sum().clamp_min(1.0)
            distill_error = (h - reference_h) ** 2
            loss = loss + self.cbf_distill_weight * torch.sum(
                distill_error * distill_weight) / denominator
        if reference_u is not None and self.controller_distill_weight > 0.0:
            loss = loss + self.controller_distill_weight * torch.mean(
                (u - reference_u) ** 2)
        if self.operational_barrier_weight > 0.0:
            _, dangerous_mask, _ = self.get_mask(state, obstacle)
            non_dangerous_state = ~torch.any(dangerous_mask > 0.0, dim=1)
            if torch.any(non_dangerous_state):
                worst_h = torch.min(h, dim=1)[0]
                operational_loss = torch.relu(
                    self.operational_barrier_margin
                    - worst_h[non_dangerous_state]
                ).mean()
                loss = loss + self.operational_barrier_weight * operational_loss
        return loss

    def train_controller_for_barrier_recovery(
            self, batch_size=1024, opt_iter=100, eps_deriv=0.03,
            barrier_margin=0.0, barrier_weight=2.0, deriv_weight=1.0):
        loss_total = 0.0
        accuracy_total = np.zeros((2,), dtype=np.float32)
        for _ in range(opt_iter):
            state, obstacle, u_nominal, state_next, state_error = \
                self.dataset.sample_data(batch_size)
            state = torch.from_numpy(state)
            obstacle = torch.from_numpy(obstacle)
            u_nominal = torch.from_numpy(u_nominal)
            state_next = torch.from_numpy(state_next)
            state_error = torch.from_numpy(state_error)
            if self.gpu_id >= 0:
                state = state.cuda(self.gpu_id)
                obstacle = obstacle.cuda(self.gpu_id)
                u_nominal = u_nominal.cuda(self.gpu_id)
                state_next = state_next.cuda(self.gpu_id)
                state_error = state_error.cuda(self.gpu_id)

            h = self.cbf(state, obstacle).detach()
            u = self.controller(state, obstacle, u_nominal, state_error)
            dsdt_nominal = self.nominal_dynamics(state, u)
            state_next_nominal = state + dsdt_nominal * self.dt
            state_next_with_grad = (
                state_next_nominal + (state_next - state_next_nominal).detach())
            h_next = self.cbf(state_next_with_grad, obstacle)
            deriv_cond = (h_next - h) / self.dt + h

            worst_h_next = torch.min(h_next, dim=1)[0]
            worst_deriv = torch.min(deriv_cond, dim=1)[0]
            barrier_loss = torch.relu(barrier_margin - worst_h_next).mean()
            deriv_loss = torch.relu(eps_deriv - worst_deriv).mean()
            action_loss = torch.mean((u - u_nominal) ** 2)
            loss = (
                barrier_weight * barrier_loss
                + deriv_weight * deriv_loss
                + self.action_loss_weight * action_loss
                + self.controller_regularization_loss(
                    state, obstacle, u_nominal, state_error, u)
            )

            self.controller_optimizer.zero_grad()
            loss.backward()
            self.controller_optimizer.step()
            loss_total += float(loss.detach().cpu().item())
            accuracy_total[0] += float(
                (worst_h_next >= 0.0).float().mean().detach().cpu().item())
            accuracy_total[1] += float(
                (worst_deriv >= 0.0).float().mean().detach().cpu().item())

        return loss_total / opt_iter, accuracy_total / opt_iter
    def get_mask(self, state, obstacle):
        if self.use_euclidean:
            return super().get_mask(state, obstacle)
        yaw = state[:, 2:3].unsqueeze(-1)
        yaw_vector = torch.cat([torch.cos(yaw), torch.sin(yaw)], dim=1)
        yaw_perp = torch.cat([-torch.sin(yaw), torch.cos(yaw)], dim=1)

        state_position = torch.unsqueeze(state, 2)[:, :2]
        obstacle_position = obstacle.permute(0, 2, 1)[:, :2]
        difference = obstacle_position - state_position
        front_distance = torch.abs(torch.sum(difference * yaw_vector, dim=1))
        side_distance = torch.abs(torch.sum(difference * yaw_perp, dim=1))

        dangerous_mask = torch.logical_and(
            front_distance <= self.dang_dist * 1.2,
            side_distance <= self.dang_dist * 0.8,
        ).float()
        safe_mask = torch.logical_or(
            front_distance >= self.safe_dist * 1.2,
            side_distance >= self.safe_dist * 0.8,
        ).float()
        middle_mask = (1.0 - safe_mask) * (1.0 - dangerous_mask)
        return safe_mask, dangerous_mask, middle_mask

class ShipRepairDataset(Dataset):

    def __init__(self, *args, use_euclidean=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_euclidean = use_euclidean

    def add_data(self, state, obstacle, u_nominal, state_next, state_error):
        if self.use_euclidean:
            return super().add_data(state, obstacle, u_nominal, state_next, state_error)
        yaw = float(state[2])
        yaw_vector = np.array([np.cos(yaw), np.sin(yaw)])
        yaw_perp = np.array([-np.sin(yaw), np.cos(yaw)])
        difference = obstacle[:, :2] - state[:2]
        front_distance = np.abs(np.sum(difference * yaw_vector, axis=1))
        side_distance = np.abs(np.sum(difference * yaw_perp, axis=1))
        dangerous = np.any(np.logical_and(
            front_distance <= self.dang_dist * 1.2,
            side_distance <= self.dang_dist * 0.8,
        ))
        safe = np.all(np.logical_or(
            front_distance >= self.safe_dist * 1.2,
            side_distance >= self.safe_dist * 0.8,
        ))

        data = [
            np.copy(state).astype(np.float32),
            np.copy(obstacle).astype(np.float32),
            np.copy(u_nominal).astype(np.float32),
            np.copy(state_next).astype(np.float32),
            np.copy(state_error).astype(np.float32),
        ]
        if dangerous:
            buffer = self.buffer_dang
        elif safe:
            buffer = self.buffer_safe
        else:
            buffer = self.buffer_mid
        buffer.append(data)
        del buffer[:-self.buffer_size]

def load_transition_arrays(source):
    if isinstance(source, (str, os.PathLike)):
        with np.load(source, allow_pickle=False) as data:
            return dict(data)
    if hasattr(source, 'arrays'):
        return source.arrays()
    return source

def add_transition_arrays(dataset, arrays, indices, repeats=1):
    required = ['state', 'obstacle', 'u_nominal', 'state_next', 'state_error']
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError('transition data is missing keys: {}'.format(missing))
    for _ in range(repeats):
        for index in indices:
            dataset.add_data(
                arrays['state'][index], arrays['obstacle'][index],
                arrays['u_nominal'][index], arrays['state_next'][index],
                arrays['state_error'][index],
            )

def validate_ship_dataset(dataset):
    sizes = {
        'safe': len(dataset.buffer_safe),
        'dangerous': len(dataset.buffer_dang),
        'middle': len(dataset.buffer_mid),
    }
    empty = [name for name, size in sizes.items() if size == 0]
    if empty:
        raise ValueError('empty replay categories: {}'.format(empty))
    return sizes

def _danger_boundary_score(state, obstacle, safe_dist, dang_dist, use_euclidean=False):
    state = np.asarray(state)
    obstacle = np.asarray(obstacle)
    scores = np.zeros(state.shape[0], dtype=np.float32)
    for i in range(state.shape[0]):
        diff = obstacle[i, :, :2] - state[i, :2]
        if use_euclidean:
            normalized = np.linalg.norm(diff, axis=1) / max(float(dang_dist), 1e-8)
            inside = normalized <= 1.0
            scores[i] = float(np.max(normalized[inside])) if np.any(inside) else 0.0
        else:
            yaw = float(state[i, 2])
            forward = np.array([np.cos(yaw), np.sin(yaw)])
            lateral = np.array([-np.sin(yaw), np.cos(yaw)])
            front = np.abs(diff @ forward) / max(float(dang_dist) * 1.2, 1e-8)
            side = np.abs(diff @ lateral) / max(float(dang_dist) * 0.8, 1e-8)
            normalized = np.maximum(front, side)
            inside = normalized <= 1.0
            scores[i] = float(np.max(normalized[inside])) if np.any(inside) else 0.0
    return scores

def build_ship_repair_dataset(
        replay, violations=None, replay_limit=10000,
        violation_repeats=3, seed=0, buffer_size=None,
        safe_dist=3.0, dang_dist=0.7, hard_negative_repeats=0,
        hard_negative_fraction=0.5, use_euclidean=False):
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
        boundary_score = _danger_boundary_score(
            violations['state'], violations['obstacle'], safe_dist, dang_dist,
            use_euclidean=use_euclidean)
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

    dataset = ShipRepairDataset(
        n_state=6, k_obstacle=8, m_control=2, n_pos=2,
        buffer_size=buffer_size, safe_dist=safe_dist, dang_dist=dang_dist,
        use_euclidean=use_euclidean,
    )
    add_transition_arrays(dataset, replay_arrays, replay_indices)
    if violation_count > 0:
        add_transition_arrays(
            dataset, violations, np.arange(violation_count),
            repeats=violation_repeats,
        )
    if hard_negative_count > 0:
        add_transition_arrays(
            dataset, violations, hard_indices, repeats=hard_negative_repeats)
    sizes = validate_ship_dataset(dataset)
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

def full_retrain_ship(
        cbf, controller, dataset, nominal_dynamics,
        rounds=3, opt_iter=100, batch_size=1024,
        eps=0.1, eps_deriv=0.03, eps_action=0.2,
        gpu_id=-1, action_loss_weight=0.02,
        safe_dist=3.0, dang_dist=0.7,
        reference_cbf=None, reference_controller=None,
        cbf_distill_weight=0.0, cbf_distill_danger_weight=1.0,
        controller_distill_weight=0.0,
        controller_only_opt_iter=0, controller_barrier_margin=0.0,
        operational_barrier_weight=0.0, operational_barrier_margin=0.0,
        c2_separation_weight=1.0,
        controller_barrier_weight=2.0, controller_deriv_weight=1.0,
        checkpoint_prefix=None, use_euclidean=False, **kwargs):

    if reference_cbf is None:
        reference_cbf = copy.deepcopy(cbf).eval()
    if reference_controller is None:
        reference_controller = copy.deepcopy(controller).eval()
    trainer = ShipFullRetrainTrainer(
        controller, cbf, dataset, nominal_dynamics,
        n_pos=2, safe_dist=safe_dist, dang_dist=dang_dist,
        action_loss_weight=action_loss_weight, gpu_id=gpu_id,
        reference_cbf=reference_cbf,
        reference_controller=reference_controller,
        cbf_distill_weight=cbf_distill_weight,
        cbf_distill_danger_weight=cbf_distill_danger_weight,
        controller_distill_weight=controller_distill_weight,
        operational_barrier_weight=operational_barrier_weight,
        operational_barrier_margin=operational_barrier_margin,
        c2_separation_weight=c2_separation_weight,
        use_euclidean=use_euclidean,
    )

    trainer.robust_margin_fn = kwargs.get('robust_margin_fn')
    trainer.robust_margin_weight = float(kwargs.get('robust_margin_weight', 0.0) or 0.0)
    trainer.robust_rho_target = float(kwargs.get('robust_rho_target', 0.0) or 0.0)

    trainer.grad_norm_weight = float(kwargs.get('grad_norm_weight', 0.0) or 0.0)
    controller.train()
    cbf.train()
    history = []
    for repair_round in range(rounds):
        loss, accuracy = trainer.train_cbf_and_controller(
            batch_size=batch_size,
            opt_iter=opt_iter,
            eps=eps,
            eps_deriv=eps_deriv,
            eps_action=eps_action,
        )
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
            'loss': float(loss),
            'accuracy': np.asarray(accuracy, dtype=np.float32),
            'controller_loss': (
                None if controller_loss is None else float(controller_loss)),
            'controller_accuracy': (
                None if controller_accuracy is None
                else np.asarray(controller_accuracy, dtype=np.float32)),
        })

        if checkpoint_prefix is not None:
            import torch as _torch
            round_idx = repair_round + 1
            _torch.save(cbf.state_dict(),
                        '{}_round_{:02d}_cbf.pth'.format(checkpoint_prefix, round_idx))
            _torch.save(controller.state_dict(),
                        '{}_round_{:02d}_controller.pth'.format(checkpoint_prefix, round_idx))
    controller.eval()
    cbf.eval()
    return history
