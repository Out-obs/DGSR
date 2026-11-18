# Training losses and optimiser steps for the certificate and the controller.

import torch
from torch import nn
import os

import numpy as np

NO_C1_LOSS = os.environ.get("DSH_NO_C1_LOSS", "0") == "1"

class Trainer(object):

    def __init__(self,
                 controller,
                 cbf,
                 dataset,
                 nominal_dynamics,
                 n_pos,
                 dt=0.1,
                 safe_dist=1,
                 dang_dist=0.6,
                 action_loss_weight=0.08,
                 gpu_id=-1,
                 lr_decay_stepsize=-1):
        self.controller = controller
        self.cbf = cbf
        self.dataset = dataset
        self.nominal_dynamics = nominal_dynamics
        self.controller_optimizer = torch.optim.Adam(
            self.controller.parameters(), lr=5e-4, weight_decay=1e-5)
        self.cbf_optimizer = torch.optim.Adam(
            self.cbf.parameters(), lr=1e-4, weight_decay=1e-5)
        self.n_pos = n_pos
        self.dt = dt
        self.safe_dist = safe_dist
        self.dang_dist = dang_dist
        self.action_loss_weight = action_loss_weight

        self.robust_margin_fn = None
        self.robust_margin_weight = 0.0
        self.robust_rho_target = 0.0

        self.grad_norm_weight = 0.0

        self.gpu_id = gpu_id

        self.lr_decay_stepsize = lr_decay_stepsize
        if lr_decay_stepsize >= 0:
            self.cbf_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.cbf_optimizer, step_size=lr_decay_stepsize, gamma=0.5)
            self.controller_lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.controller_optimizer, step_size=lr_decay_stepsize, gamma=0.5)

    def clearance_penalty(self, state_next, obstacle):
        weight = float(getattr(self, "clearance_weight", 0.0))
        if weight <= 0.0:
            return 0.0
        margin = float(getattr(self, "clearance_margin", 0.6))
        pos = state_next[:, :self.n_pos].unsqueeze(1)
        obs = obstacle[:, :, :self.n_pos]
        dist = torch.norm(pos - obs, dim=2)
        min_dist = dist.min(dim=1).values
        return weight * torch.mean(nn.ReLU()(margin - min_dist))

    def train_cbf(self, batch_size=256, opt_iter=50, eps=0.1):

        loss_np = 0.0
        acc_np = np.zeros((5,), dtype=np.float32)

        for i in range(opt_iter):

            state, obstacle, u_nominal, state_next, state_error = self.dataset.sample_data(batch_size)
            state = torch.from_numpy(state)
            obstacle = torch.from_numpy(obstacle)
            state_next = torch.from_numpy(state_next)

            if self.gpu_id >= 0:
                state = state.cuda(self.gpu_id)
                obstacle = obstacle.cuda(self.gpu_id)
                state_next = state_next.cuda(self.gpu_id)

            safe_mask, dang_mask, mid_mask = self.get_mask(state, obstacle)
            h = self.cbf(state, obstacle)

            num_safe = torch.sum(safe_mask)
            num_dang = torch.sum(dang_mask)
            num_mid = torch.sum(mid_mask)

            if NO_C1_LOSS:
                loss_h_safe = h.new_zeros(())
            else:
                loss_h_safe = torch.sum(nn.ReLU()(eps - h) * safe_mask) / (1e-5 + num_safe)
            loss_h_dang = torch.sum(nn.ReLU()(h + eps) * dang_mask) / (1e-5 + num_dang)
            loss_h_dang = loss_h_dang * float(getattr(self, "c2_separation_weight", 1.0))

            acc_h_safe = torch.sum((h >= 0).float() * safe_mask) / (1e-5 + num_safe)
            acc_h_dang = torch.sum((h < 0).float() * dang_mask) / (1e-5 + num_dang)

            h_next = self.cbf(state_next, obstacle)
            deriv_cond = (h_next - h) / self.dt + h

            loss_deriv_safe = torch.sum(nn.ReLU()(-deriv_cond) * safe_mask) / (1e-5 + num_safe)
            loss_deriv_dang = torch.sum(nn.ReLU()(-deriv_cond) * dang_mask) / (1e-5 + num_dang)
            loss_deriv_mid = torch.sum(nn.ReLU()(-deriv_cond) * mid_mask) / (1e-5 + num_mid)

            acc_deriv_safe = torch.sum((deriv_cond > 0).float() * safe_mask) / (1e-5 + num_safe)
            acc_deriv_dang = torch.sum((deriv_cond > 0).float() * dang_mask) / (1e-5 + num_dang)
            acc_deriv_mid = torch.sum((deriv_cond > 0).float() * mid_mask) / (1e-5 + num_mid)

            loss = loss_h_safe + loss_h_dang + loss_deriv_safe + loss_deriv_dang + loss_deriv_mid

            self.cbf_optimizer.zero_grad()
            loss.backward()
            self.cbf_optimizer.step()

            acc_np[0] += acc_h_safe.detach().cpu().numpy()
            acc_np[1] += acc_h_dang.detach().cpu().numpy()

            acc_np[2] += acc_deriv_safe.detach().cpu().numpy()
            acc_np[3] += acc_deriv_dang.detach().cpu().numpy()
            acc_np[4] += acc_deriv_mid.detach().cpu().numpy()

            loss_np += loss.detach().cpu().numpy()

        acc_np = acc_np / opt_iter
        loss_np = loss_np / opt_iter
        return loss_np, acc_np

    def train_controller(self, batch_size=256, opt_iter=50, eps=0.1):

        loss_np = 0.0
        acc_np = np.zeros((3,), dtype=np.float32)

        for i in range(opt_iter):

            state, obstacle, u_nominal, state_next, state_error = self.dataset.sample_data(batch_size)
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

            safe_mask, dang_mask, mid_mask = self.get_mask(state, obstacle)

            h = self.cbf(state, obstacle)

            u = self.controller(state, obstacle, u_nominal, state_error)
            dsdt_nominal = self.nominal_dynamics(state, u)
            state_next_nominal = state + dsdt_nominal * self.dt

            state_next_with_grad = state_next_nominal + (state_next - state_next_nominal).detach()

            h_next = self.cbf(state_next_with_grad, obstacle)
            deriv_cond = (h_next - h) / self.dt + h

            num_safe = torch.sum(safe_mask)
            num_dang = torch.sum(dang_mask)
            num_mid = torch.sum(mid_mask)

            loss_deriv_safe = torch.sum(nn.ReLU()(-deriv_cond) * safe_mask) / (1e-5 + num_safe)
            loss_deriv_dang = torch.sum(nn.ReLU()(-deriv_cond) * dang_mask) / (1e-5 + num_dang)
            loss_deriv_mid = torch.sum(nn.ReLU()(-deriv_cond) * mid_mask) / (1e-5 + num_mid)

            acc_deriv_safe = torch.sum((deriv_cond > 0).float() * safe_mask) / (1e-5 + num_safe)
            acc_deriv_dang = torch.sum((deriv_cond > 0).float() * dang_mask) / (1e-5 + num_dang)
            acc_deriv_mid = torch.sum((deriv_cond > 0).float() * mid_mask) / (1e-5 + num_mid)

            loss_action = torch.mean((u - u_nominal)**2)

            loss = loss_deriv_safe + loss_deriv_dang + loss_deriv_mid + loss_action * self.action_loss_weight
            loss = loss + self.clearance_penalty(state_next_with_grad, obstacle)
            loss = loss + self.controller_regularization_loss(
                state, obstacle, u_nominal, state_error, u)

            self.controller_optimizer.zero_grad()
            loss.backward()
            self.controller_optimizer.step()

            acc_np[0] += acc_deriv_safe.detach().cpu().numpy()
            acc_np[1] += acc_deriv_dang.detach().cpu().numpy()
            acc_np[2] += acc_deriv_mid.detach().cpu().numpy()

            loss_np += loss.detach().cpu().numpy()

        acc_np = acc_np / opt_iter
        loss_np = loss_np / opt_iter
        return loss_np, acc_np

    def robust_margin_loss(self, state, obstacle, u):
        fn = getattr(self, "robust_margin_fn", None)
        weight = float(getattr(self, "robust_margin_weight", 0.0))
        if fn is None or weight <= 0.0:
            return state.new_zeros(())
        rho, h = fn(state, obstacle, u)
        active = h.detach() >= 0.0
        if not bool(active.any()):
            return state.new_zeros(())
        target = float(getattr(self, "robust_rho_target", 0.0))
        masked = rho.masked_fill(~active, float("inf"))
        worst = masked.min(dim=1).values
        valid = torch.isfinite(worst)
        if not bool(valid.any()):
            return state.new_zeros(())
        return weight * torch.relu(target - worst[valid]).mean()

    def grad_norm_penalty(self, state, obstacle):
        weight = float(getattr(self, "grad_norm_weight", 0.0))
        if weight <= 0.0:
            return state.new_zeros(())
        state_g = state.detach().clone().requires_grad_(True)
        h = self.cbf(state_g, obstacle)
        grads = []
        for k in range(h.shape[1]):
            grads.append(torch.autograd.grad(
                h[:, k].sum(), state_g, retain_graph=True,
                create_graph=True)[0])
        grads = torch.stack(grads, dim=1)
        return weight * grads.pow(2).sum(dim=2).mean()

    def train_cbf_and_controller(self, batch_size=1024, opt_iter=100, eps=0.1, eps_deriv=0.03, eps_action=0.2):

        loss_np = 0.0
        acc_np = np.zeros((5,), dtype=np.float32)

        for i in range(opt_iter):

            state, obstacle, u_nominal, state_next, state_error = self.dataset.sample_data(batch_size)
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

            safe_mask, dang_mask, mid_mask = self.get_mask(state, obstacle)

            h = self.cbf(state, obstacle)

            u = self.controller(state, obstacle, u_nominal, state_error)
            dsdt_nominal = self.nominal_dynamics(state, u)
            state_next_nominal = state + dsdt_nominal * self.dt

            state_next_with_grad = state_next_nominal + (state_next - state_next_nominal).detach()

            h_next = self.cbf(state_next_with_grad, obstacle)
            deriv_cond = (h_next - h) / self.dt + h

            num_safe = torch.sum(safe_mask)
            num_dang = torch.sum(dang_mask)
            num_mid = torch.sum(mid_mask)

            if NO_C1_LOSS:

                loss_h_safe = h.new_zeros(())
            else:
                loss_h_safe = torch.sum(nn.ReLU()(eps - h) * safe_mask) / (1e-5 + num_safe)
            loss_h_dang = torch.sum(nn.ReLU()(h + eps) * dang_mask) / (1e-5 + num_dang)
            loss_h_dang = loss_h_dang * float(getattr(self, "c2_separation_weight", 1.0))

            acc_h_safe = torch.sum((h >= 0).float() * safe_mask) / (1e-5 + num_safe)
            acc_h_dang = torch.sum((h < 0).float() * dang_mask) / (1e-5 + num_dang)

            loss_deriv_safe = torch.sum(nn.ReLU()(eps_deriv - deriv_cond) * safe_mask) / (1e-5 + num_safe)
            loss_deriv_dang = torch.sum(nn.ReLU()(eps_deriv - deriv_cond) * dang_mask) / (1e-5 + num_dang)
            loss_deriv_mid = torch.sum(nn.ReLU()(eps_deriv - deriv_cond) * mid_mask) / (1e-5 + num_mid)

            acc_deriv_safe = torch.sum((deriv_cond > 0).float() * safe_mask) / (1e-5 + num_safe)
            acc_deriv_dang = torch.sum((deriv_cond > 0).float() * dang_mask) / (1e-5 + num_dang)
            acc_deriv_mid = torch.sum((deriv_cond > 0).float() * mid_mask) / (1e-5 + num_mid)

            loss_action = torch.mean(nn.ReLU()(torch.abs(u - u_nominal) - eps_action))

            loss = loss_h_safe + loss_h_dang + loss_deriv_safe + loss_deriv_dang + loss_deriv_mid + loss_action * self.action_loss_weight
            loss = loss + self.clearance_penalty(state_next_with_grad, obstacle)
            loss = loss + self.joint_regularization_loss(
                state, obstacle, u_nominal, state_error, h, u)
            loss = loss + self.robust_margin_loss(state, obstacle, u)
            loss = loss + self.grad_norm_penalty(state, obstacle)

            self.controller_optimizer.zero_grad()
            self.cbf_optimizer.zero_grad()

            loss.backward()

            self.controller_optimizer.step()
            self.cbf_optimizer.step()

            acc_np[0] += acc_h_safe.detach().cpu().numpy()
            acc_np[1] += acc_h_dang.detach().cpu().numpy()

            acc_np[2] += acc_deriv_safe.detach().cpu().numpy()
            acc_np[3] += acc_deriv_dang.detach().cpu().numpy()
            acc_np[4] += acc_deriv_mid.detach().cpu().numpy()

            loss_np += loss.detach().cpu().numpy()

        acc_np = acc_np / opt_iter
        loss_np = loss_np / opt_iter

        if self.lr_decay_stepsize >= 0:

            self.cbf_lr_scheduler.step()
            self.controller_lr_scheduler.step()

        return loss_np, acc_np

    def controller_regularization_loss(
            self, state, obstacle, u_nominal, state_error, u):
        return u.new_zeros(())

    def joint_regularization_loss(
            self, state, obstacle, u_nominal, state_error, h, u):
        return h.new_zeros(())

    def get_mask(self, state, obstacle):
        state = torch.unsqueeze(state, 2)[:, :self.n_pos]
        obstacle = obstacle.permute(0, 2, 1)[:, :self.n_pos]
        dist = torch.norm(state - obstacle, dim=1)

        safe_mask = (dist >= self.safe_dist).float()
        dang_mask = (dist <= self.dang_dist).float()
        mid_mask = (1 - safe_mask) * (1 - dang_mask)

        return safe_mask, dang_mask, mid_mask
