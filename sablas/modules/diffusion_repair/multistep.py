# Multi-step differentiable barrier recovery for the Ship controller.

import torch

def train_controller_multistep_barrier(
        trainer, diffusion, batch_size=1024, opt_iter=50, horizon=5,
        num_samples=4, barrier_margin=0.0, barrier_weight=2.0,
        eps_deriv=0.03, deriv_weight=1.0):
    loss_total = 0.0
    accuracy_total = 0.0
    for _ in range(opt_iter):
        state, obstacle, u_nominal, state_next, state_error = \
            trainer.dataset.sample_data(batch_size)
        state = torch.from_numpy(state)
        obstacle = torch.from_numpy(obstacle)
        u_nominal = torch.from_numpy(u_nominal)
        state_error = torch.from_numpy(state_error)
        if trainer.gpu_id >= 0:
            state = state.cuda(trainer.gpu_id)
            obstacle = obstacle.cuda(trainer.gpu_id)
            u_nominal = u_nominal.cuda(trainer.gpu_id)
            state_error = state_error.cuda(trainer.gpu_id)

        predicted_state = state
        barrier_loss = state.new_zeros(())
        deriv_loss = state.new_zeros(())
        action_loss = state.new_zeros(())
        positive_count = 0.0
        for _ in range(horizon):
            h = trainer.cbf(predicted_state, obstacle)
            u = trainer.controller(
                predicted_state, obstacle, u_nominal, state_error)
            dsdt_nominal = trainer.nominal_dynamics(predicted_state, u)
            state_next_nominal = predicted_state + dsdt_nominal * trainer.dt
            if diffusion is not None:
                with torch.no_grad():
                    vector_samples = diffusion.sample_vectors(
                        predicted_state, u, num_samples=num_samples)
                    mean_vector = vector_samples.mean(dim=1)
                    nominal_vector = (
                        state_next_nominal - predicted_state) / trainer.dt
                    correction = (mean_vector - nominal_vector) * trainer.dt
                state_next_predict = state_next_nominal + correction
            else:
                state_next_predict = state_next_nominal
            h_next = trainer.cbf(state_next_predict, obstacle)
            deriv_cond = (h_next - h) / trainer.dt + h
            barrier_loss = barrier_loss + torch.relu(
                barrier_margin - torch.min(h_next, dim=1)[0]).mean()
            deriv_loss = deriv_loss + torch.relu(
                eps_deriv - torch.min(deriv_cond, dim=1)[0]).mean()
            action_loss = action_loss + torch.mean((u - u_nominal) ** 2)
            positive_count += float(
                (torch.min(h_next, dim=1)[0] >= 0.0)
                .float().mean().detach().cpu().item())
            predicted_state = state_next_predict
            state_error = torch.zeros_like(state_error)

        barrier_loss = barrier_loss / horizon
        deriv_loss = deriv_loss / horizon
        action_loss = action_loss / horizon
        loss = (
            barrier_weight * barrier_loss
            + deriv_weight * deriv_loss
            + trainer.action_loss_weight * action_loss)
        trainer.controller_optimizer.zero_grad()
        loss.backward()
        trainer.controller_optimizer.step()
        loss_total += float(loss.detach().cpu().item())
        accuracy_total += positive_count / horizon

    return loss_total / opt_iter, accuracy_total / opt_iter
