# Smoke test for the diffusion-repair pipeline (no pretrained weights needed).

import os
import sys
sys.path.insert(1, os.path.abspath('.'))

import numpy as np
import torch
from torch import nn

from modules.diffusion_repair.diffusion import ConditionalVectorFieldDiffusion
from modules.diffusion_repair.repair import ResidualController, freeze_module, repair_loss
from modules.diffusion_repair.robust_verify import (
    certificate_condition_masks,
    classify_violation_type,
    robust_barrier_margin,
)
from modules.diffusion_repair.vector_dataset import VectorFieldDataset

class LinearCBF(nn.Module):

    def forward(self, state, obstacle):
        del obstacle
        return state[:, :1]

class ZeroController(nn.Module):

    def forward(self, state, obstacle, u_nominal, state_error):
        del state, obstacle, state_error
        return torch.zeros_like(u_nominal)

def main():
    torch.manual_seed(0)
    np.random.seed(0)

    dataset = VectorFieldDataset(n_state=2, k_obstacle=1, m_control=1, dt=0.1)
    dataset.add_transition(
        state=np.array([1.0, 2.0], dtype=np.float32),
        obstacle=np.zeros((1, 2), dtype=np.float32),
        action_exec=np.array([0.5], dtype=np.float32),
        u_nominal=np.array([0.25], dtype=np.float32),
        state_next=np.array([1.1, 1.8], dtype=np.float32),
        state_error=np.zeros((2,), dtype=np.float32),
        dt=0.1,
    )
    vector = dataset.arrays()["vector"][0]
    assert np.allclose(vector, np.array([1.0, -2.0], dtype=np.float32))

    diffusion = ConditionalVectorFieldDiffusion(
        state_dim=2,
        action_dim=1,
        vector_dim=2,
        hidden_dim=32,
        time_dim=8,
        diffusion_steps=4,
    )
    diffusion.set_normalization(dataset.normalization_stats())
    state = torch.tensor([[1.0, 2.0], [0.5, -0.5]], dtype=torch.float32)
    action = torch.tensor([[0.2], [0.1]], dtype=torch.float32)
    vectors = diffusion.sample_vectors(state, action, num_samples=3)
    assert vectors.shape == (2, 3, 2)
    assert torch.isfinite(vectors).all()

    cbf = LinearCBF()
    obstacle = torch.zeros((2, 1, 2), dtype=torch.float32)
    result = robust_barrier_margin(
        cbf,
        state,
        obstacle,
        action,
        diffusion,
        num_samples=3,
        q_alpha=1.0,
        vector_samples=vectors,
    )
    assert result["rho"].shape == (2, 1)
    assert torch.isfinite(result["rho"]).all()

    safe_mask, dang_mask = certificate_condition_masks(
        state,
        obstacle,
        safe_dist=0.1,
        dang_dist=0.01,
        mask_type="distance",
    )
    assert safe_mask.shape == (2, 1)
    assert dang_mask.shape == (2, 1)
    types, actions, failures = classify_violation_type(
        np.array([False, True]),
        np.array([False, False]),
        np.array([True, True]),
    )
    assert actions[0] == "repair_residual_controller"
    assert actions[1] == "retrain_cbf_controller"
    assert failures[0] == "condition3"

    base_controller = ZeroController()
    residual = ResidualController(n_state=2, k_obstacle=1, m_control=1, output_scale=0.1)
    freeze_module(cbf)
    freeze_module(base_controller)
    freeze_module(diffusion)
    loss, stats = repair_loss(
        cbf,
        base_controller,
        residual,
        diffusion,
        state,
        obstacle,
        torch.zeros((2, 1), dtype=torch.float32),
        torch.zeros((2, 2), dtype=torch.float32),
        num_samples=2,
    )
    assert torch.isfinite(loss)
    assert "safety_loss" in stats
    print("smoke_diffusion_repair passed")

if __name__ == "__main__":
    main()
