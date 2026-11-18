# Analytic (torch) version of the TRUE drone dynamics, usable as the training dynamics.

import numpy as np
import torch
import torch.nn as nn

class TrueDynamics(nn.Module):

    def __init__(self, A, B, beta=3.0, centre=10.0, L=20.0, clip=0.05):
        super().__init__()
        self.register_buffer("A", torch.as_tensor(A, dtype=torch.float32))
        self.register_buffer("B", torch.as_tensor(B, dtype=torch.float32))
        self.beta = float(beta)
        self.centre = float(centre)
        self.L = float(L)
        self.clip = float(clip)

    def forward(self, state, action):
        drift = state @ self.A.T
        g = 1.0 - self.beta * ((state[:, :3] - self.centre) ** 2).sum(1) / (self.L ** 2)
        g = g.clamp(self.clip, 1.0)
        return drift + g.unsqueeze(1) * (action @ self.B.T)

def save_checkpoint(path, A, B, beta, centre=10.0, L=20.0):
    torch.save({"kind": "true_dynamics", "A": np.asarray(A), "B": np.asarray(B),
                "beta": beta, "centre": centre, "L": L}, path)
