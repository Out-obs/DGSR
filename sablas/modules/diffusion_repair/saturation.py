# Saturation-aware nominal model.

import torch

def make_offset_fn(A, B, dt=0.1, max_speed=0.5, max_theta=None):
    if max_theta is None:
        max_theta = 3.141592653589793 / 6.0
    A_t = torch.as_tensor(A, dtype=torch.float32)
    B_t = torch.as_tensor(B, dtype=torch.float32)

    def offset_fn(state, action):
        A_local = A_t.to(state.device, state.dtype)
        B_local = B_t.to(state.device, state.dtype)
        lin = state @ A_local.T + action @ B_local.T
        nxt = state + lin * dt
        nxt = torch.cat([
            nxt[:, :3],
            torch.clamp(nxt[:, 3:6], -max_speed, max_speed),
            torch.clamp(nxt[:, 6:], -max_theta, max_theta),
        ], dim=1)
        return (nxt - state) / dt - lin

    return offset_fn
