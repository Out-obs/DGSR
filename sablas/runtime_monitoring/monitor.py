# CertPM and PredPM as described in the runtime-monitoring paper.

import numpy as np
import torch

def _h(cbf, state, obstacle, device="cpu"):
    with torch.no_grad():
        s = torch.from_numpy(np.asarray(state, dtype=np.float32).reshape(1, -1)).to(device)
        o = torch.from_numpy(np.asarray(obstacle, dtype=np.float32).reshape(1, *np.shape(obstacle))).to(device)
        return cbf(s, o).reshape(-1).cpu().numpy()

def property_violation(state, obstacle, safe_dist=0.6, n_pos=3):
    d = np.linalg.norm(np.asarray(obstacle)[:, :n_pos] - np.asarray(state)[:n_pos], axis=1)
    return bool(d.min() < safe_dist)

def condition_violation(h_now, h_next, dt, scalar=False):
    if scalar:
        a = np.array([float(np.min(h_now))])
        b = np.array([float(np.min(h_next))])
    else:
        a, b = h_now, h_next
    h_now, h_next = a, b
    nonneg = float(np.min(h_now)) >= 0.0
    deriv = float(np.min((h_next - h_now) / dt + h_now)) >= 0.0
    return (not nonneg) or (not deriv), dict(min_h=float(np.min(h_now)),
                                             min_deriv=float(np.min((h_next - h_now) / dt + h_now)))

def certpm_warning(cbf, state, obstacle, state_next, obstacle_next, dt, safe_dist=0.6,
                   include_property=True, scalar=False):
    prop = property_violation(state, obstacle, safe_dist)
    h_now = _h(cbf, state, obstacle)
    h_next = _h(cbf, state_next, obstacle_next)
    cond, info = condition_violation(h_now, h_next, dt, scalar=scalar)

    bad = cond or (prop and include_property)
    return bool(bad), dict(property=prop, condition=cond, **info)

def predpm_warning(cbf, state, obstacle, velocity, dt, horizon=10, tau=3,
                   safe_dist=0.6, n_pos=3):
    s = np.asarray(state, dtype=np.float64).copy()
    v = np.asarray(velocity, dtype=np.float64).copy()
    h_prev = float(np.min(_h(cbf, s, obstacle)))
    for step in range(1, horizon + 1):
        s = s + dt * v
        bad = property_violation(s, obstacle, safe_dist, n_pos)
        h_now = float(np.min(_h(cbf, s, obstacle)))
        cond = h_now < 0.0 or (h_now - h_prev) / dt + h_prev < 0.0
        h_prev = h_now
        if bad or cond:
            return step <= tau, dict(time_to_violation=step, property=bool(bad), condition=bool(cond))
    return False, dict(time_to_violation=None, property=False, condition=False)
