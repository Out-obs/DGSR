# Monitor-driven repair of a SABLAS policy and certificate (CertPM / PredPM).

import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from envs.env_drone import Drone
from modules.dataset import Dataset
from modules.network import CBF, NNController
from modules.trainer import Trainer
from runtime_monitoring.monitor import (certpm_warning, predpm_warning,
                                       _h)

def load_models(cbf_path, ctl_path, device="cpu"):
    cbf = CBF(8, 8, 3)
    cbf.load_state_dict(torch.load(cbf_path, map_location=device))
    ctl = NNController(8, 8, 3)
    ctl.load_state_dict(torch.load(ctl_path, map_location=device))
    return cbf, ctl

def rollout(args, cbf, ctl, monitor):
    env = Drone(total_obstacle=args.total_obstacle)
    state, obstacle, goal = env.reset()
    state_error = np.zeros((8,), dtype=np.float32)
    dt = float(env.dt)
    warned = []
    total = 0
    prev_state = None
    prev_obstacle = None
    while total < args.monitor_steps:
        u_nominal = env.nominal_controller(state, goal).astype(np.float32)
        with torch.no_grad():
            u = ctl(torch.from_numpy(state.reshape(1, 8).astype(np.float32)),
                    torch.from_numpy(obstacle.reshape(1, 8, 8).astype(np.float32)),
                    torch.from_numpy(u_nominal.reshape(1, 3).astype(np.float32)),
                    torch.from_numpy(state_error.reshape(1, 8).astype(np.float32)))
        u = np.squeeze(u.numpy())
        state_next, state_nominal_next, obstacle_next, goal_next, done = env.step(u)

        if monitor == "certpm":
            bad, info = certpm_warning(cbf, state, obstacle, state_next, obstacle_next,
                                       dt, args.safe_dist,
                                       include_property=args.warn_on_property,
                                       scalar=args.scalar_condition)
        else:
            vel = (state - prev_state) / dt if prev_state is not None else np.zeros(8)
            bad, info = predpm_warning(cbf, state, obstacle, vel, dt,
                                       horizon=args.pred_horizon, tau=args.pred_tau,
                                       safe_dist=args.safe_dist)
        if bad:
            warned.append(dict(state=np.copy(state), obstacle=np.copy(obstacle),
                               u_nominal=np.copy(u_nominal), state_next=np.copy(state_next),
                               state_error=np.copy(state_error)))
        prev_state, prev_obstacle = np.copy(state), np.copy(obstacle)
        state_error = ((state_next - state_nominal_next) / dt).astype(np.float32)
        state, obstacle, goal = state_next, obstacle_next, goal_next
        total += 1
        if done:
            state, obstacle, goal = env.reset()
            state_error = np.zeros((8,), dtype=np.float32)
            prev_state, prev_obstacle = None, None
    return warned

def build_dataset(base_data, warned, seed=0):
    ds = Dataset(n_state=8, k_obstacle=8, m_control=3, n_pos=3)
    z = np.load(base_data, allow_pickle=True)
    n = len(z["state"])
    idx = np.random.RandomState(seed).permutation(n)
    for i in idx:
        ds.add_data(z["state"][i], z["obstacle"][i], z["u_nominal"][i],
                    z["state_next"][i], z["state_error"][i])
    for w in warned:
        ds.add_data(w["state"], w["obstacle"], w["u_nominal"], w["state_next"], w["state_error"])
    return ds

def repair(args, method):
    cbf, ctl = load_models(args.cbf, args.controller)
    monitor = "predpm" if method == "predpm" else "certpm"
    history = []
    all_warned = []
    stop_reason = "rounds_exhausted"
    for rnd in range(1, args.max_rounds + 1):
        warned = rollout(args, cbf, ctl, monitor)
        print("  round %d: monitor warned %d / %d steps" % (rnd, len(warned), args.monitor_steps),
              flush=True)
        history.append(len(warned))
        if not warned:
            stop_reason = "no_violations"
            print("  stopping: monitor found no violations", flush=True)
            break

        all_warned.extend(warned)
        ds = build_dataset(args.train_data, all_warned, seed=args.seed + rnd)
        env = Drone(total_obstacle=args.total_obstacle)

        trainer = Trainer(ctl, cbf, ds, env.nominal_dynamics_torch, n_pos=3,
                          dt=float(env.dt), safe_dist=1.0, dang_dist=0.6,
                          action_loss_weight=args.action_loss_weight)
        trainer.train_cbf_and_controller(batch_size=args.batch_size, opt_iter=args.opt_iter,
                                        eps=0.0, eps_deriv=0.0, eps_action=0.0)
    torch.save(cbf.state_dict(), args.out_cbf)
    torch.save(ctl.state_dict(), args.out_controller)
    print("repair finished: reason=%s rounds=%d warns=%s" % (stop_reason, len(history), history),
          flush=True)
    return stop_reason, history

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--monitor", required=True, choices=["certpm", "predpm"])
    ap.add_argument("--cbf", default="./data/gp30dt002/models/init200k_cbf_weights.pth")
    ap.add_argument("--controller", default="./data/gp30dt002/models/init200k_controller_weights.pth")
    ap.add_argument("--train_data", default="./data/gp30dt002/data/train.npz")
    ap.add_argument("--out_cbf", required=True)
    ap.add_argument("--out_controller", required=True)
    ap.add_argument("--monitor_steps", type=int, default=4000)
    ap.add_argument("--max_rounds", type=int, default=6)
    ap.add_argument("--opt_iter", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--action_loss_weight", type=float, default=0.1)
    ap.add_argument("--safe_dist", type=float, default=0.6)
    ap.add_argument("--pred_horizon", type=int, default=10)
    ap.add_argument("--pred_tau", type=int, default=3)
    ap.add_argument("--warn-on-property", dest="warn_on_property", action="store_true",
                    default=True, help="count property violations as warnings (paper default)")
    ap.add_argument("--no-warn-on-property", dest="warn_on_property", action="store_false")
    ap.add_argument("--scalar-condition", dest="scalar_condition", action="store_true",
                    default=False, help="apply the certificate conditions to min_k h_k")
    ap.add_argument("--total_obstacle", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_cbf)), exist_ok=True)
    repair(args, args.monitor)

if __name__ == "__main__":
    main()
