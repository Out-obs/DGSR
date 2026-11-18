# SABLAS training entry point for the drone (unchanged recipe, configurable budget).

import argparse
import os
import sys

sys.path.insert(1, os.path.abspath('.'))

import numpy as np
import torch

from envs.env_drone import Drone
from modules import config
from modules import utils
from modules.dataset import Dataset
from modules.trainer import Trainer
from modules.network import CBF, NNController

np.set_printoptions(4)

def main(estimated_param=None, steps=None, out_prefix='./data/drone',
         action_loss_weight=0.1):
    env = Drone(estimated_param=estimated_param)

    nn_controller = NNController(n_state=8, k_obstacle=8, m_control=3)
    cbf = CBF(n_state=8, k_obstacle=8, m_control=3)
    dataset = Dataset(n_state=8, k_obstacle=8, m_control=3, n_pos=3)
    trainer = Trainer(nn_controller, cbf, dataset, env.nominal_dynamics_torch, n_pos=3,
                      action_loss_weight=action_loss_weight)
    state, obstacle, goal = env.reset()

    state_error = np.zeros((8,), dtype=np.float32)

    safety_rate = 0.0
    goal_reached = 0.0

    dt = float(env.dt) if hasattr(env, 'dt') else 0.1

    DATA_LIMIT = int(os.environ.get("DSH_DATA_LIMIT", "0"))
    total_steps = int(steps) if steps is not None else int(config.TRAIN_STEPS)
    for i in range(total_steps):
        u_nominal = env.nominal_controller(state, goal)
        u = nn_controller(
            torch.from_numpy(state.reshape(1, 8).astype(np.float32)),
            torch.from_numpy(obstacle.reshape(1, 8, 8).astype(np.float32)),
            torch.from_numpy(u_nominal.reshape(1, 3).astype(np.float32)),
            torch.from_numpy(state_error.reshape(1, 8).astype(np.float32)))
        u = np.squeeze(u.detach().cpu().numpy())
        state_next, state_nominal_next, obstacle_next, goal_next, done = env.step(u)

        if DATA_LIMIT <= 0 or i < DATA_LIMIT:
            dataset.add_data(state, obstacle, u_nominal, state_next, state_error)

        is_safe = int(utils.is_safe(state, obstacle, n_pos=3, dang_dist=0.6))
        safety_rate = safety_rate * (1 - 1e-4) + is_safe * 1e-4

        state_error = (state_next - state_nominal_next) / dt

        state = state_next
        obstacle = obstacle_next
        goal = goal_next

        if np.mod(i, config.POLICY_UPDATE_INTERVAL) == 0 and i > 0:
            loss_np, acc_np = trainer.train_cbf_and_controller()
            print('step: {}, train h and u, loss: {:.3f}, safety rate: {:.3f}, '
                  'goal reached: {:.3f}, acc: {}'.format(
                      i, loss_np, safety_rate, goal_reached, acc_np))

            torch.save(cbf.state_dict(), out_prefix + '_cbf_weights.pth')
            torch.save(nn_controller.state_dict(), out_prefix + '_controller_weights.pth')

            _snap = int(os.environ.get('DSH_SNAPSHOT_EVERY', '0'))
            if _snap > 0 and i % _snap == 0:
                torch.save(cbf.state_dict(), '%s_step%07d_cbf.pth' % (out_prefix, i))
                torch.save(nn_controller.state_dict(),
                           '%s_step%07d_controller.pth' % (out_prefix, i))

        if done:
            dist = np.linalg.norm(state[:3] - goal[:3])
            goal_reached = goal_reached * (1 - 1e-2) + (dist < 2.0) * 1e-2
            state, obstacle, goal = env.reset()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=None,
                        help='training steps (default: config.TRAIN_STEPS)')
    parser.add_argument('--param', type=str, default='./data/estimated_model_drone.npz',
                        help='identified nominal model used for state-error feedback')
    parser.add_argument('--action_loss_weight', type=float, default=0.1,
                        help='SABLAS goal term on (u-u_nominal)^2: larger = more goal-seeking, less safe')
    parser.add_argument('--out_prefix', type=str, default='./data/drone',
                        help='weights are written to <prefix>_cbf_weights.pth etc.')
    args = parser.parse_args()

    param = np.load(open(args.param, 'rb'))
    main(param, steps=args.steps, out_prefix=args.out_prefix,
         action_loss_weight=args.action_loss_weight)
