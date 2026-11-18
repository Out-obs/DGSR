# Shared environment configuration for Ship/Valley experiments.

import os

from envs.env_ship import FixedTrajectoryShip, MovingShip, River, Ship, Valley
from modules import utils

def is_safe_ship(state, obstacle, dang_dist=0.7):
    return bool(utils.is_safe(state, obstacle, n_pos=2, dang_dist=dang_dist))

def add_ship_env_arguments(parser, default_env='moving_ship'):
    parser.add_argument(
        '--env', choices=['ship', 'moving_ship', 'fixed_trajectory_ship', 'river', 'valley'],
        default=default_env,
        help='fixed_trajectory_ship: NPCs follow pre-recorded paths (Yu et al. style).',
    )
    parser.add_argument(
        '--trajectory_path', default='./data/fixed_traj/ship_npc_trajectories.npz',
        help='Pre-recorded NPC trajectory for fixed_trajectory_ship env.',
    )
    parser.add_argument(
        '--preplanned_traj', default='./data/ship_reference_traj.json',
        help='Preplanned ValleyEnv trajectories; must contain ego plus NPC ships.',
    )
    parser.add_argument('--npc_speed', type=float, default=0.05)
    parser.add_argument('--expected_num_npc', type=int, default=32)
    parser.add_argument('--perception_range', type=float, default=12.0)
    parser.add_argument('--random_permute', type=int, choices=[0, 1], default=1)
    parser.add_argument('--total_obstacle', type=int, default=32)
    parser.add_argument('--env_max_steps', type=int, default=2000)
    return parser

def make_ship_env(args, gpu_id=-1, estimated_param=None):
    env_name = getattr(args, 'env', 'moving_ship')
    max_steps = int(getattr(args, 'env_max_steps', 2000))

    if env_name == 'ship':
        return Ship(
            total_obstacle=int(getattr(args, 'total_obstacle', 32)),
            max_steps=max_steps,
            gpu_id=gpu_id,
            estimated_param=estimated_param,
        )
    if env_name == 'moving_ship':
        return MovingShip(
            npc_speed=float(getattr(args, 'npc_speed', 0.05)),
            total_obstacle=int(getattr(args, 'total_obstacle', 32)),
            max_steps=max_steps,
            gpu_id=gpu_id,
            estimated_param=estimated_param,
        )
    if env_name == 'fixed_trajectory_ship':
        traj_path = getattr(args, 'trajectory_path',
                            './data/fixed_traj/ship_npc_trajectories.npz')
        return FixedTrajectoryShip(
            trajectory_path=traj_path,
            npc_speed=float(getattr(args, 'npc_speed', 0.05)),
            total_obstacle=int(getattr(args, 'total_obstacle', 32)),
            max_steps=max_steps,
            gpu_id=gpu_id,
            estimated_param=estimated_param,
        )
    if env_name == 'river':
        return River(
            total_obstacle=int(getattr(args, 'total_obstacle', 32)),
            max_steps=max_steps,
            gpu_id=gpu_id,
            estimated_param=estimated_param,
        )
    if env_name != 'valley':
        raise ValueError('unknown Ship benchmark environment: {}'.format(env_name))

    preplanned_traj = getattr(args, 'preplanned_traj', None)
    if not preplanned_traj or not os.path.exists(preplanned_traj):
        raise FileNotFoundError(
            'ValleyEnv requires an existing --preplanned_traj file; got {}'.format(
                preplanned_traj))
    env = Valley(
        preplanned_traj=preplanned_traj,
        npc_speed=float(getattr(args, 'npc_speed', 0.2)),
        perception_range=float(getattr(args, 'perception_range', 12.0)),
        max_steps=max_steps,
        gpu_id=gpu_id,
        estimated_param=estimated_param,
        random_permute=bool(getattr(args, 'random_permute', 1)),
    )
    expected_num_npc = int(getattr(args, 'expected_num_npc', 32))
    if expected_num_npc >= 0 and env.num_npc != expected_num_npc:
        raise ValueError(
            'ValleyEnv trajectory file contains {} NPC ships, expected {}.'.format(
                env.num_npc, expected_num_npc))
    return env
