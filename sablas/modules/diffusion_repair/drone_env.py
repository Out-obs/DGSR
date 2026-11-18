# Shared environment configuration for Drone experiments.

import os
import numpy as np

from envs.env_drone import Drone, MovingDrone
from modules import utils

def is_safe_drone(state, obstacle, dang_dist=0.6):
    return bool(utils.is_safe(state, obstacle, n_pos=3, dang_dist=dang_dist))

def add_drone_env_arguments(parser, default_env='moving_drone'):
    parser.add_argument(
        '--env', choices=['drone', 'moving_drone', 'city'], default=default_env,
        help='moving_drone: 1024-NPC constant-velocity moving obstacles.',
    )
    parser.add_argument('--npc_speed', type=float, default=0.05)
    parser.add_argument('--total_obstacle', type=int, default=1024)
    parser.add_argument('--env_max_steps', type=int, default=500)
    parser.add_argument('--param', type=str, default='./data/estimated_model_drone.npz',
                        help='System identification parameters (A, B matrices).')
    return parser

def make_drone_env(args, gpu_id=-1, estimated_param=None):
    env_name = getattr(args, 'env', 'moving_drone')
    max_steps = int(getattr(args, 'env_max_steps', 500))
    npc_speed = float(getattr(args, 'npc_speed', 0.05))
    total_obstacle = int(getattr(args, 'total_obstacle', 1024))

    if estimated_param is None:
        param_path = getattr(args, 'param', './data/estimated_model_drone.npz')
        if os.path.exists(param_path):
            estimated_param = dict(np.load(param_path, allow_pickle=True))

    if env_name == 'drone':
        return Drone(
            total_obstacle=total_obstacle,
            max_steps=max_steps,
            estimated_param=estimated_param,
        )
    if env_name == 'moving_drone':
        return MovingDrone(
            npc_speed=npc_speed,
            total_obstacle=total_obstacle,
            max_steps=max_steps,
            estimated_param=estimated_param,
        )
    if env_name == 'city':
        from envs.env_drone import City
        return City(
            total_obstacle=total_obstacle,
            max_steps=max_steps,
            estimated_param=estimated_param,
        )
    raise ValueError('unknown Drone environment: {}'.format(env_name))
