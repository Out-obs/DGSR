# Implementation file of the released artefact.

import os
import sys
sys.path.insert(1, os.path.abspath('.'))

import json
import random
import numpy as np
import torch
import pickle
from modules import gurobi_milp
gurobi_milp.setM(1e6)
from modules.gurobi_milp import plan, interpolate

def get_distance_between_agents_and_obs(X, obstacles):
    obs = np.expand_dims(np.array(obstacles).T, [0,1])
    centers = (obs[:,:,:3,:] + obs[:,:,3:,:]) / 2
    half_width = (obs[:,:,3:,:] - obs[:,:,:3,:]) / 2
    X = np.expand_dims(X[:,:,:3], -1)
    error = np.maximum(0, np.abs((X - centers)) - half_width)
    error = np.sqrt((error ** 2).sum(axis=2))
    error = error.min(axis=-1)
    return error

def get_scene(num_agents, obstacle_bboxs, env_size=10000, bloat_factor=1.5, mutual_min_dist=1.2, clearance_waypoints=100.):

    bboxs = obstacle_bboxs
    obstacles = []
    for bbox in bboxs:
        A = np.array([[-1, 0,  0],
                      [ 1, 0,  0],
                      [0, -1,  0],
                      [0,  1,  0],
                      [0,  0, -1],
                      [0,  0,  1]])
        b = bbox.T.reshape(-1) * np.array([-1,1,-1,1,-1,1])
        obstacles.append([A, b])
    bboxs = np.array(bboxs)
    area_bound = np.array([bboxs[:,0,:].min(axis=0), bboxs[:,1,:].max(axis=0)]).T.tolist()
    area_bound[2][1] = 11.
    area_bound = np.clip(area_bound, -env_size/2, env_size/2)

    waypoints = []
    B = np.array(area_bound)
    obstacles_old_representation = bboxs.reshape([bboxs.shape[0], -1])
    bloat_factor = bloat_factor * 1.1
    num_waypoints = 3
    num_samples_trial = 100
    for _ in range(num_agents):
        waypoints.append([])
        for i in range(num_waypoints):
            while True:
                p = B[:,0:1] + bloat_factor + (
                    B[:,1:2] - B[:,0:1] - 2 * bloat_factor
                    ) * np.random.rand(B.shape[0], num_samples_trial)
                p = p.T
                dist = get_distance_between_agents_and_obs(
                    p.reshape(num_samples_trial, 1, B.shape[0]),
                    obstacles_old_representation).squeeze()
                conditions = [dist > bloat_factor * np.sqrt(B.shape[0]),]

                if len(waypoints[-1]) > 0:
                    conditions.append(np.sqrt(((
                        p - waypoints[-1][-1].reshape(1, -1))**2
                        ).sum(axis=1)) < clearance_waypoints)
                if i == 0 or i == num_waypoints-1:
                    if len(waypoints) > 1:
                        others = np.array([w[i] for w in waypoints[:-1]])
                        dist = p.reshape(num_samples_trial, 1, B.shape[0]
                        ) - others.reshape(1, others.shape[0], others.shape[1])
                        dist = np.sqrt((dist**2).sum(axis=2)).min(axis=1)
                        conditions.append(dist > mutual_min_dist)
                conditions = np.logical_and.reduce(np.array(conditions), axis=0)
                idx = np.where(conditions)[0]
                if len(idx) > 0:
                    break
            waypoints[-1].append(p[idx[0], :])
    return obstacles, obstacles_old_representation, area_bound, waypoints

DYN_MODE = __import__('os').environ.get('DSH_DYN_MODE', 'linear').lower()
DRAG_C = float(__import__('os').environ.get('DSH_DRAG', 2.0))

THRUST_BETA = float(__import__('os').environ.get('DSH_BETA', 2.0))

class Drone(object):

    def __init__(self,
                 dt=None,

                 k_obstacle=8,
                 total_obstacle=500,
                 env_size=20,
                 safe_dist=1,
                 max_steps=500,
        max_speed=float(__import__("os").environ.get("DSH_MAX_SPEED", 0.5)),
                 max_theta=np.pi/6,
                 noise_std=0.1,
                 estimated_param=None):
        assert total_obstacle >= k_obstacle

        self.dt = float(dt) if dt is not None else float(
            __import__("os").environ.get("DSH_DT", 0.1))
        self.k_obstacle = k_obstacle
        self.total_obstacle = total_obstacle
        self.env_size = env_size
        self.safe_dist = safe_dist
        self.max_steps = max_steps
        self.max_speed = max_speed
        self.max_theta = max_theta
        self.noise_std = noise_std

        self.A_real = [[0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0]]

        self.B_real = [[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 1],
                       [1, 0, 0],
                       [0, 1, 0]]

        if estimated_param is not None:

            self.A_nominal = estimated_param['A']
            self.B_nominal = estimated_param['B']
        else:

            self.A_nominal = [[0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0]]
            self.B_nominal = [[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 1],
                          [1, 0, 0],
                          [0, 1, 0]]

        self.K = np.array([[1, 0, 0, 2.41,    0,    0, 2.41,    0],
                           [0, 1, 0,    0, 2.41,    0,    0, 2.41],
                           [0, 0, 1,    0,    0, 1.73,    0,    0]])
        self.noise = np.random.normal(size=(8,)) * self.noise_std

    def reset(self):
        self.obstacle = np.random.uniform(
            low=0, high=self.env_size, size=(self.total_obstacle, 3))
        state = np.random.uniform(low=0, high=self.env_size, size=(3,))
        while np.amin(np.linalg.norm(self.obstacle - state, axis=1)) < self.safe_dist:
            state = np.random.uniform(low=0, high=self.env_size, size=(3,))
        state = np.concatenate([state, np.zeros((5,))])
        self.state = state
        obstacle = self.get_obstacle(state)
        goal = np.random.uniform(low=0, high=self.env_size, size=(3,))
        while np.amin(np.linalg.norm(self.obstacle - goal, axis=1)) < self.safe_dist:
            goal = np.random.uniform(low=0, high=self.env_size, size=(3,))
        goal = np.concatenate([goal, np.zeros((5,))])
        self.goal = goal

        self.noise = np.random.normal(size=(8,)) * self.noise_std
        self.num_steps = 0
        return state, obstacle, goal

    def step(self, u):
        dsdt = self.uncertain_dynamics(self.state, u)
        noise = self.get_noise()
        state = self.state + (dsdt + noise) * self.dt
        state[3:6] = np.clip(state[3:6], -self.max_speed, self.max_speed)
        state[6:] = np.clip(state[6:], -self.max_theta, self.max_theta)

        dsdt_nominal = self.nominal_dynamics(self.state, u)
        state_nominal = self.state + dsdt_nominal * self.dt
        state_nominal[3:6] = np.clip(state_nominal[3:6], -self.max_speed, self.max_speed)
        state_nominal[6:] = np.clip(state_nominal[6:], -self.max_theta, self.max_theta)

        obstacle = self.get_obstacle(state)
        goal = self.get_goal(state)
        self.state = state
        done = np.linalg.norm(state[:3] - goal[:3]) < self.safe_dist or self.num_steps > self.max_steps
        self.num_steps = self.num_steps + 1
        return state, state_nominal, obstacle, goal, done

    def uncertain_dynamics(self, state, u):
        A = np.array(self.A_real, dtype=np.float32)
        B = np.array(self.B_real, dtype=np.float32)
        if DYN_MODE == 'gain_pos':

            p = np.asarray(state[:3], dtype=np.float32)
            centre = 0.5 * float(getattr(self, 'env_size', 20.0))
            g = 1.0 - THRUST_BETA * float(np.sum((p - centre) ** 2)) / (
                float(getattr(self, 'env_size', 20.0)) ** 2)
            g = float(np.clip(g, 0.05, 1.0))
            dsdt = A.dot(state) + g * B.dot(u)
            return dsdt
        if DYN_MODE == 'thrust_sat':

            v = np.asarray(state[3:6], dtype=np.float32)
            gain = max(0.0, 1.0 - THRUST_BETA * float(v.dot(v)))
            dsdt = A.dot(state) + gain * B.dot(u)
            return dsdt
        if DYN_MODE == 'thrust2':

            v = np.asarray(state[3:6], dtype=np.float32)
            gain = 1.0 + THRUST_BETA * float(v.dot(v))
            dsdt = A.dot(state) + gain * B.dot(u)
            return dsdt
        dsdt = A.dot(state) + B.dot(u)
        if DYN_MODE in ('drag2', 'drag3'):

            v = np.asarray(state[3:6], dtype=np.float32)
            speed = float(np.linalg.norm(v))
            scale = speed if DYN_MODE == 'drag2' else speed * speed
            dsdt[3:6] = dsdt[3:6] - DRAG_C * scale * v

        return dsdt

    def nominal_dynamics(self, state, u):
        A = np.array(self.A_nominal, dtype=np.float32)
        B = np.array(self.B_nominal, dtype=np.float32)
        dsdt = A.dot(state) + B.dot(u)

        return dsdt

    def dynamics_spec(self):
        return {
            "mode": DYN_MODE,
            "A": np.array(self.A_real, dtype=np.float64),
            "B": np.array(self.B_real, dtype=np.float64),
            "drag": DRAG_C,
            "beta": THRUST_BETA,
            "dt": float(self.dt),
            "max_speed": float(self.max_speed),
            "max_theta": float(self.max_theta),
            "n_state": 8,
            "m_control": 3,
        }

    def nominal_dynamics_torch(self, state, u):
        A = np.array(self.A_nominal, dtype=np.float32)
        A_T = torch.from_numpy(A.T)

        B = np.array(self.B_nominal, dtype=np.float32)
        B_T = torch.from_numpy(B.T)

        dsdt = torch.matmul(state, A_T) + torch.matmul(u, B_T)

        return dsdt

    def nominal_controller(self, state, goal, u_norm_max=0.5):
        K = self.K
        u_nominal = -K.dot(state - goal)
        norm = np.linalg.norm(u_nominal)
        if norm > u_norm_max:
            u_nominal = u_nominal / norm * u_norm_max
        return u_nominal

    def get_obstacle(self, state):
        dist = np.linalg.norm(self.obstacle - state[:3], axis=1)
        argsort = np.argsort(dist)[:self.k_obstacle]
        obstacle = self.obstacle[argsort]
        obstacle = np.concatenate([obstacle, np.zeros((self.k_obstacle, 5))], axis=1)
        return obstacle

    def get_goal(self, state):
        return self.goal

    def get_noise(self):
        if np.random.uniform() < 0.05:
            self.noise = np.random.normal(size=(8,)) * self.noise_std
        noise = np.copy(self.noise)
        noise[:3] = 0
        return noise

class City(object):

    def __init__(self,
                 static_obstacle='data/city_buildings.json',
                 preplanned_traj=None,
                 dt=float(__import__("os").environ.get("DSH_DT", 0.1)),
                 k_obstacle=8,
                 env_size=100,
                 num_npc=1024,
                 npc_speed=1.0,
                 safe_dist=1,
                 perception_range=12,
                 max_steps=500,
                 max_speed=0.5,
                 max_theta=np.pi/6,
                 noise_std=0.1,
                 estimated_param=None,
                 random_permute=True):
        assert num_npc >= k_obstacle
        self.dt = dt
        self.k_obstacle = k_obstacle
        self.env_size = env_size
        self.num_npc = num_npc
        self.npc_speed = npc_speed
        self.safe_dist = safe_dist
        self.perception_range = perception_range
        self.max_steps = max_steps
        self.max_speed = max_speed
        self.max_theta = max_theta
        self.noise_std = noise_std
        self.random_permute = random_permute

        self.A_real = [[0, 0, 0, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 0, 0, 0],
                       [0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0]]

        self.B_real = [[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 1],
                       [1, 0, 0],
                       [0, 1, 0]]

        if estimated_param is not None:

            self.A_nominal = estimated_param['A']
            self.B_nominal = estimated_param['B']
        else:
            self.A_nominal = [[0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0]]
            self.B_nominal = [[0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 0],
                          [0, 0, 1],
                          [1, 0, 0],
                          [0, 1, 0]]

        self.K = np.array([[1, 0, 0, 2.41,    0,    0, 2.41,    0],
                           [0, 1, 0,    0, 2.41,    0,    0, 2.41],
                           [0, 0, 1,    0,    0, 1.73,    0,    0]])
        self.noise = np.random.normal(size=(8,)) * self.noise_std

        if preplanned_traj is None:

            obstacles, obstacles_old_representation, area_bound, waypoints = \
                get_scene(self.num_npc + 1, np.array(json.load(open(static_obstacle, 'r'))), env_size=self.env_size)
            self.reference_traj = []
            reference_traj_as_list = []
            for i in range(self.num_npc + 1):
                print('\n========= Planning for agent {} / {} =========\n'.format(i+1, self.num_npc + 1))
                x_init = np.concatenate([waypoints[i][0], np.zeros(5)])
                path = plan(x_init[:3], waypoints[i][1:], area_bound, obstacles)
                if path is None:
                    print('\n========= Planning failed for agent {} =========\n'.format(i+1))
                    continue
                traj, ts = interpolate(path, dt=self.dt, max_speed=self.max_speed)
                self.reference_traj.append({"traj": traj, "ts": ts})
                reference_traj_as_list.append({"traj": traj.tolist(), "ts": ts.tolist()})
            self.num_npc = len(self.reference_traj) - 1
            json.dump(reference_traj_as_list, open("data/reference_traj.json", "w"), indent=4)
        else:
            reference_traj_as_list = json.load(open(preplanned_traj))
            self.num_npc = len(reference_traj_as_list) - 1
            self.reference_traj = []
            for single_agent_traj in reference_traj_as_list:
                self.reference_traj.append({
                    "traj": np.array(single_agent_traj["traj"]),
                    "ts": np.array(single_agent_traj["ts"])})
        print('\nFinish preparing reference trajectories. Found {} NPC in total\n'.format(self.num_npc))

    def reset(self):
        self.t = 0
        if self.random_permute:
            random.shuffle(self.reference_traj)
        else:
            print('Preserve the default order of agents.')
        self.state = np.concatenate([self.reference_traj[0]['traj'][0], np.zeros(5)])
        obstacle = self.get_obstacle(self.state)
        goal = self.get_goal(self.state)
        return self.state, obstacle, goal

    def step(self, u):
        dsdt = self.uncertain_dynamics(self.state, u)
        noise = self.get_noise()
        state = self.state + (dsdt + noise) * self.dt
        state[3:6] = np.clip(state[3:6], -self.max_speed, self.max_speed)
        state[6:] = np.clip(state[6:], -self.max_theta, self.max_theta)

        dsdt_nominal = self.nominal_dynamics(self.state, u)
        state_nominal = self.state + dsdt_nominal * self.dt
        state_nominal[3:6] = np.clip(state_nominal[3:6], -self.max_speed, self.max_speed)
        state_nominal[6:] = np.clip(state_nominal[6:], -self.max_theta, self.max_theta)

        obstacle = self.get_obstacle(state)
        goal = self.get_goal(state)
        self.state = state
        done = int(self.t / self.dt) == len(self.reference_traj[0]['traj'])
        if np.linalg.norm(goal[:3] - state[:3]) > 20:

            done = True
        self.t = min(self.t + self.dt, len(self.reference_traj[0]['traj']) * self.dt)
        return state, state_nominal, obstacle, goal, done

    def uncertain_dynamics(self, state, u):
        A = np.array(self.A_real, dtype=np.float32)
        B = np.array(self.B_real, dtype=np.float32)
        dsdt = A.dot(state) + B.dot(u)

        return dsdt

    def nominal_dynamics(self, state, u):
        A = np.array(self.A_nominal, dtype=np.float32)
        B = np.array(self.B_nominal, dtype=np.float32)
        dsdt = A.dot(state) + B.dot(u)

        return dsdt

    def nominal_dynamics_torch(self, state, u):
        A = np.array(self.A_nominal, dtype=np.float32)
        A_T = torch.from_numpy(A.T)

        B = np.array(self.B_nominal, dtype=np.float32)
        B_T = torch.from_numpy(B.T)

        dsdt = torch.matmul(state, A_T) + torch.matmul(u, B_T)

        return dsdt

    def nominal_controller(self, state, goal, u_norm_max=0.5):
        K = self.K
        u_nominal = -K.dot(state - goal)
        norm = np.linalg.norm(u_nominal)
        if norm > u_norm_max:
            u_nominal = u_nominal / norm * u_norm_max
        return u_nominal

    def get_obstacle(self, state):
        pos = state[:3]
        ind_max = int(self.npc_speed * self.t / self.dt)
        obstacle = []
        for i in range(1, self.num_npc):
            ind = min(ind_max, len(self.reference_traj[i]['traj']) - 1)
            obstacle_pos = self.reference_traj[i]['traj'][ind][:3]
            if np.linalg.norm(pos - obstacle_pos) < self.perception_range:
                obstacle.append(obstacle_pos)

        if len(obstacle) < self.k_obstacle:
            if len(obstacle) == 0:
                obstacle.append(state[:3] + np.array([10, 10, 10]))
            while len(obstacle) < self.k_obstacle:
                obstacle.append(obstacle[-1])
        elif len(obstacle) > self.k_obstacle:
            obstacle = np.array(obstacle)
            dist = np.linalg.norm(obstacle - pos, axis=1)
            indices = np.argsort(dist)
            obstacle = obstacle[indices[:self.k_obstacle]]
        else:
            pass
        obstacle = np.concatenate([np.array(obstacle), np.zeros((self.k_obstacle, 5))], axis=1)
        return obstacle

    def get_goal(self, state, forward_t=10.0):
        ind = int((self.t + forward_t) / self.dt)
        ind = min(ind, len(self.reference_traj[0]['traj']) - 1)
        goal = self.reference_traj[0]['traj'][ind]
        goal = np.concatenate([goal, np.zeros((5,))])
        return goal

    def get_noise(self):
        if np.random.uniform() < 0.05:
            self.noise = np.random.normal(size=(8,)) * self.noise_std
        noise = np.copy(self.noise)
        noise[:3] = 0
        return noise

    def get_all_agent_state(self):
        ind_max = int(self.npc_speed * self.t / self.dt)
        all_agent_state = [self.state]
        for i in range(1, self.num_npc):
            ind = min(ind_max, len(self.reference_traj[i]['traj']) - 1)
            obstacle_pos = self.reference_traj[i]['traj'][ind][:3]
            obstacle_state = np.concatenate([obstacle_pos, np.zeros(5,)])
            all_agent_state.append(obstacle_state)
        all_agent_state = np.array(all_agent_state)
        return all_agent_state

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    env = City(preplanned_traj='data/reference_traj.json')
    fig = plt.figure()

    ax = plt.axes(projection ='3d')

    for single_agent_traj in env.reference_traj:
        traj = single_agent_traj["traj"]
        ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2])

    ax.set_title('Trajectories for all agents')
    plt.show()

    state, obstacle, goal = env.reset()
    print('Starting position: {}'.format(state[:3]))
    while True:
        u = env.nominal_controller(state, goal)
        state, state_nominal, obstacle, goal, done = env.step(u)
        print('Distance to goal: {:.2f}'.format(np.linalg.norm(state[:3] - goal[:3])))
        if done:
            break
    print('End position: {}'.format(state[:3]))

class MovingDrone(Drone):

    def __init__(self, npc_speed=0.05, **kwargs):
        super(MovingDrone, self).__init__(**kwargs)
        self.npc_speed = float(npc_speed)
        self.obstacle_velocity = None

    def reset(self):
        state, obstacle, goal = super(MovingDrone, self).reset()
        directions = np.random.normal(size=(self.total_obstacle, 3))
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / np.maximum(norms, 1e-8)
        self.obstacle_velocity = self.npc_speed * directions
        return state, obstacle, goal

    def _move_obstacles(self):
        if self.obstacle_velocity is None or self.npc_speed <= 0.0:
            return
        next_position = self.obstacle + self.obstacle_velocity * self.dt
        for axis in range(3):
            reflected = np.logical_or(
                next_position[:, axis] < 0.0,
                next_position[:, axis] > self.env_size,
            )
            self.obstacle_velocity[reflected, axis] *= -1.0
        self.obstacle = np.clip(next_position, 0.0, self.env_size)

    def step(self, u):
        dsdt = self.uncertain_dynamics(self.state, u)
        noise = self.get_noise()
        state = self.state + (dsdt + noise) * self.dt
        state[3:6] = np.clip(state[3:6], -self.max_speed, self.max_speed)
        state[6:] = np.clip(state[6:], -self.max_theta, self.max_theta)

        dsdt_nominal = self.nominal_dynamics(self.state, u)
        state_nominal = self.state + dsdt_nominal * self.dt
        state_nominal[3:6] = np.clip(state_nominal[3:6], -self.max_speed, self.max_speed)
        state_nominal[6:] = np.clip(state_nominal[6:], -self.max_theta, self.max_theta)

        self._move_obstacles()
        obstacle = self.get_obstacle(state)
        goal = self.get_goal(state)
        self.state = state
        done = (np.linalg.norm(state[:3] - goal[:3]) < self.safe_dist
                or self.num_steps > self.max_steps)
        self.num_steps = self.num_steps + 1
        return state, state_nominal, obstacle, goal, done
