# Transition dataset for vector-field learning.

import numpy as np
import torch

class VectorFieldDataset(object):

    def __init__(self, n_state, k_obstacle, m_control, dt=0.1):
        self.n_state = n_state
        self.k_obstacle = k_obstacle
        self.m_control = m_control
        self.dt = dt
        self._arrays = None
        self._states = []
        self._obstacles = []
        self._obstacle_nexts = []
        self._actions = []
        self._u_nominals = []
        self._state_nexts = []
        self._state_errors = []
        self._dts = []
        self._episode_ids = []

    def add_transition(
            self, state, obstacle, action_exec, u_nominal, state_next,
            state_error, dt=None, obstacle_next=None, episode_id=None):
        self._arrays = None
        dt = self.dt if dt is None else dt
        self._states.append(np.copy(state).astype(np.float32))
        self._obstacles.append(np.copy(obstacle).astype(np.float32))
        if obstacle_next is None:
            obstacle_next = obstacle
        self._obstacle_nexts.append(
            np.copy(obstacle_next).astype(np.float32))
        self._actions.append(np.copy(action_exec).astype(np.float32))
        self._u_nominals.append(np.copy(u_nominal).astype(np.float32))
        self._state_nexts.append(np.copy(state_next).astype(np.float32))
        self._state_errors.append(np.copy(state_error).astype(np.float32))
        self._dts.append(np.float32(dt))
        self._episode_ids.append(-1 if episode_id is None else int(episode_id))

    @property
    def num_samples(self):
        return self.arrays()["state"].shape[0]

    def arrays(self):
        if self._arrays is not None:
            return self._arrays

        state = self._stack_or_empty(self._states, (0, self.n_state))
        obstacle = self._stack_or_empty(self._obstacles, (0, self.k_obstacle, self.n_state))
        obstacle_next = self._stack_or_empty(
            self._obstacle_nexts, (0, self.k_obstacle, self.n_state))
        action = self._stack_or_empty(self._actions, (0, self.m_control))
        u_nominal = self._stack_or_empty(self._u_nominals, (0, self.m_control))
        state_next = self._stack_or_empty(self._state_nexts, (0, self.n_state))
        state_error = self._stack_or_empty(self._state_errors, (0, self.n_state))
        dt = np.array(self._dts, dtype=np.float32)
        if dt.shape[0] == 0:
            dt = np.zeros((0,), dtype=np.float32)
        episode_ids = np.asarray(self._episode_ids, dtype=np.int64)
        if episode_ids.shape[0] == 0:
            episode_ids = np.zeros((0,), dtype=np.int64)
        vector = (state_next - state) / np.maximum(dt.reshape(-1, 1), 1e-8)
        self._arrays = {
            "state": state,
            "obstacle": obstacle,
            "obstacle_next": obstacle_next,
            "action": action,
            "u_nominal": u_nominal,
            "state_next": state_next,
            "state_error": state_error,
            "dt": dt,
            "episode_id": episode_ids,
            "vector": vector.astype(np.float32),
        }
        return self._arrays

    def sample_batch(self, batch_size, indices=None, device=None):
        arrays = self.arrays()
        if indices is None:
            indices = np.arange(arrays["state"].shape[0])
        batch_indices = np.random.choice(indices, size=batch_size, replace=len(indices) < batch_size)
        batch = {}
        for key in ["state", "obstacle", "obstacle_next", "action", "u_nominal", "state_next", "state_error", "vector"]:
            value = torch.from_numpy(arrays[key][batch_indices]).float()
            if device is not None:
                value = value.to(device)
            batch[key] = value
        return batch

    def split_indices(self, calibration_fraction=0.2, seed=0):
        from modules.diffusion_repair.experiment_protocol import split_indices_by_group
        if not 0.0 < float(calibration_fraction) < 1.0:
            raise ValueError("calibration_fraction must be in (0, 1)")
        episode_ids = self.arrays().get("episode_id")
        if episode_ids is not None and np.all(episode_ids >= 0):
            return split_indices_by_group(episode_ids, calibration_fraction, seed)
        raise ValueError(
            "episode IDs are required for a leakage-free split; "
            "recollect data or provide an explicit independent calibration dataset")

    def normalization_stats(self, indices=None, eps=1e-6):
        arrays = self.arrays()
        if indices is None:
            indices = np.arange(arrays["state"].shape[0])
        stats = {}
        for source, prefix in [("state", "state"), ("action", "action"), ("vector", "vector")]:
            value = arrays[source][indices]
            stats[prefix + "_mean"] = value.mean(axis=0).astype(np.float32)
            stats[prefix + "_std"] = np.maximum(value.std(axis=0), eps).astype(np.float32)
        return stats

    def save(self, path):
        arrays = self.arrays()
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path):
        data = np.load(path)
        state = data["state"].astype(np.float32)
        obstacle = data["obstacle"].astype(np.float32)
        action = data["action"].astype(np.float32)
        dt = data["dt"].astype(np.float32)
        dataset = cls(
            n_state=state.shape[1],
            k_obstacle=obstacle.shape[1],
            m_control=action.shape[1],
            dt=float(dt[0]) if dt.shape[0] > 0 else 0.1,
        )
        state_next = data["state_next"].astype(np.float32)
        obstacle_next = (
            data["obstacle_next"].astype(np.float32)
            if "obstacle_next" in data else obstacle.copy()
        )
        vector = data["vector"].astype(np.float32) if "vector" in data else (
            state_next - state) / np.maximum(dt.reshape(-1, 1), 1e-8)
        dataset._arrays = {
            "state": state,
            "obstacle": obstacle,
            "obstacle_next": obstacle_next,
            "action": action,
            "u_nominal": data["u_nominal"].astype(np.float32),
            "state_next": state_next,
            "state_error": data["state_error"].astype(np.float32),
            "dt": dt,
            "episode_id": (
                data["episode_id"].astype(np.int64)
                if "episode_id" in data
                else np.full(state.shape[0], -1, dtype=np.int64)
            ),
            "vector": vector.astype(np.float32),
        }
        return dataset

    @staticmethod
    def _stack_or_empty(values, shape):
        if len(values) == 0:
            return np.zeros(shape, dtype=np.float32)
        return np.stack(values, axis=0).astype(np.float32)
