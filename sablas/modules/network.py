# Neural network definitions for the controller and the barrier certificate.

import torch
from torch import nn
import numpy as np

class CBF(nn.Module):

    def __init__(self, n_state, k_obstacle, m_control, preprocess_func=None):
        super().__init__()
        self.n_state = n_state
        self.k_obstacle = k_obstacle
        self.m_control = m_control
        self.preprocess_func = preprocess_func

        self.conv0 = nn.Conv1d(n_state, 64, 1)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, 1, 1)
        self.activation = nn.ReLU()
        self.output_activation = nn.Tanh()

    def forward(self, state, obstacle):
        state = torch.unsqueeze(state, 2)
        obstacle = obstacle.permute(0, 2, 1)
        state_diff = state - obstacle

        if self.preprocess_func is not None:
            state_diff = self.preprocess_func(state_diff)

        x = self.activation(self.conv0(state_diff))
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.conv4(x)
        h = torch.squeeze(x, dim=1)
        return h

class NNController(nn.Module):

    def __init__(self, n_state, k_obstacle, m_control, preprocess_func=None, output_scale=1.0):
        super().__init__()
        self.n_state = n_state
        self.k_obstacle = k_obstacle
        self.m_control = m_control
        self.preprocess_func = preprocess_func

        self.conv0 = nn.Conv1d(n_state, 64, 1)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.fc0 = nn.Linear(128 + m_control + n_state, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, m_control)
        self.activation = nn.ReLU()
        self.output_activation = nn.Tanh()
        self.output_scale = output_scale

    def forward(self, state, obstacle, u_nominal, state_error):
        state = torch.unsqueeze(state, 2)
        obstacle = obstacle.permute(0, 2, 1)
        state_diff = state - obstacle

        if self.preprocess_func is not None:
            state_diff = self.preprocess_func(state_diff)
            state_error = self.preprocess_func(state_error)

        x = self.activation(self.conv0(state_diff))
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x, _ = torch.max(x, dim=2)
        x = torch.cat([x, u_nominal, state_error], dim=1)
        x = self.activation(self.fc0(x))
        x = self.activation(self.fc1(x))
        x = self.output_activation(self.fc2(x)) * self.output_scale
        u = x + u_nominal
        return u

class ControlAffineDynamics(nn.Module):

    def __init__(self, n_state, m_control, preprocess_func=None, n_extended_state=0):
        super().__init__()
        self.n_state = n_state
        self.m_control = m_control
        self.preprocess_func = preprocess_func
        self.n_extended_state = n_extended_state

        self.f_fc0 = nn.Linear(n_state + n_extended_state, 64)
        self.f_fc1 = nn.Linear(64, 128)
        self.f_fc2 = nn.Linear(128, n_state)

        self.B_fc0 = nn.Linear(n_state + n_extended_state, 64)
        self.B_fc1 = nn.Linear(64, 128)
        self.B_fc2 = nn.Linear(128, n_state * m_control)

        self.activation = nn.Tanh()

    def forward(self, state, u):
        if self.preprocess_func is not None:
            state = self.preprocess_func(state)

        x = self.activation(self.f_fc0(state))
        x = self.activation(self.f_fc1(x))
        f = self.f_fc2(x)

        x = self.activation(self.B_fc0(state))
        x = self.activation(self.B_fc1(x))
        B = self.B_fc2(x).view(-1, self.n_state, self.m_control)

        return f, B
