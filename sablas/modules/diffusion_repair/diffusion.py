# Conditional vector-field diffusion model  p(v | x, u).

import math

import torch
from torch import nn
import torch.nn.functional as F

def sinusoidal_time_embedding(timestep, dim):
    half_dim = dim // 2
    exponent = -math.log(10000.0) * torch.arange(
        half_dim, device=timestep.device, dtype=torch.float32) / max(half_dim - 1, 1)
    angles = timestep.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding

class ConditionalVectorFieldDiffusion(nn.Module):

    def __init__(
            self,
            state_dim,
            action_dim,
            vector_dim=None,
            hidden_dim=256,
            time_dim=64,
            diffusion_steps=20,
            beta_start=1e-4,
            beta_end=2e-2):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.vector_dim = state_dim if vector_dim is None else vector_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.diffusion_steps = diffusion_steps

        input_dim = self.vector_dim + self.state_dim + self.action_dim + time_dim
        self.noise_model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.vector_dim),
        )

        betas = torch.linspace(beta_start, beta_end, diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([
            torch.ones(1, dtype=torch.float32), alpha_bars[:-1]], dim=0)
        posterior_variance = (
            betas * (1.0 - alpha_bars_prev)
            / (1.0 - alpha_bars).clamp_min(1e-12)
        )
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer(
            "posterior_variance", posterior_variance, persistent=False)
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_std", torch.ones(action_dim))
        self.register_buffer("vector_mean", torch.zeros(self.vector_dim))
        self.register_buffer("vector_std", torch.ones(self.vector_dim))
        self.q_alpha = None
        self.q_alpha_episode_max = None
        self.credible_scale = 1.0
        self.calibration_metadata = {}

    def set_normalization(self, stats):
        self.state_mean.copy_(torch.as_tensor(stats["state_mean"], dtype=torch.float32, device=self.state_mean.device))
        self.state_std.copy_(torch.as_tensor(stats["state_std"], dtype=torch.float32, device=self.state_std.device))
        self.action_mean.copy_(torch.as_tensor(stats["action_mean"], dtype=torch.float32, device=self.action_mean.device))
        self.action_std.copy_(torch.as_tensor(stats["action_std"], dtype=torch.float32, device=self.action_std.device))
        self.vector_mean.copy_(torch.as_tensor(stats["vector_mean"], dtype=torch.float32, device=self.vector_mean.device))
        self.vector_std.copy_(torch.as_tensor(stats["vector_std"], dtype=torch.float32, device=self.vector_std.device))

    def get_normalization(self):
        return {
            "state_mean": self.state_mean.detach().cpu().numpy(),
            "state_std": self.state_std.detach().cpu().numpy(),
            "action_mean": self.action_mean.detach().cpu().numpy(),
            "action_std": self.action_std.detach().cpu().numpy(),
            "vector_mean": self.vector_mean.detach().cpu().numpy(),
            "vector_std": self.vector_std.detach().cpu().numpy(),
        }

    def normalize_state(self, state):
        return (state - self.state_mean) / self.state_std.clamp_min(1e-6)

    def normalize_action(self, action):
        return (action - self.action_mean) / self.action_std.clamp_min(1e-6)

    def normalize_vector(self, vector):
        return (vector - self.vector_mean) / self.vector_std.clamp_min(1e-6)

    def denormalize_vector(self, vector):
        return vector * self.vector_std.clamp_min(1e-6) + self.vector_mean

    def predict_noise(self, vector_t, timestep, state, action):
        if timestep.ndim == 0:
            timestep = timestep.repeat(vector_t.shape[0])
        state = self.normalize_state(state)
        action = self.normalize_action(action)
        time_embedding = sinusoidal_time_embedding(timestep, self.time_dim)
        x = torch.cat([vector_t, state, action, time_embedding], dim=1)
        return self.noise_model(x)

    def training_loss(self, state, action, vector):
        batch_size = state.shape[0]
        vector = self.normalize_vector(vector)
        timestep = torch.randint(
            0, self.diffusion_steps, (batch_size,), device=state.device, dtype=torch.long)
        noise = torch.randn_like(vector)
        alpha_bar = self.alpha_bars[timestep].unsqueeze(1)
        vector_t = torch.sqrt(alpha_bar) * vector + torch.sqrt(1.0 - alpha_bar) * noise
        noise_pred = self.predict_noise(vector_t, timestep, state, action)
        return F.mse_loss(noise_pred, noise)

    def sample_vectors(self, state, action, num_samples=32):
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        batch_size = state.shape[0]
        state_rep = state.repeat_interleave(num_samples, dim=0)
        action_rep = action.repeat_interleave(num_samples, dim=0)
        vector = torch.randn(
            batch_size * num_samples, self.vector_dim, device=state.device, dtype=state.dtype)

        for step in reversed(range(self.diffusion_steps)):
            timestep = torch.full(
                (batch_size * num_samples,), step, device=state.device, dtype=torch.long)
            beta = self.betas[step]
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            noise_pred = self.predict_noise(vector, timestep, state_rep, action_rep)
            vector = (vector - beta / torch.sqrt(1.0 - alpha_bar) * noise_pred) / torch.sqrt(alpha)
            if step > 0:
                variance = self.posterior_variance[step].clamp_min(1e-12)
                vector = vector + torch.sqrt(variance) * torch.randn_like(vector)

        vector = self.denormalize_vector(vector)
        return vector.view(batch_size, num_samples, self.vector_dim)

    def checkpoint(self):
        return {
            "state_dict": self.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "vector_dim": self.vector_dim,
            "time_dim": self.time_dim,
            "hidden_dim": self.hidden_dim,
            "diffusion_steps": self.diffusion_steps,
            "q_alpha": self.q_alpha,
            "q_alpha_episode_max": self.q_alpha_episode_max,
            "credible_scale": self.credible_scale,
            "calibration_metadata": self.calibration_metadata,
        }

def load_diffusion_checkpoint(path, map_location=None):
    checkpoint = torch.load(path, map_location=map_location)
    model = ConditionalVectorFieldDiffusion(
        state_dim=checkpoint["state_dim"],
        action_dim=checkpoint["action_dim"],
        vector_dim=checkpoint["vector_dim"],
        hidden_dim=checkpoint.get("hidden_dim", 256),
        time_dim=checkpoint.get("time_dim", 64),
        diffusion_steps=checkpoint.get("diffusion_steps", 20),
    )
    if map_location is not None:
        model = model.to(map_location)
    model.load_state_dict(checkpoint["state_dict"])
    model.q_alpha = checkpoint.get("q_alpha", None)
    model.q_alpha_episode_max = checkpoint.get("q_alpha_episode_max", None)
    model.calibration_metadata = checkpoint.get("calibration_metadata", {})
    model.credible_scale = checkpoint.get("credible_scale", 1.0)
    return model

class ConditionalVectorFieldRegressor(nn.Module):

    model_type = "calibrated_residual_mlp"

    def __init__(self, state_dim, action_dim, vector_dim=None, hidden_dim=256):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.vector_dim = self.state_dim if vector_dim is None else int(vector_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.SiLU(),
            nn.Linear(self.hidden_dim, self.vector_dim),
        )
        self.register_buffer("state_mean", torch.zeros(self.state_dim))
        self.register_buffer("state_std", torch.ones(self.state_dim))
        self.register_buffer("action_mean", torch.zeros(self.action_dim))
        self.register_buffer("action_std", torch.ones(self.action_dim))
        self.register_buffer("vector_mean", torch.zeros(self.vector_dim))
        self.register_buffer("vector_std", torch.ones(self.vector_dim))
        self.register_buffer("residual_covariance", torch.eye(self.vector_dim))
        self.q_alpha = None
        self.q_alpha_episode_max = None
        self.credible_scale = 1.0
        self.calibration_metadata = {}

    def set_normalization(self, stats):
        for name in ("state_mean", "state_std", "action_mean", "action_std",
                     "vector_mean", "vector_std"):
            target = getattr(self, name)
            target.copy_(torch.as_tensor(stats[name], dtype=target.dtype, device=target.device))

    def normalized_mean(self, state, action):
        x = (state - self.state_mean) / self.state_std.clamp_min(1e-6)
        u = (action - self.action_mean) / self.action_std.clamp_min(1e-6)
        return self.net(torch.cat([x, u], dim=-1))

    def predict_mean(self, state, action):
        return self.normalized_mean(state, action) * self.vector_std + self.vector_mean

    def training_loss(self, state, action, vector):
        target = (vector - self.vector_mean) / self.vector_std.clamp_min(1e-6)
        return F.mse_loss(self.normalized_mean(state, action), target)

    def conditional_moments(self, state, action):
        mean = self.predict_mean(state, action)
        covariance = self.residual_covariance.to(device=state.device, dtype=state.dtype)
        covariance = covariance.unsqueeze(0).expand(state.shape[0], -1, -1)
        return mean, covariance

    def sample_vectors(self, state, action, num_samples=32):
        mean = self.predict_mean(state, action)
        covariance = self.residual_covariance.to(device=state.device, dtype=state.dtype)
        covariance = covariance * float(self.credible_scale)
        eye = torch.eye(self.vector_dim, device=state.device, dtype=state.dtype)
        chol = torch.linalg.cholesky(covariance + 1e-6 * eye)
        noise = torch.randn(
            state.shape[0], int(num_samples), self.vector_dim,
            device=state.device, dtype=state.dtype)
        return mean.unsqueeze(1) + torch.einsum("bmi,ji->bmj", noise, chol)

    def checkpoint(self):
        return {
            "model_type": self.model_type,
            "state_dict": self.state_dict(),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "vector_dim": self.vector_dim,
            "hidden_dim": self.hidden_dim,
            "q_alpha": self.q_alpha,
            "q_alpha_episode_max": self.q_alpha_episode_max,
            "credible_scale": self.credible_scale,
            "calibration_metadata": self.calibration_metadata,
        }

def load_uncertainty_model(path, map_location=None):
    checkpoint = torch.load(path, map_location=map_location)
    if checkpoint.get("model_type") == ConditionalVectorFieldRegressor.model_type:
        model = ConditionalVectorFieldRegressor(
            state_dim=checkpoint["state_dim"],
            action_dim=checkpoint["action_dim"],
            vector_dim=checkpoint["vector_dim"],
            hidden_dim=checkpoint.get("hidden_dim", 256),
        )
        if map_location is not None:
            model = model.to(map_location)
        model.load_state_dict(checkpoint["state_dict"])
        model.q_alpha = checkpoint.get("q_alpha")
        model.q_alpha_episode_max = checkpoint.get("q_alpha_episode_max")
        model.calibration_metadata = checkpoint.get("calibration_metadata", {})
        model.credible_scale = checkpoint.get("credible_scale", 1.0)
        return model
    return load_diffusion_checkpoint(path, map_location=map_location)
