"""
Diffusion process and noise scheduling.

Implements:
- Noise schedules (linear, cosine, sigmoid)
- Forward diffusion (adding noise)
- Reverse diffusion (DDPM and DDIM sampling)
"""

from typing import Tuple, Optional
import math

import torch
import torch.nn as nn
import numpy as np


class NoiseScheduler:
    """
    Noise scheduler for diffusion models.

    Supports multiple schedule types:
    - linear: Linear increase from beta_start to beta_end
    - cosine: Smoother schedule (recommended)
    - sigmoid: S-curve schedule
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule_type: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        s: float = 0.008  # Cosine schedule offset
    ):
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type

        # Compute betas
        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            betas = self._cosine_beta_schedule(num_timesteps, s)
        elif schedule_type == "sigmoid":
            betas = self._sigmoid_beta_schedule(num_timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        # Store all diffusion coefficients
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]),
            self.alphas_cumprod[:-1]
        ])

        # Pre-computed values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # Posterior variance (for DDPM sampling)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )

        # Posterior mean coefficients
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as in 'Improved DDPM'."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def _sigmoid_beta_schedule(
        self,
        timesteps: int,
        start: float,
        end: float
    ) -> torch.Tensor:
        """Sigmoid schedule."""
        betas = torch.linspace(-6, 6, timesteps)
        return torch.sigmoid(betas) * (end - start) + start

    def to(self, device: torch.device) -> 'NoiseScheduler':
        """Move all tensors to device."""
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        return self

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0).

        x_t = sqrt(alpha_cumprod_t) * x_0 + sqrt(1 - alpha_cumprod_t) * noise

        Args:
            x_0: Original data (batch, dim)
            noise: Gaussian noise (batch, dim)
            t: Timesteps (batch,)

        Returns:
            x_t: Noisy data at timestep t
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def get_velocity(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute velocity target for v-prediction.

        v = sqrt(alpha_cumprod) * noise - sqrt(1 - alpha_cumprod) * x_0
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_alpha * noise - sqrt_one_minus_alpha * x_0

    def predict_x0_from_noise(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor
    ) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise."""
        return (
            self.sqrt_recip_alphas_cumprod[t].view(-1, 1) * x_t -
            self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1) * noise
        )

    def predict_x0_from_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        velocity: torch.Tensor
    ) -> torch.Tensor:
        """Predict x_0 from x_t and predicted velocity."""
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        return sqrt_alpha * x_t - sqrt_one_minus_alpha * velocity

    def q_posterior_mean_variance(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute posterior q(x_{t-1} | x_t, x_0).

        Returns mean and variance.
        """
        posterior_mean = (
            self.posterior_mean_coef1[t].view(-1, 1) * x_0 +
            self.posterior_mean_coef2[t].view(-1, 1) * x_t
        )
        posterior_variance = self.posterior_variance[t].view(-1, 1)
        return posterior_mean, posterior_variance


class DiffusionProcess:
    """
    Diffusion process manager.

    Handles training and sampling.
    """

    def __init__(
        self,
        scheduler: NoiseScheduler,
        prediction_type: str = "epsilon"  # "epsilon", "v", "x_0"
    ):
        self.scheduler = scheduler
        self.prediction_type = prediction_type

    def training_losses(
        self,
        denoising_net: nn.Module,
        x_0: torch.Tensor,
        condition: torch.Tensor,
        t: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute training loss.

        Args:
            denoising_net: Denoising network
            x_0: Original data (batch, 1)
            condition: Condition embedding (batch, cond_dim)
            t: Optional timesteps (batch,), sampled if not provided

        Returns:
            loss: Scalar loss tensor
            info: Dict with additional info
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Sample timesteps if not provided
        if t is None:
            t = torch.randint(
                0, self.scheduler.num_timesteps,
                (batch_size,),
                device=device
            )

        # Sample noise
        noise = torch.randn_like(x_0)

        # Forward diffusion
        x_t = self.scheduler.add_noise(x_0, noise, t)

        # Predict
        predicted = denoising_net(x_t, t, condition)

        # Compute target based on prediction type
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "v":
            target = self.scheduler.get_velocity(x_0, noise, t)
        elif self.prediction_type == "x_0":
            target = x_0
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # MSE loss
        loss = nn.functional.mse_loss(predicted, target)

        info = {
            't': t,
            'predicted': predicted,
            'target': target,
            'x_t': x_t
        }

        return loss, info

    @torch.no_grad()
    def ddpm_sample_step(
        self,
        denoising_net: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Single DDPM sampling step.

        Args:
            denoising_net: Denoising network
            x_t: Current noisy sample
            t: Current timestep (scalar or batch)
            condition: Condition embedding

        Returns:
            x_{t-1}: Sample at previous timestep
        """
        # Ensure t is a tensor
        if isinstance(t, int):
            t = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)

        # Predict
        predicted = denoising_net(x_t, t, condition)

        # Get x_0 prediction based on prediction type
        if self.prediction_type == "epsilon":
            pred_x0 = self.scheduler.predict_x0_from_noise(x_t, t, predicted)
        elif self.prediction_type == "v":
            pred_x0 = self.scheduler.predict_x0_from_velocity(x_t, t, predicted)
        elif self.prediction_type == "x_0":
            pred_x0 = predicted
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # Clip predicted x_0 for stability
        pred_x0 = torch.clamp(pred_x0, -10, 10)

        # Get posterior mean
        posterior_mean, posterior_variance = self.scheduler.q_posterior_mean_variance(
            pred_x0, x_t, t
        )

        # Sample x_{t-1}
        noise = torch.randn_like(x_t)

        # No noise at t=0
        nonzero_mask = (t != 0).float().view(-1, 1)
        x_prev = posterior_mean + nonzero_mask * torch.sqrt(posterior_variance) * noise

        return x_prev

    @torch.no_grad()
    def ddpm_sample(
        self,
        denoising_net: nn.Module,
        condition: torch.Tensor,
        num_samples: int = 1
    ) -> torch.Tensor:
        """
        Full DDPM sampling.

        Args:
            denoising_net: Denoising network
            condition: Condition embedding (batch, cond_dim)
            num_samples: Number of samples per condition

        Returns:
            samples: (batch * num_samples, 1)
        """
        batch_size = condition.shape[0]
        device = condition.device

        # Expand condition for multiple samples
        if num_samples > 1:
            condition = condition.repeat_interleave(num_samples, dim=0)

        total_samples = batch_size * num_samples

        # Start from pure noise
        x_t = torch.randn(total_samples, 1, device=device)

        # Reverse diffusion
        for t in reversed(range(self.scheduler.num_timesteps)):
            x_t = self.ddpm_sample_step(denoising_net, x_t, t, condition)

        return x_t

    @torch.no_grad()
    def ddim_sample(
        self,
        denoising_net: nn.Module,
        condition: torch.Tensor,
        num_samples: int = 1,
        num_steps: int = 50,
        eta: float = 0.0  # 0 = deterministic, 1 = DDPM
    ) -> torch.Tensor:
        """
        DDIM sampling (faster).

        Args:
            denoising_net: Denoising network
            condition: Condition embedding (batch, cond_dim)
            num_samples: Number of samples per condition
            num_steps: Number of sampling steps (< num_timesteps)
            eta: Stochasticity parameter (0 = deterministic)

        Returns:
            samples: (batch * num_samples, 1)
        """
        batch_size = condition.shape[0]
        device = condition.device

        # Expand condition for multiple samples
        if num_samples > 1:
            condition = condition.repeat_interleave(num_samples, dim=0)

        total_samples = batch_size * num_samples

        # Create timestep sequence
        step_size = self.scheduler.num_timesteps // num_steps
        timesteps = list(range(0, self.scheduler.num_timesteps, step_size))[::-1]

        # Start from pure noise
        x_t = torch.randn(total_samples, 1, device=device)

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((total_samples,), t, device=device, dtype=torch.long)

            # Predict
            predicted = denoising_net(x_t, t_tensor, condition)

            # Get x_0 prediction
            if self.prediction_type == "epsilon":
                pred_x0 = self.scheduler.predict_x0_from_noise(x_t, t_tensor, predicted)
                pred_noise = predicted
            elif self.prediction_type == "v":
                pred_x0 = self.scheduler.predict_x0_from_velocity(x_t, t_tensor, predicted)
                # Compute corresponding noise
                sqrt_alpha = self.scheduler.sqrt_alphas_cumprod[t_tensor].view(-1, 1)
                sqrt_one_minus_alpha = self.scheduler.sqrt_one_minus_alphas_cumprod[t_tensor].view(-1, 1)
                pred_noise = (x_t - sqrt_alpha * pred_x0) / sqrt_one_minus_alpha
            elif self.prediction_type == "x_0":
                pred_x0 = predicted
                sqrt_alpha = self.scheduler.sqrt_alphas_cumprod[t_tensor].view(-1, 1)
                sqrt_one_minus_alpha = self.scheduler.sqrt_one_minus_alphas_cumprod[t_tensor].view(-1, 1)
                pred_noise = (x_t - sqrt_alpha * pred_x0) / sqrt_one_minus_alpha
            else:
                raise ValueError(f"Unknown prediction type: {self.prediction_type}")

            # Clip for stability
            pred_x0 = torch.clamp(pred_x0, -10, 10)

            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]

                alpha_cumprod = self.scheduler.alphas_cumprod[t]
                alpha_cumprod_prev = self.scheduler.alphas_cumprod[t_prev]

                # DDIM update
                sigma = eta * torch.sqrt(
                    (1 - alpha_cumprod_prev) / (1 - alpha_cumprod) *
                    (1 - alpha_cumprod / alpha_cumprod_prev)
                )

                # Predicted direction
                pred_dir = torch.sqrt(1 - alpha_cumprod_prev - sigma ** 2) * pred_noise

                # Sample x_{t-1}
                x_t = torch.sqrt(alpha_cumprod_prev) * pred_x0 + pred_dir

                if eta > 0:
                    noise = torch.randn_like(x_t)
                    x_t = x_t + sigma * noise
            else:
                # Last step: return pred_x0
                x_t = pred_x0

        return x_t

    @torch.no_grad()
    def sample(
        self,
        denoising_net: nn.Module,
        condition: torch.Tensor,
        num_samples: int = 1,
        use_ddim: bool = True,
        ddim_steps: int = 50,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Sample from the diffusion model.

        Args:
            denoising_net: Denoising network
            condition: Condition embedding (batch, cond_dim)
            num_samples: Number of samples per condition
            use_ddim: Use DDIM (faster) or DDPM sampling
            ddim_steps: Number of DDIM steps
            eta: DDIM stochasticity

        Returns:
            samples: (batch * num_samples, 1)
        """
        if use_ddim:
            return self.ddim_sample(
                denoising_net, condition, num_samples, ddim_steps, eta
            )
        else:
            return self.ddpm_sample(denoising_net, condition, num_samples)
