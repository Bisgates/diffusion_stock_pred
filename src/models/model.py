"""
Full stock diffusion model.

Combines:
- Condition encoder (Transformer/LSTM)
- Diffusion process (noise scheduling, sampling)
- Denoising network (conditional MLP)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .condition_encoder import create_encoder, LightweightTransformerEncoder, LSTMEncoder
from .diffusion import NoiseScheduler, DiffusionProcess
from .denoising_net import create_denoiser, ConditionalDenoisingMLP


@dataclass
class ModelConfig:
    """Configuration for the full model."""

    # Condition encoder
    encoder_type: str = "transformer"  # "transformer", "lstm", "convlstm"
    input_features: int = 4
    seq_length: int = 120
    encoder_dim: int = 64
    encoder_layers: int = 2
    encoder_heads: int = 4

    # Diffusion model
    diffusion_steps: int = 500
    noise_schedule: str = "cosine"
    prediction_type: str = "epsilon"  # "epsilon", "v", "x_0"

    # Denoising network
    denoiser_type: str = "mlp"  # "mlp", "cross_attn"
    denoising_hidden_dim: int = 128
    denoising_blocks: int = 4
    time_embedding_dim: int = 64

    # Regularization
    dropout: float = 0.1


class StockDiffusionModel(nn.Module):
    """
    Complete stock price distribution prediction model.

    Components:
    1. Condition Encoder: Encodes historical price sequence
    2. Noise Scheduler: Manages diffusion noise schedule
    3. Denoising Network: Predicts noise/velocity
    4. Diffusion Process: Handles training and sampling
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Create condition encoder
        self.encoder = create_encoder(
            encoder_type=config.encoder_type,
            input_dim=config.input_features,
            hidden_dim=config.encoder_dim,
            num_layers=config.encoder_layers,
            dropout=config.dropout,
            nhead=config.encoder_heads if config.encoder_type == "transformer" else 4,
            max_seq_len=config.seq_length + 50
        )

        # Create noise scheduler
        self.scheduler = NoiseScheduler(
            num_timesteps=config.diffusion_steps,
            schedule_type=config.noise_schedule
        )

        # Create diffusion process
        self.diffusion = DiffusionProcess(
            scheduler=self.scheduler,
            prediction_type=config.prediction_type
        )

        # Create denoising network
        self.denoiser = create_denoiser(
            denoiser_type=config.denoiser_type,
            input_dim=1,  # Predicting 1D return
            hidden_dim=config.denoising_hidden_dim,
            cond_dim=config.encoder_dim,
            time_dim=config.time_embedding_dim,
            num_blocks=config.denoising_blocks,
            dropout=config.dropout
        )

    def to(self, device: torch.device) -> 'StockDiffusionModel':
        """Move model and scheduler to device."""
        super().to(device)
        self.scheduler.to(device)
        return self

    def forward(
        self,
        x_sequence: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass.

        Args:
            x_sequence: (batch, seq_len, n_features) historical sequence
            target: (batch, 1) target return
            mask: (batch, seq_len) optional validity mask

        Returns:
            dict with 'loss' and other training metrics
        """
        # Encode condition from historical sequence
        condition = self.encoder(x_sequence, mask)

        # Compute diffusion loss
        loss, info = self.diffusion.training_losses(
            self.denoiser,
            target,
            condition
        )

        return {
            'loss': loss,
            'condition': condition,
            **info
        }

    @torch.no_grad()
    def sample(
        self,
        x_sequence: torch.Tensor,
        num_samples: int = 100,
        mask: Optional[torch.Tensor] = None,
        use_ddim: bool = True,
        ddim_steps: int = 50,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Sample from the predicted distribution.

        Args:
            x_sequence: (batch, seq_len, n_features) historical sequence
            num_samples: Number of samples per sequence
            mask: Optional validity mask
            use_ddim: Use DDIM (faster) or DDPM sampling
            ddim_steps: Number of DDIM steps
            eta: DDIM stochasticity (0 = deterministic)

        Returns:
            samples: (batch, num_samples) sampled returns
        """
        self.eval()
        batch_size = x_sequence.shape[0]

        # Encode condition
        condition = self.encoder(x_sequence, mask)

        # Sample from diffusion
        samples = self.diffusion.sample(
            self.denoiser,
            condition,
            num_samples=num_samples,
            use_ddim=use_ddim,
            ddim_steps=ddim_steps,
            eta=eta
        )

        # Reshape to (batch, num_samples)
        samples = samples.view(batch_size, num_samples)

        return samples

    @torch.no_grad()
    def predict_distribution(
        self,
        x_sequence: torch.Tensor,
        num_samples: int = 100,
        mask: Optional[torch.Tensor] = None,
        use_ddim: bool = True,
        ddim_steps: int = 50
    ) -> Dict[str, torch.Tensor]:
        """
        Predict return distribution with statistics.

        Args:
            x_sequence: (batch, seq_len, n_features) historical sequence
            num_samples: Number of samples
            mask: Optional validity mask
            use_ddim: Use DDIM sampling
            ddim_steps: Number of DDIM steps

        Returns:
            dict with samples and distribution statistics
        """
        samples = self.sample(
            x_sequence, num_samples, mask, use_ddim, ddim_steps
        )

        # Compute statistics
        stats = self.compute_distribution_stats(samples)
        stats['samples'] = samples

        return stats

    @staticmethod
    def compute_distribution_stats(
        samples: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute distribution statistics from samples.

        Args:
            samples: (batch, num_samples) sampled values

        Returns:
            dict with mean, std, quantiles
        """
        return {
            'mean': samples.mean(dim=1),
            'std': samples.std(dim=1),
            'median': samples.median(dim=1).values,
            'q05': torch.quantile(samples, 0.05, dim=1),
            'q10': torch.quantile(samples, 0.10, dim=1),
            'q25': torch.quantile(samples, 0.25, dim=1),
            'q75': torch.quantile(samples, 0.75, dim=1),
            'q90': torch.quantile(samples, 0.90, dim=1),
            'q95': torch.quantile(samples, 0.95, dim=1),
        }

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config: Optional[ModelConfig] = None) -> StockDiffusionModel:
    """
    Factory function to create model.

    Args:
        config: Model configuration (uses defaults if None)

    Returns:
        StockDiffusionModel instance
    """
    if config is None:
        config = ModelConfig()
    return StockDiffusionModel(config)


def create_default_model(
    input_features: int = 4,
    seq_length: int = 120,
    device: str = "cpu"
) -> StockDiffusionModel:
    """
    Create model with default MacBook-friendly configuration.

    Args:
        input_features: Number of input features
        seq_length: Input sequence length
        device: Device to place model on

    Returns:
        Model on specified device
    """
    config = ModelConfig(
        input_features=input_features,
        seq_length=seq_length,
        encoder_type="transformer",
        encoder_dim=64,
        encoder_layers=2,
        encoder_heads=4,
        diffusion_steps=500,
        noise_schedule="cosine",
        prediction_type="epsilon",
        denoising_hidden_dim=128,
        denoising_blocks=4,
        dropout=0.1
    )

    model = StockDiffusionModel(config)
    model = model.to(device)

    return model
