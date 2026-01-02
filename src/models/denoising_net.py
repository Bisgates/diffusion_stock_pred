"""
Conditional denoising network.

Uses MLP architecture with FiLM conditioning.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time step embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings.

        Args:
            t: (batch,) timesteps

        Returns:
            emb: (batch, dim) embeddings
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        # Handle odd dimensions
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        return emb


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.

    Modulates hidden features using condition vector:
    output = x * (1 + scale) + shift
    """

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.scale = nn.Linear(cond_dim, hidden_dim)
        self.shift = nn.Linear(cond_dim, hidden_dim)

        # Initialize to identity transform
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation.

        Args:
            x: (batch, hidden_dim) features
            cond: (batch, cond_dim) condition

        Returns:
            Modulated features
        """
        scale = self.scale(cond)
        shift = self.shift(cond)
        return x * (1 + scale) + shift


class ConditionalMLPBlock(nn.Module):
    """MLP block with FiLM conditioning."""

    def __init__(
        self,
        hidden_dim: int,
        cond_dim: int,
        dropout: float = 0.1,
        expansion: int = 2
    ):
        super().__init__()

        self.norm = nn.LayerNorm(hidden_dim)
        self.film = FiLMLayer(hidden_dim, cond_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * expansion, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Forward with residual connection.

        Args:
            x: (batch, hidden_dim)
            cond: (batch, cond_dim)

        Returns:
            Output with residual
        """
        residual = x

        x = self.norm(x)
        x = self.film(x, cond)
        x = self.mlp(x)

        return x + residual


class ConditionalDenoisingMLP(nn.Module):
    """
    Conditional denoising MLP.

    Takes noisy sample, timestep, and condition to predict noise/velocity.

    Architecture:
    - Input projection
    - Time embedding + condition concatenation
    - Stack of conditional MLP blocks with FiLM
    - Output projection
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        cond_dim: int = 64,
        time_dim: int = 64,
        num_blocks: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim)
        )

        # Combined condition dimension
        total_cond_dim = cond_dim + time_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Conditional MLP blocks
        self.blocks = nn.ModuleList([
            ConditionalMLPBlock(hidden_dim, total_cond_dim, dropout)
            for _ in range(num_blocks)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

        # Initialize output to small values
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise/velocity from noisy sample.

        Args:
            x: (batch, 1) noisy sample
            t: (batch,) diffusion timestep
            condition: (batch, cond_dim) condition from encoder

        Returns:
            predicted: (batch, 1) predicted noise/velocity
        """
        # Time embedding
        t_emb = self.time_embed(t)

        # Combine condition and time
        combined_cond = torch.cat([condition, t_emb], dim=-1)

        # Input projection
        h = self.input_proj(x)

        # Apply conditional blocks
        for block in self.blocks:
            h = block(h, combined_cond)

        # Output projection
        output = self.output_proj(h)

        return output


class CrossAttentionDenoisingMLP(nn.Module):
    """
    Alternative: Denoising MLP with cross-attention conditioning.

    More expressive than FiLM but slightly more expensive.
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        cond_dim: int = 64,
        time_dim: int = 64,
        num_blocks: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, hidden_dim)
        )

        # Condition projection
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Blocks with cross-attention
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(hidden_dim),
                'cross_attn': nn.MultiheadAttention(
                    hidden_dim, num_heads, dropout=dropout, batch_first=True
                ),
                'norm2': nn.LayerNorm(hidden_dim),
                'mlp': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Dropout(dropout)
                )
            }))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict noise/velocity with cross-attention.

        Args:
            x: (batch, 1) noisy sample
            t: (batch,) diffusion timestep
            condition: (batch, cond_dim) condition

        Returns:
            predicted: (batch, 1) predicted noise/velocity
        """
        batch_size = x.shape[0]

        # Embeddings
        t_emb = self.time_embed(t)  # (batch, hidden_dim)
        cond_emb = self.cond_proj(condition)  # (batch, hidden_dim)

        # Create key-value pairs from condition + time
        kv = torch.stack([cond_emb, t_emb], dim=1)  # (batch, 2, hidden_dim)

        # Input as query
        h = self.input_proj(x).unsqueeze(1)  # (batch, 1, hidden_dim)

        # Apply blocks
        for block in self.blocks:
            # Cross-attention
            h_norm = block['norm1'](h)
            attn_out, _ = block['cross_attn'](h_norm, kv, kv)
            h = h + attn_out

            # MLP
            h_norm = block['norm2'](h)
            h = h + block['mlp'](h_norm)

        # Output
        h = h.squeeze(1)  # (batch, hidden_dim)
        output = self.output_proj(h)

        return output


def create_denoiser(
    denoiser_type: str = "mlp",
    input_dim: int = 1,
    hidden_dim: int = 128,
    cond_dim: int = 64,
    time_dim: int = 64,
    num_blocks: int = 4,
    dropout: float = 0.1,
    **kwargs
) -> nn.Module:
    """
    Factory function to create denoiser.

    Args:
        denoiser_type: "mlp" or "cross_attn"
        Other args passed to constructor

    Returns:
        Denoiser module
    """
    if denoiser_type == "mlp":
        return ConditionalDenoisingMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            cond_dim=cond_dim,
            time_dim=time_dim,
            num_blocks=num_blocks,
            dropout=dropout
        )
    elif denoiser_type == "cross_attn":
        return CrossAttentionDenoisingMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            cond_dim=cond_dim,
            time_dim=time_dim,
            num_blocks=num_blocks,
            dropout=dropout,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown denoiser type: {denoiser_type}")
