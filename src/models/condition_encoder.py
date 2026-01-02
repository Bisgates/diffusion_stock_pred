"""
Condition encoder for historical price sequences.

Provides lightweight Transformer and LSTM encoders optimized for MacBook.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LightweightTransformerEncoder(nn.Module):
    """
    Lightweight Transformer encoder for sequence conditioning.

    Optimized for MacBook:
    - Small model dimension (64)
    - Few layers (2)
    - Few attention heads (4)
    """

    def __init__(
        self,
        input_dim: int = 4,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 150
    ):
        super().__init__()

        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode historical sequence into condition embedding.

        Args:
            x: (batch, seq_len, input_dim) input features
            mask: (batch, seq_len) padding mask (True = valid, False = padded)

        Returns:
            condition: (batch, d_model) condition embedding
        """
        # Project to model dimension
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create attention mask if provided
        if mask is not None:
            # TransformerEncoder expects True for positions to mask
            key_padding_mask = ~mask.bool()
        else:
            key_padding_mask = None

        # Encode
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        # Use last timestep as condition (most recent information)
        condition = x[:, -1, :]

        # Project output
        condition = self.output_projection(condition)

        return condition


class LSTMEncoder(nn.Module):
    """
    LSTM encoder as a lighter alternative.

    Even more efficient than Transformer for resource-constrained environments.
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Output dimension
        output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # Output projection
        self.output_projection = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, hidden_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode historical sequence into condition embedding.

        Args:
            x: (batch, seq_len, input_dim) input features
            mask: (batch, seq_len) padding mask (not used directly, kept for API consistency)

        Returns:
            condition: (batch, hidden_dim) condition embedding
        """
        # LSTM forward
        output, (h_n, c_n) = self.lstm(x)

        # Use final hidden state
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            condition = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            condition = h_n[-1]

        # Project output
        condition = self.output_projection(condition)

        return condition


class ConvLSTMEncoder(nn.Module):
    """
    CNN + LSTM encoder for capturing local and sequential patterns.

    Combines:
    - 1D CNN for local pattern extraction
    - LSTM for sequential modeling
    """

    def __init__(
        self,
        input_dim: int = 4,
        conv_channels: int = 32,
        hidden_dim: int = 64,
        kernel_size: int = 5,
        num_lstm_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 1D CNN for local patterns
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # LSTM for sequential patterns
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode historical sequence.

        Args:
            x: (batch, seq_len, input_dim) input features

        Returns:
            condition: (batch, hidden_dim) condition embedding
        """
        # Conv1d expects (batch, channels, seq_len)
        x = x.transpose(1, 2)

        # Apply CNN
        x = self.conv(x)

        # Back to (batch, seq_len, channels)
        x = x.transpose(1, 2)

        # LSTM
        output, (h_n, c_n) = self.lstm(x)

        # Use final hidden state
        condition = h_n[-1]

        # Project output
        condition = self.output_projection(condition)

        return condition


def create_encoder(
    encoder_type: str = "transformer",
    input_dim: int = 4,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    **kwargs
) -> nn.Module:
    """
    Factory function to create encoder.

    Args:
        encoder_type: "transformer", "lstm", or "convlstm"
        input_dim: Input feature dimension
        hidden_dim: Hidden/model dimension
        num_layers: Number of layers
        dropout: Dropout rate

    Returns:
        Encoder module
    """
    if encoder_type == "transformer":
        return LightweightTransformerEncoder(
            input_dim=input_dim,
            d_model=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs
        )
    elif encoder_type == "lstm":
        return LSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs
        )
    elif encoder_type == "convlstm":
        return ConvLSTMEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
