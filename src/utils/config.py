"""
Configuration management utilities.

Provides:
- YAML configuration loading/saving
- Configuration merging
- Default configurations
"""

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        path: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config or {}


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.

    Args:
        base: Base configuration
        override: Override values

    Returns:
        Merged configuration
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for MacBook training."""
    return {
        'data': {
            'data_dir': 'data_1s',
            'sequence_length': 120,
            'prediction_horizon': 30,
            'stride': 1,
            'input_features': [
                'log_return',
                'volatility_5s',
                'spread',
                'volume_zscore'
            ],
            'batch_size': 128,
            'num_workers': 0,  # MPS works better with 0 workers
        },
        'model': {
            'encoder_type': 'transformer',
            'encoder_dim': 64,
            'encoder_layers': 2,
            'encoder_heads': 4,
            'diffusion_steps': 500,
            'noise_schedule': 'cosine',
            'prediction_type': 'epsilon',
            'denoiser_type': 'mlp',
            'denoising_hidden_dim': 128,
            'denoising_blocks': 4,
            'time_embedding_dim': 64,
            'dropout': 0.1,
        },
        'training': {
            'epochs': 50,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'gradient_accumulation': 2,
            'warmup_steps': 500,
            'scheduler_type': 'cosine',
            'min_lr': 1e-6,
            'max_grad_norm': 1.0,
            'early_stopping_patience': 10,
            'device': 'mps',
        },
        'inference': {
            'num_samples': 100,
            'use_ddim': True,
            'ddim_steps': 50,
            'eta': 0.0,
        },
        'output': {
            'checkpoint_dir': 'outputs/checkpoints',
            'log_dir': 'outputs/logs',
            'save_every_epochs': 5,
        }
    }


def dataclass_to_dict(obj: Any) -> Dict[str, Any]:
    """Convert dataclass to dictionary."""
    if hasattr(obj, '__dataclass_fields__'):
        return asdict(obj)
    return obj


class ConfigManager:
    """
    Configuration manager for experiments.

    Handles loading, merging, and validating configurations.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize config manager.

        Args:
            config_path: Optional path to config file
        """
        self.config = get_default_config()

        if config_path:
            user_config = load_config(config_path)
            self.config = merge_configs(self.config, user_config)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value by key.

        Supports dot notation: "model.encoder_dim"
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set config value by key.

        Supports dot notation.
        """
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save(self, path: Union[str, Path]) -> None:
        """Save current configuration."""
        save_config(self.config, path)

    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary."""
        return self.config.copy()
