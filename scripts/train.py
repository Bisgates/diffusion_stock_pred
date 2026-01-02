#!/usr/bin/env python3
"""
Training script for stock diffusion model.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --quick  # Quick test with limited data
    python scripts/train.py --stride 5 --max-samples 100000  # Memory efficient
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import gc

from src.data.dataset import create_datasets, create_dataloaders, SampleConfig
from src.data.preprocessing import PreprocessConfig
from src.models.model import StockDiffusionModel, ModelConfig
from src.training.trainer import Trainer, TrainingConfig
from src.utils.config import load_config, get_default_config, merge_configs


def parse_args():
    parser = argparse.ArgumentParser(description="Train stock diffusion model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (mps, cuda, cpu)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with limited data"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride for sliding window (larger = fewer samples, less memory)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per split (None = all)"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        default=None,
        help="Specific symbols to use (e.g., AMD TSLA AAPL)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging"
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Custom wandb run name"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config = get_default_config()
    config_path = Path(args.config)
    if config_path.exists():
        user_config = load_config(config_path)
        config = merge_configs(config, user_config)

    # Override with command line args
    if args.device:
        config['training']['device'] = args.device
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.no_wandb:
        config['training']['use_wandb'] = False
    if args.wandb_name:
        config['training']['wandb_run_name'] = args.wandb_name

    # Memory optimization settings
    stride = args.stride or config['data'].get('stride', 1)
    max_samples = args.max_samples

    # Quick mode for testing
    if args.quick:
        max_samples = 5000
        stride = 10
        config['training']['epochs'] = 3
        print("Quick mode: stride=10, max_samples=5000, epochs=3")

    print("=" * 60)
    print("STOCK DIFFUSION MODEL TRAINING")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Device: {config['training']['device']}")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch size: {config['data']['batch_size']} x {config['training']['gradient_accumulation']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Stride: {stride}")
    print(f"  Max samples: {max_samples or 'unlimited'}")
    print(f"  Symbols: {args.symbols or 'all'}")
    print(f"  wandb: {config['training']['use_wandb']}")
    print()

    # Check device availability
    device = config['training']['device']
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"
        config['training']['device'] = device
    elif device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
        config['training']['device'] = device

    # Create datasets
    print("Loading and preprocessing data...")
    sample_config = SampleConfig(
        sequence_length=config['data']['sequence_length'],
        prediction_horizon=config['data']['prediction_horizon'],
        stride=stride,
        input_features=config['data']['input_features']
    )

    train_dataset, val_dataset, test_dataset, metadata = create_datasets(
        data_dir=project_root / config['data']['data_dir'],
        sample_config=sample_config,
        symbols=args.symbols,
        max_samples_per_split=max_samples,
        verbose=True
    )

    # Force garbage collection after data loading
    gc.collect()

    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )

    # Create model
    print("\nCreating model...")
    model_config = ModelConfig(
        encoder_type=config['model']['encoder_type'],
        input_features=len(config['data']['input_features']),
        seq_length=config['data']['sequence_length'],
        encoder_dim=config['model']['encoder_dim'],
        encoder_layers=config['model']['encoder_layers'],
        encoder_heads=config['model']['encoder_heads'],
        diffusion_steps=config['model']['diffusion_steps'],
        noise_schedule=config['model']['noise_schedule'],
        prediction_type=config['model']['prediction_type'],
        denoiser_type=config['model']['denoiser_type'],
        denoising_hidden_dim=config['model']['denoising_hidden_dim'],
        denoising_blocks=config['model']['denoising_blocks'],
        time_embedding_dim=config['model']['time_embedding_dim'],
        dropout=config['model']['dropout']
    )

    model = StockDiffusionModel(model_config)
    print(f"Model parameters: {model.get_num_params():,}")

    # Create trainer
    training_config = TrainingConfig(
        epochs=config['training']['epochs'],
        batch_size=config['data']['batch_size'],
        gradient_accumulation=config['training']['gradient_accumulation'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        scheduler_type=config['training']['scheduler_type'],
        min_lr=config['training']['min_lr'],
        max_grad_norm=config['training']['max_grad_norm'],
        early_stopping_patience=config['training']['early_stopping_patience'],
        device=config['training']['device'],
        checkpoint_dir=Path(config['output']['checkpoint_dir']),
        save_every_epochs=config['output']['save_every_epochs'],
        keep_last_n=config['output']['keep_last_n'],
        use_wandb=config['training']['use_wandb'],
        wandb_project=config['training']['wandb_project'],
        wandb_run_name=config['training'].get('wandb_run_name'),
        log_every_steps=10  # Log every 10 steps
    )

    trainer = Trainer(model, train_loader, val_loader, training_config)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(Path(args.resume))

    # Train
    print("\nStarting training...")
    history = trainer.train()

    print("\nTraining complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")
    print(f"Checkpoints saved to: {training_config.checkpoint_dir}")


if __name__ == "__main__":
    main()
