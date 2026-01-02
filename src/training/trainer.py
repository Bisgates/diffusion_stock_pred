"""
Training infrastructure with wandb integration.

Provides:
- Trainer class with training loop
- Per-iteration wandb logging
- Gradient accumulation
- Learning rate scheduling
- Checkpointing
- Early stopping
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, List
import time
import json
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""

    # Basic
    epochs: int = 50
    batch_size: int = 128
    gradient_accumulation: int = 2

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8

    # Learning rate schedule
    warmup_steps: int = 500
    scheduler_type: str = "cosine"
    min_lr: float = 1e-6

    # Gradient clipping
    max_grad_norm: float = 1.0

    # Checkpointing
    checkpoint_dir: Path = field(default_factory=lambda: Path("outputs/checkpoints"))
    save_every_epochs: int = 5
    keep_last_n: int = 3

    # Validation
    val_every_epochs: int = 1
    val_every_steps: Optional[int] = None  # Validate every N steps (overrides epochs)
    early_stopping_patience: int = 10

    # Device
    device: str = "mps"

    # Logging
    log_every_steps: int = 10  # Log every N steps
    use_wandb: bool = True
    wandb_project: str = "stock-diffusion"
    wandb_run_name: Optional[str] = None


class Trainer:
    """
    Model trainer with wandb integration.

    Logs every iteration to wandb for detailed monitoring.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Move model to device
        self.device = torch.device(config.device)
        self.model = self.model.to(self.device)

        # Create optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps
        )

        # Create learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history: List[Dict] = []

        # Loss tracking for averaging
        self.running_loss = 0.0
        self.running_loss_count = 0

        # Create checkpoint directory
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize wandb
        self.wandb_run = None
        if config.use_wandb:
            self._init_wandb()

        logger.info(f"Trainer initialized on {self.device}")
        logger.info(f"Model parameters: {self.model.get_num_params():,}")

    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        total_steps = len(self.train_loader) * self.config.epochs // self.config.gradient_accumulation

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=self.config.warmup_steps
        )

        if self.config.scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, total_steps - self.config.warmup_steps),
                eta_min=self.config.min_lr
            )
        elif self.config.scheduler_type == "linear":
            main_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=self.config.min_lr / self.config.learning_rate,
                total_iters=max(1, total_steps - self.config.warmup_steps)
            )
        else:
            main_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=1.0,
                total_iters=total_steps
            )

        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[self.config.warmup_steps]
        )

        return scheduler

    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            import wandb

            run_name = self.config.wandb_run_name or f"run_{time.strftime('%Y%m%d_%H%M%S')}"

            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                config={
                    'model_params': self.model.get_num_params(),
                    'batch_size': self.config.batch_size,
                    'effective_batch_size': self.config.batch_size * self.config.gradient_accumulation,
                    'learning_rate': self.config.learning_rate,
                    'epochs': self.config.epochs,
                    'device': self.config.device,
                    'gradient_accumulation': self.config.gradient_accumulation,
                    'warmup_steps': self.config.warmup_steps,
                }
            )
            logger.info(f"wandb initialized: {self.config.wandb_project}/{run_name}")

        except ImportError:
            logger.warning("wandb not installed, disabling logging")
            self.config.use_wandb = False
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            self.config.use_wandb = False

    def _log_to_wandb(self, metrics: Dict, step: Optional[int] = None):
        """Log metrics to wandb."""
        if self.wandb_run is not None:
            import wandb
            wandb.log(metrics, step=step or self.global_step)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with per-iteration logging."""
        self.model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        epoch_start = time.time()

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            batch_start = time.time()

            # Move to device
            x_seq = batch['input'].to(self.device)
            target = batch['target'].to(self.device)

            # Forward pass
            outputs = self.model(x_seq, target)
            loss = outputs['loss'] / self.config.gradient_accumulation

            # Backward pass
            loss.backward()

            # Track loss
            batch_loss = outputs['loss'].item()
            self.running_loss += batch_loss
            self.running_loss_count += 1
            epoch_loss += batch_loss
            epoch_batches += 1

            # Gradient accumulation step
            if (batch_idx + 1) % self.config.gradient_accumulation == 0:
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # Per-step logging
                if self.global_step % self.config.log_every_steps == 0:
                    avg_loss = self.running_loss / max(1, self.running_loss_count)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    batch_time = time.time() - batch_start

                    log_dict = {
                        'train/loss': avg_loss,
                        'train/loss_batch': batch_loss,
                        'train/lr': current_lr,
                        'train/grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        'train/epoch': self.current_epoch,
                        'train/step': self.global_step,
                        'train/batch_time': batch_time,
                    }

                    self._log_to_wandb(log_dict)

                    # Console logging
                    logger.info(
                        f"Step {self.global_step:6d} | "
                        f"Loss: {avg_loss:.6f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Grad: {grad_norm:.4f}"
                    )

                    # Reset running average
                    self.running_loss = 0.0
                    self.running_loss_count = 0

                # Validation during training (optional)
                if self.config.val_every_steps and self.global_step % self.config.val_every_steps == 0:
                    val_metrics = self.validate()
                    self._check_improvement(val_metrics['val_loss'])

        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / max(1, epoch_batches)

        return {
            'train_loss': avg_epoch_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
            'epoch_time': epoch_time,
            'batches': epoch_batches
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            x_seq = batch['input'].to(self.device)
            target = batch['target'].to(self.device)

            outputs = self.model(x_seq, target)
            total_loss += outputs['loss'].item()
            num_batches += 1

        val_loss = total_loss / max(1, num_batches)

        # Log to wandb
        log_dict = {
            'val/loss': val_loss,
            'val/epoch': self.current_epoch,
            'val/step': self.global_step,
        }
        self._log_to_wandb(log_dict)

        logger.info(f"Validation | Loss: {val_loss:.6f}")

        self.model.train()
        return {'val_loss': val_loss}

    def _check_improvement(self, val_loss: float) -> bool:
        """Check if validation loss improved."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self.save_checkpoint('best.pt')
            logger.info(f"New best model! Val loss: {val_loss:.6f}")
            return True
        else:
            self.patience_counter += 1
            return False

    def train(self) -> Dict[str, List]:
        """Full training loop."""
        logger.info("=" * 60)
        logger.info("STARTING TRAINING")
        logger.info("=" * 60)
        logger.info(f"Device: {self.device}")
        logger.info(f"Epochs: {self.config.epochs}")
        logger.info(f"Batch size: {self.config.batch_size} x {self.config.gradient_accumulation}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"Train batches: {len(self.train_loader)}")
        logger.info(f"Val batches: {len(self.val_loader)}")
        logger.info("=" * 60)

        training_start = time.time()

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            logger.info(f"\n{'='*20} Epoch {epoch}/{self.config.epochs-1} {'='*20}")

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = {}
            if epoch % self.config.val_every_epochs == 0:
                val_metrics = self.validate()

                # Early stopping check
                improved = self._check_improvement(val_metrics['val_loss'])

                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

            # Save periodic checkpoint
            if epoch % self.config.save_every_epochs == 0:
                self.save_checkpoint(f'epoch_{epoch:04d}.pt')
                self._cleanup_old_checkpoints()

            # Record history
            epoch_time = time.time() - epoch_start
            record = {
                'epoch': epoch,
                'train_loss': train_metrics['train_loss'],
                'val_loss': val_metrics.get('val_loss'),
                'lr': train_metrics['lr'],
                'time': epoch_time,
                'global_step': self.global_step
            }
            self.training_history.append(record)

            # Log epoch summary to wandb
            epoch_log = {
                'epoch/train_loss': train_metrics['train_loss'],
                'epoch/val_loss': val_metrics.get('val_loss'),
                'epoch/lr': train_metrics['lr'],
                'epoch/time': epoch_time,
                'epoch/number': epoch,
            }
            self._log_to_wandb(epoch_log)

            # Console summary
            val_str = f"{val_metrics.get('val_loss', 'N/A'):.6f}" if val_metrics else "N/A"
            logger.info(
                f"Epoch {epoch} complete | "
                f"Train: {train_metrics['train_loss']:.6f} | "
                f"Val: {val_str} | "
                f"Time: {epoch_time:.1f}s"
            )

        # Training complete
        total_time = time.time() - training_start
        logger.info("=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        logger.info(f"Total steps: {self.global_step}")
        logger.info("=" * 60)

        # Save final checkpoint
        self.save_checkpoint('final.pt')
        self._save_history()

        # Close wandb
        if self.wandb_run is not None:
            import wandb
            wandb.finish()

        return {
            'train_loss': [r['train_loss'] for r in self.training_history],
            'val_loss': [r['val_loss'] for r in self.training_history if r['val_loss']],
            'lr': [r['lr'] for r in self.training_history]
        }

    def save_checkpoint(self, filename: str) -> Path:
        """Save model checkpoint."""
        path = self.config.checkpoint_dir / filename
        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }, path)
        logger.info(f"Checkpoint saved: {path}")
        return path

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Checkpoint loaded: {path}")

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints."""
        checkpoints = sorted(
            self.config.checkpoint_dir.glob('epoch_*.pt'),
            key=lambda p: int(p.stem.split('_')[1])
        )
        for ckpt in checkpoints[:-self.config.keep_last_n]:
            ckpt.unlink()

    def _save_history(self):
        """Save training history to JSON."""
        history_path = self.config.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[TrainingConfig] = None
) -> Dict[str, List]:
    """Convenience function to train a model."""
    if config is None:
        config = TrainingConfig()

    trainer = Trainer(model, train_loader, val_loader, config)
    return trainer.train()
