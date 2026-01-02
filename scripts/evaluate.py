#!/usr/bin/env python3
"""
Evaluation script for trained stock diffusion model.

Usage:
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best.pt
    python scripts/evaluate.py --checkpoint outputs/checkpoints/best.pt --num-samples 200
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from src.data.dataset import create_datasets, create_dataloaders, SampleConfig
from src.models.model import StockDiffusionModel, ModelConfig
from src.inference.sampler import DistributionSampler
from src.utils.config import load_config, get_default_config, merge_configs
from src.utils.metrics import Evaluator, print_evaluation_report, compute_baseline_crps


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate stock diffusion model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
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
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples for distribution estimation"
    )
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=50,
        help="Number of DDIM sampling steps"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Which data split to evaluate on"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for detailed results (JSON)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config = get_default_config()
    if Path(args.config).exists():
        user_config = load_config(args.config)
        config = merge_configs(config, user_config)

    # Override device
    device = args.device or config['training']['device']
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("=" * 60)
    print("STOCK DIFFUSION MODEL EVALUATION")
    print("=" * 60)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Samples: {args.num_samples}")
    print(f"DDIM steps: {args.ddim_steps}")
    print()

    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Create model with same config
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
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model parameters: {model.get_num_params():,}")
    print(f"Trained for {checkpoint['epoch']} epochs")
    print(f"Best val loss: {checkpoint['best_val_loss']:.6f}")

    # Load data
    print("\nLoading data...")
    sample_config = SampleConfig(
        sequence_length=config['data']['sequence_length'],
        prediction_horizon=config['data']['prediction_horizon'],
        stride=config['data']['stride'],
        input_features=config['data']['input_features']
    )

    train_dataset, val_dataset, test_dataset, metadata = create_datasets(
        data_dir=project_root / config['data']['data_dir'],
        sample_config=sample_config,
        verbose=True
    )

    # Select dataset
    if args.split == "test":
        eval_dataset = test_dataset
        print(f"\nEvaluating on test set ({len(test_dataset)} samples)")
    else:
        eval_dataset = val_dataset
        print(f"\nEvaluating on validation set ({len(val_dataset)} samples)")

    _, _, test_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        eval_dataset,
        batch_size=config['data']['batch_size'],
        num_workers=0
    )

    # Create sampler
    sampler = DistributionSampler(
        model,
        device=device,
        num_samples=args.num_samples,
        use_ddim=True,
        ddim_steps=args.ddim_steps
    )

    # Run predictions
    print("\nRunning predictions...")
    results = sampler.predict_batch(
        test_loader,
        return_samples=True,
        show_progress=True
    )

    # Evaluate
    print("\nComputing metrics...")
    evaluator = Evaluator()
    metrics = evaluator.evaluate(
        results['samples'],
        results['actuals']
    )

    # Compute baseline CRPS
    baseline_crps = compute_baseline_crps(results['actuals'])

    # Print report
    print_evaluation_report(metrics, baseline_crps)

    # Additional analysis
    print("\nDistribution Analysis:")
    print(f"  Actual returns - mean: {np.mean(results['actuals']):.6f}, std: {np.std(results['actuals']):.6f}")
    print(f"  Predicted returns - mean: {np.mean(results['predictions']):.6f}, std: {np.std(results['predictions']):.6f}")

    # Analyze by quantile
    print("\nPerformance by return magnitude:")
    quantile_metrics = evaluator.evaluate_by_quantile(
        results['samples'],
        results['actuals'],
        n_quantiles=5
    )

    for q_name, q_metrics in quantile_metrics.items():
        print(f"  {q_name}: MAE={q_metrics['mae']:.6f}, Dir.Acc={q_metrics['direction_accuracy']:.3f}, n={q_metrics['n_samples']}")

    # Save detailed results if requested
    if args.output:
        import json

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            'checkpoint': args.checkpoint,
            'split': args.split,
            'num_samples': args.num_samples,
            'ddim_steps': args.ddim_steps,
            'metrics': {k: float(v) for k, v in metrics.items()},
            'baseline_crps': float(baseline_crps),
            'quantile_metrics': {
                k: {kk: float(vv) if not isinstance(vv, int) else vv for kk, vv in v.items()}
                for k, v in quantile_metrics.items()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nDetailed results saved to: {output_path}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
