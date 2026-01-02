"""
Inference sampler for distribution prediction.

Provides:
- Batch prediction
- Distribution statistics computation
- Price conversion
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class DistributionSampler:
    """
    Sampler for generating price distribution predictions.

    Handles:
    - Efficient batch sampling
    - Distribution statistics
    - Price conversion from returns
    """

    def __init__(
        self,
        model,
        device: str = "mps",
        num_samples: int = 100,
        use_ddim: bool = True,
        ddim_steps: int = 50,
        eta: float = 0.0
    ):
        """
        Initialize sampler.

        Args:
            model: Trained StockDiffusionModel
            device: Device to run inference on
            num_samples: Number of samples per prediction
            use_ddim: Use DDIM (faster) or DDPM sampling
            ddim_steps: Number of DDIM steps
            eta: DDIM stochasticity
        """
        self.model = model.to(device).eval()
        self.device = device
        self.num_samples = num_samples
        self.use_ddim = use_ddim
        self.ddim_steps = ddim_steps
        self.eta = eta

    @torch.no_grad()
    def predict_single(
        self,
        x_sequence: torch.Tensor,
        current_price: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Predict distribution for a single sequence.

        Args:
            x_sequence: (1, seq_len, n_features) historical sequence
            current_price: Current price for return-to-price conversion

        Returns:
            dict with samples and statistics
        """
        x_sequence = x_sequence.to(self.device)

        # Sample returns
        return_samples = self.model.sample(
            x_sequence,
            num_samples=self.num_samples,
            use_ddim=self.use_ddim,
            ddim_steps=self.ddim_steps
        )

        return_samples = return_samples.squeeze(0).cpu().numpy()

        # Compute statistics
        stats = {
            'return_samples': return_samples,
            'return_mean': np.mean(return_samples),
            'return_std': np.std(return_samples),
            'return_median': np.median(return_samples),
            'return_q05': np.percentile(return_samples, 5),
            'return_q25': np.percentile(return_samples, 25),
            'return_q75': np.percentile(return_samples, 75),
            'return_q95': np.percentile(return_samples, 95),
        }

        # Convert to prices if current price provided
        if current_price is not None:
            # For log returns: price = current_price * exp(return)
            price_samples = current_price * np.exp(return_samples)

            stats.update({
                'price_samples': price_samples,
                'price_mean': np.mean(price_samples),
                'price_std': np.std(price_samples),
                'price_median': np.median(price_samples),
                'price_q05': np.percentile(price_samples, 5),
                'price_q25': np.percentile(price_samples, 25),
                'price_q75': np.percentile(price_samples, 75),
                'price_q95': np.percentile(price_samples, 95),
            })

        return stats

    @torch.no_grad()
    def predict_batch(
        self,
        dataloader: DataLoader,
        return_samples: bool = False,
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Batch prediction for evaluation.

        Args:
            dataloader: DataLoader with test data
            return_samples: Whether to return all samples (memory intensive)
            show_progress: Show progress bar

        Returns:
            dict with predictions and actuals
        """
        all_samples = []
        all_predictions = []
        all_actuals = []

        loader = tqdm(dataloader, desc="Predicting") if show_progress else dataloader

        for batch in loader:
            x_seq = batch['input'].to(self.device)
            target = batch['target'].cpu().numpy().squeeze()

            # Sample
            samples = self.model.sample(
                x_seq,
                num_samples=self.num_samples,
                use_ddim=self.use_ddim,
                ddim_steps=self.ddim_steps
            ).cpu().numpy()

            # Mean as point prediction
            predictions = samples.mean(axis=1)

            all_predictions.append(predictions)
            all_actuals.append(target)

            if return_samples:
                all_samples.append(samples)

        results = {
            'predictions': np.concatenate(all_predictions),
            'actuals': np.concatenate(all_actuals)
        }

        if return_samples:
            results['samples'] = np.concatenate(all_samples)

        return results

    def get_distribution_percentiles(
        self,
        samples: np.ndarray,
        percentiles: List[float] = [5, 10, 25, 50, 75, 90, 95]
    ) -> Dict[str, np.ndarray]:
        """
        Compute percentiles from samples.

        Args:
            samples: (n_obs, n_samples) array
            percentiles: List of percentiles to compute

        Returns:
            dict with percentile arrays
        """
        result = {}
        for p in percentiles:
            result[f'p{p}'] = np.percentile(samples, p, axis=1)
        return result


class StreamingSampler:
    """
    Memory-efficient sampler for large datasets.

    Processes data in chunks to avoid memory issues.
    """

    def __init__(
        self,
        model,
        device: str = "mps",
        num_samples: int = 100,
        use_ddim: bool = True,
        ddim_steps: int = 50
    ):
        self.sampler = DistributionSampler(
            model, device, num_samples, use_ddim, ddim_steps
        )

    @torch.no_grad()
    def evaluate_streaming(
        self,
        dataloader: DataLoader,
        evaluator,
        chunk_size: int = 1000
    ) -> Dict[str, float]:
        """
        Evaluate with streaming to save memory.

        Args:
            dataloader: Data loader
            evaluator: Evaluator instance
            chunk_size: Number of samples to process at once

        Returns:
            Aggregated metrics
        """
        from src.utils.metrics import Evaluator

        # Accumulate results
        all_samples = []
        all_actuals = []
        metrics_sum = {}
        n_chunks = 0

        current_samples = []
        current_actuals = []

        for batch in tqdm(dataloader, desc="Evaluating"):
            x_seq = batch['input'].to(self.sampler.device)
            target = batch['target'].cpu().numpy().squeeze()

            samples = self.sampler.model.sample(
                x_seq,
                num_samples=self.sampler.num_samples,
                use_ddim=self.sampler.use_ddim,
                ddim_steps=self.sampler.ddim_steps
            ).cpu().numpy()

            current_samples.append(samples)
            current_actuals.append(target)

            # Process chunk
            total_current = sum(len(s) for s in current_samples)
            if total_current >= chunk_size:
                chunk_samples = np.concatenate(current_samples)
                chunk_actuals = np.concatenate(current_actuals)

                chunk_metrics = evaluator.evaluate(chunk_samples, chunk_actuals)

                # Accumulate
                for k, v in chunk_metrics.items():
                    if k not in metrics_sum:
                        metrics_sum[k] = 0
                    metrics_sum[k] += v * len(chunk_actuals)

                n_chunks += len(chunk_actuals)
                current_samples = []
                current_actuals = []

        # Process remaining
        if current_samples:
            chunk_samples = np.concatenate(current_samples)
            chunk_actuals = np.concatenate(current_actuals)
            chunk_metrics = evaluator.evaluate(chunk_samples, chunk_actuals)

            for k, v in chunk_metrics.items():
                if k not in metrics_sum:
                    metrics_sum[k] = 0
                metrics_sum[k] += v * len(chunk_actuals)
            n_chunks += len(chunk_actuals)

        # Average
        return {k: v / n_chunks for k, v in metrics_sum.items()}
