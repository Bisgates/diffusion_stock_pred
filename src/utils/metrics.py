"""
Evaluation metrics for distribution prediction.

Provides:
- Distribution metrics (CRPS, calibration, sharpness)
- Point prediction metrics (MAE, RMSE, direction accuracy)
- Comprehensive evaluation
"""

from typing import Dict, Optional, Tuple
import numpy as np


class DistributionMetrics:
    """Metrics for evaluating predicted distributions."""

    @staticmethod
    def crps(
        samples: np.ndarray,
        observations: np.ndarray
    ) -> float:
        """
        Continuous Ranked Probability Score.

        Measures how well the predicted distribution matches observations.
        Lower is better.

        Args:
            samples: (n_obs, n_samples) predicted samples
            observations: (n_obs,) actual values

        Returns:
            Mean CRPS
        """
        n_obs, n_samples = samples.shape

        crps_values = []
        for i in range(n_obs):
            sample = np.sort(samples[i])
            obs = observations[i]

            # Compute CRPS using the definition:
            # CRPS = E|X - y| - 0.5 * E|X - X'|
            # where X, X' are independent samples from the predicted distribution

            # Term 1: E|X - y|
            term1 = np.mean(np.abs(sample - obs))

            # Term 2: E|X - X'| (approximated)
            # For sorted samples, this can be computed efficiently
            n = len(sample)
            indices = np.arange(n)
            term2 = 2 * np.sum((2 * indices - n + 1) * sample) / (n * (n - 1))

            crps = term1 - 0.5 * term2
            crps_values.append(crps)

        return np.mean(crps_values)

    @staticmethod
    def crps_gaussian(
        mu: np.ndarray,
        sigma: np.ndarray,
        observations: np.ndarray
    ) -> float:
        """
        CRPS for Gaussian predictions (closed form).

        Args:
            mu: (n_obs,) predicted means
            sigma: (n_obs,) predicted stds
            observations: (n_obs,) actual values

        Returns:
            Mean CRPS
        """
        from scipy import stats

        z = (observations - mu) / sigma
        crps = sigma * (
            z * (2 * stats.norm.cdf(z) - 1) +
            2 * stats.norm.pdf(z) -
            1 / np.sqrt(np.pi)
        )
        return np.mean(crps)

    @staticmethod
    def calibration(
        samples: np.ndarray,
        observations: np.ndarray,
        quantiles: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Calibration assessment.

        Checks if X% prediction intervals contain X% of actual values.

        Args:
            samples: (n_obs, n_samples) predicted samples
            observations: (n_obs,) actual values
            quantiles: Quantile levels to check

        Returns:
            dict with expected, observed coverages, and calibration error
        """
        if quantiles is None:
            quantiles = np.linspace(0.1, 0.9, 9)

        observed_coverage = []

        for q in quantiles:
            # Compute predicted quantile for each observation
            pred_q = np.percentile(samples, q * 100, axis=1)
            # Check how many observations fall below this quantile
            coverage = np.mean(observations <= pred_q)
            observed_coverage.append(coverage)

        observed_coverage = np.array(observed_coverage)
        calibration_error = np.mean(np.abs(observed_coverage - quantiles))

        return {
            'quantiles': quantiles,
            'expected': quantiles,
            'observed': observed_coverage,
            'calibration_error': calibration_error
        }

    @staticmethod
    def sharpness(samples: np.ndarray) -> Dict[str, float]:
        """
        Sharpness assessment.

        Measures width of prediction intervals.
        Narrower is better (while maintaining calibration).

        Args:
            samples: (n_obs, n_samples) predicted samples

        Returns:
            dict with interval widths
        """
        # 90% prediction interval
        q95 = np.percentile(samples, 95, axis=1)
        q05 = np.percentile(samples, 5, axis=1)
        pi90_width = np.mean(q95 - q05)

        # 50% prediction interval
        q75 = np.percentile(samples, 75, axis=1)
        q25 = np.percentile(samples, 25, axis=1)
        pi50_width = np.mean(q75 - q25)

        # Standard deviation
        mean_std = np.mean(np.std(samples, axis=1))

        return {
            'pi90_width': pi90_width,
            'pi50_width': pi50_width,
            'mean_std': mean_std
        }

    @staticmethod
    def coverage(
        samples: np.ndarray,
        observations: np.ndarray,
        confidence: float = 0.9
    ) -> float:
        """
        Compute coverage probability for given confidence level.

        Args:
            samples: (n_obs, n_samples) predicted samples
            observations: (n_obs,) actual values
            confidence: Confidence level (e.g., 0.9 for 90%)

        Returns:
            Empirical coverage probability
        """
        alpha = 1 - confidence
        lower = np.percentile(samples, alpha / 2 * 100, axis=1)
        upper = np.percentile(samples, (1 - alpha / 2) * 100, axis=1)

        covered = (observations >= lower) & (observations <= upper)
        return np.mean(covered)


class PointMetrics:
    """Metrics for point predictions."""

    @staticmethod
    def mae(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Mean Absolute Error."""
        return np.mean(np.abs(predictions - actuals))

    @staticmethod
    def rmse(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(np.mean((predictions - actuals) ** 2))

    @staticmethod
    def mape(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = actuals != 0
        return np.mean(np.abs((predictions[mask] - actuals[mask]) / actuals[mask])) * 100

    @staticmethod
    def direction_accuracy(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Direction (sign) accuracy."""
        pred_sign = np.sign(predictions)
        actual_sign = np.sign(actuals)
        return np.mean(pred_sign == actual_sign)

    @staticmethod
    def correlation(predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Pearson correlation coefficient."""
        if len(predictions) < 2:
            return 0.0
        corr = np.corrcoef(predictions, actuals)[0, 1]
        return corr if not np.isnan(corr) else 0.0

    @staticmethod
    def compute_all(
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> Dict[str, float]:
        """Compute all point metrics."""
        return {
            'mae': PointMetrics.mae(predictions, actuals),
            'rmse': PointMetrics.rmse(predictions, actuals),
            'direction_accuracy': PointMetrics.direction_accuracy(predictions, actuals),
            'correlation': PointMetrics.correlation(predictions, actuals)
        }


class Evaluator:
    """Comprehensive model evaluator."""

    def __init__(self):
        self.dist_metrics = DistributionMetrics()
        self.point_metrics = PointMetrics()

    def evaluate(
        self,
        samples: np.ndarray,
        observations: np.ndarray,
        predictions: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Full evaluation of predictions.

        Args:
            samples: (n_obs, n_samples) predicted distribution samples
            observations: (n_obs,) actual values
            predictions: (n_obs,) point predictions (default: mean of samples)

        Returns:
            dict with all metrics
        """
        if predictions is None:
            predictions = samples.mean(axis=1)

        # Distribution metrics
        crps = self.dist_metrics.crps(samples, observations)
        calibration = self.dist_metrics.calibration(samples, observations)
        sharpness = self.dist_metrics.sharpness(samples)
        coverage_90 = self.dist_metrics.coverage(samples, observations, 0.9)

        # Point metrics
        point = self.point_metrics.compute_all(predictions, observations)

        return {
            # Distribution metrics
            'crps': crps,
            'calibration_error': calibration['calibration_error'],
            'coverage_90': coverage_90,
            'pi90_width': sharpness['pi90_width'],
            'pi50_width': sharpness['pi50_width'],
            'mean_std': sharpness['mean_std'],

            # Point metrics
            'mae': point['mae'],
            'rmse': point['rmse'],
            'direction_accuracy': point['direction_accuracy'],
            'correlation': point['correlation']
        }

    def evaluate_by_quantile(
        self,
        samples: np.ndarray,
        observations: np.ndarray,
        n_quantiles: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate by observation quantiles.

        Useful for analyzing performance across different market conditions.

        Args:
            samples: (n_obs, n_samples) predicted samples
            observations: (n_obs,) actual values
            n_quantiles: Number of quantile groups

        Returns:
            dict with metrics per quantile group
        """
        # Compute absolute observation values for grouping
        abs_obs = np.abs(observations)
        quantile_boundaries = np.percentile(abs_obs, np.linspace(0, 100, n_quantiles + 1))

        results = {}
        for i in range(n_quantiles):
            mask = (abs_obs >= quantile_boundaries[i]) & (abs_obs < quantile_boundaries[i + 1])
            if mask.sum() > 0:
                group_name = f'q{i+1}'
                results[group_name] = self.evaluate(
                    samples[mask],
                    observations[mask]
                )
                results[group_name]['n_samples'] = mask.sum()

        return results


def compute_baseline_crps(observations: np.ndarray) -> float:
    """
    Compute baseline CRPS using historical volatility.

    Assumes a Gaussian distribution with mean=0 and std=historical std.
    """
    mu = np.zeros_like(observations)
    sigma = np.full_like(observations, np.std(observations))
    return DistributionMetrics.crps_gaussian(mu, sigma, observations)


def print_evaluation_report(metrics: Dict[str, float], baseline_crps: Optional[float] = None) -> None:
    """Print formatted evaluation report."""
    print("\n" + "=" * 50)
    print("EVALUATION REPORT")
    print("=" * 50)

    print("\nDistribution Metrics:")
    print(f"  CRPS:              {metrics['crps']:.6f}")
    if baseline_crps:
        improvement = (baseline_crps - metrics['crps']) / baseline_crps * 100
        print(f"  CRPS (baseline):   {baseline_crps:.6f} ({improvement:+.1f}% improvement)")
    print(f"  Calibration Error: {metrics['calibration_error']:.4f}")
    print(f"  90% Coverage:      {metrics['coverage_90']:.4f}")
    print(f"  90% PI Width:      {metrics['pi90_width']:.6f}")
    print(f"  50% PI Width:      {metrics['pi50_width']:.6f}")

    print("\nPoint Prediction Metrics:")
    print(f"  MAE:               {metrics['mae']:.6f}")
    print(f"  RMSE:              {metrics['rmse']:.6f}")
    print(f"  Direction Acc:     {metrics['direction_accuracy']:.4f} ({metrics['direction_accuracy']*100:.1f}%)")
    print(f"  Correlation:       {metrics['correlation']:.4f}")

    print("=" * 50 + "\n")
