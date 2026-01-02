"""
Data preprocessing and feature engineering.

Handles:
- Resampling to regular 1-second intervals
- Feature computation (returns, volatility, etc.)
- Normalization
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd


@dataclass
class PreprocessConfig:
    """Configuration for data preprocessing."""

    # Resampling
    resample_freq: str = "1s"  # Resample frequency
    fill_method: str = "ffill"  # Forward fill
    max_gap_seconds: int = 60  # Max gap before marking invalid

    # Feature engineering
    use_returns: bool = True
    return_type: str = "log"  # "log" or "simple"

    # Rolling windows for features
    volatility_windows: List[int] = field(default_factory=lambda: [5, 30])
    sma_windows: List[int] = field(default_factory=lambda: [30])

    # Normalization
    normalize_method: str = "zscore"  # "zscore", "minmax", "robust"
    clip_outliers: bool = True
    clip_std: float = 5.0  # Clip at +/- N std


class DataPreprocessor:
    """
    Data preprocessor for stock data.

    Handles:
    1. Resampling to regular 1-second intervals
    2. Gap detection and handling
    3. Feature computation
    4. Normalization
    """

    def __init__(self, config: PreprocessConfig):
        self.config = config

    def resample_to_regular(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Resample to regular 1-second intervals.

        Args:
            df: DataFrame with 'timestamp' index and price columns

        Returns:
            resampled_df: Regular 1-second interval data
            valid_mask: Boolean series indicating original data points
        """
        if df.empty:
            return df, pd.Series(dtype=bool)

        # Ensure timestamp is index
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')

        # Get time range
        start_time = df.index.min().floor('s')
        end_time = df.index.max().ceil('s')

        # Create regular time index
        regular_index = pd.date_range(
            start=start_time,
            end=end_time,
            freq=self.config.resample_freq,
            tz='UTC'
        )

        # Create valid mask (True where we have original data)
        valid_mask = pd.Series(False, index=regular_index)
        original_times = df.index.floor('s')
        valid_mask.loc[valid_mask.index.isin(original_times)] = True

        # Resample using last value within each second
        resampled = df.resample('1s').last()

        # Reindex to regular grid
        resampled = resampled.reindex(regular_index)

        # Forward fill missing values
        resampled = resampled.ffill()

        # Mark large gaps as invalid
        gap_seconds = (~valid_mask).astype(int).groupby(
            (valid_mask != valid_mask.shift()).cumsum()
        ).cumsum()

        # Reset valid mask for gaps exceeding threshold
        valid_mask = valid_mask | (gap_seconds <= self.config.max_gap_seconds)

        # Actually, we want valid_mask to be True only where we have actual data
        # or where the gap is small enough to trust forward-filled values
        cumulative_gap = (~valid_mask.astype(bool)).astype(int)

        # Calculate running gap length
        gap_groups = (valid_mask != valid_mask.shift()).cumsum()
        gap_lengths = valid_mask.groupby(gap_groups).transform('size')
        gap_lengths = gap_lengths.where(~valid_mask, 0)

        # Keep only points within acceptable gap
        for idx in resampled.index:
            if not valid_mask.loc[idx]:
                # Find distance to last valid point
                prev_valid = valid_mask.loc[:idx][valid_mask.loc[:idx]].index
                if len(prev_valid) > 0:
                    gap = (idx - prev_valid[-1]).total_seconds()
                    if gap <= self.config.max_gap_seconds:
                        valid_mask.loc[idx] = True

        return resampled, valid_mask

    def compute_returns(self, prices: pd.Series) -> pd.Series:
        """
        Compute returns from price series.

        Args:
            prices: Price series

        Returns:
            Returns series
        """
        if self.config.return_type == "log":
            # Log return: log(P_t / P_{t-1})
            returns = np.log(prices / prices.shift(1))
        else:
            # Simple return: (P_t - P_{t-1}) / P_{t-1}
            returns = prices.pct_change()

        return returns

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute technical features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with additional feature columns
        """
        result = df.copy()

        # Ensure we have close prices
        if 'close' not in result.columns:
            raise ValueError("DataFrame must have 'close' column")

        # Log returns
        result['log_return'] = self.compute_returns(result['close'])

        # Volatility features (rolling std of returns)
        for window in self.config.volatility_windows:
            col_name = f'volatility_{window}s'
            result[col_name] = result['log_return'].rolling(
                window=window, min_periods=1
            ).std()

        # Simple moving averages
        for window in self.config.sma_windows:
            col_name = f'sma_{window}s'
            result[col_name] = result['close'].rolling(
                window=window, min_periods=1
            ).mean()

        # Price deviation from SMA
        if 30 in self.config.sma_windows:
            result['price_deviation'] = (
                result['close'] - result['sma_30s']
            ) / result['sma_30s']

        # Spread (normalized range)
        if 'high' in result.columns and 'low' in result.columns:
            result['spread'] = (result['high'] - result['low']) / result['close']

        # Volume features
        if 'volume' in result.columns:
            # Log volume (avoid log(0))
            result['log_volume'] = np.log1p(result['volume'])

            # Volume z-score within rolling window
            vol_mean = result['volume'].rolling(window=120, min_periods=1).mean()
            vol_std = result['volume'].rolling(window=120, min_periods=1).std()
            result['volume_zscore'] = (result['volume'] - vol_mean) / (vol_std + 1e-8)

        # Fill NaN values at the beginning
        result = result.ffill().bfill()

        return result

    def normalize_features(
        self,
        features: np.ndarray,
        method: Optional[str] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Normalize features.

        Args:
            features: Array of shape (n_samples, n_features) or (seq_len, n_features)
            method: Normalization method (default: config value)

        Returns:
            normalized: Normalized features
            stats: Dictionary with normalization statistics
        """
        method = method or self.config.normalize_method

        if method == "zscore":
            mean = np.nanmean(features, axis=0, keepdims=True)
            std = np.nanstd(features, axis=0, keepdims=True) + 1e-8
            normalized = (features - mean) / std
            stats = {'mean': mean, 'std': std}

        elif method == "minmax":
            min_val = np.nanmin(features, axis=0, keepdims=True)
            max_val = np.nanmax(features, axis=0, keepdims=True)
            normalized = (features - min_val) / (max_val - min_val + 1e-8)
            stats = {'min': min_val, 'max': max_val}

        elif method == "robust":
            median = np.nanmedian(features, axis=0, keepdims=True)
            q75 = np.nanpercentile(features, 75, axis=0, keepdims=True)
            q25 = np.nanpercentile(features, 25, axis=0, keepdims=True)
            iqr = q75 - q25 + 1e-8
            normalized = (features - median) / iqr
            stats = {'median': median, 'iqr': iqr}

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Clip outliers
        if self.config.clip_outliers:
            normalized = np.clip(
                normalized,
                -self.config.clip_std,
                self.config.clip_std
            )

        return normalized, stats

    def process_day(
        self,
        df: pd.DataFrame,
        compute_features: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Process a full day of data.

        Args:
            df: Raw DataFrame from loader
            compute_features: Whether to compute additional features

        Returns:
            processed_df: Processed DataFrame with features
            valid_mask: Boolean series indicating valid data points
        """
        if df.empty:
            return df, pd.Series(dtype=bool)

        # Resample to regular intervals
        resampled, valid_mask = self.resample_to_regular(df)

        # Compute features if requested
        if compute_features:
            resampled = self.compute_features(resampled)

        return resampled, valid_mask

    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names."""
        features = ['log_return']

        for window in self.config.volatility_windows:
            features.append(f'volatility_{window}s')

        features.extend(['spread', 'volume_zscore'])

        return features


def prepare_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_length: int = 120,
    prediction_horizon: int = 30,
    stride: int = 1,
    valid_mask: Optional[pd.Series] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[pd.Timestamp]]:
    """
    Prepare sequences for training.

    Args:
        df: Processed DataFrame with features
        feature_cols: List of feature column names to use
        seq_length: Length of input sequence
        prediction_horizon: How far ahead to predict
        stride: Step size between samples
        valid_mask: Boolean mask of valid data points

    Returns:
        X: Input sequences (n_samples, seq_length, n_features)
        y: Target values (n_samples,)
        sample_valid: Boolean mask for valid samples (n_samples,)
        timestamps: Timestamp of each sample's prediction point
    """
    n_rows = len(df)
    n_features = len(feature_cols)

    # Extract feature array
    features = df[feature_cols].values

    # Compute target (return at t+horizon)
    target_col = 'log_return' if 'log_return' in df.columns else feature_cols[0]
    close_prices = df['close'].values

    # Target is log return from current to t+horizon
    future_prices = np.roll(close_prices, -prediction_horizon)
    targets = np.log(future_prices / close_prices)

    # Create valid mask if not provided
    if valid_mask is None:
        valid_mask = pd.Series(True, index=df.index)

    valid_arr = valid_mask.values

    # Generate samples
    X_list = []
    y_list = []
    valid_list = []
    timestamp_list = []

    # Need seq_length history + prediction_horizon future
    for i in range(seq_length - 1, n_rows - prediction_horizon, stride):
        # Extract sequence
        seq_start = i - seq_length + 1
        seq_end = i + 1

        X_seq = features[seq_start:seq_end]
        y_val = targets[i]

        # Check validity: all points in sequence + target point must be valid
        seq_valid = valid_arr[seq_start:seq_end].all()
        target_valid = valid_arr[i + prediction_horizon] if i + prediction_horizon < len(valid_arr) else False

        # Also check for NaN
        has_nan = np.isnan(X_seq).any() or np.isnan(y_val)

        sample_valid = seq_valid and target_valid and not has_nan

        X_list.append(X_seq)
        y_list.append(y_val)
        valid_list.append(sample_valid)
        timestamp_list.append(df.index[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    sample_valid = np.array(valid_list, dtype=bool)

    return X, y, sample_valid, timestamp_list
