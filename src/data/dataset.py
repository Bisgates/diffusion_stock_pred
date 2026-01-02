"""
PyTorch Dataset for stock diffusion model.

Memory-efficient implementation with:
- Incremental preprocessing to disk
- Memory-mapped data loading
- Progress monitoring
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
import gc
import os

from .loader import StockDataLoader, DataConfig
from .preprocessing import DataPreprocessor, PreprocessConfig, prepare_sequences


@dataclass
class SampleConfig:
    """Configuration for sample generation."""

    sequence_length: int = 120  # Input sequence length (seconds)
    prediction_horizon: int = 30  # Prediction horizon (seconds)
    stride: int = 1  # Sliding window stride

    # Features to use
    input_features: List[str] = field(default_factory=lambda: [
        'log_return',
        'volatility_5s',
        'spread',
        'volume_zscore'
    ])

    # Target
    target_type: str = "return"  # "return" or "price"

    # Data quality
    min_valid_ratio: float = 0.9  # Min ratio of valid points in sequence


class StockDiffusionDataset(Dataset):
    """
    PyTorch Dataset for stock diffusion model.

    Supports both in-memory and memory-mapped data.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        normalize: bool = True,
        norm_stats: Optional[dict] = None
    ):
        """
        Initialize dataset.

        Args:
            X: Input sequences (n_samples, seq_length, n_features)
            y: Target values (n_samples,)
            valid_mask: Sample validity mask (n_samples,)
            normalize: Whether to normalize inputs
            norm_stats: Pre-computed normalization statistics
        """
        # Filter to valid samples
        if valid_mask is not None:
            X = X[valid_mask]
            y = y[valid_mask]

        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

        # Normalize inputs
        if normalize:
            if norm_stats is not None:
                self.norm_stats = norm_stats
            else:
                # Compute stats from data
                self.norm_stats = self._compute_norm_stats(self.X)

            self.X = self._normalize(self.X, self.norm_stats)
        else:
            self.norm_stats = None

        # Normalize targets (returns) - typically already small
        self.y_mean = np.mean(self.y)
        self.y_std = np.std(self.y) + 1e-8

        # Convert to tensors
        self.X_tensor = torch.from_numpy(self.X)
        self.y_tensor = torch.from_numpy(self.y).unsqueeze(-1)

    def _compute_norm_stats(self, X: np.ndarray) -> dict:
        """Compute normalization statistics."""
        flat = X.reshape(-1, X.shape[-1])
        return {
            'mean': np.nanmean(flat, axis=0),
            'std': np.nanstd(flat, axis=0) + 1e-8
        }

    def _normalize(self, X: np.ndarray, stats: dict) -> np.ndarray:
        """Apply normalization."""
        return (X - stats['mean']) / stats['std']

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input': self.X_tensor[idx],
            'target': self.y_tensor[idx]
        }

    def get_norm_stats(self) -> dict:
        """Get normalization statistics for inference."""
        return self.norm_stats


class MemmapDataset(Dataset):
    """
    Memory-mapped dataset for large data.

    Loads data from disk on-demand to save memory.
    """

    def __init__(
        self,
        X_path: Path,
        y_path: Path,
        shape_X: Tuple[int, int, int],
        shape_y: Tuple[int],
        norm_stats: Optional[dict] = None
    ):
        self.X_mmap = np.memmap(X_path, dtype=np.float32, mode='r', shape=shape_X)
        self.y_mmap = np.memmap(y_path, dtype=np.float32, mode='r', shape=shape_y)
        self.norm_stats = norm_stats

    def __len__(self) -> int:
        return len(self.X_mmap)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        X = self.X_mmap[idx].copy()
        y = self.y_mmap[idx].copy()

        return {
            'input': torch.from_numpy(X),
            'target': torch.tensor([y], dtype=torch.float32)
        }


class DataSplitter:
    """Time-series data splitter."""

    def __init__(
        self,
        train_ratio: float = 0.75,
        val_ratio: float = 0.125,
        test_ratio: float = 0.125
    ):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split_by_date(
        self,
        dates: List[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        n_dates = len(dates)
        n_train = int(n_dates * self.train_ratio)
        n_val = int(n_dates * self.val_ratio)

        train_dates = dates[:n_train]
        val_dates = dates[n_train:n_train + n_val]
        test_dates = dates[n_train + n_val:]

        return train_dates, val_dates, test_dates


def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def process_and_save_incremental(
    dates: List[str],
    loader: StockDataLoader,
    preprocessor: DataPreprocessor,
    sample_config: SampleConfig,
    output_dir: Path,
    split_name: str,
    symbols: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[Path, Path, int, dict]:
    """
    Process data incrementally and save to disk.

    Returns paths to memmap files and shape info.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # First pass: count total samples
    total_samples = 0
    sample_counts = []

    if verbose:
        print(f"[{split_name}] Counting samples...")

    for date in tqdm(dates, desc=f"Counting {split_name}", disable=not verbose):
        available_symbols = loader.get_available_symbols(date)
        if symbols:
            day_symbols = [s for s in symbols if s in available_symbols]
        else:
            day_symbols = available_symbols

        day_count = 0
        for symbol in day_symbols:
            df = loader.load_symbol_day(symbol, date, filter_market_hours=True)
            if df is None or len(df) < sample_config.sequence_length + sample_config.prediction_horizon:
                continue

            # Estimate samples for this symbol/day
            n_samples = max(0, (len(df) - sample_config.sequence_length - sample_config.prediction_horizon) // sample_config.stride + 1)
            day_count += n_samples

        sample_counts.append((date, day_count))
        total_samples += day_count

    if total_samples == 0:
        raise ValueError(f"No samples found for {split_name}")

    if verbose:
        print(f"[{split_name}] Total samples: {total_samples:,}")

    # Create memmap files
    n_features = len(sample_config.input_features)
    seq_len = sample_config.sequence_length

    X_path = output_dir / f"{split_name}_X.dat"
    y_path = output_dir / f"{split_name}_y.dat"

    X_mmap = np.memmap(X_path, dtype=np.float32, mode='w+', shape=(total_samples, seq_len, n_features))
    y_mmap = np.memmap(y_path, dtype=np.float32, mode='w+', shape=(total_samples,))

    # Second pass: process and write
    current_idx = 0
    stats_accum = {'sum': None, 'sum_sq': None, 'count': 0}

    if verbose:
        print(f"[{split_name}] Processing and saving...")

    for date in tqdm(dates, desc=f"Processing {split_name}", disable=not verbose):
        available_symbols = loader.get_available_symbols(date)
        if symbols:
            day_symbols = [s for s in symbols if s in available_symbols]
        else:
            day_symbols = available_symbols

        for symbol in day_symbols:
            df = loader.load_symbol_day(symbol, date, filter_market_hours=True)
            if df is None or len(df) < sample_config.sequence_length + sample_config.prediction_horizon:
                continue

            try:
                # Preprocess
                processed, valid_mask = preprocessor.process_day(df)

                # Generate sequences
                X, y, sample_valid, _ = prepare_sequences(
                    processed,
                    sample_config.input_features,
                    seq_length=sample_config.sequence_length,
                    prediction_horizon=sample_config.prediction_horizon,
                    stride=sample_config.stride,
                    valid_mask=valid_mask
                )

                # Filter valid
                X = X[sample_valid]
                y = y[sample_valid]

                if len(X) == 0:
                    continue

                # Write to memmap
                end_idx = min(current_idx + len(X), total_samples)
                actual_len = end_idx - current_idx

                X_mmap[current_idx:end_idx] = X[:actual_len]
                y_mmap[current_idx:end_idx] = y[:actual_len]

                # Accumulate stats for normalization
                flat = X[:actual_len].reshape(-1, n_features)
                if stats_accum['sum'] is None:
                    stats_accum['sum'] = np.sum(flat, axis=0)
                    stats_accum['sum_sq'] = np.sum(flat ** 2, axis=0)
                else:
                    stats_accum['sum'] += np.sum(flat, axis=0)
                    stats_accum['sum_sq'] += np.sum(flat ** 2, axis=0)
                stats_accum['count'] += len(flat)

                current_idx = end_idx

            except Exception as e:
                if verbose:
                    print(f"Warning: Error processing {symbol}/{date}: {e}")
                continue

        # Periodic memory cleanup
        gc.collect()

    # Truncate if we have fewer samples than estimated
    actual_samples = current_idx
    if actual_samples < total_samples:
        if verbose:
            print(f"[{split_name}] Actual samples: {actual_samples:,} (estimated {total_samples:,})")

        # Resize memmap files
        del X_mmap
        del y_mmap

        X_mmap = np.memmap(X_path, dtype=np.float32, mode='r+', shape=(total_samples, seq_len, n_features))
        y_mmap = np.memmap(y_path, dtype=np.float32, mode='r+', shape=(total_samples,))

        # Read actual data
        X_data = X_mmap[:actual_samples].copy()
        y_data = y_mmap[:actual_samples].copy()

        del X_mmap
        del y_mmap

        # Remove old files and create new ones
        X_path.unlink()
        y_path.unlink()

        X_mmap = np.memmap(X_path, dtype=np.float32, mode='w+', shape=(actual_samples, seq_len, n_features))
        y_mmap = np.memmap(y_path, dtype=np.float32, mode='w+', shape=(actual_samples,))

        X_mmap[:] = X_data
        y_mmap[:] = y_data

        total_samples = actual_samples

    # Flush to disk
    del X_mmap
    del y_mmap
    gc.collect()

    # Compute normalization stats
    mean = stats_accum['sum'] / stats_accum['count']
    var = stats_accum['sum_sq'] / stats_accum['count'] - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-8))

    norm_stats = {'mean': mean.astype(np.float32), 'std': std.astype(np.float32)}

    return X_path, y_path, total_samples, norm_stats


def create_datasets(
    data_dir: Union[str, Path],
    sample_config: Optional[SampleConfig] = None,
    preprocess_config: Optional[PreprocessConfig] = None,
    symbols: Optional[List[str]] = None,
    max_samples_per_split: Optional[int] = None,
    verbose: bool = True,
    cache_dir: Optional[Union[str, Path]] = None,
    use_memmap: bool = False
) -> Tuple[Dataset, Dataset, Dataset, dict]:
    """
    Create train/val/test datasets from raw data.

    Args:
        data_dir: Path to data_1s directory
        sample_config: Sample generation config
        preprocess_config: Preprocessing config
        symbols: List of symbols to use (None = all)
        max_samples_per_split: Max samples per split (for debugging)
        verbose: Print progress
        cache_dir: Directory to cache processed data
        use_memmap: Use memory-mapped files (for large datasets)

    Returns:
        train_dataset, val_dataset, test_dataset, metadata
    """
    sample_config = sample_config or SampleConfig()
    preprocess_config = preprocess_config or PreprocessConfig()

    # Initialize components
    data_config = DataConfig(data_dir=Path(data_dir))
    loader = StockDataLoader(data_config)
    preprocessor = DataPreprocessor(preprocess_config)
    splitter = DataSplitter()

    # Get available dates and split
    all_dates = loader.get_available_dates()
    train_dates, val_dates, test_dates = splitter.split_by_date(all_dates)

    if verbose:
        print(f"Date splits: Train={len(train_dates)}, Val={len(val_dates)}, Test={len(test_dates)}")
        print(f"Train: {train_dates[0]} - {train_dates[-1]}")
        print(f"Val: {val_dates[0]} - {val_dates[-1]}")
        print(f"Test: {test_dates[0]} - {test_dates[-1]}")

    def process_dates(dates: List[str], desc: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process a set of dates and return sequences (memory-efficient)."""
        X_all, y_all, valid_all = [], [], []
        n_features = len(sample_config.input_features)

        date_iter = tqdm(dates, desc=desc) if verbose else dates

        for date in date_iter:
            available_symbols = loader.get_available_symbols(date)
            if symbols:
                day_symbols = [s for s in symbols if s in available_symbols]
            else:
                day_symbols = available_symbols

            for symbol in day_symbols:
                df = loader.load_symbol_day(symbol, date, filter_market_hours=True)
                if df is None or len(df) < sample_config.sequence_length + sample_config.prediction_horizon:
                    continue

                try:
                    processed, valid_mask = preprocessor.process_day(df)
                    X, y, sample_valid, _ = prepare_sequences(
                        processed,
                        sample_config.input_features,
                        seq_length=sample_config.sequence_length,
                        prediction_horizon=sample_config.prediction_horizon,
                        stride=sample_config.stride,
                        valid_mask=valid_mask
                    )

                    if len(X) > 0:
                        # Filter to valid and convert to float32
                        X = X[sample_valid].astype(np.float32)
                        y = y[sample_valid].astype(np.float32)

                        X_all.append(X)
                        y_all.append(y)

                except Exception as e:
                    continue

            # Memory cleanup after each date
            gc.collect()

        if not X_all:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=bool)

        X_combined = np.concatenate(X_all, axis=0)
        y_combined = np.concatenate(y_all, axis=0)
        valid_combined = np.ones(len(X_combined), dtype=bool)

        # Free memory
        del X_all, y_all
        gc.collect()

        return X_combined, y_combined, valid_combined

    # Process each split
    if verbose:
        print(f"\nProcessing datasets...")

    X_train, y_train, valid_train = process_dates(train_dates, "Processing train")
    gc.collect()

    X_val, y_val, valid_val = process_dates(val_dates, "Processing val")
    gc.collect()

    X_test, y_test, valid_test = process_dates(test_dates, "Processing test")
    gc.collect()

    if verbose:
        print(f"\nRaw sample counts:")
        print(f"  Train: {len(X_train):,}")
        print(f"  Val: {len(X_val):,}")
        print(f"  Test: {len(X_test):,}")

    # Limit samples if requested
    if max_samples_per_split:
        if len(X_train) > max_samples_per_split:
            idx = np.random.choice(len(X_train), max_samples_per_split, replace=False)
            X_train, y_train = X_train[idx], y_train[idx]
        if len(X_val) > max_samples_per_split:
            idx = np.random.choice(len(X_val), max_samples_per_split, replace=False)
            X_val, y_val = X_val[idx], y_val[idx]
        if len(X_test) > max_samples_per_split:
            idx = np.random.choice(len(X_test), max_samples_per_split, replace=False)
            X_test, y_test = X_test[idx], y_test[idx]

    # Create datasets
    train_dataset = StockDiffusionDataset(X_train, y_train, normalize=True)
    norm_stats = train_dataset.get_norm_stats()

    # Free training arrays after creating dataset
    del X_train, y_train
    gc.collect()

    val_dataset = StockDiffusionDataset(X_val, y_val, normalize=True, norm_stats=norm_stats)
    del X_val, y_val
    gc.collect()

    test_dataset = StockDiffusionDataset(X_test, y_test, normalize=True, norm_stats=norm_stats)
    del X_test, y_test
    gc.collect()

    metadata = {
        'train_dates': train_dates,
        'val_dates': val_dates,
        'test_dates': test_dates,
        'norm_stats': norm_stats,
        'feature_names': sample_config.input_features,
        'seq_length': sample_config.sequence_length,
        'prediction_horizon': sample_config.prediction_horizon,
        'y_stats': {
            'mean': train_dataset.y_mean,
            'std': train_dataset.y_std
        }
    }

    if verbose:
        print(f"\nFinal dataset sizes:")
        print(f"  Train: {len(train_dataset):,}")
        print(f"  Val: {len(val_dataset):,}")
        print(f"  Test: {len(test_dataset):,}")

    return train_dataset, val_dataset, test_dataset, metadata


def create_datasets_memmap(
    data_dir: Union[str, Path],
    cache_dir: Union[str, Path],
    sample_config: Optional[SampleConfig] = None,
    preprocess_config: Optional[PreprocessConfig] = None,
    symbols: Optional[List[str]] = None,
    verbose: bool = True,
    force_reprocess: bool = False
) -> Tuple[Dataset, Dataset, Dataset, dict]:
    """
    Create datasets using memory-mapped files for large data.

    This is more memory-efficient but slower for first run.
    """
    sample_config = sample_config or SampleConfig()
    preprocess_config = preprocess_config or PreprocessConfig()
    cache_dir = Path(cache_dir)

    # Initialize components
    data_config = DataConfig(data_dir=Path(data_dir))
    loader = StockDataLoader(data_config)
    preprocessor = DataPreprocessor(preprocess_config)
    splitter = DataSplitter()

    all_dates = loader.get_available_dates()
    train_dates, val_dates, test_dates = splitter.split_by_date(all_dates)

    if verbose:
        print(f"Date splits: Train={len(train_dates)}, Val={len(val_dates)}, Test={len(test_dates)}")

    # Check if cache exists
    metadata_path = cache_dir / "metadata.npz"
    if metadata_path.exists() and not force_reprocess:
        if verbose:
            print("Loading from cache...")

        meta = np.load(metadata_path, allow_pickle=True)
        norm_stats = meta['norm_stats'].item()
        shapes = meta['shapes'].item()

        train_dataset = MemmapDataset(
            cache_dir / "train_X.dat",
            cache_dir / "train_y.dat",
            shapes['train_X'],
            shapes['train_y'],
            norm_stats
        )
        val_dataset = MemmapDataset(
            cache_dir / "val_X.dat",
            cache_dir / "val_y.dat",
            shapes['val_X'],
            shapes['val_y'],
            norm_stats
        )
        test_dataset = MemmapDataset(
            cache_dir / "test_X.dat",
            cache_dir / "test_y.dat",
            shapes['test_X'],
            shapes['test_y'],
            norm_stats
        )

        metadata = {
            'norm_stats': norm_stats,
            'feature_names': sample_config.input_features,
            'seq_length': sample_config.sequence_length,
            'prediction_horizon': sample_config.prediction_horizon,
        }

        return train_dataset, val_dataset, test_dataset, metadata

    # Process and save to cache
    if verbose:
        print("Processing data to cache (this may take a while)...")

    # Process train first to get normalization stats
    train_X_path, train_y_path, train_n, norm_stats = process_and_save_incremental(
        train_dates, loader, preprocessor, sample_config, cache_dir, "train", symbols, verbose
    )

    val_X_path, val_y_path, val_n, _ = process_and_save_incremental(
        val_dates, loader, preprocessor, sample_config, cache_dir, "val", symbols, verbose
    )

    test_X_path, test_y_path, test_n, _ = process_and_save_incremental(
        test_dates, loader, preprocessor, sample_config, cache_dir, "test", symbols, verbose
    )

    # Apply normalization to saved data
    if verbose:
        print("Applying normalization...")

    for split_name, n_samples in [("train", train_n), ("val", val_n), ("test", test_n)]:
        X_path = cache_dir / f"{split_name}_X.dat"
        shape = (n_samples, sample_config.sequence_length, len(sample_config.input_features))
        X_mmap = np.memmap(X_path, dtype=np.float32, mode='r+', shape=shape)

        # Normalize in chunks
        chunk_size = 10000
        for i in range(0, n_samples, chunk_size):
            end = min(i + chunk_size, n_samples)
            X_mmap[i:end] = (X_mmap[i:end] - norm_stats['mean']) / norm_stats['std']

        del X_mmap
        gc.collect()

    # Save metadata
    shapes = {
        'train_X': (train_n, sample_config.sequence_length, len(sample_config.input_features)),
        'train_y': (train_n,),
        'val_X': (val_n, sample_config.sequence_length, len(sample_config.input_features)),
        'val_y': (val_n,),
        'test_X': (test_n, sample_config.sequence_length, len(sample_config.input_features)),
        'test_y': (test_n,),
    }

    np.savez(
        metadata_path,
        norm_stats=norm_stats,
        shapes=shapes
    )

    # Create datasets
    train_dataset = MemmapDataset(train_X_path, train_y_path, shapes['train_X'], shapes['train_y'], norm_stats)
    val_dataset = MemmapDataset(val_X_path, val_y_path, shapes['val_X'], shapes['val_y'], norm_stats)
    test_dataset = MemmapDataset(test_X_path, test_y_path, shapes['test_X'], shapes['test_y'], norm_stats)

    metadata = {
        'norm_stats': norm_stats,
        'feature_names': sample_config.input_features,
        'seq_length': sample_config.sequence_length,
        'prediction_horizon': sample_config.prediction_horizon,
    }

    if verbose:
        print(f"\nDataset sizes: Train={train_n:,}, Val={val_n:,}, Test={test_n:,}")

    return train_dataset, val_dataset, test_dataset, metadata


def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 128,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train/val/test datasets."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,  # Disable for MPS
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader, test_loader
