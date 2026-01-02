"""
Data loader for stock CSV files.

Handles loading, parsing, and filtering of 1-second bar data.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, time
import warnings


@dataclass
class DataConfig:
    """Configuration for data loading."""

    data_dir: Path = field(default_factory=lambda: Path("data_1s"))

    # Market hours in UTC (9:30-16:00 ET = 13:30-20:00 UTC)
    market_open_utc: time = field(default_factory=lambda: time(13, 30, 0))
    market_close_utc: time = field(default_factory=lambda: time(20, 0, 0))

    # Model parameters
    sequence_length: int = 120  # seconds of history
    prediction_horizon: int = 30  # seconds ahead to predict

    # Data quality
    min_coverage_ratio: float = 0.8  # minimum data coverage in a window


class StockDataLoader:
    """
    Loader for stock CSV data.

    Handles:
    - Scanning date directories
    - Loading and parsing CSV files
    - Timezone handling (UTC)
    - Market hours filtering
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self._date_dirs: List[str] = []
        self._symbol_cache: Dict[str, List[str]] = {}
        self._scan_directories()

    def _scan_directories(self) -> None:
        """Scan data directory for date folders."""
        data_path = Path(self.config.data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")

        # Find all YYYYMMDD directories
        self._date_dirs = sorted([
            d.name for d in data_path.iterdir()
            if d.is_dir() and d.name.isdigit() and len(d.name) == 8
        ])

        if not self._date_dirs:
            raise ValueError(f"No date directories found in {data_path}")

    def get_available_dates(self) -> List[str]:
        """Get list of available trading dates."""
        return self._date_dirs.copy()

    def get_available_symbols(self, date: str) -> List[str]:
        """Get list of available symbols for a given date."""
        if date in self._symbol_cache:
            return self._symbol_cache[date]

        date_path = Path(self.config.data_dir) / date
        if not date_path.exists():
            return []

        symbols = sorted([
            f.stem for f in date_path.glob("*.csv")
        ])
        self._symbol_cache[date] = symbols
        return symbols

    def load_symbol_day(
        self,
        symbol: str,
        date: str,
        filter_market_hours: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Load data for a single symbol on a single day.

        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")
            date: Date string in YYYYMMDD format
            filter_market_hours: If True, only return market hours data

        Returns:
            DataFrame with columns: [timestamp, close, high, low, volume, amount]
            or None if file doesn't exist
        """
        file_path = Path(self.config.data_dir) / date / f"{symbol}.csv"

        if not file_path.exists():
            return None

        try:
            # Read CSV
            df = pd.read_csv(file_path)

            # Parse timestamps - bob is the bar open begin time
            df['timestamp'] = pd.to_datetime(df['bob'], utc=True)

            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Select and rename columns
            df = df[['timestamp', 'close', 'high', 'low', 'volume', 'amount']].copy()

            # Filter to market hours if requested
            if filter_market_hours:
                df = self._filter_market_hours(df)

            if len(df) == 0:
                return None

            return df

        except Exception as e:
            warnings.warn(f"Error loading {file_path}: {e}")
            return None

    def _filter_market_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame to market hours only (UTC 13:30-20:00)."""
        if df.empty:
            return df

        # Extract time component
        times = df['timestamp'].dt.time

        # Filter to market hours
        mask = (times >= self.config.market_open_utc) & (times < self.config.market_close_utc)

        return df[mask].reset_index(drop=True)

    def load_multiple_days(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        filter_market_hours: bool = True
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load data for multiple symbols across multiple days.

        Args:
            symbols: List of symbols to load (None = all available)
            start_date: Start date YYYYMMDD (None = first available)
            end_date: End date YYYYMMDD (None = last available)
            filter_market_hours: If True, only return market hours data

        Returns:
            Nested dict: {symbol: {date: DataFrame}}
        """
        # Determine date range
        dates = self._date_dirs
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]

        if not dates:
            return {}

        result: Dict[str, Dict[str, pd.DataFrame]] = {}

        for date in dates:
            available_symbols = self.get_available_symbols(date)

            # Filter symbols if specified
            if symbols:
                day_symbols = [s for s in symbols if s in available_symbols]
            else:
                day_symbols = available_symbols

            for symbol in day_symbols:
                df = self.load_symbol_day(symbol, date, filter_market_hours)

                if df is not None and len(df) > 0:
                    if symbol not in result:
                        result[symbol] = {}
                    result[symbol][date] = df

        return result

    def load_all_data_flat(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        filter_market_hours: bool = True
    ) -> pd.DataFrame:
        """
        Load all data into a single flat DataFrame.

        Returns:
            DataFrame with columns: [symbol, date, timestamp, close, high, low, volume, amount]
        """
        all_dfs = []

        # Determine date range
        dates = self._date_dirs
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]

        for date in dates:
            available_symbols = self.get_available_symbols(date)

            if symbols:
                day_symbols = [s for s in symbols if s in available_symbols]
            else:
                day_symbols = available_symbols

            for symbol in day_symbols:
                df = self.load_symbol_day(symbol, date, filter_market_hours)

                if df is not None and len(df) > 0:
                    df['symbol'] = symbol
                    df['date'] = date
                    all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()

        result = pd.concat(all_dfs, ignore_index=True)
        return result[['symbol', 'date', 'timestamp', 'close', 'high', 'low', 'volume', 'amount']]

    def get_data_summary(self) -> pd.DataFrame:
        """Get summary statistics of available data."""
        records = []

        for date in self._date_dirs:
            symbols = self.get_available_symbols(date)
            records.append({
                'date': date,
                'num_symbols': len(symbols),
                'symbols': ','.join(symbols[:5]) + ('...' if len(symbols) > 5 else '')
            })

        return pd.DataFrame(records)
