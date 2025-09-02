"""Data imputation module for handling missing values."""

import logging
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from statsmodels.tsa.statespace.sarimax import SARIMAX

logger = logging.getLogger(__name__)


class DataImputer:
    """Handle missing data imputation using various methods."""
    
    def __init__(self, method: str = 'mice', config: Optional[Dict] = None):
        """
        Initialize DataImputer.
        
        Args:
            method: Imputation method ('mice', 'kalman', 'simple', 'forward_fill')
            config: Optional configuration dictionary
        """
        self.method = method
        self.config = config or {}
        self.imputer = None
        self.models = {}
        
    def analyze_missing_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Analyze patterns of missing data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with missing data statistics
        """
        stats = {
            'total_missing': df.isnull().sum().sum(),
            'total_cells': df.size,
            'missing_percentage': (df.isnull().sum().sum() / df.size) * 100,
            'columns_missing': {}
        }
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                stats['columns_missing'][col] = {
                    'count': missing_count,
                    'percentage': (missing_count / len(df)) * 100
                }
        
        # Analyze missing data patterns
        missing_mask = df.isnull()
        
        # Check for complete rows/columns
        stats['complete_rows'] = (~missing_mask.any(axis=1)).sum()
        stats['complete_columns'] = (~missing_mask.any(axis=0)).sum()
        
        # Check for patterns (e.g., blocks of missing data)
        if 'datetime' in df.columns:
            df_sorted = df.sort_values('datetime')
            missing_blocks = self._find_missing_blocks(df_sorted)
            stats['missing_blocks'] = missing_blocks
        
        return stats
    
    def _find_missing_blocks(self, df: pd.DataFrame) -> List[Dict]:
        """Find contiguous blocks of missing data."""
        blocks = []
        
        for col in df.columns:
            if col == 'datetime':
                continue
            
            missing_mask = df[col].isnull()
            if not missing_mask.any():
                continue
            
            # Find start and end of missing blocks
            diff = missing_mask.astype(int).diff()
            starts = df.index[diff == 1].tolist()
            ends = df.index[diff == -1].tolist()
            
            # Handle edge cases
            if missing_mask.iloc[0]:
                starts = [df.index[0]] + starts
            if missing_mask.iloc[-1]:
                ends = ends + [df.index[-1]]
            
            for start, end in zip(starts, ends):
                block_size = end - start + 1
                if block_size > 1:  # Only record blocks larger than 1
                    blocks.append({
                        'column': col,
                        'start_idx': start,
                        'end_idx': end,
                        'size': block_size
                    })
        
        return blocks
    
    def impute(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Impute missing values using specified method.
        
        Args:
            df: Input DataFrame
            columns: Columns to impute (None for all)
            
        Returns:
            DataFrame with imputed values
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_imputed = df.copy()
        
        if self.method == 'mice':
            df_imputed = self._impute_mice(df_imputed, columns)
        elif self.method == 'kalman':
            df_imputed = self._impute_kalman(df_imputed, columns)
        elif self.method == 'simple':
            df_imputed = self._impute_simple(df_imputed, columns)
        elif self.method == 'forward_fill':
            df_imputed = self._impute_forward_fill(df_imputed, columns)
        else:
            raise ValueError(f"Unknown imputation method: {self.method}")
        
        return df_imputed
    
    def _impute_mice(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Impute using Multiple Imputation by Chained Equations (MICE).
        
        Args:
            df: Input DataFrame
            columns: Columns to impute
            
        Returns:
            DataFrame with imputed values
        """
        logger.info("Performing MICE imputation...")
        
        # Configure MICE imputer
        max_iter = self.config.get('mice_max_iter', 10)
        n_nearest = self.config.get('mice_n_nearest', 5)
        
        imputer = IterativeImputer(
            max_iter=max_iter,
            n_nearest_features=n_nearest,
            random_state=42,
            verbose=0
        )
        
        # Fit and transform
        df[columns] = imputer.fit_transform(df[columns])
        self.imputer = imputer
        
        return df
    
    def _impute_kalman(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Impute using Kalman filtering/smoothing.
        
        Args:
            df: Input DataFrame
            columns: Columns to impute
            
        Returns:
            DataFrame with imputed values
        """
        logger.info("Performing Kalman filter imputation...")
        
        if 'datetime' not in df.columns:
            raise ValueError("Kalman imputation requires datetime column")
        
        df = df.sort_values('datetime')
        
        for col in columns:
            if df[col].isnull().all():
                continue
            
            # Create a time series
            ts = df.set_index('datetime')[col]
            
            # Find non-missing values
            non_missing_mask = ~ts.isnull()
            
            if non_missing_mask.sum() < 10:
                # Not enough data for Kalman filter
                logger.warning(f"Not enough data for Kalman filter in {col}, using forward fill")
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
                continue
            
            try:
                # Fit SARIMAX model (state space model)
                model = SARIMAX(
                    ts[non_missing_mask],
                    order=(1, 0, 1),  # ARMA(1,1)
                    seasonal_order=(0, 0, 0, 0),  # No seasonality for simplicity
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                
                results = model.fit(disp=False)
                
                # Get filtered and smoothed values
                filtered = results.fittedvalues
                
                # Interpolate missing values
                ts_imputed = ts.copy()
                ts_imputed[ts.isnull()] = np.nan
                
                # Use model to predict missing values
                for idx in ts[ts.isnull()].index:
                    # Simple interpolation based on nearest fitted values
                    nearest_past = filtered[filtered.index < idx]
                    nearest_future = filtered[filtered.index > idx]
                    
                    if len(nearest_past) > 0 and len(nearest_future) > 0:
                        # Linear interpolation between nearest points
                        ts_imputed[idx] = (nearest_past.iloc[-1] + nearest_future.iloc[0]) / 2
                    elif len(nearest_past) > 0:
                        ts_imputed[idx] = nearest_past.iloc[-1]
                    elif len(nearest_future) > 0:
                        ts_imputed[idx] = nearest_future.iloc[0]
                
                df[col] = ts_imputed.values
                
            except Exception as e:
                logger.error(f"Kalman filter failed for {col}: {e}, using forward fill")
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def _impute_simple(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Simple imputation using mean/median/mode.
        
        Args:
            df: Input DataFrame
            columns: Columns to impute
            
        Returns:
            DataFrame with imputed values
        """
        strategy = self.config.get('simple_strategy', 'mean')
        
        imputer = SimpleImputer(strategy=strategy)
        df[columns] = imputer.fit_transform(df[columns])
        self.imputer = imputer
        
        return df
    
    def _impute_forward_fill(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Forward fill imputation for time series.
        
        Args:
            df: Input DataFrame
            columns: Columns to impute
            
        Returns:
            DataFrame with imputed values
        """
        if 'datetime' in df.columns:
            df = df.sort_values('datetime')
        
        limit = self.config.get('forward_fill_limit', 3)
        
        for col in columns:
            df[col] = df[col].fillna(method='ffill', limit=limit)
            df[col] = df[col].fillna(method='bfill', limit=limit)
        
        return df
    
    def validate_imputation(self, df_original: pd.DataFrame, df_imputed: pd.DataFrame) -> Dict:
        """
        Validate imputation quality.
        
        Args:
            df_original: Original DataFrame with missing values
            df_imputed: Imputed DataFrame
            
        Returns:
            Dictionary with validation metrics
        """
        metrics = {}
        
        # Check if all missing values were imputed
        remaining_missing = df_imputed.isnull().sum().sum()
        metrics['remaining_missing'] = remaining_missing
        metrics['imputation_complete'] = (remaining_missing == 0)
        
        # Compare distributions (for columns that had missing values)
        for col in df_original.columns:
            if df_original[col].isnull().any():
                original_values = df_original[col].dropna()
                imputed_values = df_imputed[col][df_original[col].isnull()]
                
                if len(original_values) > 0 and len(imputed_values) > 0:
                    metrics[f'{col}_mean_diff'] = abs(
                        original_values.mean() - imputed_values.mean()
                    )
                    metrics[f'{col}_std_diff'] = abs(
                        original_values.std() - imputed_values.std()
                    )
        
        return metrics