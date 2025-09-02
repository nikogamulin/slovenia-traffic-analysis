"""Data preprocessing module for traffic data."""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess traffic data for analysis."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize DataPreprocessor.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.outlier_method = self.config.get('outlier_method', 'iqr')
        self.outlier_threshold = self.config.get('outlier_threshold', 3)
        self.scaler = None
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by removing duplicates and invalid records.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        initial_shape = df.shape
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Remove records with invalid timestamps
        if 'datetime' in df.columns:
            df = df.dropna(subset=['datetime'])
            df = df.sort_values('datetime')
        
        # Remove records with all zeros (likely sensor failures)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df[(df[numeric_cols] != 0).any(axis=1)]
        
        final_shape = df.shape
        logger.info(f"Cleaned data from {initial_shape} to {final_shape}")
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Detect outliers in specified columns.
        
        Args:
            df: Input DataFrame
            columns: Columns to check for outliers
            
        Returns:
            DataFrame with outlier flags
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if self.outlier_method == 'iqr':
                outlier_mask = self._detect_outliers_iqr(df[col])
            elif self.outlier_method == 'zscore':
                outlier_mask = self._detect_outliers_zscore(df[col])
            else:
                outlier_mask = pd.Series([False] * len(df))
            
            df[f'{col}_outlier'] = outlier_mask
        
        return df
    
    def _detect_outliers_iqr(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    def _detect_outliers_zscore(self, series: pd.Series) -> pd.Series:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(series.dropna()))
        return pd.Series(z_scores > self.outlier_threshold, index=series.dropna().index).reindex(series.index, fill_value=False)
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'cap') -> pd.DataFrame:
        """
        Handle outliers using specified method.
        
        Args:
            df: Input DataFrame with outlier flags
            method: Method to handle outliers ('remove', 'cap', 'transform')
            
        Returns:
            DataFrame with outliers handled
        """
        df = df.copy()
        outlier_cols = [col for col in df.columns if col.endswith('_outlier')]
        
        for outlier_col in outlier_cols:
            base_col = outlier_col.replace('_outlier', '')
            
            if base_col not in df.columns:
                continue
            
            if method == 'remove':
                df = df[~df[outlier_col]]
            elif method == 'cap':
                # Cap at 95th and 5th percentiles
                lower = df[base_col].quantile(0.05)
                upper = df[base_col].quantile(0.95)
                df.loc[df[outlier_col], base_col] = df.loc[df[outlier_col], base_col].clip(lower, upper)
            elif method == 'transform':
                # Log transformation for positive values
                if (df[base_col] > 0).all():
                    df[f'{base_col}_log'] = np.log1p(df[base_col])
        
        # Drop outlier flag columns
        df = df.drop(columns=outlier_cols)
        
        return df
    
    def normalize_features(self, df: pd.DataFrame, columns: Optional[List[str]] = None,
                          method: str = 'standard') -> pd.DataFrame:
        """
        Normalize specified features.
        
        Args:
            df: Input DataFrame
            columns: Columns to normalize
            method: Normalization method ('standard', 'robust', 'minmax')
            
        Returns:
            DataFrame with normalized features
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df = df.copy()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        
        # Fit and transform
        df[columns] = scaler.fit_transform(df[columns])
        self.scaler = scaler  # Store for inverse transformation
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional time-based features.
        
        Args:
            df: Input DataFrame with datetime column
            
        Returns:
            DataFrame with additional time features
        """
        if 'datetime' not in df.columns:
            logger.warning("No datetime column found, skipping time features")
            return df
        
        df = df.copy()
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Peak hour indicators
        df['is_morning_peak'] = df['hour'].isin([7, 8, 9])
        df['is_evening_peak'] = df['hour'].isin([16, 17, 18, 19])
        df['is_night'] = df['hour'].isin(range(22, 24)) | df['hour'].isin(range(0, 6))
        
        # Season
        df['season'] = pd.cut(df['month'], bins=[0, 3, 6, 9, 12], 
                              labels=['Winter', 'Spring', 'Summer', 'Autumn'])
        
        return df
    
    def aggregate_temporal(self, df: pd.DataFrame, freq: str = 'H',
                          agg_dict: Optional[Dict] = None) -> pd.DataFrame:
        """
        Aggregate data to specified temporal frequency.
        
        Args:
            df: Input DataFrame
            freq: Frequency for aggregation ('H', 'D', 'W', 'M')
            agg_dict: Aggregation dictionary
            
        Returns:
            Aggregated DataFrame
        """
        if 'datetime' not in df.columns:
            raise ValueError("datetime column required for temporal aggregation")
        
        if agg_dict is None:
            # Default aggregation
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            agg_dict = {col: 'mean' for col in numeric_cols}
            
            # Special aggregations
            if 'Total_All_Lanes' in agg_dict:
                agg_dict['Total_All_Lanes'] = 'sum'
            if 'Trucks_7.5t' in agg_dict:
                agg_dict['Trucks_7.5t'] = 'sum'
        
        # Set datetime as index
        df = df.set_index('datetime')
        
        # Group by frequency and aggregate
        aggregated = df.resample(freq).agg(agg_dict)
        
        # Reset index
        aggregated = aggregated.reset_index()
        
        return aggregated
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2,
                  validation_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets for time series.
        
        Args:
            df: Input DataFrame
            test_size: Proportion for test set
            validation_size: Proportion for validation set
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if 'datetime' in df.columns:
            df = df.sort_values('datetime')
        
        n = len(df)
        train_end = int(n * (1 - test_size - validation_size))
        val_end = int(n * (1 - test_size))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        logger.info(f"Split data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
        return train_df, val_df, test_df