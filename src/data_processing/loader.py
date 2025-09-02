"""Data loading module for DARS traffic data."""

import os
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and merge DARS traffic data from CSV files."""
    
    def __init__(self, base_path: Union[str, Path], config: Optional[Dict] = None):
        """
        Initialize DataLoader.
        
        Args:
            base_path: Base path to data directory
            config: Optional configuration dictionary
        """
        self.base_path = Path(base_path)
        self.config = config or {}
        self.encoding = self.config.get('encoding', 'utf-8')
        self.date_format = self.config.get('date_format', '%Y-%m-%d')
        self.time_format = self.config.get('time_format', '%H:%M:%S')
        
        # Column mappings based on legenda.txt
        self.column_mappings = {
            'vignette_vehicles': ['Vignette_1', 'Vignette_2'],
            'toll_vehicles': ['Toll_1', 'Toll_2', 'Toll_3'],
            'heavy_vehicles': ['Trucks_7.5t'],
            'lane_data': ['Lane_1', 'Lane_2', 'Lane_3'],
            'speed_data': ['Speed_Lane_1', 'Speed_Lane_2', 'Speed_Lane_3']
        }
        
        # Status codes for data quality
        self.status_codes = {
            'X': 'Missing records',
            'E': 'Loop error',
            'P': 'Power outage',
            'O': 'Cabinet door open'
        }
    
    def load_csv_files(self, pattern: str = "*.csv") -> pd.DataFrame:
        """
        Load all CSV files matching pattern.
        
        Args:
            pattern: File pattern to match
            
        Returns:
            Merged DataFrame with all data
        """
        file_paths = glob.glob(str(self.base_path / pattern))
        
        if not file_paths:
            raise FileNotFoundError(f"No files found matching {pattern} in {self.base_path}")
        
        logger.info(f"Found {len(file_paths)} files to load")
        
        dataframes = []
        for file_path in tqdm(file_paths, desc="Loading CSV files"):
            try:
                df = self._load_single_file(file_path)
                dataframes.append(df)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        if not dataframes:
            raise ValueError("No data could be loaded from files")
        
        # Merge all dataframes
        merged_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Loaded {len(merged_df)} records from {len(dataframes)} files")
        
        return merged_df
    
    def _load_single_file(self, file_path: str) -> pd.DataFrame:
        """
        Load a single CSV file and extract metadata.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with data and metadata
        """
        # Extract metadata from filename
        filename = Path(file_path).stem
        metadata = self._extract_metadata(filename)
        
        # Load CSV
        df = pd.read_csv(file_path, encoding=self.encoding)
        
        # Add metadata columns
        for key, value in metadata.items():
            df[key] = value
        
        # Parse datetime if columns exist
        if 'Date' in df.columns and 'Time' in df.columns:
            df['datetime'] = pd.to_datetime(
                df['Date'] + ' ' + df['Time'],
                format=f"{self.date_format} {self.time_format}"
            )
        
        return df
    
    def _extract_metadata(self, filename: str) -> Dict:
        """
        Extract metadata from filename.
        
        Args:
            filename: Name of the file
            
        Returns:
            Dictionary with metadata
        """
        metadata = {}
        
        # Parse road ID and date from filename
        # Expected format: vehicle_count_ROADID_YYYYMMDD.csv
        parts = filename.split('_')
        
        if len(parts) >= 4:
            metadata['road_id'] = parts[-2]
            metadata['file_date'] = parts[-1]
        
        return metadata
    
    def create_unified_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create unified dataset with engineered features.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        # Calculate total vehicles
        if 'Total_All_Lanes' not in df.columns:
            lane_cols = [col for col in df.columns if col.startswith('Lane_')]
            if lane_cols:
                df['Total_All_Lanes'] = df[lane_cols].sum(axis=1)
        
        # Calculate vehicle class totals
        vignette_cols = [col for col in df.columns if col in self.column_mappings['vignette_vehicles']]
        if vignette_cols:
            df['VV_Total'] = df[vignette_cols].sum(axis=1)
        
        toll_cols = [col for col in df.columns if col in self.column_mappings['toll_vehicles']]
        if toll_cols:
            df['TV_Total'] = df[toll_cols].sum(axis=1)
        
        # Calculate heavy vehicle proportion
        if 'Trucks_7.5t' in df.columns and 'Total_All_Lanes' in df.columns:
            df['HV_Proportion'] = df['Trucks_7.5t'] / df['Total_All_Lanes'].replace(0, np.nan)
        
        # Calculate traffic density (if speed data available)
        if 'Total_All_Lanes' in df.columns and 'Avg_Speed' in df.columns:
            df['Traffic_Density'] = df['Total_All_Lanes'] / df['Avg_Speed'].replace(0, np.nan)
        
        # Add temporal features
        if 'datetime' in df.columns:
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['week'] = df['datetime'].dt.isocalendar().week
            df['month'] = df['datetime'].dt.month
            df['year'] = df['datetime'].dt.year
            df['is_weekend'] = df['day_of_week'].isin([5, 6])
        
        return df
    
    def validate_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate loaded data and return statistics.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation statistics
        """
        stats = {
            'total_records': len(df),
            'date_range': None,
            'missing_data_percentage': None,
            'unique_roads': None,
            'columns': list(df.columns)
        }
        
        if 'datetime' in df.columns:
            stats['date_range'] = (df['datetime'].min(), df['datetime'].max())
        
        # Calculate missing data percentage
        missing_percentage = (df.isnull().sum() / len(df)) * 100
        stats['missing_data_percentage'] = missing_percentage.to_dict()
        
        if 'road_id' in df.columns:
            stats['unique_roads'] = df['road_id'].nunique()
        
        # Check for data quality issues
        quality_issues = []
        
        # Check for negative values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if (df[col] < 0).any():
                quality_issues.append(f"Negative values found in {col}")
        
        # Check for unrealistic speeds
        if 'Avg_Speed' in df.columns:
            if (df['Avg_Speed'] > 200).any():
                quality_issues.append("Unrealistic speeds (>200 km/h) detected")
        
        stats['quality_issues'] = quality_issues
        
        return stats