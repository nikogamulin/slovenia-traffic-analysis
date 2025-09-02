"""Tests for data loading module."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from src.data_processing.loader import DataLoader


class TestDataLoader:
    """Test suite for DataLoader class."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory with sample CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample CSV files
            dates = pd.date_range('2024-01-01', periods=24, freq='H')
            
            for i in range(3):
                df = pd.DataFrame({
                    'Date': dates.strftime('%Y-%m-%d'),
                    'Time': dates.strftime('%H:%M:%S'),
                    'Total_All_Lanes': np.random.poisson(500, 24),
                    'Avg_Speed': np.random.normal(80, 10, 24),
                    'Trucks_7.5t': np.random.poisson(50, 24),
                    'Vignette_1': np.random.poisson(200, 24),
                    'Vignette_2': np.random.poisson(150, 24),
                    'Lane_1': np.random.poisson(150, 24),
                    'Lane_2': np.random.poisson(200, 24),
                    'Lane_3': np.random.poisson(150, 24)
                })
                
                filename = f"vehicle_count_0071_2024010{i+1}.csv"
                filepath = Path(tmpdir) / filename
                df.to_csv(filepath, index=False)
            
            yield tmpdir
    
    def test_initialization(self, temp_data_dir):
        """Test DataLoader initialization."""
        loader = DataLoader(temp_data_dir)
        assert loader.base_path == Path(temp_data_dir)
        assert loader.encoding == 'utf-8'
        assert 'vignette_vehicles' in loader.column_mappings
    
    def test_load_csv_files(self, temp_data_dir):
        """Test loading CSV files."""
        loader = DataLoader(temp_data_dir)
        df = loader.load_csv_files()
        
        assert not df.empty
        assert len(df) == 72  # 3 files * 24 records
        assert 'Total_All_Lanes' in df.columns
        assert 'Avg_Speed' in df.columns
    
    def test_load_csv_files_no_files(self):
        """Test loading when no files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = DataLoader(tmpdir)
            with pytest.raises(FileNotFoundError):
                loader.load_csv_files("nonexistent*.csv")
    
    def test_extract_metadata(self, temp_data_dir):
        """Test metadata extraction from filename."""
        loader = DataLoader(temp_data_dir)
        metadata = loader._extract_metadata("vehicle_count_0071_20240101")
        
        assert metadata['road_id'] == '0071'
        assert metadata['file_date'] == '20240101'
    
    def test_create_unified_dataset(self, temp_data_dir):
        """Test unified dataset creation."""
        loader = DataLoader(temp_data_dir)
        df = loader.load_csv_files()
        df_unified = loader.create_unified_dataset(df)
        
        # Check engineered features
        assert 'VV_Total' in df_unified.columns
        assert 'HV_Proportion' in df_unified.columns
        assert 'hour' in df_unified.columns
        assert 'day_of_week' in df_unified.columns
        assert 'is_weekend' in df_unified.columns
        
        # Check calculations
        vv_total = df_unified['Vignette_1'] + df_unified['Vignette_2']
        assert np.allclose(df_unified['VV_Total'], vv_total)
    
    def test_validate_data(self, temp_data_dir):
        """Test data validation."""
        loader = DataLoader(temp_data_dir)
        df = loader.load_csv_files()
        df = loader.create_unified_dataset(df)
        
        stats = loader.validate_data(df)
        
        assert 'total_records' in stats
        assert stats['total_records'] == len(df)
        assert 'date_range' in stats
        assert 'missing_data_percentage' in stats
        assert 'quality_issues' in stats
    
    def test_handle_missing_values(self, temp_data_dir):
        """Test handling of missing values in data."""
        loader = DataLoader(temp_data_dir)
        df = loader.load_csv_files()
        
        # Introduce some missing values
        df.loc[5:10, 'Avg_Speed'] = np.nan
        df.loc[15:20, 'Total_All_Lanes'] = np.nan
        
        stats = loader.validate_data(df)
        missing_pct = stats['missing_data_percentage']
        
        assert missing_pct['Avg_Speed'] > 0
        assert missing_pct['Total_All_Lanes'] > 0