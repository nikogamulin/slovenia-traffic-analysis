"""Pytest configuration and shared fixtures."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile


@pytest.fixture
def sample_traffic_data():
    """Generate sample traffic data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    
    df = pd.DataFrame({
        'datetime': dates,
        'Total_All_Lanes': np.random.poisson(500, 100),
        'Avg_Speed': np.random.normal(80, 15, 100),
        'Trucks_7.5t': np.random.poisson(50, 100),
        'Vignette_1': np.random.poisson(200, 100),
        'Vignette_2': np.random.poisson(150, 100),
        'Toll_1': np.random.poisson(30, 100),
        'Toll_2': np.random.poisson(20, 100),
        'Toll_3': np.random.poisson(10, 100),
        'Lane_1': np.random.poisson(150, 100),
        'Lane_2': np.random.poisson(200, 100),
        'Lane_3': np.random.poisson(150, 100),
        'Speed_Lane_1': np.random.normal(75, 10, 100),
        'Speed_Lane_2': np.random.normal(80, 10, 100),
        'Speed_Lane_3': np.random.normal(85, 10, 100),
        'road_id': np.random.choice(['0071', '0171', '0161', '0011'], 100)
    })
    
    # Add time features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    
    return df


@pytest.fixture
def sample_config():
    """Sample configuration dictionary."""
    return {
        'data': {
            'processing': {
                'chunk_size': 1000,
                'missing_data_threshold': 0.3,
                'imputation_method': 'mice',
                'outlier_detection': True,
                'outlier_method': 'iqr'
            }
        },
        'analysis': {
            'bayesian': {
                'mcmc': {
                    'chains': 4,
                    'samples': 1000,
                    'warmup': 500
                }
            }
        }
    }


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)