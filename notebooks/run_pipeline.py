
#!/usr/bin/env python3
"""Example usage of the traffic data pipeline."""

import pandas as pd
import numpy as np
from pipeline_modules.pipeline import TrafficDataPipeline

def main():
    # Generate test data
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='H')
    
    test_data = pd.DataFrame({
        'timestamp': dates,
        'vehicle_count': np.random.poisson(100, n_samples),
        'avg_speed': np.random.normal(80, 15, n_samples),
        'occupancy': np.random.beta(2, 5, n_samples) * 100,
        'temperature': np.random.normal(15, 10, n_samples),
        'precipitation': np.random.exponential(2, n_samples)
    })
    
    # Initialize and run pipeline
    pipeline = TrafficDataPipeline()
    results = pipeline.run(test_data)
    
    print(f"Pipeline success: {results['success']}")
    if results['success']:
        print(f"Final data shape: {results['final_data'].shape}")

if __name__ == '__main__':
    main()
