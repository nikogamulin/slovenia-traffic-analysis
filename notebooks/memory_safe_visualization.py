"""
Memory-safe visualization to replace visualization cells 9, 11, 13
This version properly manages matplotlib memory and prevents accumulation
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import gc
from contextlib import contextmanager

@contextmanager
def memory_safe_figure(*args, **kwargs):
    """Context manager for memory-safe matplotlib figures."""
    fig = plt.figure(*args, **kwargs)
    try:
        yield fig
    finally:
        plt.close(fig)
        plt.clf()
        plt.cla()
        gc.collect()

def create_performance_summary_safe(metrics_df: pd.DataFrame, max_points: int = 1000):
    """Memory-safe version of performance summary visualization."""
    
    if len(metrics_df) == 0:
        print("No metrics to visualize")
        return
    
    # Limit data points to prevent memory issues
    if len(metrics_df) > max_points:
        metrics_df = metrics_df.sample(n=max_points)
        print(f"Sampling {max_points} points for visualization")
    
    with memory_safe_figure(figsize=(15, 8)) as fig:
        fig.suptitle('Vehicle Count Models - Performance Analysis', fontsize=14, fontweight='bold')
        
        # Create 2x2 grid instead of 2x3 to reduce memory
        axes = fig.subplots(2, 2)
        
        # RMSE comparison
        axes[0, 0].bar(metrics_df['sensor_id'], metrics_df['rmse'], color='steelblue', alpha=0.7)
        axes[0, 0].set_title('RMSE by Sensor', fontsize=12)
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # R² comparison
        axes[0, 1].bar(metrics_df['sensor_id'], metrics_df['r2'], color='green', alpha=0.7)
        axes[0, 1].axhline(y=0.9, color='r', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('R² Score by Sensor', fontsize=12)
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # RMSE percentage with color coding
        colors = ['green' if x < 15 else 'red' for x in metrics_df['rmse_percentage']]
        axes[1, 0].bar(metrics_df['sensor_id'], metrics_df['rmse_percentage'], color=colors, alpha=0.7)
        axes[1, 0].axhline(y=15, color='black', linestyle='--', linewidth=2, alpha=0.8)
        axes[1, 0].set_title('RMSE as % of Mean', fontsize=12)
        axes[1, 0].set_ylabel('RMSE %')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Distribution of RMSE %
        axes[1, 1].hist(metrics_df['rmse_percentage'], bins=min(15, len(metrics_df)//2), 
                       edgecolor='black', alpha=0.7, color='skyblue')
        axes[1, 1].axvline(x=15, color='red', linestyle='--', linewidth=2, alpha=0.8)
        mean_rmse = metrics_df['rmse_percentage'].mean()
        axes[1, 1].axvline(x=mean_rmse, color='blue', linestyle='--', linewidth=2, alpha=0.8)
        axes[1, 1].set_title('Distribution of RMSE %', fontsize=12)
        axes[1, 1].set_xlabel('RMSE as % of Mean')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def create_feature_importance_safe(feature_importance_dict: dict, top_n: int = 15):
    """Memory-safe feature importance visualization."""
    
    # Calculate average importance with memory efficiency
    avg_importance = []
    for feature, values in feature_importance_dict.items():
        avg_importance.append({
            'feature': feature, 
            'avg_importance': np.mean(values), 
            'std_importance': np.std(values)
        })
    
    avg_importance_df = pd.DataFrame(avg_importance).sort_values('avg_importance', ascending=False)
    top_features = avg_importance_df.head(top_n)
    
    with memory_safe_figure(figsize=(10, 6)) as fig:
        plt.barh(range(len(top_features)), top_features['avg_importance'], 
                 xerr=top_features['std_importance'], capsize=3, alpha=0.7, color='steelblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Average Feature Importance')
        plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.show()
    
    return top_features

def create_sample_predictions_safe(predictions_summary: dict, 
                                 actual_data: np.ndarray = None, 
                                 predicted_data: np.ndarray = None,
                                 sample_size: int = 500):
    """Memory-safe sample predictions visualization."""
    
    if actual_data is None or predicted_data is None:
        print("No prediction data available for visualization")
        return
    
    # Limit sample size to prevent memory issues
    n_samples = min(sample_size, len(actual_data))
    indices = np.random.choice(len(actual_data), n_samples, replace=False)
    
    actual_sample = actual_data[indices]
    predicted_sample = predicted_data[indices]
    
    with memory_safe_figure(figsize=(12, 8)) as fig:
        fig.suptitle(f'Model Performance - {predictions_summary.get("sensor_id", "Unknown")}', 
                    fontsize=14, fontweight='bold')
        
        axes = fig.subplots(2, 2)
        
        # Scatter plot (sampled)
        axes[0, 0].scatter(actual_sample, predicted_sample, alpha=0.6, s=10, color='steelblue')
        min_val = min(actual_sample.min(), predicted_sample.min())
        max_val = max(actual_sample.max(), predicted_sample.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0, 0].set_title('Predictions vs Actual (Sampled)', fontweight='bold')
        axes[0, 0].set_xlabel('Actual Vehicle Count')
        axes[0, 0].set_ylabel('Predicted Vehicle Count')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Error distribution
        errors = actual_sample - predicted_sample
        axes[0, 1].hist(errors, bins=min(30, len(errors)//10), edgecolor='black', alpha=0.7, color='coral')
        axes[0, 1].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_title('Error Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('Prediction Error')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Time series (first part of sample)
        time_sample_size = min(168, len(actual_sample))  # 1 week max
        time_indices = range(time_sample_size)
        axes[1, 0].plot(time_indices, actual_sample[:time_sample_size], 
                       label='Actual', alpha=0.8, linewidth=1.5, color='blue')
        axes[1, 0].plot(time_indices, predicted_sample[:time_sample_size], 
                       label='Predicted', alpha=0.8, linewidth=1.5, color='orange')
        axes[1, 0].set_title('Time Series Comparison', fontweight='bold')
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Vehicle Count')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance metrics text
        axes[1, 1].text(0.1, 0.8, f"R² Score: {predictions_summary.get('r2', 0):.3f}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.6, f"RMSE: {predictions_summary.get('rmse', 0):.2f}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.4, f"RMSE %: {predictions_summary.get('rmse_percentage', 0):.1f}%", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.2, f"Sample Size: {n_samples}", 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Performance Summary', fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # Cleanup
    del actual_sample, predicted_sample, errors
    gc.collect()

# Memory monitoring utility
def check_memory_usage():
    """Check current memory usage."""
    import psutil
    memory = psutil.virtual_memory()
    print(f"Memory usage: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB)")
    return memory.percent

if __name__ == "__main__":
    # Test memory-safe visualization
    print("Testing memory-safe visualization...")
    check_memory_usage()
    
    # Create sample data
    sample_metrics = pd.DataFrame({
        'sensor_id': [f'TEST_{i:03d}' for i in range(5)],
        'rmse': np.random.uniform(5, 25, 5),
        'mae': np.random.uniform(3, 20, 5),
        'r2': np.random.uniform(0.7, 0.95, 5),
        'mape': np.random.uniform(8, 30, 5),
        'rmse_percentage': np.random.uniform(8, 30, 5)
    })
    
    create_performance_summary_safe(sample_metrics)
    check_memory_usage()
    
    print("Memory-safe visualization test completed.")