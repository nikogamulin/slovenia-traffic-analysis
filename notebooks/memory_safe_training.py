"""
Memory-safe model training to replace the training loop in cell 7
This version prevents memory leaks and limits resource usage
"""
import gc
import psutil
import numpy as np
import pandas as pd
from pathlib import Path

def train_sensor_model_memory_safe(sensor_id: str, 
                                  data: pd.DataFrame,
                                  target_col: str = 'vehicle_count',
                                  test_size: float = 0.2,
                                  val_size: float = 0.1,
                                  max_memory_mb: int = 1000):
    """Memory-safe version of train_sensor_model with resource monitoring."""
    
    # Check available memory before starting
    available_memory = psutil.virtual_memory().available / 1024**2  # MB
    if available_memory < max_memory_mb:
        raise MemoryError(f"Insufficient memory: {available_memory:.1f}MB available, {max_memory_mb}MB required")
    
    # Filter data for specific sensor and immediately drop unused columns
    sensor_df = data[data['sensor_id'] == sensor_id].copy()
    del data  # Immediately free reference to full dataset
    
    # Initialize framework with memory-safe parameters
    framework = XGBoostFramework(
        task_type='regression',
        model_name=f'{sensor_id}_{target_col}',
        save_dir='./models/vehicle_count'
    )
    
    # Prepare features with explicit cleanup
    exclude_cols = ['timestamp', 'sensor_id', 'sensor_type', 'location', target_col]
    feature_cols = [col for col in sensor_df.columns if col not in exclude_cols]
    
    X, y = framework.prepare_features(
        sensor_df,
        target_col=target_col,
        feature_cols=feature_cols
    )
    
    # Immediately cleanup sensor_df after feature extraction
    del sensor_df
    gc.collect()
    
    # Time-based split with memory-efficient indexing
    n_samples = len(X)
    train_idx = int(n_samples * (1 - test_size - val_size))
    val_idx = int(n_samples * (1 - test_size))
    
    # Create splits without copying data
    X_train = X[:train_idx]
    y_train = y[:train_idx]
    X_val = X[train_idx:val_idx]
    y_val = y[train_idx:val_idx]
    X_test = X[val_idx:]
    y_test = y[val_idx:]
    
    # Memory-safe training parameters (limited resources)
    params = framework.get_default_params()
    params.update({
        'n_estimators': 100,      # Reduced from 200
        'max_depth': 6,           # Reduced from 8  
        'learning_rate': 0.1,     # Increased for faster convergence
        'n_jobs': 2,              # Limited CPU cores (was -1)
        'max_delta_step': 1,      # Prevent extreme updates
        'tree_method': 'hist',    # Memory-efficient tree construction
        'max_bin': 256,           # Reduce memory for histogram
    })
    
    # Train with memory monitoring
    print(f"  Memory before training: {psutil.virtual_memory().percent:.1f}%")
    
    framework.train(
        X_train, y_train,
        X_val, y_val,
        params=params,
        early_stopping_rounds=15  # Reduced from 20
    )
    
    print(f"  Memory after training: {psutil.virtual_memory().percent:.1f}%")
    
    # Evaluate on test set
    metrics = framework.evaluate(X_test, y_test)
    
    # Create lightweight predictions (no DataFrame overhead)
    y_pred = framework.model.predict(X_test)
    metrics['mean_target'] = float(y_test.mean())
    metrics['rmse_percentage'] = float((metrics['rmse'] / metrics['mean_target']) * 100)
    
    # Create minimal predictions summary (not full DataFrame)
    predictions_summary = {
        'sensor_id': sensor_id,
        'n_predictions': len(y_test),
        'rmse': float(metrics['rmse']),
        'mae': float(metrics['mae']),
        'r2': float(metrics['r2']),
        'rmse_percentage': float(metrics['rmse_percentage'])
    }
    
    # Immediate cleanup of training data
    del X, y, X_train, y_train, X_val, y_val, X_test, y_test, y_pred
    gc.collect()
    
    print(f"  Memory after cleanup: {psutil.virtual_memory().percent:.1f}%")
    
    return framework, metrics, predictions_summary

def memory_safe_training_loop(sensor_data: pd.DataFrame, max_models: int = None):
    """Memory-safe training loop with resource monitoring."""
    
    print("Starting memory-safe training loop...")
    print(f"Initial memory usage: {psutil.virtual_memory().percent:.1f}%")
    
    all_models = {}
    all_metrics = []
    all_predictions = []
    
    sensor_ids = sensor_data['sensor_id'].unique()
    if max_models:
        sensor_ids = sensor_ids[:max_models]
    
    for i, sensor_id in enumerate(sensor_ids):
        print(f"\nTraining model {i+1}/{len(sensor_ids)} for {sensor_id}...")
        
        # Check memory before each model
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 85:  # Stop if memory usage exceeds 85%
            print(f"⚠️  Memory usage too high ({memory_percent:.1f}%), stopping training")
            break
        
        try:
            # Train model with memory monitoring
            framework, metrics, predictions_summary = train_sensor_model_memory_safe(
                sensor_id, sensor_data.copy()  # Pass copy to avoid reference issues
            )
            
            all_models[sensor_id] = framework
            all_metrics.append({
                'sensor_id': sensor_id,
                **metrics
            })
            all_predictions.append(predictions_summary)
            
            # Print results
            print(f"  ✓ Model trained successfully")
            print(f"  RMSE: {metrics['rmse']:.2f}")
            print(f"  R²: {metrics['r2']:.3f}")
            print(f"  RMSE %: {metrics['rmse_percentage']:.1f}%")
            
            # Force garbage collection after each model
            if hasattr(framework.model, 'get_booster'):
                # Clear XGBoost internal structures
                booster = framework.model.get_booster()
                booster.dump_model  # This can help clear internal memory
                del booster
            
            gc.collect()
            
            # Optional: Save model immediately to free memory
            if len(all_models) % 3 == 0:  # Save every 3 models
                print(f"  Saving models to disk to free memory...")
                for saved_sensor_id, saved_framework in all_models.items():
                    try:
                        saved_framework.save_model()
                        # Clear the model from memory after saving
                        saved_framework.model = None
                        saved_framework.scaler = None
                    except Exception as e:
                        print(f"    Warning: Could not save {saved_sensor_id}: {e}")
                gc.collect()
            
        except Exception as e:
            print(f"  ❌ Error training model: {e}")
            gc.collect()  # Cleanup even on failure
            continue
    
    return all_models, all_metrics, all_predictions

if __name__ == "__main__":
    # Test memory-safe training
    from memory_optimized_data_generation import generate_memory_efficient_sensor_data
    
    print("Testing memory-safe training...")
    data = generate_memory_efficient_sensor_data(n_sensors=3, n_days=30)
    models, metrics, predictions = memory_safe_training_loop(data, max_models=3)
    print(f"Trained {len(models)} models successfully")