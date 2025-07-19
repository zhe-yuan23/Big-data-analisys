# Online Updating Huber Robust Regression

A robust regression implementation using Huber loss for online learning with batch processing capabilities. This project demonstrates how to build a robust regression model that can handle outliers effectively while supporting incremental learning.

## Features

- **Robust to Outliers**: Uses Huber loss function which is less sensitive to outliers than squared loss
- **Online Learning**: Supports batch-wise training for large datasets
- **GPU Acceleration**: CUDA-optimized implementation with CuPy for significant speedup
- **Advanced Data Pipeline**: Asynchronous data loading with background threading
- **Memory Transfer Optimization**: Overlapped CPU↔GPU memory transfers using CUDA streams
- **Regularization**: Includes L2 regularization to prevent overfitting
- **Flexible**: Supports both with and without intercept fitting
- **Performance Monitoring**: Detailed timing statistics and GPU utilization tracking

## Installation

```bash
pip install numpy scipy scikit-learn pandas matplotlib seaborn

# For GPU support (optional)
pip install cupy-cuda11x  # or cupy-cuda12x depending on your CUDA version
```

## Usage

### Basic CPU Implementation

```python
from robust_huber_regression import RobustOnlineHuberRegressor
import numpy as np

# Initialize the model
model = RobustOnlineHuberRegressor(
    k=1.345,           # Huber loss threshold
    fit_intercept=True, 
    reg_param=1e-4     # Regularization strength
)

# Train in batches
for X_batch, y_batch in data_batches:
    model.fit_batch(X_batch, y_batch)

# Finalize training
model.finalize()

# Make predictions
predictions = model.predict(X_test)
```

### GPU Implementation with Data Pipeline

```python
from robust_huber_regression_gpu import RobustOnlineHuberRegressorGPU, DataPipeline
import cupy as cp
import time

# Initialize GPU model
model = RobustOnlineHuberRegressorGPU(
    k=1.345, 
    fit_intercept=True, 
    reg_param=1e-4
)

# Initialize data pipeline with optimized parameters
pipeline = DataPipeline(
    X_train_scaled, 
    y_train, 
    batch_size=50000,      # Larger batch sizes for GPU efficiency
    max_queue_size=3       # Queue size for background processing
)

# Start asynchronous data pipeline
start_time = time.time()
pipeline.start_pipeline()

# Training loop with overlapped memory transfers
batch_count = 0
gpu_time = 0
prev_stream = None

while True:
    # Get batch from queue (non-blocking with background thread)
    batch_data = pipeline.get_batch()
    if batch_data is None:
        break
        
    X_batch, y_batch, batch_id, stream = batch_data
    
    # Synchronize previous GPU computation
    if prev_stream is not None:
        prev_stream.synchronize()
    
    # GPU computation (overlapped with next batch memory transfer)
    gpu_start = time.time()
    model.fit_batch(X_batch, y_batch, stream)
    gpu_time += time.time() - gpu_start
    
    pipeline.mark_batch_done()
    batch_count += 1
    prev_stream = stream

# Final synchronization
if prev_stream is not None:
    prev_stream.synchronize()

# Stop pipeline and finalize model
pipeline.stop_pipeline()
model.finalize()

# Performance statistics
total_time = time.time() - start_time
print(f"總時間: {total_time:.2f}秒")
print(f"GPU 運算時間: {gpu_time:.2f}秒") 
print(f"GPU 使用時間佔比: {gpu_time/total_time*100:.1f}%")
```

## Algorithm Details

### Huber Loss Function

The Huber loss function combines the best properties of squared loss and absolute loss:

$$
L(r) = \begin{cases}
0.5 r^2, & \text{if } |r| \leq k \\
k |r| - 0.5 k^2, & \text{if } |r| > k
\end{cases}
$$

Where `r` is the residual and `k` is the threshold parameter.

### GPU Optimization Strategy

The GPU implementation uses several optimization techniques:

1. **Asynchronous Data Pipeline**: Background thread handles data preparation while GPU computes
2. **CUDA Streams**: Non-blocking memory transfers overlapped with computation
3. **Manual Gradient Descent**: Custom implementation optimized for GPU parallel computation
4. **Memory Management**: Efficient CuPy array handling with proper synchronization

### Online Learning Approach

The model uses an online learning approach where:
1. For each batch, it computes optimal parameters using L-BFGS-B optimization
2. Updates cumulative matrices A_total and b_total
3. Final parameters are computed by solving the regularized normal equation

## Parameters

- **k** (float, default=1.345): Huber loss threshold. Smaller values make the model more robust to outliers
- **fit_intercept** (bool, default=True): Whether to fit an intercept term
- **reg_param** (float, default=1e-4): L2 regularization parameter

## Data Preprocessing Features

- **Outlier Detection**: Visual detection using boxplots with statistical removal
- **Missing Value Handling**: Intelligent imputation with mean/median strategies
- **Categorical Encoding**: LabelEncoder for categorical variables
- **Low-Frequency Filtering**: Remove categories with insufficient samples
- **Data Standardization**: StandardScaler for numerical stability
- **Data Validation**: Comprehensive NaN/Inf checking for GPU compatibility

## Visualization

The code includes visualization capabilities:
- Outlier detection boxplots
- Prediction vs actual value scatter plots
- Detailed performance timing statistics
- GPU utilization monitoring
- Training progress indicators

## Performance

Performance improvements with GPU implementation:
- **CPU Version**: Good for smaller datasets, simple deployment
- **GPU Version**: speedup on large datasets with optimized memory pipeline
- **Memory Efficiency**: Background data preparation eliminates GPU idle time
- **Scalability**: Handles datasets too large for memory through streaming

## File Structure

```
├── Online updating Huber robust regression.py        # CPU implementation
├── Online updating Huber robust regression on GPU.py # GPU implementation with pipeline
└── README.md                                         # This file
```

## Requirements

### Basic Requirements
- Python 3.7+
- NumPy
- SciPy
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn

### GPU Requirements
- CUDA-compatible GPU
- CuPy (cupy-cuda11x or cupy-cuda12x)
- Sufficient GPU memory for batch processing

## References

- Online Updating Huber Robust Regression for Big Data Streams
