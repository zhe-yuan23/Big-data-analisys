# Online Updating Huber Robust Regression

A robust regression implementation using Huber loss for online learning with batch processing.  
This project demonstrates both **CPU** and **GPU** versions that can handle outliers effectively while supporting incremental learning.

## Features
- **Robust to Outliers**: Uses Huber loss, less sensitive to outliers than squared loss
- **Online Learning**: Batch-wise training for large datasets
- **GPU Acceleration**: CUDA-optimized with CuPy for speedup
- **Asynchronous Data Pipeline**: Background threading + CUDA streams
- **Regularization**: L2 penalty to prevent overfitting
- **Performance Monitoring**: Timing statistics and GPU utilization

## Installation

```bash
pip install numpy scipy scikit-learn pandas matplotlib seaborn

# For GPU support (optional)
pip install cupy-cuda11x  # or cupy-cuda12x depending on your CUDA version
```
## Dataset
playground-series-s5e4/train.csv
You can download it from Kaggle [Playground Series S5E4](https://www.kaggle.com/competitions/playground-series-s5e4).
## Usage

### Basic CPU Implementation

```python
from robust_huber_regression import RobustOnlineHuberRegressor
import numpy as np

# Initialize the model
from Online_updating_Huber_robust_regression import RobustOnlineHuberRegressor
import numpy as np

model = RobustOnlineHuberRegressor(
    k=1.345,
    fit_intercept=True, 
    reg_param=1e-4
)

for X_batch, y_batch in data_batches:
    model.fit_batch(X_batch, y_batch)

model.finalize()
predictions = model.predict(X_test)
```

### GPU Implementation with Data Pipeline

```python
from Online_updating_Huber_robust_regression_on_GPU import RobustOnlineHuberRegressorGPU, DataPipeline
import cupy as cp
import time

model = RobustOnlineHuberRegressorGPU(k=1.345, fit_intercept=True, reg_param=1e-4)
pipeline = DataPipeline(X_train_scaled, y_train, batch_size=50000, max_queue_size=3)

start_time = time.time()
pipeline.start_pipeline()

batch_count, gpu_time, prev_stream = 0, 0, None
while True:
    batch_data = pipeline.get_batch()
    if batch_data is None:
        break
    
    # batch_data = (X_batch_gpu, y_batch_gpu, batch_id, stream)
    X_batch, y_batch, batch_id, stream = batch_data
    
    if prev_stream is not None:
        prev_stream.synchronize()
    
    gpu_start = time.time()
    model.fit_batch(X_batch, y_batch, stream)
    gpu_time += time.time() - gpu_start
    
    pipeline.mark_batch_done()
    batch_count += 1
    prev_stream = stream

if prev_stream is not None:
    prev_stream.synchronize()

pipeline.stop_pipeline()
model.finalize()
```

## Algorithm Details
### CPU Version
Initialization: Least Squares solution

Optimization: Uses L-BFGS-B (via SciPy minimize) to optimize Huber loss

Update Rule: Accumulate matrices A_total and b_total across batches

### GPU Version
Initialization: Least Squares solution

Optimization: Custom gradient descent (100 iterations, fixed step size)

Pipeline: Background thread loads data → Asynchronous CPU→GPU transfer → Overlapped GPU computation with CUDA streams

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
Online-Updating-Huber-Robust-Regression/
├── README.md                                          # This file
├── huber_regressor.py                                 # 核心演算法實作
├── Online updating Huber robust regression.py         # CPU version
└── Online updating Huber robust regression on GPU.py  # GPU version                                   
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
- Kaggle Playground Series S5E4
