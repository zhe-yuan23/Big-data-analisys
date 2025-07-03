# Online Updating Huber Robust Regression

A robust regression implementation using Huber loss for online learning with batch processing capabilities. This project demonstrates how to build a robust regression model that can handle outliers effectively while supporting incremental learning.

## Features

- **Robust to Outliers**: Uses Huber loss function which is less sensitive to outliers than squared loss
- **Online Learning**: Supports batch-wise training for large datasets
- **Regularization**: Includes L2 regularization to prevent overfitting
- **Flexible**: Supports both with and without intercept fitting
- **Efficient**: Uses matrix operations for fast computation

## Installation

```bash
pip install numpy scipy scikit-learn pandas matplotlib seaborn
```

## Usage

### Basic Example

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

### Complete Pipeline (Podcast Dataset Example)

The repository includes a complete example using a podcast dataset:

```python
# Data preprocessing
df = preprocess_data(df)  # Handle categorical variables
X = df.drop(columns=['target_column', 'id_columns'])
y = df['target_column']

# Split and scale data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Train model
huber_model = RobustOnlineHuberRegressor(k=1.345, fit_intercept=True, reg_param=1e-4)

batch_size = 10000
for i in range(0, len(X_train_scaled), batch_size):
    X_batch = X_train_scaled[i:i+batch_size]
    y_batch = y_train.iloc[i:i+batch_size].values
    huber_model.fit_batch(X_batch, y_batch)

huber_model.finalize()

# Evaluate
y_pred = huber_model.predict(X_val_scaled)
print(f"MAE: {mean_absolute_error(y_val, y_pred):.2f}")
print(f"RMSE: {mean_squared_error(y_val, y_pred)**0.5:.2f}")
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

The example includes comprehensive data preprocessing:
- Outlier detection and removal using boxplots
- Missing value imputation with mean/median
- Categorical variable encoding with LabelEncoder
- Data standardization with StandardScaler
- Low-frequency category removal

## Visualization

The code includes visualization capabilities:
- Outlier detection boxplots
- Prediction vs actual value scatter plots
- Model performance metrics

## Performance

The model demonstrates robust performance on the podcast dataset:
- Handles outliers effectively
- Maintains good predictive accuracy
- Scales well with large datasets through batch processing

## File Structure

```
├── Online updating Huber robust regression.py  # Main implementation
├── README.md                                   # This file
└── requirements.txt                           # Dependencies
```

## Requirements

- Python 3.7+
- NumPy
- SciPy
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## References

- Online Updating Huber Robust Regression for Big Data Streams
