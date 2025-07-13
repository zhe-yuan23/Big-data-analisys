import cupy as cp
import numpy as np
from scipy.optimize import minimize

class RobustOnlineHuberRegressor:
    def __init__(self, k=1.345, fit_intercept=True, reg_param=1e-4):
        self.k = k                           # Huber損失閾值
        self.fit_intercept = fit_intercept   # 正則化強度
        self.reg_param = reg_param           # 累積的設計矩陣
        self.A_total = None                  # 累積的目標向量
        self.b_total = None                  # 回歸係數
        self.coef_ = None                    # 截距項
        self.intercept_ = 0.0 if fit_intercept else None

    def _huber_loss(self, theta, X, y):
        residuals = y - X.dot(theta)
        loss = np.where(np.abs(residuals) <= self.k, 
                        0.5 * residuals**2,
                        self.k * np.abs(residuals) - 0.5 * self.k**2)
        return np.mean(loss)

    def predict(self, X):
        if self.fit_intercept:
            return X @ self.coef_ + self.intercept_
        else:
            return X @ self.coef_

    def fit_batch(self, X_batch, y_batch):
        # X_batch (array-like): 批次特徵矩陣
        # y_batch (array-like): 批次目標值
        # 如果需要截距項，添加全1列
        if self.fit_intercept:
            X_batch = np.column_stack([np.ones(len(X_batch)), X_batch])
        
        # 初始化累積矩陣
        if self.A_total is None:
            n_features = X_batch.shape[1]
            self.A_total = np.eye(n_features) * self.reg_param  # 初始化正則化項
            self.b_total = np.zeros(n_features)
        
        # 使用Least Squares Method作為初始解
        theta_init = np.linalg.lstsq(X_batch, y_batch, rcond=None)[0]
        
        # 使用L-BFGS-B優化Huber損失
        try:
            result = minimize(
                lambda theta: self._huber_loss(theta, X_batch, y_batch),
                theta_init,
                method='L-BFGS-B'
            )
            theta_t = result.x
        except:
            theta_t = theta_init  # 回退到Least Squares Method解

        A_t = X_batch.T @ X_batch
        self.A_total += A_t
        self.b_total += A_t @ theta_t

    def finalize(self):
        # 添加正則化項確保矩陣可逆
        self.A_total += np.eye(self.A_total.shape[0]) * self.reg_param
        self.coef_ = np.linalg.solve(self.A_total, self.b_total)
        
        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        return self
    
class RobustOnlineHuberRegressorGPU:
    def __init__(self, k=1.345, fit_intercept=True, reg_param=1e-4):
        self.k = k
        self.fit_intercept = fit_intercept
        self.reg_param = reg_param
        self.A_total = None
        self.b_total = None
        self.coef_ = None
        self.intercept_ = 0.0 if fit_intercept else None

    def _huber_loss(self, theta, X, y):
        residuals = y - X.dot(theta)
        loss = cp.where(cp.abs(residuals) <= self.k, 
                        0.5 * residuals**2,
                        self.k * cp.abs(residuals) - 0.5 * self.k**2)
        return cp.mean(loss)

    def _huber_grad(self, theta, X, y):
        residuals = y - X @ theta
        abs_r = cp.abs(residuals)
        mask = cp.where(abs_r <= self.k, 
                        1.0, 
                        self.k / abs_r)
        grad = - X.T @ (mask * residuals)
        return grad / X.shape[0]

    def predict(self, X):
        # 確保 coef 和 intercept 都是 NumPy 陣列或數值
        if isinstance(self.coef_, cp.ndarray):
            coef = cp.asnumpy(self.coef_)
        else:
            coef = self.coef_

        if isinstance(self.intercept_, cp.ndarray):
            intercept = cp.asnumpy(self.intercept_)
        else:
            intercept = self.intercept_

        return X @ coef + intercept

    def fit_batch(self, X_batch, y_batch):
        # GPU：轉換為 CuPy 陣列
        X_batch = cp.asarray(X_batch)
        y_batch = cp.asarray(y_batch)

        if self.fit_intercept:
            X_batch = cp.column_stack([cp.ones(len(X_batch)), X_batch])
        
        if self.A_total is None:
            n_features = X_batch.shape[1]
            self.A_total = cp.eye(n_features) * self.reg_param
            self.b_total = cp.zeros(n_features)
        
        # 初始值用最小二乘
        theta_t = cp.linalg.lstsq(X_batch, y_batch, rcond=None)[0]

        # 手動梯度下降
        for _ in range(100):  # 可調整 self.n_iter
            grad = self._huber_grad(theta_t, X_batch, y_batch)
            theta_t -= 0.01 * grad  # 學習率可參數化

        # 累積統計量
        A_t = X_batch.T @ X_batch
        self.A_total += A_t
        self.b_total += A_t @ theta_t

    def finalize(self):
        # 添加正則化確保可逆
        self.A_total += cp.eye(self.A_total.shape[0]) * self.reg_param
        self.coef_ = cp.linalg.solve(self.A_total, self.b_total)
        
        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        return self