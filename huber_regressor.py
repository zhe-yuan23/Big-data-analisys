import cupy as cp
import numpy as np
from scipy.optimize import minimize
import queue
import threading
from sklearn.preprocessing import StandardScaler, LabelEncoder

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
        # 初始化模型參數 (Huber 閾值k、是否要截距、正則化參數)
        self.k = k
        self.fit_intercept = fit_intercept
        self.reg_param = reg_param
        self.A_total = None
        self.b_total = None
        self.coef_ = None
        self.intercept_ = 0.0 if fit_intercept else None

    def _huber_loss(self, theta, X, y):
        """計算 Huber 損失 (小誤差平方、大誤差線性)"""
        residuals = y - X.dot(theta)
        loss = cp.where(cp.abs(residuals) <= self.k, 
                        0.5 * residuals**2,
                        self.k * cp.abs(residuals) - 0.5 * self.k**2)
        return cp.mean(loss)

    def _huber_grad(self, theta, X, y):
        """計算 Huber 損失的梯度，用來做梯度下降"""
        residuals = y - X @ theta
        abs_r = cp.abs(residuals)
        mask = cp.where(abs_r <= self.k, 
                        1.0, 
                        self.k / abs_r)
        grad = - X.T @ (mask * residuals)
        return grad / X.shape[0]

    def predict(self, X):
        """用已訓練好的係數預測 y"""
        if isinstance(self.coef_, cp.ndarray):
            coef = cp.asnumpy(self.coef_)
        else:
            coef = self.coef_

        if isinstance(self.intercept_, cp.ndarray):
            intercept = cp.asnumpy(self.intercept_)
        else:
            intercept = self.intercept_

        return X @ coef + intercept

    def fit_batch(self, X_batch, y_batch, stream=None):
        """在指定 stream 中執行 GPU 運算"""
        with stream if stream is not None else cp.cuda.Stream.null:
            if self.fit_intercept:
                X_batch = cp.column_stack([cp.ones(len(X_batch)), X_batch])
            
            if self.A_total is None:
                n_features = X_batch.shape[1]
                self.A_total = cp.eye(n_features) * self.reg_param
                self.b_total = cp.zeros(n_features)
            
            # 初始值用Least Squares Method
            theta_t = cp.linalg.lstsq(X_batch, y_batch, rcond=None)[0]

            # 手動梯度下降
            for _ in range(100):
                grad = self._huber_grad(theta_t, X_batch, y_batch)
                theta_t -= 0.01 * grad

            # 累積統計量
            A_t = X_batch.T @ X_batch
            self.A_total += A_t
            self.b_total += A_t @ theta_t

    def finalize(self):
        """根據所有批次的累積統計量，解出最終模型係數"""
        self.A_total += cp.eye(self.A_total.shape[0]) * self.reg_param
        self.coef_ = cp.linalg.solve(self.A_total, self.b_total)
        
        if self.fit_intercept:
            self.intercept_ = self.coef_[0]
            self.coef_ = self.coef_[1:]
        return self

class DataPipeline:
    def __init__(self, X_train, y_train, batch_size=10000, max_queue_size=3):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size
        self.data_queue = queue.Queue(maxsize=max_queue_size)
        self.finished = False
        self.total_batches = (len(X_train) + batch_size - 1) // batch_size
        
    def _data_preparation_thread(self):
        """背景執行緒：資料準備和預處理"""
        for i in range(0, len(self.X_train), self.batch_size):
            if self.finished:
                break
                
            # 步驟 1：切片資料 (CPU)
            X_batch = self.X_train[i:i+self.batch_size]
            y_batch = self.y_train.iloc[i:i+self.batch_size].values

            # 步驟 2：非同步轉換為 CuPy 陣列 (CPU→GPU)
            stream = cp.cuda.Stream(non_blocking=True)  # 建立非同步 Stream
            with stream:
                X_batch_gpu = cp.asarray(X_batch, order='C')  # Host → Device
                y_batch_gpu = cp.asarray(y_batch, order='C')

            # 步驟 3：放入佇列（包含 stream）
            batch_data = (X_batch_gpu, y_batch_gpu, i // self.batch_size + 1, stream)
            if self.data_queue.full():
               print(f"[警告] queue 已滿，背景 thread 等待中")

            self.data_queue.put(batch_data)

            if i // self.batch_size % 10 == 0:
                print(f"完成批次 {i//self.batch_size + 1}/{self.total_batches}")

        print("資料準備執行緒完成")
    
    def start_pipeline(self):
        """啟動資料pipeline"""
        # 啟動背景執行緒
        self.thread = threading.Thread(target=self._data_preparation_thread, daemon=True) # dermon 設定為背景執行序
        self.thread.start()
        
    def get_batch(self):
        """主執行緒呼叫：取得下一批資料"""
        try:
            return self.data_queue.get(timeout=30)  # 30秒超時
        except queue.Empty:
            return None
    
    def mark_batch_done(self):
        """標記批次處理完成"""
        self.data_queue.task_done()
    
    def stop_pipeline(self):
        """停止pipeline(發出結束信號並等待結束)"""
        self.finished = True
        if hasattr(self, 'thread'): # 判斷是否有thread屬性
            self.thread.join() # .join 等待thread結束再繼續

# 原始資料預處理 - 根據資料修改
def preprocess_data(df):
    """資料預處理 """
    categories = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
    for col in categories:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df