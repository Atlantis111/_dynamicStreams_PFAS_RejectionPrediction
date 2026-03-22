import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.base import clone
import warnings

warnings.filterwarnings('ignore')


class DynamicDataStreamRandomForest:
    def __init__(self, feature_columns, target_column, window_size=100, retrain_interval=50,
                 drift_detection_threshold=0.1, initial_train_size=50):
        """
        动态数据流随机森林回归器

        参数:
            feature_columns: 自变量参数列表
            target_column: 待预测参数
            window_size: 滑动窗口大小
            retrain_interval: 重新训练间隔
            drift_detection_threshold: 漂移检测阈值
            initial_train_size: 初始训练集大小
        """
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.window_size = window_size
        self.retrain_interval = retrain_interval
        self.drift_detection_threshold = drift_detection_threshold
        self.initial_train_size = initial_train_size

        # 数据存储
        self.data_buffer = []
        self.X_window = None
        self.y_window = None

        # 模型相关
        self.model = None
        self.best_params_ = None
        self.retrain_count = 0
        self.drift_detected = False

        # 评估指标记录
        self.metrics_history = {
            'train_mse': [], 'test_mse': [],
            'train_mae': [], 'test_mae': [],
            'train_r2': [], 'test_r2': [],
            'drift_scores': []
        }

    def load_initial_data(self, file_path):
        """加载初始数据集"""
        df = pd.read_excel(file_path)
        print(f"数据集加载成功，共{len(df)}条数据，{len(self.feature_columns)}个特征")
        return df

    def initialize_model(self, X_train, y_train):
        """使用网格搜索初始化随机森林模型"""
        print("正在进行初始网格搜索优化...")

        # 定义参数网格
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['auto', 'sqrt']
        }

        # 网格搜索
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_train, y_train)
        self.best_params_ = grid_search.best_params_
        self.model = grid_search.best_estimator_

        print(f"初始网格搜索完成，最佳参数: {self.best_params_}")
        return self.model

    def detect_concept_drift(self, X_new, y_new):
        """检测概念漂移"""
        if len(self.metrics_history['drift_scores']) < 2:
            return False

        # 使用最新数据计算预测误差
        y_pred = self.model.predict(X_new)
        current_error = mean_squared_error(y_new, y_pred)

        # 计算误差变化率
        if len(self.metrics_history['test_mse']) > 0:
            previous_error = np.mean(self.metrics_history['test_mse'][-3:])  # 最近3次的平均误差
            error_change = abs(current_error - previous_error) / previous_error

            # 如果误差变化超过阈值，检测到漂移
            if error_change > self.drift_detection_threshold:
                print(f"检测到概念漂移! 误差变化率: {error_change:.4f}")
                return True

        return False

    def update_data_window(self, new_data):
        """更新数据窗口"""
        if len(self.data_buffer) >= self.window_size:
            # 移除最旧的数据
            self.data_buffer = self.data_buffer[1:]

        self.data_buffer.append(new_data)

        # 更新窗口数据
        window_df = pd.DataFrame(self.data_buffer)
        self.X_window = window_df[self.feature_columns].values
        self.y_window = window_df[self.target_column].values

    def retrain_model(self):
        """重新训练模型"""
        print(f"第{self.retrain_count + 1}次重新训练...")

        if len(self.data_buffer) < self.initial_train_size:
            print("数据量不足，跳过本次重新训练")
            return

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_window, self.y_window, test_size=0.2, random_state=32
        )

        # 使用网格搜索优化参数
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }

        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        self.best_params_ = grid_search.best_params_

        # 评估模型
        train_metrics = self.evaluate_model(X_train, y_train)
        test_metrics = self.evaluate_model(X_test, y_test)

        # 记录评估指标
        self.metrics_history['train_mse'].append(train_metrics['mse'])
        self.metrics_history['train_mae'].append(train_metrics['mae'])
        self.metrics_history['train_r2'].append(train_metrics['r2'])
        self.metrics_history['test_mse'].append(test_metrics['mse'])
        self.metrics_history['test_mae'].append(test_metrics['mae'])
        self.metrics_history['test_r2'].append(test_metrics['r2'])

        self.retrain_count += 1
        print(f"重新训练完成，测试集MSE: {test_metrics['mse']:.4f}")

    def evaluate_model(self, X, y):
        """评估模型性能"""
        y_pred = self.model.predict(X)
        return {
            'mse': mean_squared_error(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }

    def process_data_stream(self, file_path, stream_size=150):
        """处理数据流"""
        # 加载数据
        df = self.load_initial_data(file_path)

        if len(df) < stream_size:
            stream_size = len(df)
            print(f"数据量不足，调整流大小为: {stream_size}")

        # 初始训练
        initial_data = df.iloc[:self.initial_train_size]
        self.data_buffer = [dict(row) for _, row in initial_data.iterrows()]
        window_df = pd.DataFrame(self.data_buffer)

        X_initial = window_df[self.feature_columns].values
        y_initial = window_df[self.target_column].values

        self.initialize_model(X_initial, y_initial)

        # 处理数据流
        print("开始处理数据流...")
        for i in range(self.initial_train_size, stream_size):
            new_sample = dict(df.iloc[i])
            self.update_data_window(new_sample)

            # 检测概念漂移
            X_new = np.array([new_sample[col] for col in self.feature_columns]).reshape(1, -1)
            y_new = np.array([new_sample[self.target_column]])

            drift_detected = self.detect_concept_drift(X_new, y_new)
            self.metrics_history['drift_scores'].append(drift_detected)

            # 定期重新训练或检测到漂移时重新训练
            if (len(self.data_buffer) % self.retrain_interval == 0) or drift_detected:
                self.retrain_model()

            # 打印进度
            if (i - self.initial_train_size + 1) % 20 == 0:
                print(f"已处理 {i - self.initial_train_size + 1} 个新样本")

        print("数据流处理完成!")

    def print_final_metrics(self):
        """打印最终评估结果"""
        if len(self.metrics_history['test_mse']) > 0:
            print("\n" + "=" * 60)
            print("最终模型评估结果:")
            print("=" * 60)

            latest_test_mse = self.metrics_history['test_mse'][-1]
            latest_test_mae = self.metrics_history['test_mae'][-1]
            latest_test_r2 = self.metrics_history['test_r2'][-1]

            print(f"测试集 - MSE: {latest_test_mse:.4f}, MAE: {latest_test_mae:.4f}, R²: {latest_test_r2:.4f}")

            if len(self.metrics_history['train_mse']) > 0:
                latest_train_mse = self.metrics_history['train_mse'][-1]
                latest_train_mae = self.metrics_history['train_mae'][-1]
                latest_train_r2 = self.metrics_history['train_r2'][-1]
                print(f"训练集 - MSE: {latest_train_mse:.4f}, MAE: {latest_train_mae:.4f}, R²: {latest_train_r2:.4f}")

            print(f"检测到的概念漂移次数: {sum(self.metrics_history['drift_scores'])}")
            print(f"总重新训练次数: {self.retrain_count}")


# 使用示例
if __name__ == "__main__":
    # 定义特征列和目标列
    feature_columns = [
        'Compound log K ow', 'WS (mg/L)',
        'MinPartialCharge', 'MaxPartialCharge', 'min projection (Å)', 'S',
        'rs/rp', '∆Gs-m (J·m-2)', 'MB contact angle (°)',
        'MB volume charge density δm (mol·m-3)',
        'Pressure (kPa)', 'Measurement time (min)', 'Initial concentration of compound (mg/L)', 'pH'
    ]
    target_column = "removal rate (%)"

    # 创建动态数据流随机森林模型
    dds_rf = DynamicDataStreamRandomForest(
        feature_columns=feature_columns,
        target_column=target_column,
        window_size=100,
        retrain_interval=30,
        drift_detection_threshold=0.8,
        initial_train_size=40
    )

    # 处理数据流
    dds_rf.process_data_stream('../../PFASdata.xlsx', stream_size=150)

    # 打印最终结果
    dds_rf.print_final_metrics()

    # 可选：保存评估历史
    metrics_df = pd.DataFrame(dds_rf.metrics_history)
    metrics_df.to_csv('dynamic_forest_metrics_history.csv', index=False)
    print("\n评估历史已保存到 'dynamic_forest_metrics_history.csv'")