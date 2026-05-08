import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
import warnings

warnings.filterwarnings('ignore')

class DynamicDataStreamRandomForest:
    def __init__(self, feature_columns, target_column, window_size=100, retrain_interval=50,
                 drift_detection_threshold=0.1, initial_train_size=50):
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.window_size = window_size
        self.retrain_interval = retrain_interval
        self.drift_detection_threshold = drift_detection_threshold
        self.initial_train_size = initial_train_size

        self.data_buffer = []
        self.X_window = None
        self.y_window = None

        self.model = None
        self.best_params_ = None
        self.retrain_count = 0
        self.drift_detected = False

        self.metrics_history = {
            'train_mse': [], 'test_mse': [],
            'train_mae': [], 'test_mae': [],
            'train_r2': [], 'test_r2': [],
            'drift_scores': []
        }

    def load_initial_data(self, file_path):
        df = pd.read_excel(file_path)
        print(f"数据集加载成功，共{len(df)}条数据，{len(self.feature_columns)}个特征")
        return df

    def initialize_model(self, X_train, y_train):
        print("正在进行初始网格搜索优化...")
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['auto', 'sqrt']
        }

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
        if len(self.metrics_history['drift_scores']) < 2:
            return False

        y_pred = self.model.predict(X_new)
        current_error = mean_squared_error(y_new, y_pred)

        if len(self.metrics_history['test_mse']) > 0:
            previous_error = np.mean(self.metrics_history['test_mse'][-3:])
            error_change = abs(current_error - previous_error) / previous_error

            if error_change > self.drift_detection_threshold:
                print(f"检测到概念漂移! 误差变化率: {error_change:.4f}")
                return True

        return False

    def update_data_window(self, new_data):
        if len(self.data_buffer) >= self.window_size:
            self.data_buffer = self.data_buffer[1:]

        self.data_buffer.append(new_data)
        window_df = pd.DataFrame(self.data_buffer)
        self.X_window = window_df[self.feature_columns].values
        self.y_window = window_df[self.target_column].values

    def retrain_model(self):
        print(f"第{self.retrain_count + 1}次重新训练...")

        if len(self.data_buffer) < self.initial_train_size:
            print("数据量不足，跳过本次重新训练")
            return

        X_train, X_test, y_train, y_test = train_test_split(
            self.X_window, self.y_window, test_size=0.2, random_state=32
        )

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

        train_metrics = self.evaluate_model(X_train, y_train)
        test_metrics = self.evaluate_model(X_test, y_test)

        self.metrics_history['train_mse'].append(train_metrics['mse'])
        self.metrics_history['train_mae'].append(train_metrics['mae'])
        self.metrics_history['train_r2'].append(train_metrics['r2'])
        self.metrics_history['test_mse'].append(test_metrics['mse'])
        self.metrics_history['test_mae'].append(test_metrics['mae'])
        self.metrics_history['test_r2'].append(test_metrics['r2'])

        self.retrain_count += 1
        print(f"重新训练完成，测试集MSE: {test_metrics['mse']:.4f}")

    def evaluate_model(self, X, y):
        y_pred = self.model.predict(X)
        return {
            'mse': mean_squared_error(y, y_pred),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }

    def process_data_stream(self, file_path, stream_size=150):
        df = self.load_initial_data(file_path)

        if len(df) < stream_size:
            stream_size = len(df)
            print(f"数据量不足，调整流大小为: {stream_size}")

        initial_data = df.iloc[:self.initial_train_size]
        self.data_buffer = [dict(row) for _, row in initial_data.iterrows()]
        window_df = pd.DataFrame(self.data_buffer)

        X_initial = window_df[self.feature_columns].values
        y_initial = window_df[self.target_column].values

        self.initialize_model(X_initial, y_initial)

        print("开始处理数据流...")
        for i in range(self.initial_train_size, stream_size):
            new_sample = dict(df.iloc[i])
            self.update_data_window(new_sample)

            X_new = np.array([new_sample[col] for col in self.feature_columns]).reshape(1, -1)
            y_new = np.array([new_sample[self.target_column]])

            drift_detected = self.detect_concept_drift(X_new, y_new)
            self.metrics_history['drift_scores'].append(drift_detected)

            if (len(self.data_buffer) % self.retrain_interval == 0) or drift_detected:
                self.retrain_model()

            if (i - self.initial_train_size + 1) % 20 == 0:
                print(f"已处理 {i - self.initial_train_size + 1} 个新样本")

        print("数据流处理完成!")

    def print_final_metrics(self):
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

if __name__ == "__main__":
    feature_columns = [
        'Compound log K ow', 'WS (mg/L)',
        'MinPartialCharge', 'MaxPartialCharge', 'min projection (Å)', 'S',
        'rs/rp', '∆Gs-m (J·m-2)', 'MB contact angle (°)',
        'MB volume charge density δm (mol·m-3)',
        'Pressure (kPa)', 'Measurement time (min)', 'Initial concentration of compound (mg/L)', 'pH'
    ]
    target_column = "removal rate (%)"

    dds_rf = DynamicDataStreamRandomForest(
        feature_columns=feature_columns,
        target_column=target_column,
        window_size=100,
        retrain_interval=30,
        drift_detection_threshold=0.8,
        initial_train_size=40
    )

    dds_rf.process_data_stream('../../PFASdata.xlsx', stream_size=150)
    dds_rf.print_final_metrics()

    metrics_df = pd.DataFrame(dds_rf.metrics_history)
    metrics_df.to_csv('dynamic_forest_metrics_history.csv', index=False)
    print("\n评估历史已保存到 'dynamic_forest_metrics_history.csv'")