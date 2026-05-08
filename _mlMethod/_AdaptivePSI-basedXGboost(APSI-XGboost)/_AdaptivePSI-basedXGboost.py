import pandas as pd, numpy as np, xgboost as xgb, warnings, os, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

warnings.filterwarnings('ignore')

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> Tuple[float, Dict]:
    expected, actual = expected[~np.isnan(expected)], actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return 0.0, {}
    combined = np.concatenate([expected, actual])
    bins = np.percentile(combined, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0, {}
    expected_hist, _ = np.histogram(expected, bins=bins)
    actual_hist, _ = np.histogram(actual, bins=bins)
    expected_hist, actual_hist = expected_hist.astype(float) + 0.001, actual_hist.astype(float) + 0.001
    expected_perc, actual_perc = expected_hist / np.sum(expected_hist), actual_hist / np.sum(actual_hist)
    psi_bins = [(actual_perc[i] - expected_perc[i]) * np.log(actual_perc[i] / expected_perc[i])
                if expected_perc[i] > 0 and actual_perc[i] > 0 else 0
                for i in range(len(expected_perc))]
    psi_total = np.sum(psi_bins)
    return psi_total, {
        'total_psi': psi_total, 'bins': bins, 'expected_percentage': expected_perc.tolist(),
        'actual_percentage': actual_perc.tolist(), 'psi_by_bin': psi_bins,
        'expected_samples': len(expected), 'actual_samples': len(actual)
    }

def calculate_feature_psi(expected_data: pd.DataFrame, actual_data: pd.DataFrame,
                          selected_columns: List[str], n_bins: int = 10) -> Dict:
    psi_results = {}
    for col in selected_columns:
        if col in expected_data.columns and col in actual_data.columns:
            psi_value, psi_detail = calculate_psi(expected_data[col].values, actual_data[col].values, n_bins)
            psi_results[col] = {'psi': psi_value, 'detail': psi_detail}
        else:
            psi_results[col] = {'psi': None, 'detail': None, 'error': f'Column {col} not found in both datasets'}
    return psi_results

class BatchPsiEnhancedXGBoost:
    def __init__(self, initial_data_path: str, selected_columns: List[str], label_column: str,
                 window_size: int = 100, max_models: int = 5, use_grid_search: bool = True,
                 use_psi_detection: bool = True, psi_threshold: float = 0.1, n_psi_bins: int = 10,
                 initial_ratio: float = 0.5):
        self.selected_columns, self.label_column = selected_columns, label_column
        self.window_size, self.max_models = window_size, max_models
        self.use_grid_search, self.use_psi_detection = use_grid_search, use_psi_detection
        self.psi_threshold, self.n_psi_bins, self.initial_ratio = psi_threshold, n_psi_bins, initial_ratio
        self.scaler, self.current_model = StandardScaler(), None
        self.model_pool, self.model_weights, self.best_params = [], [], None
        self.data_stream, self.reference_distribution = [], None
        self.model_version, self.performance_history = 1, []
        self.psi_detection_history, self.train_test_metrics_history = [], []
        self.remaining_data = None
        self._initialize_model_with_batch(initial_data_path)
        if self.use_psi_detection:
            self._initialize_reference_distribution()

    def _initialize_model_with_batch(self, file_path: str):
        print(f"正在初始化XGBoost基础模型...")
        df = pd.read_excel(file_path, engine='openpyxl')
        df_cleaned = df[self.selected_columns + [self.label_column]].dropna()
        if len(df_cleaned) == 0:
            raise ValueError("初始数据清理后无有效数据")
        initial_size = int(len(df_cleaned) * self.initial_ratio)
        initial_data = df_cleaned.iloc[:initial_size].copy()
        self.remaining_data = df_cleaned.iloc[initial_size:].copy()
        self.data_stream = initial_data.to_dict('records')
        self._train_model(self.data_stream, is_initial=True)

    def get_remaining_data_info(self) -> Dict:
        return {"total_remaining": len(self.remaining_data) if self.remaining_data is not None else 0,
                "remaining_data": self.remaining_data}

    def update_with_next_batch(self, batch_size: int = None) -> bool:
        if self.remaining_data is None or len(self.remaining_data) == 0:
            print("没有剩余数据可用于更新")
            return False
        if batch_size is None:
            batch_data = self.remaining_data
            self.remaining_data = pd.DataFrame()
        else:
            batch_size = min(batch_size, len(self.remaining_data))
            batch_data = self.remaining_data.iloc[:batch_size]
            self.remaining_data = self.remaining_data.iloc[batch_size:].reset_index(drop=True)
        print(f"\n正在处理第 {self.model_version} 批数据...")
        return self._update_with_data_batch(batch_data)

    def update_with_custom_batch(self, batch_data: pd.DataFrame) -> bool:
        print(f"\n正在处理自定义批次数据...")
        return self._update_with_data_batch(batch_data)

    def _update_with_data_batch(self, batch_data: pd.DataFrame) -> bool:
        new_cleaned = batch_data[self.selected_columns + [self.label_column]].dropna()
        if len(new_cleaned) == 0:
            print("批次数据无有效记录，跳过更新")
            return False
        new_records = new_cleaned.to_dict('records')
        self.data_stream.extend(new_records)
        if len(self.data_stream) > self.window_size:
            self.data_stream = self.data_stream[-self.window_size:]

        psi_drift_detected, psi_summary = False, {}
        if self.use_psi_detection:
            print("\n" + "=" * 50 + "\n执行PSI特征分布漂移检测\n" + "=" * 50)
            psi_drift_detected, psi_summary, _ = self._detect_psi_drift(new_records)
            if psi_summary:
                print(f"平均PSI: {psi_summary['avg_psi']:.4f}")
                print(f"最大PSI: {psi_summary['max_psi']:.4f}")
                print(f"PSI阈值: {self.psi_threshold}")
                if psi_summary['high_psi_features']:
                    print(f"高PSI特征 ({len(psi_summary['high_psi_features'])} 个):")
                    for feature_info in psi_summary['high_psi_features'][:5]:
                        print(f"  {feature_info['feature']}: PSI={feature_info['psi']:.4f}")
                print("⚠️ PSI检测到特征分布漂移" if psi_drift_detected else "✓ 特征分布稳定")

        drift_detected = psi_drift_detected
        if drift_detected or len(new_records) > len(self.data_stream) * 0.3:
            print("\n" + "=" * 50 + "\n执行完全重训练...\n" + "=" * 50)
            self.model_version += 1
            self._train_model(self.data_stream, is_initial=False)
            if self.use_psi_detection:
                self.reference_distribution = pd.DataFrame(self.data_stream)[self.selected_columns].copy()
                print("已更新参考分布")
        else:
            print("\n" + "=" * 50 + "\n执行增量学习...\n" + "=" * 50)
            self.model_version += 1
            self._train_model(new_records, is_initial=False, previous_model=self.current_model)
        self._save_model_snapshot()
        return True

    def _initialize_reference_distribution(self):
        if len(self.data_stream) > 0:
            df_ref = pd.DataFrame(self.data_stream)
            self.reference_distribution = df_ref[self.selected_columns].copy()
            print(f"已初始化参考分布，包含 {len(self.reference_distribution)} 条记录")

    def _calculate_feature_psi(self, new_data: pd.DataFrame) -> Dict:
        if self.reference_distribution is None or len(self.reference_distribution) == 0:
            return {}
        return calculate_feature_psi(self.reference_distribution, new_data, self.selected_columns, self.n_psi_bins)

    def _detect_psi_drift(self, new_records: List[Dict]) -> Tuple[bool, Dict, pd.DataFrame]:
        if not self.use_psi_detection or self.reference_distribution is None:
            return False, {}, pd.DataFrame()
        new_df = pd.DataFrame(new_records)
        if len(new_df) < 10:
            return False, {}, new_df
        psi_results = self._calculate_feature_psi(new_df)
        if not psi_results:
            return False, {}, new_df

        psi_values, high_psi_features = [], []
        for feature, result in psi_results.items():
            if result['psi'] is not None:
                psi_val = result['psi']
                psi_values.append(psi_val)
                if psi_val > self.psi_threshold:
                    high_psi_features.append({'feature': feature, 'psi': psi_val, 'threshold': self.psi_threshold})
        if not psi_values:
            return False, {}, new_df

        avg_psi, max_psi = np.mean(psi_values), np.max(psi_values)
        drift_detected = (avg_psi > self.psi_threshold) or (max_psi > self.psi_threshold * 2)
        psi_record = {
            'timestamp': datetime.now(), 'avg_psi': avg_psi, 'max_psi': max_psi,
            'psi_threshold': self.psi_threshold, 'high_psi_features': high_psi_features,
            'all_psi_values': {f: psi_results[f]['psi'] for f in self.selected_columns
                               if psi_results[f]['psi'] is not None},
            'drift_detected': drift_detected
        }
        self.psi_detection_history.append(psi_record)
        psi_summary = {'avg_psi': avg_psi, 'max_psi': max_psi, 'drift_detected': drift_detected,
                       'high_psi_features': high_psi_features, 'psi_results': psi_results}
        return drift_detected, psi_summary, new_df

    def _get_optimal_params(self, X_train: np.ndarray, y_train: np.ndarray, data_size: int) -> Dict:
        if data_size < 100:
            param_grid = {'max_depth': [3, 4, 5], 'min_child_weight': [1, 2, 3],
                          'learning_rate': [0.05, 0.1], 'subsample': [0.7, 0.8],
                          'colsample_bytree': [0.7, 0.8], 'reg_alpha': [0, 0.1], 'reg_lambda': [1, 2]}
        elif data_size < 500:
            param_grid = {'max_depth': [3, 4, 5, 6], 'min_child_weight': [1, 2, 3, 4],
                          'learning_rate': [0.01, 0.05, 0.1], 'subsample': [0.6, 0.7, 0.8, 0.9],
                          'colsample_bytree': [0.6, 0.7, 0.8, 0.9], 'reg_alpha': [0, 0.1, 0.5],
                          'reg_lambda': [0.5, 1, 2]}
        else:
            param_grid = {'max_depth': [3, 4, 5, 6, 7], 'min_child_weight': [1, 2, 3, 4, 5],
                          'learning_rate': [0.01, 0.05, 0.1, 0.15], 'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                          'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0], 'reg_alpha': [0, 0.1, 0.5, 1],
                          'reg_lambda': [0.5, 1, 1.5, 2]}

        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        cv_folds = min(5, max(3, data_size // 100))
        print(f"开始网格搜索，数据量: {data_size}，参数组合数: {len(param_grid)}")
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='r2',
                                   cv=cv_folds, verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        print(f"网格搜索完成，最佳参数: {grid_search.best_params_}")
        print(f"最佳交叉验证R²分数: {grid_search.best_score_:.4f}")
        return grid_search.best_params_

    def _train_model(self, data_batch: List[Dict], is_initial: bool = False,
                     previous_model: Optional[xgb.Booster] = None):
        df_batch = pd.DataFrame(data_batch)
        X, y = df_batch[self.selected_columns], df_batch[self.label_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if is_initial:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled, X_test_scaled = self.scaler.transform(X_train), self.scaler.transform(X_test)

        if self.use_grid_search and (is_initial or len(data_batch) > 50):
            optimal_params = self._get_optimal_params(X_train_scaled, y_train, len(data_batch))
            self.best_params = optimal_params
        else:
            if self.best_params is None:
                self.best_params = {'max_depth': 6, 'min_child_weight': 1, 'learning_rate': 0.1,
                                    'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0}
            optimal_params = self.best_params

        params = {'objective': 'reg:squarederror', 'max_depth': optimal_params['max_depth'],
                  'learning_rate': optimal_params['learning_rate'], 'subsample': optimal_params['subsample'],
                  'colsample_bytree': optimal_params['colsample_bytree'], 'reg_alpha': optimal_params['reg_alpha'],
                  'reg_lambda': optimal_params['reg_lambda'], 'random_state': 42}
        dtrain, dtest = xgb.DMatrix(self.scaler.transform(X_train), label=y_train), xgb.DMatrix(self.scaler.transform(X_test), label=y_test)

        if previous_model is not None and not is_initial:
            print("使用增量学习模式...")
            self.current_model = xgb.train(params, dtrain, num_boost_round=50, xgb_model=previous_model)
        else:
            num_round = max(50, min(200, len(data_batch) // 10))
            print(f"进行全新模型训练，迭代次数: {num_round}")
            self.current_model = xgb.train(params, dtrain, num_boost_round=num_round)

        train_metrics = self._evaluate_model_on_dataset(X_train, y_train, "训练集")
        test_metrics = self._evaluate_model_on_dataset(X_test, y_test, "测试集")
        metrics_info = {'version': self.model_version, 'training_time': datetime.now(),
                        'train_metrics': train_metrics, 'test_metrics': test_metrics,
                        'data_size': len(data_batch), 'params': params}
        self.train_test_metrics_history.append(metrics_info)
        model_info = {'version': self.model_version, 'training_time': datetime.now(),
                      'data_size': len(data_batch), 'performance': test_metrics,
                      'model': self.current_model, 'model_type': 'incremental' if previous_model else 'full',
                      'params': params}
        self.performance_history.append(model_info)
        self._manage_model_pool(model_info)
        print(f"模型版本 {self.model_version} 训练完成")
        print(f"使用参数: max_depth={params['max_depth']}, lr={params['learning_rate']}")
        print(f"训练集性能 - R²: {train_metrics['r2_score']:.4f}, RMSE: {train_metrics['rmse']:.4f}")
        print(f"测试集性能 - R²: {test_metrics['r2_score']:.4f}, RMSE: {test_metrics['rmse']:.4f}")
        overfitting_gap = train_metrics['r2_score'] - test_metrics['r2_score']
        print(f"过拟合程度(R²差距): {overfitting_gap:.4f}")

    def _evaluate_model_on_dataset(self, X: pd.DataFrame, y: pd.Series, dataset_name: str = "") -> Dict:
        if self.current_model is None:
            raise ValueError("模型未初始化")
        X_scaled = self.scaler.transform(X)
        y_pred = self.current_model.predict(xgb.DMatrix(X_scaled))
        mse, mae = mean_squared_error(y, y_pred), mean_absolute_error(y, y_pred)
        r2, rmse = r2_score(y, y_pred), np.sqrt(mse)
        mape = calculate_mape(y, y_pred)
        print(f"{dataset_name}评估指标:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        return {'mse': mse, 'mae': mae, 'r2_score': r2, 'rmse': rmse, 'mape': mape}

    def _evaluate_model(self, data_batch: List[Dict]) -> Dict:
        df_batch = pd.DataFrame(data_batch)
        return self._evaluate_model_on_dataset(df_batch[self.selected_columns], df_batch[self.label_column], "批量数据")

    def _manage_model_pool(self, new_model_info: Dict):
        self.model_pool.append(new_model_info)
        self.model_weights.append(1.0)
        if len(self.model_pool) > self.max_models:
            self.model_pool.pop(0)
            self.model_weights.pop(0)
        self._update_model_weights()

    def _update_model_weights(self):
        if len(self.model_pool) <= 1:
            return
        recent_performance = []
        for i, model_info in enumerate(self.model_pool):
            recency_weight = (i + 1) / len(self.model_pool)
            performance_weight = model_info['performance']['r2_score']
            total_weight = 0.7 * performance_weight + 0.3 * recency_weight
            recent_performance.append(total_weight)
        total = sum(recent_performance)
        self.model_weights = [w / total for w in recent_performance]

    def predict_ensemble(self, features: List[float]) -> float:
        if not self.model_pool:
            raise ValueError("模型池为空")
        X_scaled = self.scaler.transform([features])
        dtest = xgb.DMatrix(X_scaled)
        predictions = [model_info['model'].predict(dtest)[0] for model_info in self.model_pool]
        return sum(p * w for p, w in zip(predictions, self.model_weights))

    def predict(self, features: List[float]) -> float:
        if self.current_model is None:
            raise ValueError("模型未初始化")
        X_scaled = self.scaler.transform([features])
        return self.current_model.predict(xgb.DMatrix(X_scaled))[0]

    def _save_model_snapshot(self):
        snapshot_dir = "xgboost_model_snapshots"
        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_path = os.path.join(snapshot_dir, f"xgb_model_v{self.model_version}.model")
        self.current_model.save_model(snapshot_path)
        meta_path = os.path.join(snapshot_dir, f"metadata_v{self.model_version}.pkl")
        joblib.dump({'performance_history': self.performance_history,
                     'psi_detection_history': self.psi_detection_history,
                     'train_test_metrics_history': self.train_test_metrics_history,
                     'model_weights': self.model_weights,
                     'data_stream_size': len(self.data_stream),
                     'best_params': self.best_params}, meta_path)
        print(f"模型快照已保存: {snapshot_path}")

    def print_detailed_metrics_report(self):
        if not self.train_test_metrics_history:
            print("暂无评估指标记录")
            return
        print("\n" + "=" * 80 + "\n训练集和测试集评估指标详细报告\n" + "=" * 80)
        for metrics_info in self.train_test_metrics_history[-3:]:
            print(f"\n模型版本 {metrics_info['version']} ({metrics_info['training_time'].strftime('%Y-%m-%d %H:%M')}):")
            print(f"数据量: {metrics_info['data_size']} 条记录")
            train, test = metrics_info['train_metrics'], metrics_info['test_metrics']
            print("训练集指标:")
            print(f"  MSE: {train['mse']:.4f} | MAE: {train['mae']:.4f} | R²: {train['r2_score']:.4f} | RMSE: {train['rmse']:.4f} | MAPE: {train['mape']:.2f}%")
            print("测试集指标:")
            print(f"  MSE: {test['mse']:.4f} | MAE: {test['mae']:.4f} | R²: {test['r2_score']:.4f} | RMSE: {test['rmse']:.4f} | MAPE: {test['mape']:.2f}%")
            r2_gap, rmse_gap = train['r2_score'] - test['r2_score'], test['rmse'] - train['rmse']
            print(f"过拟合分析: R²差距: {r2_gap:.4f}, RMSE差距: {rmse_gap:.4f}")
            print("-" * 60)

    def print_psi_drift_report(self):
        print("\n" + "=" * 100 + "\nPSI特征漂移检测报告\n" + "=" * 100)
        if self.psi_detection_history:
            print("\nPSI特征漂移检测历史:")
            print("-" * 40)
            for i, psi_info in enumerate(self.psi_detection_history[-5:]):
                print(f"检测 {i + 1} ({psi_info['timestamp'].strftime('%Y-%m-%d %H:%M')}):")
                print(f"  平均PSI: {psi_info['avg_psi']:.4f}")
                print(f"  最大PSI: {psi_info['max_psi']:.4f}")
                print(f"  PSI阈值: {psi_info['psi_threshold']:.4f}")
                print(f"  漂移检测: {'是' if psi_info['drift_detected'] else '否'}")
                if psi_info['high_psi_features']:
                    print(f"  高PSI特征数: {len(psi_info['high_psi_features'])}")
                print("-" * 30)
        else:
            print("\nPSI漂移检测: 暂无记录")

    def visualize_psi_drift(self, last_n: int = 10):
        if not self.psi_detection_history or len(self.psi_detection_history) < 2:
            print("PSI检测历史不足，无法可视化")
            return
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        recent_history = self.psi_detection_history[-last_n:]
        timestamps = [h['timestamp'] for h in recent_history]
        avg_psi, max_psi = [h['avg_psi'] for h in recent_history], [h['max_psi'] for h in recent_history]
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(timestamps, avg_psi, 'b-o', label='平均PSI', linewidth=2)
        plt.plot(timestamps, max_psi, 'r-s', label='最大PSI', linewidth=2)
        plt.axhline(y=self.psi_threshold, color='g', linestyle='--', label=f'PSI阈值({self.psi_threshold})')
        plt.axhline(y=self.psi_threshold * 2, color='r', linestyle=':', label=f'警告阈值({self.psi_threshold * 2})')
        plt.xlabel('时间')
        plt.ylabel('PSI值')
        plt.title('PSI漂移趋势图')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.subplot(2, 2, 2)
        x, width = range(len(avg_psi)), 0.35
        plt.bar([i - width / 2 for i in x], avg_psi, width, label='平均PSI', alpha=0.7)
        plt.bar([i + width / 2 for i in x], max_psi, width, label='最大PSI', alpha=0.7)
        plt.axhline(y=self.psi_threshold, color='r', linestyle='--', linewidth=1)
        plt.xlabel('检测次数')
        plt.ylabel('PSI值')
        plt.title('PSI检测结果对比')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        plt.subplot(2, 2, 3)
        high_psi_counts = [len(h['high_psi_features']) for h in recent_history]
        plt.plot(timestamps, high_psi_counts, 'g-^', linewidth=2)
        plt.xlabel('时间')
        plt.ylabel('高PSI特征数量')
        plt.title('高PSI特征数量趋势')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.subplot(2, 2, 4)
        drift_detected = [1 if h['drift_detected'] else 0 for h in recent_history]
        plt.bar(range(len(drift_detected)), drift_detected, color=['red' if d else 'green' for d in drift_detected])
        plt.xlabel('检测次数')
        plt.ylabel('是否漂移')
        plt.title('漂移检测结果 (1=漂移, 0=正常)')
        plt.yticks([0, 1], ['正常', '漂移'])
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.show()
        print("\n最新PSI检测的详细特征分析:")
        latest_psi = self.psi_detection_history[-1]
        if 'all_psi_values' in latest_psi:
            psi_values = latest_psi['all_psi_values']
            sorted_features = sorted(psi_values.items(), key=lambda x: x[1] if x[1] is not None else 0, reverse=True)
            print(f"\n特征PSI值排名 (检测时间: {latest_psi['timestamp'].strftime('%Y-%m-%d %H:%M')}):")
            print("-" * 60)
            print(f"{'特征名':<30} {'PSI值':<10} {'状态':<10}")
            print("-" * 60)
            for feature, psi_val in sorted_features[:10]:
                if psi_val is not None:
                    status = "警告" if psi_val > self.psi_threshold else "正常"
                    print(f"{feature:<30} {psi_val:<10.4f} {status:<10}")
            if len(sorted_features) > 10:
                print(f"... 等{len(sorted_features)}个特征")

    def get_model_info(self) -> Dict:
        remaining_info = self.get_remaining_data_info()
        return {'current_version': self.model_version, 'data_stream_size': len(self.data_stream),
                'model_pool_size': len(self.model_pool), 'remaining_data_size': remaining_info['total_remaining'],
                'performance_history': self.performance_history, 'train_test_metrics_history': self.train_test_metrics_history,
                'psi_detections': len(self.psi_detection_history), 'use_psi_detection': self.use_psi_detection,
                'psi_threshold': self.psi_threshold, 'best_params': self.best_params}

def batch_psi_xgb_regression(file_path: str, selected_columns: List[str], label_column: str,
                             use_grid_search: bool = True, use_psi_detection: bool = True,
                             initial_ratio: float = 0.5) -> BatchPsiEnhancedXGBoost:
    psi_xgb = BatchPsiEnhancedXGBoost(initial_data_path=file_path, selected_columns=selected_columns,
                                      label_column=label_column, window_size=100, max_models=3,
                                      use_grid_search=use_grid_search, use_psi_detection=use_psi_detection,
                                      psi_threshold=0.1, n_psi_bins=10, initial_ratio=initial_ratio)
    model_info = psi_xgb.get_model_info()
    print("\n基于PSI漂移检测的分批XGBoost模型初始化完成:")
    print("=" * 60)
    print(f"模型版本: {model_info['current_version']}")
    print(f"数据流大小: {model_info['data_stream_size']} 条记录")
    print(f"剩余数据: {model_info['remaining_data_size']} 条记录")
    print(f"模型池大小: {model_info['model_pool_size']}")
    print(f"PSI检测: {'启用' if model_info['use_psi_detection'] else '禁用'}")
    if model_info['use_psi_detection']:
        print(f"PSI阈值: {model_info['psi_threshold']}")
    if model_info['best_params']:
        print(f"最优参数: {model_info['best_params']}")
    return psi_xgb

if __name__ == "__main__":
    file_path = "../../PFASdata.xlsx"
    selected_columns = ['Compound log K ow', 'WS (mg/L)', 'MinPartialCharge', 'MaxPartialCharge',
                        'min projection (Å)', 'S', 'rs/rp', '∆Gs-m (J·m-2)', 'MB contact angle (°)',
                        'MB volume charge density δm (mol·m-3)', 'Pressure (kPa)', 'Measurement time (min)',
                        'Initial concentration of compound (mg/L)', 'pH']
    label_column = "removal rate (%)"
    try:
        print("=" * 80 + "\n基于PSI漂移检测的分批XGBoost模型\n" + "=" * 80)
        psi_model = batch_psi_xgb_regression(file_path=file_path, selected_columns=selected_columns,
                                             label_column=label_column, use_grid_search=True,
                                             use_psi_detection=True, initial_ratio=0.5)
        psi_model.print_detailed_metrics_report()
        print("\n" + "=" * 80 + "\n开始分批更新剩余数据\n" + "=" * 80)
        batch_size, batch_count = 24, 0
        while True:
            remaining_info = psi_model.get_remaining_data_info()
            if remaining_info['total_remaining'] == 0:
                print("所有数据已处理完成!")
                break
            batch_count += 1
            print(f"\n>>> 正在处理第 {batch_count} 批数据 <<<")
            success = psi_model.update_with_next_batch(batch_size=batch_size)
            if not success:
                break
            psi_model.print_psi_drift_report()
            model_info = psi_model.get_model_info()
            print(f"\n当前模型版本: {model_info['current_version']}")
            print(f"剩余数据: {model_info['remaining_data_size']} 条记录")
        print("\n" + "=" * 80 + "\n分批更新完成!\n" + "=" * 80)
        final_info = psi_model.get_model_info()
        print(f"最终模型版本: {final_info['current_version']}")
        print(f"总训练次数: {len(final_info['train_test_metrics_history'])}")
        print(f"PSI检测次数: {final_info['psi_detections']}")
        if len(psi_model.psi_detection_history) > 0:
            psi_model.visualize_psi_drift()
    except Exception as e:
        print(f"程序执行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()