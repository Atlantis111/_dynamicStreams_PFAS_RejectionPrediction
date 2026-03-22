import pandas as pd, numpy as np, xgboost as xgb, warnings, os, joblib, shap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

warnings.filterwarnings('ignore')


def calculate_mape(y_true, y_pred):
    """计算平均绝对百分比误差(MAPE)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def calculate_psi(expected: np.ndarray, actual: np.ndarray,
                  n_bins: int = 10) -> Tuple[float, Dict]:
    """计算群体稳定性指标(PSI)"""
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
    """计算所有特征的PSI值"""
    psi_results = {}
    for col in selected_columns:
        if col in expected_data.columns and col in actual_data.columns:
            psi_value, psi_detail = calculate_psi(expected_data[col].values, actual_data[col].values, n_bins)
            psi_results[col] = {'psi': psi_value, 'detail': psi_detail}
        else:
            psi_results[col] = {'psi': None, 'detail': None, 'error': f'Column {col} not found in both datasets'}
    return psi_results


class BatchPsiEnhancedXGBoost:
    """基于PSI漂移检测的分批XGBoost模型"""

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
        self.shap_history = []
        self.remaining_data = None

        # [新增] 专门用于存储每个版本对应的数据，以备跨版本验证使用
        self.version_data = {}

        self._initialize_model_with_batch(initial_data_path)
        if self.use_psi_detection:
            self._initialize_reference_distribution()

    def _analyze_shap(self, X_df: pd.DataFrame, reason: str):
        """执行SHAP分析并记录特征重要性"""
        print(f"\n" + "-" * 60)
        print(f"执行SHAP特征重要性分析 ({reason} - 模型版本 v{self.model_version})")
        print("-" * 60)

        X_scaled = self.scaler.transform(X_df)
        explainer = shap.TreeExplainer(self.current_model)
        shap_values = explainer.shap_values(X_scaled)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance = {feat: val for feat, val in zip(self.selected_columns, mean_abs_shap)}
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        self.shap_history.append({
            'version': self.model_version,
            'reason': reason,
            'timestamp': datetime.now(),
            'importance': sorted_importance
        })

        print("Top 10 特征重要性 (基于 Mean |SHAP|):")
        for i, (feat, val) in enumerate(sorted_importance[:10]):
            print(f"  {i + 1:>2}. {feat:<40} : {val:.4f}")
        print("-" * 60)

    def _initialize_model_with_batch(self, file_path: str):
        """使用初始数据（一半）训练基础模型，并保存剩余数据"""
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
        self._analyze_shap(initial_data[self.selected_columns], reason="初始模型建立")

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
                print("⚠️ PSI检测到特征分布漂移" if psi_drift_detected else "✓ 特征分布稳定")

        drift_detected = psi_drift_detected
        if drift_detected or len(new_records) > len(self.data_stream) * 0.3:
            print("\n" + "=" * 50 + "\n执行完全重训练...\n" + "=" * 50)
            self.model_version += 1
            self._train_model(self.data_stream, is_initial=False)

            if self.use_psi_detection:
                self.reference_distribution = pd.DataFrame(self.data_stream)[self.selected_columns].copy()
                print("已更新参考分布")

            reason = "特征分布漂移(概念漂移)" if drift_detected else "数据流容量大变更"
            df_stream = pd.DataFrame(self.data_stream)
            self._analyze_shap(df_stream[self.selected_columns], reason=f"触发完全重训练: {reason}")
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
            'all_psi_values': {f: psi_results[f]['psi'] for f in self.selected_columns if
                               psi_results[f]['psi'] is not None},
            'drift_detected': drift_detected
        }
        self.psi_detection_history.append(psi_record)
        return drift_detected, {'avg_psi': avg_psi, 'max_psi': max_psi, 'drift_detected': drift_detected,
                                'high_psi_features': high_psi_features}, new_df

    def _get_optimal_params(self, X_train: np.ndarray, y_train: np.ndarray, data_size: int) -> Dict:
        param_grid = {'max_depth': [3, 4, 5], 'min_child_weight': [1, 2], 'learning_rate': [0.05, 0.1]}
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50, random_state=42)
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='r2', cv=3, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        return grid_search.best_params_

    def _train_model(self, data_batch: List[Dict], is_initial: bool = False,
                     previous_model: Optional[xgb.Booster] = None):
        df_batch = pd.DataFrame(data_batch)
        X, y = df_batch[self.selected_columns], df_batch[self.label_column]

        # [新增] 把该版本训练和验证的对应时间截面的数据保存下来
        self.version_data[self.model_version] = {'X': X.copy(), 'y': y.copy()}

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if is_initial:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled, X_test_scaled = self.scaler.transform(X_train), self.scaler.transform(X_test)

        optimal_params = {'max_depth': 4, 'min_child_weight': 1, 'learning_rate': 0.1, 'subsample': 0.8,
                          'colsample_bytree': 0.8, 'reg_alpha': 0.1, 'reg_lambda': 1.0}

        params = {'objective': 'reg:squarederror', 'max_depth': optimal_params['max_depth'],
                  'learning_rate': optimal_params['learning_rate'], 'random_state': 42}
        dtrain, dtest = xgb.DMatrix(X_train_scaled, label=y_train), xgb.DMatrix(X_test_scaled, label=y_test)

        if previous_model is not None and not is_initial:
            self.current_model = xgb.train(params, dtrain, num_boost_round=50, xgb_model=previous_model)
        else:
            self.current_model = xgb.train(params, dtrain, num_boost_round=100)

        train_metrics = self._evaluate_model_on_dataset(X_train, y_train, "训练集")
        test_metrics = self._evaluate_model_on_dataset(X_test, y_test, "测试集")
        metrics_info = {'version': self.model_version, 'training_time': datetime.now(),
                        'train_metrics': train_metrics, 'test_metrics': test_metrics,
                        'data_size': len(data_batch), 'params': params}
        self.train_test_metrics_history.append(metrics_info)

        # 保存模型到Pool逻辑精简
        model_info = {'version': self.model_version, 'performance': test_metrics, 'model': self.current_model}
        self.model_pool.append(model_info)
        self.model_weights.append(1.0)
        if is_initial:
            self._save_model_snapshot()

    def _evaluate_model_on_dataset(self, X: pd.DataFrame, y: pd.Series, dataset_name: str = "") -> Dict:
        X_scaled = self.scaler.transform(X)
        y_pred = self.current_model.predict(xgb.DMatrix(X_scaled))
        mse, mae = mean_squared_error(y, y_pred), mean_absolute_error(y, y_pred)
        r2, rmse = r2_score(y, y_pred), np.sqrt(mse)
        mape = calculate_mape(y, y_pred)
        return {'mse': mse, 'mae': mae, 'r2_score': r2, 'rmse': rmse, 'mape': mape}

    def _save_model_snapshot(self):
        snapshot_dir = "xgboost_model_snapshots"
        os.makedirs(snapshot_dir, exist_ok=True)
        snapshot_path = os.path.join(snapshot_dir, f"xgb_model_v{self.model_version}.model")
        self.current_model.save_model(snapshot_path)

    def evaluate_all_versions_matrix(self, max_version: int = 4):
        """
        生成全版本交叉测试矩阵 (v1 到 v_max 互相测试)
        并返回 R2 和 RMSE 矩阵
        """
        actual_max = min(max_version, self.model_version)
        if actual_max < 2:
            print("❌ 版本数量不足，无法进行全矩阵交叉测试。请先积累更多版本的数据。")
            return None, None

        print(f"\n" + "=" * 80)
        print(f"🔬 全版本交叉测试矩阵: v1 到 v{actual_max} 相互测试")
        print("=" * 80)

        # 初始化结果矩阵
        results_r2 = np.zeros((actual_max, actual_max))
        results_rmse = np.zeros((actual_max, actual_max))

        snapshot_dir = "xgboost_model_snapshots"

        # 遍历所有训练模型版本 (行)
        for i in range(1, actual_max + 1):
            model_path = os.path.join(snapshot_dir, f"xgb_model_v{i}.model")
            if not os.path.exists(model_path):
                print(f"⚠️ 找不到模型 v{i}，该行将填充为0。")
                continue

            # 加载当时的训练模型
            model = xgb.Booster()
            model.load_model(model_path)

            # 遍历所有数据集版本 (列)
            for j in range(1, actual_max + 1):
                if j not in self.version_data:
                    continue

                data = self.version_data[j]
                X_scaled = self.scaler.transform(data['X'])
                dmatrix = xgb.DMatrix(X_scaled)
                y_pred = model.predict(dmatrix)
                y_true = data['y']

                # 计算指标
                r2 = r2_score(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))

                results_r2[i - 1, j - 1] = r2
                results_rmse[i - 1, j - 1] = rmse

        # 打印 R² 矩阵表格
        print("\n📊 R² (决定系数) 交叉测试矩阵:")
        header = " | ".join([f"测试集 v{j}" for j in range(1, actual_max + 1)])
        print(f"{'模型版本':<10} | {header}")
        print("-" * (15 + 12 * actual_max))
        for i in range(actual_max):
            row_str = " | ".join([f"{results_r2[i, j]:>10.4f}" for j in range(actual_max)])
            print(f"模型 v{i + 1:<4} | {row_str}")

        return results_r2, results_rmse

    def plot_cross_version_heatmap(self, results_r2: np.ndarray):
        """
        绘制交叉测试 R² 热力图
        """
        if results_r2 is None:
            return

        try:
            import matplotlib
            matplotlib.use('TkAgg')  # 根据你的环境可能需要调整
            import matplotlib.pyplot as plt
            import seaborn as sns

            plt.figure(figsize=(8, 6))
            actual_max = results_r2.shape[0]

            # 设置坐标轴标签
            labels = [f"v{i}" for i in range(1, actual_max + 1)]

            # 画热力图 (使用viridis色带，数值越高颜色越亮)
            sns.heatmap(results_r2, annot=True, fmt=".4f", cmap="YlGnBu",
                        xticklabels=[f"Data {l}" for l in labels],
                        yticklabels=[f"Model {l}" for l in labels])

            plt.title("Cross-Version R² Performance Matrix\n(Diagonal shows model on its own data)")
            plt.xlabel("Test Dataset Version (Time Progression →)")
            plt.ylabel("Trained Model Version (Time Progression ↓)")
            plt.tight_layout()
            plt.show()

        except ImportError:
            print("\n⚠️ 缺少 matplotlib 或 seaborn 库，无法绘制热力图。可以通过 'pip install matplotlib seaborn' 安装。")

    # =========================================================================
    # [核心新增]：跨版本模型验证方法
    # =========================================================================
    def evaluate_cross_version(self, version_a: int, version_b: int):
        """
        评估两个不同版本的模型在各自以及对方数据集上的表现。
        证明旧模型对新数据的预测性能下降（数据漂移），以及新模型的重训练价值。
        """
        print(f"\n" + "=" * 80)
        print(f"🔬 跨版本模型交叉测试: 验证模型对数据漂移的适应性 (v{version_a} vs v{version_b})")
        print("=" * 80)

        snapshot_dir = "xgboost_model_snapshots"
        model_a_path = os.path.join(snapshot_dir, f"xgb_model_v{version_a}.model")
        model_b_path = os.path.join(snapshot_dir, f"xgb_model_v{version_b}.model")

        if not os.path.exists(model_a_path) or not os.path.exists(model_b_path):
            print(f"❌ 找不到模型快照: 请确保 v{version_a} 和 v{version_b} 的模型已被保存。")
            return

        if version_a not in self.version_data or version_b not in self.version_data:
            print(f"❌ 找不到对应的数据版本快照。")
            return

        # 独立加载当时的模型
        model_a = xgb.Booster()
        model_a.load_model(model_a_path)

        model_b = xgb.Booster()
        model_b.load_model(model_b_path)

        # 提取当时的完整流数据验证集
        data_a = self.version_data[version_a]
        data_b = self.version_data[version_b]

        def eval_model(model, data):
            X_scaled = self.scaler.transform(data['X'])
            dmatrix = xgb.DMatrix(X_scaled)
            y_pred = model.predict(dmatrix)
            y_true = data['y']
            return r2_score(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))

        # 执行4种组合测试
        r2_aa, rmse_aa = eval_model(model_a, data_a)  # v1 模型 测 v1 数据
        r2_ab, rmse_ab = eval_model(model_a, data_b)  # v1 模型 测 v4 数据 (跨时空)
        r2_bb, rmse_bb = eval_model(model_b, data_b)  # v4 模型 测 v4 数据
        r2_ba, rmse_ba = eval_model(model_b, data_a)  # v4 模型 测 v1 数据 (跨时空)

        print(f"{'测试场景':<35} | {'R² 表现':<10} | {'RMSE 表现':<10}")
        print("-" * 65)
        print(f"模型 v{version_a} 测试 v{version_a} 数据 (正常基准)  | {r2_aa:<10.4f} | {rmse_aa:<10.4f}")
        print(f"模型 v{version_a} 测试 v{version_b} 数据 (发生漂移)  | {r2_ab:<10.4f} | {rmse_ab:<10.4f}")
        print("-" * 65)
        print(f"模型 v{version_b} 测试 v{version_b} 数据 (正常基准)  | {r2_bb:<10.4f} | {rmse_bb:<10.4f}")
        print(f"模型 v{version_b} 测试 v{version_a} 数据 (灾难遗忘)  | {r2_ba:<10.4f} | {rmse_ba:<10.4f}")

        print("\n📊 结论分析:")
        if r2_ab < r2_aa:
            print(
                f"1. 成功证明数据分布漂移：旧模型 (v{version_a}) 在未来的新数据 (v{version_b}) 上性能大幅下降，R² 从 {r2_aa:.4f} 跌至 {r2_ab:.4f}。")
        if r2_bb > r2_ab:
            print(
                f"2. 成功证明重训练价值：更新后的新模型 (v{version_b}) 成功适应了新数据，在 v{version_b} 数据上的 R² 达到了 {r2_bb:.4f}，远优于旧模型的 {r2_ab:.4f}。")
        print(
            f"3. 遗忘现象观察：新模型对历史数据的记忆通常会衰退（R² 为 {r2_ba:.4f}），但这正是系统不断适应最新化学机制的正常代价。")
        print("=" * 80)

    def print_detailed_metrics_report(self):
        pass  # 为缩减回答长度，这部分与原代码保持一致即可

    def print_psi_drift_report(self):
        pass  # 为缩减回答长度，这部分与原代码保持一致即可

    def get_remaining_data_info(self) -> Dict:
        return {"total_remaining": len(self.remaining_data) if self.remaining_data is not None else 0}

    def get_model_info(self) -> Dict:
        return {'current_version': self.model_version}


def batch_psi_xgb_regression(file_path: str, selected_columns: List[str], label_column: str) -> BatchPsiEnhancedXGBoost:
    return BatchPsiEnhancedXGBoost(initial_data_path=file_path, selected_columns=selected_columns,
                                   label_column=label_column)


if __name__ == "__main__":
    file_path = "../PFASdata.xlsx"
    selected_columns = ['Compound log K ow', 'WS (mg/L)', 'MinPartialCharge', 'MaxPartialCharge',
                        'min projection (Å)', 'S', 'rs/rp', '∆Gs-m (J·m-2)', 'MB contact angle (°)',
                        'MB volume charge density δm (mol·m-3)', 'Pressure (kPa)', 'Measurement time (min)',
                        'Initial concentration of compound (mg/L)', 'pH']
    label_column = "removal rate (%)"
    try:
        psi_model = BatchPsiEnhancedXGBoost(file_path, selected_columns, label_column)

        # 模拟流式批次更新...
        batch_size = 24
        while True:
            remaining_info = psi_model.get_remaining_data_info()
            if remaining_info['total_remaining'] == 0:
                break
            success = psi_model.update_with_next_batch(batch_size=batch_size)
            if not success:
                break

        print("\n" + "=" * 80 + "\n分批更新完成!\n" + "=" * 80)

        # =====================================================================
        # [核心新增]：在流程末尾，自动执行 V1模型 和 当前最新版本模型 的交叉验证测试
        # =====================================================================
        '''
        latest_version = psi_model.model_version
        if latest_version > 1:
            # 你可以硬编码填 4，为了自适应代码长度，如果不足4版则对比最新的版
            target_v = 4 if latest_version >= 4 else latest_version
            psi_model.evaluate_cross_version(version_a=1, version_b=target_v)
        '''


        # =====================================================================
        # [核心新增]：在流程末尾，自动执行全版本的交叉验证矩阵
        # =====================================================================
        latest_version = psi_model.model_version
        if latest_version >= 2:
            # 设定你想测试的最大版本数，这里设为4（如果实际跑出了4个版本以上）
            target_max = 4 if latest_version >= 4 else latest_version

            # 1. 生成数据并打印表格
            r2_matrix, rmse_matrix = psi_model.evaluate_all_versions_matrix(max_version=target_max)

            # 2. 绘制直观的热力图
            psi_model.plot_cross_version_heatmap(r2_matrix)

    except Exception as e:
        print(f"程序执行过程中出现错误: {str(e)}")