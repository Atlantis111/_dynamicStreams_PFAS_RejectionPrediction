import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
import os
from datetime import datetime
import joblib
from scipy import stats
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


def calculate_mape(y_true, y_pred):
    """计算平均绝对百分比误差(MAPE)"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


class DynamicXGBoost:
    """
    面向数据流的动态XGBoost模型
    集成增量学习、漂移检测和自适应更新机制
    """

    def __init__(self, initial_data_path, selected_columns, label_column,
                 window_size=100, drift_threshold=0.05, performance_threshold=0.02,
                 max_models=5, use_drift_detection=True, use_grid_search=True):
        """
        初始化动态XGBoost模型

        参数:
        initial_data_path: 初始数据文件路径
        selected_columns: 特征列名列表
        label_column: 目标变量列名
        window_size: 滑动窗口大小
        drift_threshold: 漂移检测阈值
        performance_threshold: 性能下降阈值
        max_models: 最大模型保存数量
        use_drift_detection: 是否启用漂移检测
        use_grid_search: 是否启用网格搜索优化参数
        """
        self.selected_columns = selected_columns
        self.label_column = label_column
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        self.max_models = max_models
        self.use_drift_detection = use_drift_detection
        self.use_grid_search = use_grid_search

        # 模型组件初始化
        self.scaler = StandardScaler()
        self.current_model = None
        self.model_pool = []  # 模型池，用于集成学习
        self.model_weights = []  # 模型权重
        self.best_params = None  # 存储最优参数组合

        # 数据流管理
        self.data_stream = []
        self.model_version = 1
        self.performance_history = []
        self.drift_detection_history = []
        self.train_test_metrics_history = []  # 新增：存储训练集和测试集评估指标

        # 用初始数据训练基础模型
        self._initialize_model(initial_data_path)

    def _initialize_model(self, file_path):
        """使用初始数据训练基础模型"""
        print("正在初始化XGBoost基础模型...")
        df = pd.read_excel(file_path, engine='openpyxl')
        df_cleaned = df[self.selected_columns + [self.label_column]].dropna()

        if len(df_cleaned) == 0:
            raise ValueError("初始数据清理后无有效数据")

        # 存储到数据流
        self.data_stream = df_cleaned.to_dict('records')
        print(f"初始数据加载完成，共 {len(self.data_stream)} 条记录")

        # 训练初始模型
        self._train_model(self.data_stream, is_initial=True)

    def _get_optimal_params(self, X_train, y_train, data_size):
        """
        使用网格搜索找到最优参数组合[1,2](@ref)
        根据数据量调整搜索策略以避免过拟合
        """
        # 根据数据量选择参数搜索范围
        if data_size < 100:  # 小数据量使用保守参数
            param_grid = {
                'max_depth': [3, 4, 5],
                'min_child_weight': [1, 2, 3],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.7, 0.8],
                'colsample_bytree': [0.7, 0.8],
                'reg_alpha': [0, 0.1],
                'reg_lambda': [1, 2]
            }
        elif data_size < 500:  # 中等数据量
            param_grid = {
                'max_depth': [3, 4, 5, 6],
                'min_child_weight': [1, 2, 3, 4],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.6, 0.7, 0.8, 0.9],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                'reg_alpha': [0, 0.1, 0.5],
                'reg_lambda': [0.5, 1, 2]
            }
        else:  # 大数据量使用更细粒度的搜索
            param_grid = {
                'max_depth': [3, 4, 5, 6, 7],
                'min_child_weight': [1, 2, 3, 4, 5],
                'learning_rate': [0.01, 0.05, 0.1, 0.15],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1],
                'reg_lambda': [0.5, 1, 1.5, 2]
            }

        # 创建基础模型[1,2](@ref)
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            random_state=42
        )

        # 根据数据量调整交叉验证折数[1](@ref)
        cv_folds = min(5, max(3, data_size // 100))

        print(f"开始网格搜索，数据量: {data_size}，参数组合数: {len(param_grid)}")

        # 执行网格搜索[1,2](@ref)
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring='r2',  # 优先优化R²分数
            cv=cv_folds,
            verbose=1,
            n_jobs=-1  # 使用所有可用的CPU核心
        )

        grid_search.fit(X_train, y_train)

        print(f"网格搜索完成，最佳参数: {grid_search.best_params_}")
        print(f"最佳交叉验证R²分数: {grid_search.best_score_:.4f}")

        return grid_search.best_params_

    def _train_model(self, data_batch, is_initial=False, previous_model=None):
        """训练XGBoost模型，支持增量学习和网格搜索优化"""
        df_batch = pd.DataFrame(data_batch)
        X = df_batch[self.selected_columns]
        y = df_batch[self.label_column]

        # 划分训练集和测试集[4](@ref)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # 数据标准化
        if is_initial:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_train_scaled = self.scaler.transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

        # 仅在初始训练或数据量足够时使用网格搜索[1,2](@ref)
        if self.use_grid_search and (is_initial or len(data_batch) > 50):
            # 使用网格搜索获取最优参数[1](@ref)
            optimal_params = self._get_optimal_params(X_train_scaled, y_train, len(data_batch))
            self.best_params = optimal_params
        else:
            # 使用之前的最优参数或默认参数[3](@ref)
            if self.best_params is None:
                self.best_params = {
                    'max_depth': 6,
                    'min_child_weight': 1,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0
                }
            optimal_params = self.best_params

        # 构建最终参数[1,3](@ref)
        params = {
            'objective': 'reg:squarederror',
            'max_depth': optimal_params['max_depth'],
            'learning_rate': optimal_params['learning_rate'],
            'subsample': optimal_params['subsample'],
            'colsample_bytree': optimal_params['colsample_bytree'],
            'reg_alpha': optimal_params['reg_alpha'],
            'reg_lambda': optimal_params['reg_lambda'],
            'random_state': 42
        }

        # 创建DMatrix[4](@ref)
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
        dtest = xgb.DMatrix(X_test_scaled, label=y_test)

        # 增量学习：如果提供了已有模型，在其基础上继续训练[5](@ref)
        if previous_model is not None and not is_initial:
            print("使用增量学习模式...")
            self.current_model = xgb.train(
                params,
                dtrain,
                num_boost_round=50,  # 新增的树数量
                xgb_model=previous_model  # 关键：在原有模型基础上继续训练
            )
        else:
            # 全新训练，根据数据量调整迭代次数[3](@ref)
            num_round = max(50, min(200, len(data_batch) // 10))
            print(f"进行全新模型训练，迭代次数: {num_round}")
            self.current_model = xgb.train(
                params,
                dtrain,
                num_boost_round=num_round
            )

        # 评估模型性能（训练集和测试集）[1,4](@ref)
        train_metrics = self._evaluate_model_on_dataset(X_train, y_train, "训练集")
        test_metrics = self._evaluate_model_on_dataset(X_test, y_test, "测试集")

        # 记录训练集和测试集评估指标
        metrics_info = {
            'version': self.model_version,
            'training_time': datetime.now(),
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'data_size': len(data_batch),
            'params': params  # 记录使用的参数
        }
        self.train_test_metrics_history.append(metrics_info)

        # 记录模型信息（使用测试集性能作为主要参考）
        performance = test_metrics
        training_time = datetime.now()
        model_info = {
            'version': self.model_version,
            'training_time': training_time,
            'data_size': len(data_batch),
            'performance': performance,
            'model': self.current_model,
            'model_type': 'incremental' if previous_model else 'full',
            'params': params
        }

        self.performance_history.append(model_info)

        # 管理模型池
        self._manage_model_pool(model_info)

        print(f"模型版本 {self.model_version} 训练完成")
        print(f"使用参数: max_depth={params['max_depth']}, lr={params['learning_rate']}")
        print(
            f"训练集性能 - R²: {train_metrics['r2_score']:.4f}, RMSE: {train_metrics['rmse']:.4f}")
        print(
            f"测试集性能 - R²: {test_metrics['r2_score']:.4f}, RMSE: {test_metrics['rmse']:.4f}")

        # 过拟合分析
        overfitting_gap = train_metrics['r2_score'] - test_metrics['r2_score']
        print(f"过拟合程度(R²差距): {overfitting_gap:.4f}")

    def _evaluate_model_on_dataset(self, X, y, dataset_name=""):
        """在指定数据集上评估模型性能[1,4](@ref)"""
        if self.current_model is None:
            raise ValueError("模型未初始化")

        X_scaled = self.scaler.transform(X)
        dtest = xgb.DMatrix(X_scaled)

        y_pred = self.current_model.predict(dtest)

        # 计算评估指标[1,4](@ref)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mse)
        mape = calculate_mape(y, y_pred)

        # 打印评估指标[1](@ref)
        print(f"{dataset_name}评估指标:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAPE: {mape:.2f}%")

        return {
            'mse': mse,
            'mae': mae,
            'r2_score': r2,
            'rmse': rmse,
            'mape': mape
        }

    # 其余方法保持不变（为了简洁省略，实际使用时需要保留）
    def _evaluate_model(self, data_batch):
        """评估模型性能（兼容接口）"""
        df_batch = pd.DataFrame(data_batch)
        X = df_batch[self.selected_columns]
        y = df_batch[self.label_column]

        return self._evaluate_model_on_dataset(X, y, "批量数据")

    def print_detailed_metrics_report(self):
        """打印详细的评估指标报告[1,4](@ref)"""
        if not self.train_test_metrics_history:
            print("暂无评估指标记录")
            return

        print("\n" + "=" * 80)
        print("训练集和测试集评估指标详细报告")
        print("=" * 80)

        for i, metrics_info in enumerate(self.train_test_metrics_history[-3:]):  # 最近3次训练
            print(f"\n模型版本 {metrics_info['version']} ({metrics_info['training_time'].strftime('%Y-%m-%d %H:%M')}):")
            print(f"数据量: {metrics_info['data_size']} 条记录")

            # 训练集指标
            train = metrics_info['train_metrics']
            print("训练集指标:")
            print(
                f"  MSE: {train['mse']:.4f} | MAE: {train['mae']:.4f} | R²: {train['r2_score']:.4f} | RMSE: {train['rmse']:.4f} | MAPE: {train['mape']:.2f}%")

            # 测试集指标
            test = metrics_info['test_metrics']
            print("测试集指标:")
            print(
                f"  MSE: {test['mse']:.4f} | MAE: {test['mae']:.4f} | R²: {test['r2_score']:.4f} | RMSE: {test['rmse']:.4f} | MAPE: {test['mape']:.2f}%")

            # 过拟合分析
            r2_gap = train['r2_score'] - test['r2_score']
            rmse_gap = test['rmse'] - train['rmse']
            print(f"过拟合分析: R²差距: {r2_gap:.4f}, RMSE差距: {rmse_gap:.4f}")
            print("-" * 60)

    def _detect_concept_drift(self, new_records):
        """检测概念漂移"""
        if len(new_records) < 10:  # 新数据太少，不进行漂移检测
            return False

        # 使用新数据评估当前模型性能
        new_performance = self._evaluate_model(new_records)
        current_performance = self.performance_history[-1]['performance']

        # 性能变化检测
        r2_change = current_performance['r2_score'] - new_performance['r2_score']
        rmse_change = new_performance['rmse'] - current_performance['rmse']

        print(f"漂移检测 - R²变化: {r2_change:.4f}, RMSE变化: {rmse_change:.4f}")

        # 记录漂移检测结果
        drift_info = {
            'timestamp': datetime.now(),
            'r2_change': r2_change,
            'rmse_change': rmse_change,
            'threshold': self.drift_threshold
        }
        self.drift_detection_history.append(drift_info)

        # 判断是否发生显著漂移
        significant_drift = (r2_change > self.drift_threshold or
                             rmse_change > self.drift_threshold)

        if significant_drift:
            print("⚠️ 检测到概念漂移，触发模型更新机制")

        return significant_drift

    def _manage_model_pool(self, new_model_info):
        """管理模型池，实现模型集成"""
        # 添加新模型到模型池
        self.model_pool.append(new_model_info)
        self.model_weights.append(1.0)  # 初始权重

        # 限制模型池大小
        if len(self.model_pool) > self.max_models:
            # 移除最旧的模型
            self.model_pool.pop(0)
            self.model_weights.pop(0)

        # 更新模型权重（基于性能）
        self._update_model_weights()

    def _update_model_weights(self):
        """基于模型性能更新权重"""
        if len(self.model_pool) <= 1:
            return

        # 基于最近性能更新权重
        recent_performance = []
        for i, model_info in enumerate(self.model_pool):
            # 给较新的模型更高权重
            recency_weight = (i + 1) / len(self.model_pool)
            performance_weight = model_info['performance']['r2_score']

            # 综合权重
            total_weight = 0.7 * performance_weight + 0.3 * recency_weight
            recent_performance.append(total_weight)

        # 归一化权重
        total = sum(recent_performance)
        self.model_weights = [w / total for w in recent_performance]

    def update_with_new_data(self, new_data_file):
        """
        用新数据更新模型，支持多种输入格式
        """
        print(f"\n正在处理新数据文件: {new_data_file}")

        # 读取新数据
        if isinstance(new_data_file, str):
            new_df = pd.read_excel(new_data_file, engine='openpyxl')
        else:
            new_df = new_data_file

        new_cleaned = new_df[self.selected_columns + [self.label_column]].dropna()

        if len(new_cleaned) == 0:
            print("新数据无有效记录，跳过更新")
            return False

        # 转换为记录格式并添加到数据流
        new_records = new_cleaned.to_dict('records')

        # 应用滑动窗口
        self.data_stream.extend(new_records)
        if len(self.data_stream) > self.window_size:
            self.data_stream = self.data_stream[-self.window_size:]

        print(f"数据流更新完成，当前大小: {len(self.data_stream)} 条记录")

        # 概念漂移检测
        drift_detected = False
        if self.use_drift_detection:
            drift_detected = self._detect_concept_drift(new_records)

        # 决定更新策略
        if drift_detected or len(new_records) > len(self.data_stream) * 0.3:
            # 显著漂移或大量新数据：完全重训练
            print("执行完全重训练...")
            self.model_version += 1
            self._train_model(self.data_stream, is_initial=False)
        else:
            # 轻微变化：增量学习
            print("执行增量学习...")
            self.model_version += 1
            self._train_model(
                new_records,
                is_initial=False,
                previous_model=self.current_model
            )

        self._save_model_snapshot()
        return True

    def predict_ensemble(self, features):
        """使用模型集成进行预测"""
        if not self.model_pool:
            raise ValueError("模型池为空")

        X_scaled = self.scaler.transform([features])
        dtest = xgb.DMatrix(X_scaled)

        predictions = []
        for i, model_info in enumerate(self.model_pool):
            pred = model_info['model'].predict(dtest)[0]
            predictions.append(pred)

        # 加权平均
        weighted_pred = sum(p * w for p, w in zip(predictions, self.model_weights))
        return weighted_pred

    def predict(self, features):
        """使用当前模型进行预测"""
        if self.current_model is None:
            raise ValueError("模型未初始化")

        X_scaled = self.scaler.transform([features])
        dtest = xgb.DMatrix(X_scaled)
        return self.current_model.predict(dtest)[0]

    def _save_model_snapshot(self):
        """保存模型快照"""
        snapshot_dir = "xgboost_model_snapshots"
        os.makedirs(snapshot_dir, exist_ok=True)

        snapshot_path = os.path.join(snapshot_dir, f"xgb_model_v{self.model_version}.model")
        self.current_model.save_model(snapshot_path)

        # 保存元数据
        meta_path = os.path.join(snapshot_dir, f"metadata_v{self.model_version}.pkl")
        joblib.dump({
            'performance_history': self.performance_history,
            'drift_detection_history': self.drift_detection_history,
            'train_test_metrics_history': self.train_test_metrics_history,
            'model_weights': self.model_weights,
            'data_stream_size': len(self.data_stream),
            'best_params': self.best_params  # 保存最优参数
        }, meta_path)

        print(f"模型快照已保存: {snapshot_path}")

    def print_drift_report(self):
        """打印漂移检测报告"""
        if not self.drift_detection_history:
            print("暂无漂移检测记录")
            return

        print("\n" + "=" * 70)
        print("概念漂移检测报告")
        print("=" * 70)

        for i, drift_info in enumerate(self.drift_detection_history[-5:]):  # 最近5次检测
            print(f"检测 {i + 1} ({drift_info['timestamp'].strftime('%Y-%m-%d %H:%M')}):")
            print(f"  R²变化: {drift_info['r2_change']:.4f}")
            print(f"  RMSE变化: {drift_info['rmse_change']:.4f}")
            print(f"  阈值: {drift_info['threshold']:.4f}")
            print("-" * 40)

    def get_model_info(self):
        """获取模型当前状态信息"""
        return {
            'current_version': self.model_version,
            'data_stream_size': len(self.data_stream),
            'model_pool_size': len(self.model_pool),
            'performance_history': self.performance_history,
            'train_test_metrics_history': self.train_test_metrics_history,
            'drift_detections': len(self.drift_detection_history),
            'best_params': self.best_params
        }


def dynamic_xgb_regression(file_path, selected_columns, label_column, use_grid_search=True):
    """
    动态XGBoost回归（兼容接口）
    """
    # 创建动态XGBoost实例
    dynamic_xgb = DynamicXGBoost(
        initial_data_path=file_path,
        selected_columns=selected_columns,
        label_column=label_column,
        window_size=100,
        drift_threshold=0.05,
        performance_threshold=0.02,
        max_models=3,
        use_drift_detection=True,
        use_grid_search=use_grid_search
    )

    # 输出初始模型信息
    model_info = dynamic_xgb.get_model_info()
    print("\n初始XGBoost模型训练完成:")
    print("=" * 50)
    print(f"模型版本: {model_info['current_version']}")
    print(f"数据量: {model_info['data_stream_size']}")
    print(f"模型池大小: {model_info['model_pool_size']}")
    if model_info['best_params']:
        print(f"最优参数: {model_info['best_params']}")

    return dynamic_xgb


# 使用示例
if __name__ == "__main__":
    # 配置参数
    file_path = "../../PFASdata.xlsx"
    selected_columns = [
        'Compound log K ow', 'WS (mg/L)',
        'MinPartialCharge', 'MaxPartialCharge', 'min projection (Å)', 'S',
        'rs/rp', '∆Gs-m (J·m-2)', 'MB contact angle (°)',
        'MB volume charge density δm (mol·m-3)',
        'Pressure (kPa)', 'Measurement time (min)', 'Initial concentration of compound (mg/L)', 'pH'
    ]
    label_column = "removal rate (%)"

    try:
        # 创建动态XGBoost模型（启用网格搜索）
        dynamic_model = dynamic_xgb_regression(file_path, selected_columns, label_column, use_grid_search=True)

        # 打印详细的评估指标报告
        dynamic_model.print_detailed_metrics_report()


        # 打印漂移报告
        dynamic_model.print_drift_report()

        print("\n动态XGBoost模型已就绪，支持持续学习和漂移检测!")

    except Exception as e:
        print(f"程序执行过程中出现错误: {str(e)}")
        import traceback

        traceback.print_exc()