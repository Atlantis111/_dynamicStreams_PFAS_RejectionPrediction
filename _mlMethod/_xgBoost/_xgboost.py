import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import warnings
import joblib

warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path):
    data = pd.read_excel(file_path)
    feature_columns = [
        'Compound log K ow', 'WS (mg/L)', 'MinPartialCharge', 'MaxPartialCharge', 'min projection (Å)', 'S',
        'rs/rp', '∆Gs-m (J·m-2)', 'MB contact angle (°)',
        'MB volume charge density δm (mol·m-3)',
        'Pressure (kPa)', 'Measurement time (min)', 'Initial concentration of compound (mg/L)', 'pH'
    ]
    target_column = "removal rate (%)"

    missing_features = [col for col in feature_columns if col not in data.columns]
    if missing_features:
        print(f"警告: 以下特征列在数据集中不存在: {missing_features}")
        feature_columns = [col for col in feature_columns if col in data.columns]

    if target_column not in data.columns:
        raise ValueError(f"目标列 '{target_column}' 在数据集中不存在")

    X = data[feature_columns]
    y = data[target_column]

    X = X.fillna(X.mean())
    y = y.fillna(y.mean())

    return X, y, feature_columns

def feature_engineering(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def evaluate_train_set(model, X_train, y_train):
    y_train_pred = model.predict(X_train)

    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    print("\n=== 训练集评估结果 ===")
    print(f"训练集均方误差 (MSE): {train_mse:.4f}")
    print(f"训练集均方根误差 (RMSE): {train_rmse:.4f}")
    print(f"训练集平均绝对误差 (MAE): {train_mae:.4f}")
    print(f"训练集决定系数 (R²): {train_r2:.4f}")

    return {'MSE': train_mse, 'RMSE': train_rmse, 'MAE': train_mae, 'R2': train_r2}, y_train_pred

def build_xgboost_model(X_train, y_train):
    xgb_model = xgb.XGBRegressor(random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [1, 2, 4, 6],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.2, 0.5, 0.7],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 1.5, 2]
    }

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    print("开始网格搜索...")
    grid_search.fit(X_train, y_train)

    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数: {-grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== 测试集评估结果 ===")
    print(f"测试集均方误差 (MSE): {mse:.4f}")
    print(f"测试集均方根误差 (RMSE): {rmse:.4f}")
    print(f"测试集平均绝对误差 (MAE): {mae:.4f}")
    print(f"测试集决定系数 (R²): {r2:.4f}")

    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}, y_pred

def cross_validation_evaluation(model, X, y, cv=5):
    print("\n=== 交叉验证结果 ===")

    scoring_metrics = {
        'neg_mean_squared_error': '负均方误差',
        'neg_mean_absolute_error': '负平均绝对误差',
        'r2': '决定系数'
    }

    cv_results = {}
    for metric, name in scoring_metrics.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
        if 'neg_' in metric:
            scores = -scores
            if metric == 'neg_mean_squared_error':
                cv_results['MSE'] = scores.mean()
                print(f"交叉验证 MSE: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
            else:
                cv_results['MAE'] = scores.mean()
                print(f"交叉验证 MAE: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        else:
            cv_results[metric.upper()] = scores.mean()
            print(f"交叉验证 {name}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

    return cv_results

def plot_enhanced_results(y_train, y_train_pred, y_test, y_test_pred, feature_columns, model):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].scatter(y_train, y_train_pred, alpha=0.6, color='blue', label='训练集')
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('实际值')
    axes[0, 0].set_ylabel('预测值')
    axes[0, 0].set_title('训练集: 实际值 vs 预测值')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(y_test, y_test_pred, alpha=0.6, color='green', label='测试集')
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('实际值')
    axes[0, 1].set_ylabel('预测值')
    axes[0, 1].set_title('测试集: 实际值 vs 预测值')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    train_residuals = y_train - y_train_pred
    test_residuals = y_test - y_test_pred

    axes[0, 2].scatter(y_train_pred, train_residuals, alpha=0.6, color='blue', label='训练集残差')
    axes[0, 2].scatter(y_test_pred, test_residuals, alpha=0.6, color='green', label='测试集残差')
    axes[0, 2].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 2].set_xlabel('预测值')
    axes[0, 2].set_ylabel('残差')
    axes[0, 2].set_title('训练集 vs 测试集残差对比')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    feature_importance = model.feature_importances_
    indices = np.argsort(feature_importance)[::-1]

    axes[1, 0].barh(range(len(indices)), feature_importance[indices], color='orange')
    axes[1, 0].set_yticks(range(len(indices)))
    axes[1, 0].set_yticklabels([feature_columns[i] for i in indices])
    axes[1, 0].set_xlabel('特征重要性')
    axes[1, 0].set_title('特征重要性排序')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].hist(y_train, alpha=0.7, label='训练集实际值', bins=20, color='blue')
    axes[1, 1].hist(y_train_pred, alpha=0.7, label='训练集预测值', bins=20, color='lightblue')
    axes[1, 1].set_xlabel('截留率')
    axes[1, 1].set_ylabel('频数')
    axes[1, 1].set_title('训练集: 实际值与预测值分布')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].hist(y_test, alpha=0.7, label='测试集实际值', bins=20, color='green')
    axes[1, 2].hist(y_test_pred, alpha=0.7, label='测试集预测值', bins=20, color='lightgreen')
    axes[1, 2].set_xlabel('截留率')
    axes[1, 2].set_ylabel('频数')
    axes[1, 2].set_title('测试集: 实际值与预测值分布')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def main():
    try:
        print("步骤1: 加载数据...")
        X, y, feature_columns = load_and_preprocess_data('../../PFASdata.xlsx')
        print(f"数据形状: {X.shape}")
        print(f"特征数量: {len(feature_columns)}")

        print("\n步骤2: 特征工程和数据标准化...")
        X_scaled, y, scaler = feature_engineering(X, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        print(f"训练集大小: {X_train.shape}")
        print(f"测试集大小: {X_test.shape}")

        print("\n步骤3: 模型训练和网格搜索...")
        best_model, best_params = build_xgboost_model(X_train, y_train)

        print("\n步骤4: 训练集评估...")
        train_metrics, y_train_pred = evaluate_train_set(best_model, X_train, y_train)

        print("\n步骤5: 测试集评估...")
        test_metrics, y_test_pred = evaluate_model(best_model, X_test, y_test)

        cv_metrics = cross_validation_evaluation(best_model, X_scaled, y)

        print("\n步骤6: 生成可视化结果...")
        plot_enhanced_results(y_train.values, y_train_pred, y_test.values, y_test_pred,
                              feature_columns, best_model)

        print("\n=== 过拟合分析 ===")
        overfitting_gap = train_metrics['R2'] - test_metrics['R2']
        print(f"训练集与测试集R²差距: {overfitting_gap:.4f}")

        if overfitting_gap > 0.1:
            print("警告: 可能存在过拟合现象，建议增加正则化参数或减少模型复杂度")
        elif overfitting_gap < 0.01:
            print("模型表现均衡，过拟合风险较低")
        else:
            print("模型表现正常，略有差异")

        joblib.dump(best_model, 'xgboost_removal_rate_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        print("\n模型已保存为 'xgboost_removal_rate_model.pkl'")
        print("数据标准化器已保存为 'scaler.pkl'")

        print("\n" + "=" * 60)
        print("模型训练完成！综合评估报告")
        print("=" * 60)
        print(f"最佳参数: {best_params}")
        print(f"训练集 R²: {train_metrics['R2']:.4f}")
        print(f"测试集 R²: {test_metrics['R2']:.4f}")
        print(f"训练集 RMSE: {train_metrics['RMSE']:.4f}")
        print(f"测试集 RMSE: {test_metrics['RMSE']:.4f}")
        print(f"训练集与测试集R²差距: {overfitting_gap:.4f}")

    except FileNotFoundError:
        print("错误: 未找到文件 'PFASdata.xlsx'")
        print("请确保文件存在于当前目录中，或修改文件路径")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        print("请检查数据文件格式和内容是否正确")

if __name__ == "__main__":
    main()