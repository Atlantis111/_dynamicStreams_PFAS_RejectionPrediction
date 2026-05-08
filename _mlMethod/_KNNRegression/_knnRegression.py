import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
import warnings

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

    print("数据基本信息：")
    print(f"数据集形状: {data.shape}")
    print(f"特征数量: {len(feature_columns)}")
    print(f"目标变量: {target_column}")
    print("\n缺失值统计：")
    print(data[feature_columns + [target_column]].isnull().sum())

    data_clean = data.dropna(subset=feature_columns + [target_column])
    X = data_clean[feature_columns]
    y = data_clean[target_column]

    print(f"\n清洗后数据形状: {X.shape}")
    print(f"目标变量统计:")
    print(y.describe())

    return X, y, feature_columns, target_column


def build_knn_model(X_train, y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor())
    ])

    param_grid = {
        'knn__n_neighbors': range(3, 15),
        'knn__weights': ['uniform', 'distance'],
        'knn__p': [1, 2]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    print("开始网格搜索...")
    grid_search.fit(X_train, y_train)

    print("网格搜索完成！")
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数 (负MSE): {grid_search.best_score_:.4f}")

    return grid_search


def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\n" + "=" * 50)
    print("训练集评估结果:")
    print("=" * 50)
    print(f"均方误差 (MSE): {train_mse:.4f}")
    print(f"均方根误差 (RMSE): {train_rmse:.4f}")
    print(f"平均绝对误差 (MAE): {train_mae:.4f}")
    print(f"决定系数 (R²): {train_r2:.4f}")

    print("\n" + "=" * 50)
    print("测试集评估结果:")
    print("=" * 50)
    print(f"均方误差 (MSE): {test_mse:.4f}")
    print(f"均方根误差 (RMSE): {test_rmse:.4f}")
    print(f"平均绝对误差 (MAE): {test_mae:.4f}")
    print(f"决定系数 (R²): {test_r2:.4f}")

    mse_gap = train_mse - test_mse
    r2_gap = train_r2 - test_r2

    print("\n" + "=" * 50)
    print("过拟合分析:")
    print("=" * 50)
    print(f"MSE差距 (训练集-测试集): {mse_gap:.4f}")
    print(f"R²差距 (训练集-测试集): {r2_gap:.4f}")

    if train_r2 > test_r2 + 0.1:
        print("⚠️ 可能存在过拟合现象")
    elif test_r2 > train_r2 + 0.1:
        print("ℹ️ 可能存在欠拟合现象")
    else:
        print("✅ 模型拟合程度较为平衡")

    return y_test_pred, train_mse, train_rmse, train_mae, train_r2, test_mse, test_rmse, test_mae, test_r2


def perform_cross_validation(model, X, y, cv=10):
    print(f"\n开始{cv}折交叉验证...")
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    cv_rmse_scores = np.sqrt(-cv_scores)

    print("交叉验证结果:")
    print(f"各折RMSE: {cv_rmse_scores}")
    print(f"平均RMSE: {cv_rmse_scores.mean():.4f} (±{cv_rmse_scores.std():.4f})")
    print(f"最佳RMSE: {cv_rmse_scores.min():.4f}")
    print(f"最差RMSE: {cv_rmse_scores.max():.4f}")

    return cv_rmse_scores


def plot_results(y_train, y_train_pred, y_test, y_test_pred):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].scatter(y_train, y_train_pred, alpha=0.6, color='blue')
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('真实值')
    axes[0, 0].set_ylabel('预测值')
    axes[0, 0].set_title('训练集: 预测值 vs 真实值')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(y_test, y_test_pred, alpha=0.6, color='green')
    axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('真实值')
    axes[0, 1].set_ylabel('预测值')
    axes[0, 1].set_title('测试集: 预测值 vs 真实值')
    axes[0, 1].grid(True, alpha=0.3)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    axes[0, 2].bar(['训练集', '测试集'], [train_r2, test_r2], color=['blue', 'green'], alpha=0.7)
    axes[0, 2].set_ylabel('R² Score')
    axes[0, 2].set_title('训练集 vs 测试集 R²对比')
    axes[0, 2].grid(True, alpha=0.3)

    train_residuals = y_train - y_train_pred
    axes[1, 0].scatter(y_train_pred, train_residuals, alpha=0.6, color='blue')
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('预测值')
    axes[1, 0].set_ylabel('残差')
    axes[1, 0].set_title('训练集残差图')
    axes[1, 0].grid(True, alpha=0.3)

    test_residuals = y_test - y_test_pred
    axes[1, 1].scatter(y_test_pred, test_residuals, alpha=0.6, color='green')
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('预测值')
    axes[1, 1].set_ylabel('残差')
    axes[1, 1].set_title('测试集残差图')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].hist(train_residuals, bins=20, alpha=0.5, color='blue', label='训练集', edgecolor='black')
    axes[1, 2].hist(test_residuals, bins=20, alpha=0.5, color='green', label='测试集', edgecolor='black')
    axes[1, 2].axvline(x=0, color='r', linestyle='--')
    axes[1, 2].set_xlabel('残差')
    axes[1, 2].set_ylabel('频数')
    axes[1, 2].set_title('训练集和测试集误差分布对比')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_features(X, y, feature_columns):
    data_with_target = X.copy()
    data_with_target['removal_rate'] = y

    correlation = data_with_target.corr()['removal_rate'].sort_values(ascending=False)

    print("\n特征与目标变量相关性分析:")
    for feature, corr_value in correlation.items():
        if feature != 'removal_rate':
            print(f"{feature}: {corr_value:.4f}")

    plt.figure(figsize=(12, 10))
    corr_matrix = data_with_target.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', annot_kws={'size': 8})
    plt.title('特征相关性热力图')
    plt.tight_layout()
    plt.show()

    return correlation


def main():
    print("=== KNN回归模型 - 膜截留率预测 ===\n")

    try:
        X, y, feature_columns, target_column = load_and_preprocess_data('../../PFASdata.xlsx')
    except FileNotFoundError:
        print("错误: 未找到PFASdata.xlsx文件，请确保文件路径正确")
        return
    except Exception as e:
        print(f"数据加载错误: {e}")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")

    correlation = analyze_features(X, y, feature_columns)

    grid_search = build_knn_model(X_train, y_train)
    best_model = grid_search.best_estimator_

    y_test_pred, train_mse, train_rmse, train_mae, train_r2, test_mse, test_rmse, test_mae, test_r2 = evaluate_model(
        best_model, X_train, y_train, X_test, y_test
    )

    y_train_pred = best_model.predict(X_train)

    cv_scores = perform_cross_validation(best_model, X, y)

    plot_results(y_train, y_train_pred, y_test, y_test_pred)

    print("\n" + "=" * 50)
    print("模型总结")
    print("=" * 50)
    print("最佳参数组合:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")

    print(f"\n模型性能对比:")
    print(f"训练集 R²: {train_r2:.4f}")
    print(f"测试集 R²: {test_r2:.4f}")
    print(f"训练集 RMSE: {train_rmse:.4f}")
    print(f"测试集 RMSE: {test_rmse:.4f}")
    print(f"交叉验证平均 RMSE: {cv_scores.mean():.4f}")

    train_results_df = pd.DataFrame({
        '真实值': y_train,
        '预测值': y_train_pred,
        '残差': y_train - y_train_pred,
        '数据集': '训练集'
    })

    test_results_df = pd.DataFrame({
        '真实值': y_test,
        '预测值': y_test_pred,
        '残差': y_test - y_test_pred,
        '数据集': '测试集'
    })

    results_df = pd.concat([train_results_df, test_results_df], ignore_index=True)
    results_df.to_csv('knn_prediction_results.csv', index=False)
    print(f"\n预测结果已保存到: knn_prediction_results.csv")

    return best_model, grid_search, results_df


if __name__ == "__main__":
    best_model, grid_search, results_df = main()