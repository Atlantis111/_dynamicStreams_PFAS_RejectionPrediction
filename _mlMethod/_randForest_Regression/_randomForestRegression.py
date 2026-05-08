import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import datetime

warnings.filterwarnings('ignore')
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

feature_columns = [
    'Compound log K ow', 'WS (mg/L)', 'MinPartialCharge', 'MaxPartialCharge', 'min projection (Å)', 'S',
    'rs/rp', '∆Gs-m (J·m-2)', 'MB contact angle (°)',
    'MB volume charge density δm (mol·m-3)',
    'Pressure (kPa)', 'Measurement time (min)', 'Initial concentration of compound (mg/L)', 'pH'
]
target_column = "removal rate (%)"

def load_and_preprocess_data(file_path):
    try:
        data = pd.read_excel(file_path)
        print(f"数据形状: {data.shape}")
        print(f"特征数量: {len(feature_columns)}")

        missing_values = data[feature_columns + [target_column]].isnull().sum()
        print("缺失值统计:")
        print(missing_values[missing_values > 0])

        data = data.dropna(subset=feature_columns + [target_column])
        print(f"处理缺失值后数据形状: {data.shape}")

        X = data[feature_columns]
        y = data[target_column]

        return X, y, data
    except Exception as e:
        print(f"数据加载错误: {e}")
        return None, None, None

def explore_data(X, y):
    print("\n=== 数据探索分析 ===")

    print("\n特征描述性统计:")
    print(X.describe())

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(y, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('截留率')
    plt.ylabel('频数')
    plt.title('目标变量分布')

    plt.subplot(1, 3, 2)
    correlations = X.corrwith(y).sort_values(ascending=False)
    plt.barh(range(len(correlations)), correlations.values)
    plt.yticks(range(len(correlations)), correlations.index)
    plt.xlabel('与目标变量的相关系数')
    plt.title('特征重要性（初步）')

    plt.subplot(1, 3, 3)
    corr_matrix = X.corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns)
    plt.title('特征间相关性热图')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()

    return correlations

def build_random_forest_model(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"\n=== 数据集划分 ===")
    print(f"训练集大小: {X_train.shape[0]} 样本")
    print(f"测试集大小: {X_test.shape[0]} 样本")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt']
    }

    print("\n=== 开始网格搜索 ===")
    print("参数网格:", param_grid)

    rf = RandomForestRegressor(random_state=random_state, n_jobs=-1)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_scaled, y_train)

    print("\n=== 网格搜索完成 ===")
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数 (负MSE): {grid_search.best_score_:.4f}")

    best_rf = grid_search.best_estimator_

    return best_rf, X_train_scaled, X_test_scaled, y_train, y_test, scaler, grid_search

def evaluate_model(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mse = mean_squared_error(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\n" + "=" * 60)
    print("模型评估结果")
    print("=" * 60)

    evaluation_df = pd.DataFrame({
        '数据集': ['训练集', '测试集'],
        'MSE': [train_mse, test_mse],
        'MAE': [train_mae, test_mae],
        'R²': [train_r2, test_r2]
    })

    print(evaluation_df.round(4))

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_test_pred, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title('测试集: 实际值 vs 预测值')

    plt.subplot(1, 3, 2)
    residuals = y_test - y_test_pred
    plt.scatter(y_test_pred, residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('残差分析')

    plt.subplot(1, 3, 3)
    feature_importance = model.feature_importances_
    indices = np.argsort(feature_importance)[::-1]

    plt.barh(range(len(indices)), feature_importance[indices])
    plt.yticks(range(len(indices)), [feature_columns[i] for i in indices])
    plt.xlabel('特征重要性')
    plt.title('随机森林特征重要性排名')

    plt.tight_layout()
    plt.show()

    return evaluation_df, y_test_pred, residuals

def advanced_model_analysis(model, X, y, feature_names):
    print("\n=== 高级模型分析 ===")

    cv_scores_mse = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    cv_scores_r2 = cross_val_score(model, X, y, cv=5, scoring='r2')

    print(f"5折交叉验证MSE: {cv_scores_mse.mean():.4f} (±{cv_scores_mse.std():.4f})")
    print(f"5折交叉验证R²: {cv_scores_r2.mean():.4f} (±{cv_scores_r2.std():.4f})")

    feature_importance_df = pd.DataFrame({
        '特征': feature_names,
        '重要性': model.feature_importances_
    }).sort_values('重要性', ascending=False)

    print("\n特征重要性排名:")
    print(feature_importance_df)

    train_sizes = np.linspace(0.1, 1.0, 5)
    train_scores = []
    test_scores = []

    for size in train_sizes:
        n_samples = int(size * X.shape[0])
        X_subset = X[:n_samples]
        y_subset = y[:n_samples]

        model_temp = RandomForestRegressor(**model.get_params())
        model_temp.fit(X_subset, y_subset)

        train_scores.append(model_temp.score(X_subset, y_subset))
        test_scores.append(model_temp.score(X, y))

    plt.figure(figsize=(10, 5))
    plt.plot(train_sizes, train_scores, 'o-', label='训练集得分')
    plt.plot(train_sizes, test_scores, 'o-', label='测试集得分')
    plt.xlabel('训练样本比例')
    plt.ylabel('R²得分')
    plt.title('学习曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    return feature_importance_df

def save_model_and_results(model, scaler, evaluation_results, feature_importance, file_prefix):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    model_filename = f"{file_prefix}_random_forest_model_{timestamp}.pkl"
    scaler_filename = f"{file_prefix}_scaler_{timestamp}.pkl"

    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)

    results_filename = f"{file_prefix}_model_results_{timestamp}.csv"
    evaluation_results.to_csv(results_filename, index=False)

    importance_filename = f"{file_prefix}_feature_importance_{timestamp}.csv"
    feature_importance.to_csv(importance_filename, index=False)

    print(f"\n=== 模型和结果已保存 ===")
    print(f"模型文件: {model_filename}")
    print(f"标准化器: {scaler_filename}")
    print(f"评估结果: {results_filename}")
    print(f"特征重要性: {importance_filename}")

    return {
        'model_file': model_filename,
        'scaler_file': scaler_filename,
        'results_file': results_filename,
        'importance_file': importance_filename
    }

def main():
    print("开始执行膜污染物截留率预测模型...")

    X, y, full_data = load_and_preprocess_data("../../PFASdata.xlsx")

    if X is None:
        print("数据加载失败，请检查文件路径和数据格式")
        return

    correlations = explore_data(X, y)

    best_model, X_train, X_test, y_train, y_test, scaler, grid_search = build_random_forest_model(X, y)

    evaluation_results, y_pred, residuals = evaluate_model(best_model, X_train, X_test, y_train, y_test)

    feature_importance_results = advanced_model_analysis(best_model, X, y, feature_columns)

    saved_files = save_model_and_results(
        best_model, scaler, evaluation_results,
        feature_importance_results, "PFAS_removal"
    )

    print("\n" + "=" * 80)
    print("建模完成！总结报告")
    print("=" * 80)
    print(f"数据集: {X.shape[0]} 条样本, {X.shape[1]} 个特征")
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"测试集R²: {evaluation_results.iloc[1]['R²']:.4f}")
    print(f"测试集MSE: {evaluation_results.iloc[1]['MSE']:.4f}")
    print(f"测试集MAE: {evaluation_results.iloc[1]['MAE']:.4f}")
    print(f"最重要的特征: {feature_importance_results.iloc[0]['特征']} "
          f"(重要性: {feature_importance_results.iloc[0]['重要性']:.4f})")

    return {
        'model': best_model,
        'evaluation': evaluation_results,
        'feature_importance': feature_importance_results,
        'saved_files': saved_files
    }

if __name__ == "__main__":
    results = main()