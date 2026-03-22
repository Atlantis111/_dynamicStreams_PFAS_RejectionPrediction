import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.use('TkAgg')


def plot_manual_heatmap(data_list: list,
                        x_fontsize=16,
                        y_fontsize=14,
                        annot_fontsize=16,
                        cbar_label_fontsize=14,
                        cbar_tick_fontsize=14):
    """
    支持不等长输入的 R² 交叉验证热力图绘制
    字体大小参数说明：
    - x_fontsize: 横轴标签字体大小
    - y_fontsize: 纵轴标签字体大小
    - annot_fontsize: 热力图上数字的字体大小
    - cbar_label_fontsize: 颜色条标题字体大小
    - cbar_tick_fontsize: 颜色条刻度标签字体大小
    """
    # --- 处理不等长列表 ---
    max_cols = max(len(row) for row in data_list)
    num_rows = len(data_list)
    data_matrix = np.full((num_rows, max_cols), np.nan)

    for i, row in enumerate(data_list):
        data_matrix[i, :len(row)] = row

    plt.figure(figsize=(10, 8))

    # 设置坐标轴的标签
    labels_y = [f"AP-XGboost v{i + 1}" for i in range(num_rows)]
    labels_x = [f"Dataset v{j + 1}" for j in range(max_cols)]

    # 绘制热力图
    ax = sns.heatmap(data_matrix,
                     annot=True,
                     fmt=".3f",
                     cmap="YlGnBu",
                     vmin=0.6,
                     vmax=1.0,
                     xticklabels=labels_x,
                     yticklabels=labels_y,
                     mask=np.isnan(data_matrix),
                     cbar_kws={'label': 'R² Score'})

    # 修改横轴标签字体大小
    plt.xticks(fontsize=x_fontsize)
    plt.xlabel("Test Dataset Version (Time Progression)", labelpad=10, fontsize=x_fontsize)

    # 修改纵轴标签字体大小
    plt.yticks(fontsize=y_fontsize)
    plt.ylabel("Trained Model Version (Time Progression)", labelpad=10, fontsize=y_fontsize)

    # 修改热力图方格上的数字字体大小
    for text in ax.texts:
        text.set_fontsize(annot_fontsize)

    # 修改颜色条
    cbar = ax.collections[0].colorbar
    # 修改颜色条刻度标签字体大小
    cbar.ax.tick_params(labelsize=cbar_tick_fontsize)
    # 修改颜色条标题字体大小
    cbar.ax.set_ylabel('R² Score', fontsize=cbar_label_fontsize)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("🎨 修改版热力图工具启动")
    print("=" * 60)

    manual_data = [
        [0.920],
        [0.983, 0.981],
        [0.970, 0.972, 0.975],
        [0.901, 0.925, 0.956, 0.987]
    ]

    # 调用函数，这里使用了默认字体大小
    # 如果需要调整字体大小，可以这样调用：
    # plot_manual_heatmap(manual_data,
    #                     x_fontsize=12,
    #                     y_fontsize=12,
    #                     annot_fontsize=10,
    #                     cbar_label_fontsize=12,
    #                     cbar_tick_fontsize=10)
    plot_manual_heatmap(manual_data)
    print("✅ 画图完成！")