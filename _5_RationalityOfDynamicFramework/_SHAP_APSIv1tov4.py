import matplotlib
# 尝试使用 'TkAgg' 后端，它通常兼容性较好
matplotlib.use('TkAgg')  # 或者也可以尝试 'Qt5Agg'
import matplotlib.pyplot as plt
import numpy as np

# ----------------- 数据定义 -----------------
# 为了图表美观，我们对部分带有上下标的特征名进行了 matplotlib math 格式化
data_v1 = {
    "MB contact angle (°)": 1.4326,
    "∆G$_{s-m}$ (J·m$^{-2}$)": 0.9649,
    "r$_s$/r$_p$": 0.9197,
    "Compound log K$_{ow}$": 0.4996,
    "MB vol charge density δ$_m$ (mol·m$^{-3}$)": 0.3353,
    "Initial concentration (mg/L)": 0.3081,
    "Measurement time (min)": 0.2206,
    "WS (mg/L)": 0.1386,
    "Pressure (kPa)": 0.1344,
    "min projection (Å)": 0.0541
}

data_v2 = {
    "MB contact angle (°)": 1.4438,
    "r$_s$/r$_p$": 1.3217,
    "pH": 0.7583,
    "∆G$_{s-m}$ (J·m$^{-2}$)": 0.6496,
    "MB vol charge density δ$_m$ (mol·m$^{-3}$)": 0.4464,
    "Pressure (kPa)": 0.2780,
    "Measurement time (min)": 0.2305,
    "Compound log K$_{ow}$": 0.2273,
    "WS (mg/L)": 0.1629,
    "min projection (Å)": 0.0847
}

data_v3 = {
    "r$_s$/r$_p$": 2.6393,
    "pH": 1.6839,
    "MB vol charge density δ$_m$ (mol·m$^{-3}$)": 0.8162,
    "MB contact angle (°)": 0.7441,
    "MinPartialCharge": 0.7268,
    "Measurement time (min)": 0.7178,
    "∆G$_{s-m}$ (J·m$^{-2}$)": 0.5644,
    "Initial concentration (mg/L)": 0.3382,
    "Pressure (kPa)": 0.2700,
    "MaxPartialCharge": 0.2482
}

data_v4 = {
    "r$_s$/r$_p$": 2.3201,
    "pH": 1.6817,
    "MB vol charge density δ$_m$ (mol·m$^{-3}$)": 1.4665,
    "MB contact angle (°)": 0.8863,
    "Measurement time (min)": 0.5005,
    "∆G$_{s-m}$ (J·m$^{-2}$)": 0.4852,
    "Initial concentration (mg/L)": 0.3968,
    "Compound log K$_{ow}$": 0.3962,
    "Pressure (kPa)": 0.2508,
    "min projection (Å)": 0.2248
}

# ----------------- 全局样式设置 -----------------
plt.rcParams['font.sans-serif'] = ['Arial']  # 如果没有Arial可改为 'DejaVu Sans'
plt.rcParams['font.size'] = 12

# 创建 2x2 的子图网格
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
datasets = [data_v1, data_v2, data_v3, data_v4]
titles = ['Model v1 (Initial)', 'Model v2 (Concept Drift)', 'Model v3 (Concept Drift)', 'Model v4 (Concept Drift)']
panel_labels = ['a)', 'b)', 'c)', 'd)']

# 提取参考图颜色
bar_color = '#6372f2'


# ----------------- 绘图函数 -----------------
def plot_shap_bars(ax, data, title, panel_label):
    # 将字典转为列表并按 SHAP 值升序排列（因为条形图是从下往上画的）
    features = list(data.keys())
    values = list(data.values())

    # 反转顺序，让最大的在最上面
    features.reverse()
    values.reverse()

    # 绘制水平条形图
    ax.barh(features, values, color=bar_color, height=0.6)

    # ----------------- 边框与刻度精修 -----------------
    # 设置边框粗细
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')

    # 设置刻度朝内，显示上方和右侧刻度，设置刻度线粗细
    ax.tick_params(axis='both', direction='in', length=6, width=1.5,
                   top=True, right=True, labelsize=12)

    # X轴刻度只留底部标签
    ax.tick_params(labeltop=False, labelright=False)

    # 移除 Y 轴特征名的长刻度线（只留小刻度或者不要刻度，这里为了美观保留标准长度）
    # ax.set_xlabel("Mean |SHAP| Value", fontsize=14, fontweight='bold')

    # 添加左上角的面板标签 (如 a), b) )，坐标是相对轴的比例 (0-1)
    ax.text(-0.05, 1.05, panel_label, transform=ax.transAxes,
            fontsize=18, fontweight='bold', va='bottom', ha='right')

    # 可选：设置X轴最大值，留出一点空白
    ax.set_xlim(0, max(values) * 1.1)


# ----------------- 执行绘制 -----------------
for i, ax in enumerate(axes.flatten()):
    plot_shap_bars(ax, datasets[i], titles[i], panel_labels[i])

# 调整布局以防止文本重叠
plt.tight_layout(pad=3.0)

# 保存高分辨率图片，适合直接插入论文
plt.savefig('shap_feature_importance.png', dpi=600, bbox_inches='tight')

# 显示图片
plt.show()