import matplotlib.pyplot as plt
import numpy as np

# ==============================
# 🔧【你的实验数据】
# ==============================
automamba_b0 = [34.85, 67.79]
automamba_b2 = [15.43, 70.17]
segformer_b0 = [25.31, 66.55]
segformer_b2 = [12.54, 67.82]

automamba_flops_b0 = [7.614, 30.455, 60.909, 124.928]
automamba_flops_b2 = [26.268, 107.52, 215.04, 430.08]
segformer_flops_b0 = [7.956, 44.307, 124.928, 386.048]
segformer_flops_b2 = [25.317, 151.552, 431.104, None]  # OOM

# ==============================
# 🎨 绘图
# ==============================
try:
    plt.style.use(['science', 'ieee', 'no-latex'])
except:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Liberation Serif", "Times"],
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 10,
        "axes.linewidth": 0.8,
    })

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))

# ------------------------------
# 图1: FPS vs mIoU（保持不变）
# ------------------------------
ax1 = axes[0]
auto_points = np.array([automamba_b0, automamba_b2])
seg_points = np.array([segformer_b0, segformer_b2])

color_automamba = '#2E86AB'
color_segformer = '#A23B72'
marker_b0 = 'D'   # diamond
marker_b2 = '^'   # triangle up

# 绘制
ax1.plot(auto_points[:, 0], auto_points[:, 1], color=color_automamba, linestyle='-', linewidth=1.5, zorder=2)
ax1.scatter(auto_points[0, 0], auto_points[0, 1], color=color_automamba, marker=marker_b0, s=60, edgecolors='k', linewidth=0.5, zorder=3)
ax1.scatter(auto_points[1, 0], auto_points[1, 1], color=color_automamba, marker=marker_b2, s=60, edgecolors='k', linewidth=0.5, zorder=3)

ax1.plot(seg_points[:, 0], seg_points[:, 1], color=color_segformer, linestyle='--', linewidth=1.5, zorder=2)
ax1.scatter(seg_points[0, 0], seg_points[0, 1], color=color_segformer, marker=marker_b0, s=60, edgecolors='k', linewidth=0.5, zorder=3)
ax1.scatter(seg_points[1, 0], seg_points[1, 1], color=color_segformer, marker=marker_b2, s=60, edgecolors='k', linewidth=0.5, zorder=3)

# 标注 mIoU（纯文字）
def annotate_mIoU(ax, x, y, miou, offset_y):
    ax.annotate(f'{miou:.2f}', xy=(x, y), xytext=(0, offset_y),
                textcoords='offset points', fontsize=8, ha='center', va='center', color='black')

annotate_mIoU(ax1, *automamba_b0, automamba_b0[1], 10)
annotate_mIoU(ax1, *automamba_b2, automamba_b2[1], 10)
annotate_mIoU(ax1, *segformer_b0, segformer_b0[1], -12)
annotate_mIoU(ax1, *segformer_b2, segformer_b2[1], -12)

ax1.set_xlim(10, None)
ax1.set_ylim(60, None)
ax1.set_xlabel('FPS (↑)')
ax1.set_ylabel('mIoU (\%) (↑)')
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.locator_params(axis='x', nbins=5)
ax1.locator_params(axis='y', nbins=4)

# 图例：严格匹配点形状
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color=color_automamba, linestyle='-', linewidth=1.5, label='AutoMamba'),
    Line2D([0], [0], color=color_segformer, linestyle='--', linewidth=1.5, label='SegFormer'),
    Line2D([0], [0], marker=marker_b0, color='w', markerfacecolor='gray', markersize=7, markeredgecolor='k', label='B0'),
    Line2D([0], [0], marker=marker_b2, color='w', markerfacecolor='gray', markersize=7, markeredgecolor='k', label='B2'),
]
ax1.legend(handles=legend_elements, loc='lower right', frameon=True, fancybox=False, edgecolor='black', ncol=2)

# ------------------------------
# 图2: GFlops —— 关键修复版
# ------------------------------
ax2 = axes[1]

resolutions = ['512²', '1024²', '1024×2048', '2048²']
x = np.arange(len(resolutions))

# 数据
data = {
    'AutoMamba-B0': automamba_flops_b0,
    'AutoMamba-B2': automamba_flops_b2,
    'SegFormer-B0': segformer_flops_b0,
    'SegFormer-B2': segformer_flops_b2,
}

colors = ['#2E86AB', '#2E86AB', '#A23B72', '#A23B72']
linestyles = ['-', '--', '-', '--']
markers = ['o', 's', 'o', 's']  # 注意：这里只是占位，实际绘图用 D/^

# ✅ 关键：确保绘图时使用与图1一致的 marker：B0=D, B2=^
plot_markers = {'B0': 'D', 'B2': '^'}
model_styles = {
    'AutoMamba-B0': (color_automamba, '-', plot_markers['B0']),
    'AutoMamba-B2': (color_automamba, '--', plot_markers['B2']),
    'SegFormer-B0': (color_segformer, '-', plot_markers['B0']),
    'SegFormer-B2': (color_segformer, '--', plot_markers['B2']),
}

plotted_x = {}
plotted_y = {}

for label, (color, ls, mk) in model_styles.items():
    y_vals = data[label]
    y_plot = []
    x_plot = []
    for j, y in enumerate(y_vals):
        if y is not None:
            y_plot.append(y)
            x_plot.append(x[j])
    ax2.plot(x_plot, y_plot, color=color, linestyle=ls, linewidth=1.5, zorder=2)
    ax2.scatter(x_plot, y_plot, color=color, marker=mk, s=60, edgecolors='k', linewidth=0.5, zorder=3)
    plotted_x[label] = x_plot
    plotted_y[label] = y_plot

# === 设置纵轴上限为 450 ===
ax2.set_ylim(0, 450)

# === 百分比标注（仅从 1024² 开始，红色，右侧）===
def add_red_reduction(ax, x_pos, y_auto, y_seg, side='right'):
    if y_seg == 0:
        return
    reduction = (y_seg - y_auto) / y_seg * 100
    ha = 'left' if side == 'right' else 'right'
    offset_x = 6 if side == 'right' else -4
    ax.annotate(f'↓{reduction:.1f}%',
                xy=(x_pos, y_auto),
                xytext=(offset_x, 0),
                textcoords='offset points',
                fontsize=8, ha=ha, va='center',
                color='red', fontweight='bold')

# B0: 1024², 1024×2048, 20²
for idx, side in zip([1, 2, 3], ['right', 'right', 'left']):
    add_red_reduction(ax2, x[idx], automamba_flops_b0[idx], segformer_flops_b0[idx], side=side)

# B2: 1024², 1024×2048, 20²
for idx, side in zip([1, 2], ['right', 'right']):
    add_red_reduction(ax2, x[idx], automamba_flops_b2[idx], segformer_flops_b2[idx], side=side)

# === SegFormer-B2 连线到图外 OOM ===
if len(plotted_x['SegFormer-B2']) >= 3:
    last_x = plotted_x['SegFormer-B2'][-1]   # 1024×2048 的 x
    last_y = plotted_y['SegFormer-B2'][-1]   # 对应 y

    oom_x = x[-1]  # 2048² 的 x
    oom_y = 455  # 超出 ylim 上限（450），放在图外

    # 画虚线：从最后一个点 → OOM 位置
    ax2.plot([last_x, oom_x], [last_y, 500],
             color=color_segformer, linestyle='--', linewidth=1.2, alpha=0.9)

    # 标注 OOM（红色，无框）
    ax2.text(oom_x, oom_y, 'OOM',
             color='red', fontsize=10, fontweight='bold',
             ha='center', va='bottom')

# 坐标轴
ax2.set_xticks(x)
ax2.set_xticklabels(resolutions)
ax2.set_ylabel('GFlops (↓)')
ax2.set_xlabel('Input Resolution')
ax2.grid(True, axis='y', linestyle=':', alpha=0.7)
ax2.locator_params(axis='y', nbins=5)  # 0, 100, 200, 300, 400, 450

# === 图例：严格复刻图1风格 ===
legend_elements2 = [
    Line2D([0], [0], color=color_automamba, linestyle='-', linewidth=1.5, label='AutoMamba'),
    Line2D([0], [0], color=color_segformer, linestyle='--', linewidth=1.5, label='SegFormer'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='gray', markersize=7, markeredgecolor='k', label='B0'),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=7, markeredgecolor='k', label='B2'),
]
ax2.legend(handles=legend_elements2, loc='upper left', frameon=True, fancybox=False, edgecolor='black', ncol=2)

# ------------------------------
# 全局调整
# ------------------------------
plt.tight_layout(pad=0.3)
plt.savefig('automamba_vs_segformer.png', dpi=300, bbox_inches='tight')
plt.savefig('automamba_vs_segformer.pdf', bbox_inches='tight')
print("✅ FIXED: y_max=450, OOM connected outside, legend & markers consistent.")
plt.show()