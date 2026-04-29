import numpy as np
import matplotlib.pyplot as plt

# =========================
# 中文字体设置（必须！否则乱码）
# =========================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# =========================
# Data
# =========================
stages = ['输入', '第一层编码器', '第二层编码器']
x = np.arange(len(stages))
width = 0.22

# Baseline
baseline_low  = [0.9036, 0.7429, 0.5918]
baseline_mid  = [0.0863, 0.2315, 0.3326]
baseline_high = [0.0102, 0.0256, 0.0756]

# Improved
improved_low  = [0.9036, 0.4937, 0.3643]
improved_mid  = [0.0863, 0.4477, 0.5033]
improved_high = [0.0102, 0.0586, 0.1324]

# =========================
# Plot
# =========================
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, sharey=True)

# 颜色（论文风格）
color_low  = '#4C72B0'
color_mid  = '#55A868'
color_high = '#C44E52'

def add_labels(ax, bars):
    """添加百分数标签"""
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2,
            height + 0.015,
            f'{height*100:.2f}%',
            ha='center',
            va='bottom',
            fontsize=11
        )

# -------- 基线模型 --------
ax = axes[0]
bars1 = ax.bar(x - width, baseline_low,  width, label='低频',  color=color_low)
bars2 = ax.bar(x,          baseline_mid,  width, label='中频',  color=color_mid)
bars3 = ax.bar(x + width, baseline_high, width, label='高频', color=color_high)

add_labels(ax, bars1)
add_labels(ax, bars2)
add_labels(ax, bars3)

ax.set_title('MedNeXt', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(stages, fontsize=13)
ax.set_ylabel('能量占比', fontsize=14)
ax.set_ylim(0, 1.08)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# -------- 改进模型 --------
ax = axes[1]
bars1 = ax.bar(x - width, improved_low,  width, color=color_low)
bars2 = ax.bar(x,          improved_mid,  width, color=color_mid)
bars3 = ax.bar(x + width, improved_high, width, color=color_high)

add_labels(ax, bars1)
add_labels(ax, bars2)
add_labels(ax, bars3)

ax.set_title('FD-RSM-NeXt', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(stages, fontsize=13)
ax.grid(axis='y', linestyle='--', alpha=0.3)

# =========================
# 图例
# =========================
fig.legend(['低频', '中频', '高频'],
           loc='upper center',
           ncol=3,
           fontsize=12,
           frameon=True)

# =========================
# 布局
# =========================
plt.tight_layout(rect=[0, 0, 1, 0.92])

# 保存
plt.savefig('./feature_vis_output/频域能量对比图.png', dpi=300, bbox_inches='tight')
# plt.savefig('频域能量对比图.pdf', bbox_inches='tight')

plt.show()

