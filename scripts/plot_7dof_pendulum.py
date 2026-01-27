#!/usr/bin/env python3
"""
2D 绘制 7 自由度摆（X-Z 平面），参考用户提供的可视化样式
- 零位姿态为竖直向下，初始每个关节角均为 0.2 rad
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# 字体与样式统一设置
FONT_FAMILY = 'DejaVu Sans'
TITLE_FONT_SIZE = 14
LABEL_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12
TICK_FONT_SIZE = 11
TITLE_FONT_WEIGHT = 'bold'

DEFAULT_DPI = 200

def _apply_style():
    plt.rcParams.update({
        'font.family': FONT_FAMILY,
        'axes.titlesize': TITLE_FONT_SIZE,
        'axes.titleweight': TITLE_FONT_WEIGHT,
        'axes.labelsize': LABEL_FONT_SIZE,
        'legend.fontsize': LEGEND_FONT_SIZE,
        'xtick.labelsize': TICK_FONT_SIZE,
        'ytick.labelsize': TICK_FONT_SIZE,
        'legend.frameon': True,
    })

def compute_fk_2d(q, link_length=1.0):
    """解析计算 2D 正向运动学（X-Z 平面），基座在 (0,0)，向下为负 Z。"""
    n_links = len(q)
    x = [0.0]
    z = [0.0]
    current_angle = 0.0
    for i in range(n_links):
        current_angle += q[i]
        next_x = x[-1] + link_length * np.sin(current_angle)
        next_z = z[-1] - link_length * np.cos(current_angle)
        x.append(next_x)
        z.append(next_z)
    return np.array(x), np.array(z)


def plot_7dof_chain(x, z, save_path=None, title='7-DoF Pendulum Initial Configuration'):
    _apply_style()
    fig, ax = plt.subplots(figsize=(6, 8))

    # 1. 连杆
    ax.plot(x, z, 'o-', linewidth=3, color='#34495e', markersize=8, zorder=1, label='Links')

    # 2. 关节（排除基座和末端以示区分）
    if len(x) > 2:
        ax.scatter(x[1:-1], z[1:-1], s=100, c='#e74c3c', zorder=2, label='Joints')

    # 3. 基座
    ax.scatter(x[0], z[0], s=200, marker='^', c='black', zorder=3, label='Base (Fixed)')

    # 4. 末端执行器
    ax.scatter(x[-1], z[-1], s=150, c='#2ecc71', marker='*', zorder=3, label='End-Tip')

    # 5. 零位参考线（竖直向下）
    n_links = len(x)-1
    ax.plot([0, 0], [0, -n_links*1.0], '--', color='gray', alpha=0.5, label='Zero Configuration')

    # 注释与美化
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Z Position (m)', fontsize=12)
    ax.set_title(f"{title}\n($q_i = 0.2$ rad for all joints)", fontsize=TITLE_FONT_SIZE, fontweight=TITLE_FONT_WEIGHT)
    ax.legend()

    # 累积弯曲注释（示例）
    # mid_idx = min(3, len(x)-1)
    # ax.text(x[mid_idx]+0.2, z[mid_idx], "Cumulative Curvature", fontsize=10, color='#34495e', style='italic')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=DEFAULT_DPI)
        print('Saved:', save_path)
    plt.show()


def main():
    # 参数
    n_links = 7
    link_length = 1.0
    q = np.array([0.2] * n_links)

    x, z = compute_fk_2d(q, link_length=link_length)
    fig_path = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi/fig/model/pendulum_7dof_2d.png')
    plot_7dof_chain(x, z, save_path=fig_path)


if __name__ == '__main__':
    main()
