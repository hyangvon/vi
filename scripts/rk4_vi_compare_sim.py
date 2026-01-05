#!/usr/bin/env python3
import subprocess
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 绘图统一风格与辅助函数
FIGSIZE = (10, 5)
DEFAULT_DPI = 200

# 字体与样式统一设置
FONT_FAMILY = 'DejaVu Sans'
TITLE_FONT_SIZE = 14
LABEL_FONT_SIZE = 12
LEGEND_FONT_SIZE = 12
TICK_FONT_SIZE = 11
TITLE_FONT_WEIGHT = 'bold'

def _init_fig(figsize=None):
    if figsize is None:
        figsize = FIGSIZE
    # 通过 rcParams 统一字体、标题和图例样式
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
    plt.figure(figsize=figsize)

def _save_fig(tag, filename, dpi, show=True):
    save_dir = os.path.expanduser(f"~/ros2_ws/dynamic_ws/src/vi/fig/{tag}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    print(f"Saved: {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def run_ctsvi():
    """运行 C++ 仿真节点"""
    config_file = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi/config/vi_params.yaml')

    print("Starting ctsvi simulation...")

    try:
        result = subprocess.run([
            'ros2', 'run', 'vi', 'ctsvi_ad_node',
            '--ros-args', '--params-file', config_file
        ], check=True)

        print("Simulation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Simulation failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def run_atsvi():
    """运行 C++ 仿真节点"""
    config_file = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi/config/vi_params.yaml')

    print("Starting atsvi simulation...")

    try:
        result = subprocess.run([
            'ros2', 'run', 'vi', 'atsvi_ad_node',
            '--ros-args', '--params-file', config_file
        ], check=True)

        print("Simulation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Simulation failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def run_etsvi():
    """运行 C++ 仿真节点"""
    config_file = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi/config/vi_params.yaml')

    print("Starting etsvi simulation...")

    try:
        result = subprocess.run([
            'ros2', 'run', 'vi', 'etsvi_node',
            '--ros-args', '--params-file', config_file
        ], check=True)

        print("Simulation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Simulation failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def run_rk4():
    """运行 RK4 仿真节点"""
    config_file = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi/config/vi_params.yaml')

    print("Starting rk4 simulation...")

    try:
        result = subprocess.run([
            'ros2', 'run', 'vi', 'rk4_node',
            '--ros-args', '--params-file', config_file
        ], check=True)

        print("Simulation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Simulation failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def plot_runtime_comparison(tag, dpi_set):
    """绘制ETSVI和RK4的平均运行时间对比图"""
    print("Generating runtime comparison plot...")
    base = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi/csv/')

    # 定义算法和对应的文件路径
    algorithms = {
        'etsvi': 'etsvi/avg_runtime.txt',
        'rk4': 'rk4/avg_runtime.txt'
    }

    avg_times = []
    labels = []

    for alg_name, alg_file in algorithms.items():
        try:
            with open(os.path.join(base, alg_file), 'r') as f:
                avg_time = float(f.read().strip())
                avg_times.append(avg_time)
                labels.append(alg_name.upper())
        except FileNotFoundError:
            print(f"Warning: {alg_file} not found, using default value 0")
            avg_times.append(0)
            labels.append(alg_name.upper())

    # 绘制柱状图（使用统一风格）
    _init_fig(figsize=(10, 6))
    bars = plt.bar(labels, avg_times,
                   color=['lightcoral', 'lightyellow'],
                   alpha=0.7,
                   edgecolor='black',
                   linewidth=1)

    # 在柱子上添加数值标签
    max_val = max(avg_times) if avg_times else 0
    for bar, time_val in zip(bars, avg_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_val*0.01,
                 f'{time_val:.3f}ms',
                 ha='center',
                 va='bottom',
                 fontsize=12,
                 fontweight='bold')

    plt.xlabel('Algorithms', fontsize=12)
    plt.ylabel('Average Runtime (ms)', fontsize=12)
    plt.title('Average Runtime Comparison', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    # 保存并显示（统一）
    filename = f"runtime_comparison_{tag}.png"
    _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=True)

    # 显示数值
    for label, time_val in zip(labels, avg_times):
        print(f"{label}: {time_val:.3f}ms")

def plot_results(tag, dpi_set):
    """绘制仿真结果（包含 RK4）"""
    print("Generating plots...")

    # ---------- 1. 读取数据 ----------
    base = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi/csv/')

    csv_dir_etsvi = os.path.join(base, 'etsvi')
    csv_dir_rk4   = os.path.join(base, 'rk4')

    if not os.path.exists(os.path.join(csv_dir_rk4, 'q_history.csv')):
        print("CSV files not found. Simulation may have failed.")
        return False

    try:
        tcp_etsvi = np.loadtxt(os.path.join(csv_dir_etsvi, 'ee_history.csv'), delimiter=',')
        tcp_rk4 = np.loadtxt(os.path.join(csv_dir_rk4, 'ee_history.csv'), delimiter=',')
        
        delta_energy_etsvi = np.loadtxt(os.path.join(csv_dir_etsvi, 'delta_energy_history.csv'), delimiter=',')
        delta_energy_rk4 = np.loadtxt(os.path.join(csv_dir_rk4, 'delta_energy_history.csv'), delimiter=',')
        
        time_etsvi = np.loadtxt(os.path.join(csv_dir_etsvi, 'time_history.csv'), delimiter=',')
        time_rk4 = np.loadtxt(os.path.join(csv_dir_rk4, 'time_history.csv'), delimiter=',')
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return False

    # ---------- 绘制能量曲线 ----------
    _init_fig()

    # ETSVI
    plt.plot(time_etsvi, delta_energy_etsvi, label='ΔEnergy_etsvi', linestyle='--', linewidth=2)

    # RK4
    plt.plot(time_rk4, delta_energy_rk4, label='ΔEnergy_rk4', linestyle=':', linewidth=2)

    plt.xlabel('Time [s]')
    plt.ylabel('Energy [J]')
    plt.title('Energy evolution')
    plt.legend(loc='upper left')
    plt.grid(True)
    # plt.ylim(-0.05, 0.05)
    filename = f"energy_{tag}.png"
    _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=False)

    # ---------- 绘制时间步 ----------
    
    # ---------- TCP 位置曲线 ----------
    _init_fig()
    # ETSVI
    plt.plot(time_etsvi, tcp_etsvi[:, 0], label='px_etsvi', linestyle='--', linewidth=2)
    plt.plot(time_etsvi, tcp_etsvi[:, 2], label='pz_etsvi', linestyle='--', linewidth=2)

    # RK4
    plt.plot(time_rk4, tcp_rk4[:, 0], label='px_rk4', linestyle=':', linewidth=2)
    plt.plot(time_rk4, tcp_rk4[:, 2], label='pz_rk4', linestyle=':', linewidth=2)

    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('TCP Position')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.ylim(-8, 6)
    filename = f"tcp_{tag}.png"
    _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=False)

    plot_runtime_comparison(tag, dpi_set)

    return 0

def main():
    """主函数"""
    # 运行仿真
    # if not run_etsvi():
    #     return 1

    # if not run_rk4():
    #     return 1

    # 绘制结果
    if not plot_results("etsvi_rk4_compare", 1000):
        return 1

    print("All tasks completed successfully!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
