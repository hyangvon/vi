#!/usr/bin/env python3
import subprocess
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

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

# def run_etsvi_op():
#     """运行 C++ 仿真节点"""
#     config_file = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi/config/vi_params.yaml')
#
#     print("Starting etsvi_op simulation...")
#
#     try:
#         result = subprocess.run([
#             'ros2', 'run', 'vi', 'etsvi_op_node',
#             '--ros-args', '--params-file', config_file
#         ], check=True)
#
#         print("Simulation completed successfully")
#         return True
#     except subprocess.CalledProcessError as e:
#         print(f"Simulation failed: {e}")
#         if e.stderr:
#             print(f"Error output: {e.stderr}")
#         return False

def plot_runtime_comparison(tag, dpi_set):
    """绘制三种算法的平均运行时间对比图"""
    print("Generating runtime comparison plot...")

    base = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi/csv/')

    # 定义算法和对应的文件路径
    algorithms = {
        'ctsvi_ad': 'ctsvi_ad/avg_runtime.txt',
        'atsvi_ad': 'atsvi_ad/avg_runtime.txt',
        'etsvi': 'etsvi/avg_runtime.txt'
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

    # 绘制柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, avg_times,
                   color=['skyblue', 'lightgreen', 'lightcoral'],
                   alpha=0.7,
                   edgecolor='black',
                   linewidth=1)

    # 在柱子上添加数值标签
    for bar, time_val in zip(bars, avg_times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(avg_times)*0.01,
                 f'{time_val:.3f}ms',
                 ha='center',
                 va='bottom',
                 fontsize=12,
                 fontweight='bold')

    plt.xlabel('Algorithms', fontsize=12)
    plt.ylabel('Average Runtime (ms)', fontsize=12)
    plt.title('Average Runtime Comparison', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    # 保存图片
    save_dir = os.path.expanduser(f"~/ros2_ws/dynamic_ws/src/vi/fig/{tag}")
    os.makedirs(save_dir, exist_ok=True)
    filename = f"runtime_comparison_{tag}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=dpi_set)
    print("Saved runtime comparison:", save_path)

    # 显示数值
    for label, time_val in zip(labels, avg_times):
        print(f"{label}: {time_val:.3f}ms")

    plt.show()

def plot_results(tag, dpi_set):
    """绘制仿真结果"""
    print("Generating plots...")

    # ---------- 1. 读取数据 ----------
    base = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi/csv/')

    csv_dir_ctsvi_ad = os.path.join(base, 'ctsvi_ad')
    csv_dir_atsvi_ad = os.path.join(base, 'atsvi_ad')
    csv_dir_etsvi    = os.path.join(base, 'etsvi')

    if not os.path.exists(os.path.join(csv_dir_atsvi_ad, 'q_history.csv')):
        print("CSV files not found. Simulation may have failed.")
        return False

    try:
        # q_history = np.loadtxt(os.path.join(csv_dir, 'q_history.csv'), delimiter=',')
        # if q_history.ndim == 1:
        #     q_history = q_history.reshape(-1, 1)
        tcp_ctsvi = np.loadtxt(os.path.join(csv_dir_ctsvi_ad, 'ee_history.csv'), delimiter=',')
        tcp_atsvi = np.loadtxt(os.path.join(csv_dir_atsvi_ad, 'ee_history.csv'), delimiter=',')
        tcp_etsvi = np.loadtxt(os.path.join(csv_dir_etsvi, 'ee_history.csv'), delimiter=',')
        # energy = np.loadtxt(os.path.join(csv_dir, 'energy_history.csv'), delimiter=',')
        delta_energy_ctsvi = np.loadtxt(os.path.join(csv_dir_ctsvi_ad, 'delta_energy_history.csv'), delimiter=',')
        delta_energy_atsvi = np.loadtxt(os.path.join(csv_dir_atsvi_ad, 'delta_energy_history.csv'), delimiter=',')
        delta_energy_etsvi = np.loadtxt(os.path.join(csv_dir_etsvi, 'delta_energy_history.csv'), delimiter=',')
        time_ctsvi = np.loadtxt(os.path.join(csv_dir_ctsvi_ad, 'time_history.csv'), delimiter=',')
        time_atsvi = np.loadtxt(os.path.join(csv_dir_atsvi_ad, 'time_history.csv'), delimiter=',')
        time_etsvi = np.loadtxt(os.path.join(csv_dir_etsvi, 'time_history.csv'), delimiter=',')
        # momentum = np.loadtxt(os.path.join(csv_dir, 'momentum_history.csv'), delimiter=',')
        step_atsvi = np.loadtxt(os.path.join(csv_dir_atsvi_ad, 'h_history.csv'), delimiter=',')
        step_etsvi = np.loadtxt(os.path.join(csv_dir_etsvi, 'h_history.csv'), delimiter=',')
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return False

    # print("q_history shape:", q_history.shape)

    # 自动检测自由度数
    # nq = q_history.shape[1]
    # print(f"Loaded q_history with {nq} DOFs, {q_history.shape[0]} time steps")

    # 创建图形窗口
    # plt.ion()  # 开启交互模式

    save_dir = os.path.expanduser(f"~/ros2_ws/dynamic_ws/src/vi/fig/{tag}")
    os.makedirs(save_dir, exist_ok=True)   # 自动创建目录

    # # ---------- 2. 绘制关节角随时间 ----------
    # plt.figure(figsize=(10, 5))
    # for i in range(nq):
    #     plt.plot(time, q_history[0:, i], label=f'q{i+1}')
    # plt.xlabel('Time [s]')
    # plt.ylabel('Joint angle [rad]')
    # plt.title('Joint trajectories')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # filename = f"q_{tag}.png"
    # save_path = os.path.join(save_dir, filename)
    # plt.savefig(save_path, dpi = dpi_set)
    # print("Saved:", save_path)
    # # plt.show()
    #
    # ---------- 3. 绘制能量曲线 ----------
    plt.figure(figsize=(10, 5))
    # plt.plot(time, energy, label='Total Energy')

    # CTSVI
    plt.plot(time_ctsvi, delta_energy_ctsvi, label='ΔEnergy_ctsvi', linestyle='-', linewidth=2)

    # ATSVI
    plt.plot(time_atsvi, delta_energy_atsvi, label='ΔEnergy_atsvi', linestyle='-.', linewidth=2)

    # ETSVI
    plt.plot(time_etsvi, delta_energy_etsvi, label='ΔEnergy_etsvi', linestyle='--', linewidth=2)

    plt.xlabel('Time [s]')
    plt.ylabel('Energy [J]')
    plt.title('Energy evolution')
    plt.legend()
    plt.grid(True)
    # plt.ylim(-0.05, 0.05)
    plt.tight_layout()
    filename = f"energy_{tag}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi = dpi_set)
    print("Saved:", save_path)
    # plt.show()
    #
    # # ---------- 4. 相平面图（q vs qdot） ----------
    # dt = time[1] - time[0]
    # qdot = np.diff(q_history, axis=0) / dt  # numerical derivative
    # plt.figure(figsize=(6, 6))
    # for i in range(nq):
    #     plt.plot(q_history[:-1, i], qdot[:, i], label=f'Joint {i+1}')
    # plt.xlabel('q [rad]')
    # plt.ylabel('qdot [rad/s]')
    # plt.title('Phase Portraits')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # filename = f"phase_{tag}.png"
    # save_path = os.path.join(save_dir, filename)
    # plt.savefig(save_path, dpi = dpi_set)
    # print("Saved:", save_path)
    # # plt.show()

    # ---------- 5. 绘制timestep ----------
    plt.figure(figsize=(10, 5))
    plt.plot(time_atsvi, step_atsvi, label='Time Step of atsvi', linestyle='-.', linewidth=2)
    plt.plot(time_etsvi, step_etsvi, label='Time Step of etsvi', linestyle='--', linewidth=2)
    # plt.plot(time, delta_energy, label='ΔEnergy (relative to initial)')
    plt.xlabel('Time [s]')
    plt.ylabel('Step')
    plt.title('Adaptive Time Step')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f"step_{tag}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi = dpi_set)
    print("Saved:", save_path)
    # plt.show()

    # ---------- 5. position xyz ----------
    plt.figure(figsize=(10, 5))

    # CTSVI
    plt.plot(time_ctsvi, tcp_ctsvi[:, 0], label='px_ctsvi', linestyle='-', linewidth=2)
    plt.plot(time_ctsvi, tcp_ctsvi[:, 2], label='pz_ctsvi', linestyle='-', linewidth=2)

    # ATSVI
    plt.plot(time_atsvi, tcp_atsvi[:, 0], label='px_atsvi', linestyle='-.', linewidth=2)
    plt.plot(time_atsvi, tcp_atsvi[:, 2], label='pz_atsvi', linestyle='-.', linewidth=2)

    # ETSVI
    plt.plot(time_etsvi, tcp_etsvi[:, 0], label='px_etsvi', linestyle='--', linewidth=2)
    plt.plot(time_etsvi, tcp_etsvi[:, 2], label='pz_etsvi', linestyle='--', linewidth=2)

    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('TCP Position')
    plt.legend()
    plt.grid(True)
    plt.ylim(-8, 6)
    plt.tight_layout()
    filename = f"tcp_{tag}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi = dpi_set)
    print("Saved:", save_path)
    # plt.show()

    plot_runtime_comparison(tag, dpi_set)

    # print("Plotting completed. Close all plot windows to exit.")

    # 等待用户关闭窗口
    # plt.ioff()  # 关闭交互模式
    # plt.show()  # 阻塞直到所有窗口关闭

    return 0

def main():
    """主函数"""
    # 运行仿真
    if not run_ctsvi():
        return 1

    if not run_atsvi():
        return 1

    if not run_etsvi():
        return 1

    # if not run_etsvi_op():
    #     return 1

    # 绘制结果
    if not plot_results("compare", 1000):
        return 1

    print("All tasks completed successfully!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
