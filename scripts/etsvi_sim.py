#!/usr/bin/env python3
# atsvi_sim.py
import subprocess
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

def run_simulation():
    """运行 C++ 仿真节点"""
    config_file = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi/config/vi_params.yaml')

    print("Starting estvi simulation...")
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

def plot_results(tag, dpi_set):
    """绘制仿真结果"""
    print("Generating plots...")

    # ---------- 1. 读取数据 ----------
    csv_dir = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi/csv/etsvi/')

    if not os.path.exists(os.path.join(csv_dir, 'q_history.csv')):
        print("CSV files not found. Simulation may have failed.")
        return False

    try:
        q_history = np.loadtxt(os.path.join(csv_dir, 'q_history.csv'), delimiter=',')
        if q_history.ndim == 1:
            q_history = q_history.reshape(-1, 1)
        # energy = np.loadtxt(os.path.join(csv_dir, 'energy_history.csv'), delimiter=',')
        delta_energy = np.loadtxt(os.path.join(csv_dir, 'delta_energy_history.csv'), delimiter=',')
        time = np.loadtxt(os.path.join(csv_dir, 'time_history.csv'), delimiter=',')
        # momentum = np.loadtxt(os.path.join(csv_dir, 'momentum_history.csv'), delimiter=',')
        step = np.loadtxt(os.path.join(csv_dir, 'h_history.csv'), delimiter=',')
        tcp = np.loadtxt(os.path.join(csv_dir, 'ee_history.csv'), delimiter=',')
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return False

    print("q_history shape:", q_history.shape)

    # 自动检测自由度数
    nq = q_history.shape[1]
    print(f"Loaded q_history with {nq} DOFs, {q_history.shape[0]} time steps")

    # 创建图形窗口
    # plt.ion()  # 开启交互模式

    save_dir = os.path.expanduser(f"~/ros2_ws/dynamic_ws/src/vi/fig/{tag}")
    os.makedirs(save_dir, exist_ok=True)   # 自动创建目录

    # ---------- 2. 绘制关节角随时间 ----------
    plt.figure(figsize=(10, 5))
    for i in range(nq):
        plt.plot(time, q_history[0:, i], label=f'q{i+1}')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint angle [rad]')
    plt.title('Joint trajectories')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f"q_{tag}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi = dpi_set)
    print("Saved:", save_path)
    # plt.show()

    # ---------- 3. 绘制能量曲线 ----------
    plt.figure(figsize=(10, 5))
    # plt.plot(time, energy, label='Total Energy')
    plt.plot(time, delta_energy, label='ΔEnergy (relative to initial)')
    plt.xlabel('Time [s]')
    plt.ylabel('Energy [J]')
    plt.title('Energy evolution')
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.05, 0.05)
    plt.tight_layout()
    filename = f"energy_{tag}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi = dpi_set)
    print("Saved:", save_path)
    # plt.show()

    # ---------- 4. 相平面图（q vs qdot） ----------
    dt = time[1] - time[0]
    qdot = np.diff(q_history, axis=0) / dt  # numerical derivative
    plt.figure(figsize=(6, 6))
    for i in range(nq):
        plt.plot(q_history[:-1, i], qdot[:, i], label=f'Joint {i+1}')
    plt.xlabel('q [rad]')
    plt.ylabel('qdot [rad/s]')
    plt.title('Phase Portraits')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f"phase_{tag}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi = dpi_set)
    print("Saved:", save_path)
    # plt.show()

    # ---------- 5. 绘制timestep ----------
    plt.figure(figsize=(10, 5))
    # plt.plot(time[1:], step, label='Time Step')
    plt.plot(time, step, label='Time Step')
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

    # ---------- position xyz ----------
    plt.figure(figsize=(10, 5))

    # ETSVI
    plt.plot(time, tcp[:, 0], label='px_etsvi', linestyle='--', linewidth=2)
    plt.plot(time, tcp[:, 2], label='pz_etsvi', linestyle='--', linewidth=2)

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

    print("Plotting completed. Close all plot windows to exit.")

    # 等待用户关闭窗口
    plt.ioff()  # 关闭交互模式
    plt.show()  # 阻塞直到所有窗口关闭

    return 0

def main():
    """主函数"""
    # 运行仿真
    if not run_simulation():
        return 1

    # 绘制结果
    if not plot_results("etsvi", 1000):
        return 1

    print("All tasks completed successfully!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
