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

    # 绘制柱状图（使用统一风格）
    _init_fig(figsize=(10, 6))
    bars = plt.bar(labels, avg_times,
                   color=['skyblue', 'lightgreen', 'lightcoral'],
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


def plot_runtime_vs_energy(tag, dpi_set):
    """以平均运行时间为横轴，平均能量误差为纵轴，绘制三种方法对比图并保存。"""
    print("Generating runtime vs energy plot...")
    base = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi/csv/')

    methods = {
        'CTSVI': {
            'runtime': os.path.join(base, 'ctsvi_ad', 'avg_runtime.txt'),
            'energy': os.path.join(base, 'ctsvi_ad', 'delta_energy_history.csv')
        },
        'ATSVI': {
            'runtime': os.path.join(base, 'atsvi_ad', 'avg_runtime.txt'),
            'energy': os.path.join(base, 'atsvi_ad', 'delta_energy_history.csv')
        },
        'C-ATSVI': {
            'runtime': os.path.join(base, 'etsvi', 'avg_runtime.txt'),
            'energy': os.path.join(base, 'etsvi', 'delta_energy_history.csv')
        }
    }

    xs = []
    ys = []
    labels = []
    for name, paths in methods.items():
        # read runtime
        rt = float('nan')
        if os.path.exists(paths['runtime']):
            try:
                with open(paths['runtime'], 'r') as f:
                    rt = float(f.read().strip())
            except Exception:
                rt = float('nan')

        # read energy error mean abs
        mae = float('nan')
        if os.path.exists(paths['energy']):
            try:
                arr = np.loadtxt(paths['energy'], delimiter=',')
                arr = np.atleast_1d(arr).astype(float)
                mae = float(np.mean(np.abs(arr)))
            except Exception:
                mae = float('nan')

        xs.append(rt)
        ys.append(mae)
        labels.append(name)

    # 绘图
    _init_fig(figsize=(6, 6))
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(len(labels))]
    for x, y, lbl, c in zip(xs, ys, labels, colors):
        plt.scatter(x, y, label=lbl, color=c, s=120)
        plt.text(x, y, f' {lbl}', verticalalignment='center', fontsize=11)

    plt.xlabel('Average Runtime (ms)')
    plt.ylabel('Mean Absolute Energy Error (J)')
    plt.title('Runtime vs Energy Error')
    plt.grid(True, alpha=0.3)

    filename = f"runtime_vs_energy_{tag}.png"
    _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=False)


def compute_and_save_energy_errors(tag):
    """计算 CTSVI/ATSVI/ETSVI 的能量误差平均值并保存为 CSV，同时打印到终端。
    输出保存到: src/vi/csv/<tag>/energy_error_summary_<tag>.csv
    """
    base = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi/csv/')
    methods = {
        'CTSVI_AD': os.path.join(base, 'ctsvi_ad', 'delta_energy_history.csv'),
        'ATSVI_AD': os.path.join(base, 'atsvi_ad', 'delta_energy_history.csv'),
        'ETSVI': os.path.join(base, 'etsvi', 'delta_energy_history.csv'),
    }

    results = []
    for name, path in methods.items():
        if os.path.exists(path):
            try:
                arr = np.loadtxt(path, delimiter=',')
                arr = np.atleast_1d(arr).astype(float)
                mean_abs = float(np.mean(np.abs(arr)))
                rms = float(np.sqrt(np.mean(arr**2)))
                n = int(arr.size)
            except Exception as e:
                print(f"Error reading {path}: {e}")
                mean_abs = float('nan')
                rms = float('nan')
                n = 0
        else:
            mean_abs = float('nan')
            rms = float('nan')
            n = 0
        results.append((name, mean_abs, rms, n))

    # 打印到终端
    print("Energy error summary:")
    for name, mean_abs, rms, n in results:
        print(f"- {name}: samples={n}, mean_abs_error={mean_abs:.6e}, rms={rms:.6e}")

    # 保存 CSV 到子目录 tag
    out_dir = os.path.join(base, tag)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'energy_error_summary_{tag}.csv')
    with open(out_path, 'w') as f:
        f.write('method,mean_abs_error,rms_error,samples\n')
        for name, mean_abs, rms, n in results:
            f.write(f"{name},{mean_abs},{rms},{n}\n")

    print(f"Saved energy error summary to {out_path}")
    return out_path

def plot_7dof_pendulum():
    # 参数设置
    n_links = 7
    link_length = 1.0  # 根据 URDF 文件，joint 到 joint 距离为 1.0m
    
    # 初始关节角 (每个关节都是 0.2 rad)
    q = np.array([0.2] * n_links)
    
    # 运动学正解 (Forward Kinematics) - 仅计算 X-Z 平面
    # 假设基座在 (0, 0)
    x = [0.0]
    z = [0.0]
    
    current_angle = 0.0
    
    # 循环计算每个关节的位置
    for i in range(n_links):
        # 累加角度（相对角度 -> 绝对角度）
        # 0位姿是竖直向下 (-Z)，绕 Y 轴正向旋转会将连杆向 +X 方向摆动
        current_angle += q[i]
        
        # 计算下一个关节的位置
        # x = L * sin(theta)
        # z = -L * cos(theta) (因为基准是向下)
        next_x = x[-1] + link_length * np.sin(current_angle)
        next_z = z[-1] - link_length * np.cos(current_angle)
        
        x.append(next_x)
        z.append(next_z)
        
    # --- 绘图 ---
    fig, ax = plt.subplots(figsize=(6, 8))
    
    # 1. 画连杆 (实线)
    ax.plot(x, z, 'o-', linewidth=3, color='#34495e', markersize=8, zorder=1, label='Links')
    
    # 2. 画关节 (红色圆点)
    ax.scatter(x[1:-1], z[1:-1], s=100, c='#e74c3c', zorder=2, label='Joints')
    
    # 3. 画基座 (固定点)
    ax.scatter(x[0], z[0], s=200, marker='^', c='black', zorder=3, label='Base (Fixed)')
    
    # 4. 画末端执行器 (TCP)
    ax.scatter(x[-1], z[-1], s=150, c='#2ecc71', marker='*', zorder=3, label='End Effector')
    
    # 5. 画零位参考线 (虚线)
    ax.plot([0, 0], [0, -n_links*link_length], '--', color='gray', alpha=0.5, label='Zero Configuration')

    # 标注和美化
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Z Position (m)', fontsize=12)
    ax.set_title(f'7-DoF Pendulum Initial Configuration\n($q_i = 0.2$ rad for all joints)', fontsize=14)
    ax.legend()
    
    # 添加角度累积的注释
    ax.text(x[3]+0.2, z[3], "Cumulative Curvature", fontsize=10, color='#34495e', style='italic')

    # 调整视野
    plt.tight_layout()
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

    # 输出目录将在保存时由 _save_fig 创建

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
    _init_fig()
    # plt.plot(time, energy, label='Total Energy')

    # 使用调色板和更明显的样式以增强可区分度
    cmap = plt.get_cmap('tab10')
    c_ctsvi = cmap(0)
    c_atsvi = cmap(1)
    c_etsvi = cmap(2)


    # CTSVI
    plt.plot(time_ctsvi, delta_energy_ctsvi, label='ΔEnergy of CTSVI', color=c_ctsvi, linestyle='-', linewidth=2)

    # ATSVI
    plt.plot(time_atsvi, delta_energy_atsvi, label='ΔEnergy of ATSVI', color=c_atsvi, linestyle='-.', linewidth=2)

    # ETSVI
    plt.plot(time_etsvi, delta_energy_etsvi, label='ΔEnergy of C-ATSVI', color=c_etsvi, linestyle='--', linewidth=2)

    plt.xlabel('Time [s]')
    plt.ylabel('Energy [J]')
    plt.title('Energy evolution')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.ylim(-0.015, 0.015)
    filename = f"energy_{tag}.png"
    _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=False)
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
    _init_fig()
    plt.plot(time_atsvi, step_atsvi, label='Time Step of atsvi', linestyle='-.', linewidth=2)
    plt.plot(time_etsvi, step_etsvi, label='Time Step of etsvi', linestyle='--', linewidth=2)
    plt.xlabel('Time [s]')
    plt.ylabel('Step')
    plt.title('Adaptive Time Step')
    plt.legend()
    plt.grid(True)
    filename = f"step_{tag}.png"
    _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=False)

    # ---------- 5. position xyz ----------
    _init_fig()
    # CTSVI
    # plt.plot(time_ctsvi, tcp_ctsvi[:, 0], label='px_ctsvi', linestyle='-', linewidth=2)
    plt.plot(time_ctsvi, tcp_ctsvi[:, 2], label='position Z of CTSVI', color=c_ctsvi, linestyle='-', linewidth=2)

    # ATSVI
    # plt.plot(time_atsvi, tcp_atsvi[:, 0], label='px_atsvi', linestyle='-.', linewidth=2)
    plt.plot(time_atsvi, tcp_atsvi[:, 2], label='position Z of ATSVI', color=c_atsvi, linestyle='-.', linewidth=2)
    
    # ETSVI
    # plt.plot(time_etsvi, tcp_etsvi[:, 0], label='px_etsvi', linestyle='--', linewidth=2)
    plt.plot(time_etsvi, tcp_etsvi[:, 2], label='position Z of C-ATSVI', color=c_etsvi, linestyle='--', linewidth=2)

    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('TCP Position')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.ylim(-8, -3)
    filename = f"tcp_{tag}.png"
    _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=False)

    plot_runtime_comparison(tag, dpi_set)

    # 计算并保存三种 VI 方法的能量误差平均值
    compute_and_save_energy_errors(tag)

    # 绘制 平均运行时间 vs 平均能量误差
    plot_runtime_vs_energy(tag, dpi_set)

    # print("Plotting completed. Close all plot windows to exit.")

    # 等待用户关闭窗口
    # plt.ioff()  # 关闭交互模式
    # plt.show()  # 阻塞直到所有窗口关闭

    plot_7dof_pendulum()

    return 1

def main():
    """主函数"""
    # 运行仿真
    # if not run_ctsvi():
    #     return 1

    # if not run_atsvi():
    #     return 1

    # if not run_etsvi():
    #     return 1

    # if not run_etsvi_op():
    #     return 1

    # 绘制结果
    if not plot_results("compare", 1000):
        return 1

    print("All tasks completed successfully!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
