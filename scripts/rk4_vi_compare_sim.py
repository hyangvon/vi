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


def run_pybullet_inline():
    """在当前 Python 进程中加载并运行 scripts/pybullet_sim.py 的 main()，便于一键生成 CSV 数据。"""
    script_path = os.path.join(os.path.dirname(__file__), 'pybullet_sim.py')
    if not os.path.exists(script_path):
        print(f"pybullet script not found: {script_path}")
        return False

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('pybullet_sim', script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, 'main'):
            print('Running pybullet_sim.main()...')
            module.main()
            return True
        else:
            print('pybullet_sim.py does not define main()')
            return False
    except Exception as e:
        print(f'Error running pybullet inline: {e}')
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
    csv_dir_py    = os.path.join(base, 'pybullet')

    if not os.path.exists(os.path.join(csv_dir_rk4, 'q_history.csv')):
        print("CSV files not found. Simulation may have failed.")
        return False

    # 读取 CSV（支持可选的 pybullet 目录）
    try:
        tcp_etsvi = np.loadtxt(os.path.join(csv_dir_etsvi, 'ee_history.csv'), delimiter=',')
        tcp_rk4 = np.loadtxt(os.path.join(csv_dir_rk4, 'ee_history.csv'), delimiter=',')

        delta_energy_etsvi = np.loadtxt(os.path.join(csv_dir_etsvi, 'delta_energy_history.csv'), delimiter=',')
        delta_energy_rk4 = np.loadtxt(os.path.join(csv_dir_rk4, 'delta_energy_history.csv'), delimiter=',')

        time_etsvi = np.loadtxt(os.path.join(csv_dir_etsvi, 'time_history.csv'), delimiter=',')
        time_rk4 = np.loadtxt(os.path.join(csv_dir_rk4, 'time_history.csv'), delimiter=',')

        # optional pybullet
        tcp_py = None
        delta_energy_py = None
        time_py = None
        if os.path.exists(os.path.join(csv_dir_py, 'ee_history.csv')):
            tcp_py = np.loadtxt(os.path.join(csv_dir_py, 'ee_history.csv'), delimiter=',')
        if os.path.exists(os.path.join(csv_dir_py, 'delta_energy_history.csv')):
            delta_energy_py = np.loadtxt(os.path.join(csv_dir_py, 'delta_energy_history.csv'), delimiter=',')
        if os.path.exists(os.path.join(csv_dir_py, 'time_history.csv')):
            time_py = np.loadtxt(os.path.join(csv_dir_py, 'time_history.csv'), delimiter=',')
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return False

    # ---------- 绘制能量曲线 ----------
    _init_fig()
    # 使用调色板和更明显的样式以增强可区分度
    cmap = plt.get_cmap('tab10')
    c_rk4 = cmap(0)
    c_py = cmap(1)
    c_etsvi = cmap(2)

    # 计算采样间隔以减少标记密度
    def _markevery(arr, target=50):
        try:
            n = max(1, int(len(arr) / target))
        except Exception:
            n = 1
        return n

    # RK4
    plt.plot(time_rk4, delta_energy_rk4, label='ΔEnergy of RK4', color=c_rk4,
             linestyle='-.', linewidth=2.0)

    # PyBullet (if available)
    if 'delta_energy_py' in locals() and delta_energy_py is not None and time_py is not None:
        plt.plot(time_py, delta_energy_py, label='ΔEnergy of PyBullet', color=c_py,
                 linestyle='-', linewidth=2.0)
    
    # ETSVI
    plt.plot(time_etsvi, delta_energy_etsvi, label='ΔEnergy of C-ATSVI', color=c_etsvi,
             linestyle='--', linewidth=2.0)
             
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
    # RK4
    # plt.plot(time_rk4, tcp_rk4[:, 0], label='px_rk4', color=c_rk4, linestyle=':', linewidth=1.5,
    #          marker='s', markersize=3, markevery=_markevery(time_rk4))
    plt.plot(time_rk4, tcp_rk4[:, 2], label='position Z of RK4', color=c_rk4, linestyle='-.', linewidth=2.0,
             alpha=0.9, marker=None)

    # PyBullet
    if 'tcp_py' in locals() and tcp_py is not None:
        # tcp_py may be (N,3)
        # plt.plot(time_py, tcp_py[:, 0], label='px_pybullet', color=c_py, linestyle='-.', linewidth=1.5,
        #          marker='o', markersize=3, markevery=_markevery(time_py), alpha=0.9)
        plt.plot(time_py, tcp_py[:, 2], label='position Z of PyBullet', color=c_py, linestyle='-', linewidth=2.0,
                 alpha=0.9, marker=None)

    # ETSVI (TCP position)
    # plt.plot(time_etsvi, tcp_etsvi[:, 0], label='px_etsvi', color=c_etsvi, linestyle='--', linewidth=1.5,
    #          marker='^', markersize=3, markevery=_markevery(time_etsvi))
    plt.plot(time_etsvi, tcp_etsvi[:, 2], label='position Z of C-ATSVI', color=c_etsvi, linestyle='--', linewidth=2.0,
             alpha=0.9, marker=None)

    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('TCP Position')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.ylim(-8, -3)
    filename = f"tcp_{tag}.png"
    _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=False)

    plot_runtime_comparison(tag, dpi_set)
    
    # 绘制第7关节相平面（关节编号从1开始，脚本内部使用0-based）
    plot_phase_plane("etsvi_rk4_compare", 300, joint_index=6)

    # 绘制第7关节庞加莱截面：用关节1过零上升事件触发采样
    plot_poincare_section("etsvi_rk4_compare", 300, joint_index=6, trigger_joint_index=0, surface='q=0', direction='+')

    return 1


def plot_phase_plane(tag, dpi_set, joint_index=6):
    """绘制指定关节（0-based 索引）的相平面图（q vs qdot），对比 rk4/pybullet/etsvi。"""
    print(f"Generating phase plane for joint {joint_index+1}...")
    base = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi/csv/')

    csv_dir_etsvi = os.path.join(base, 'etsvi')
    csv_dir_rk4   = os.path.join(base, 'rk4')
    csv_dir_py    = os.path.join(base, 'pybullet')

    # --- RK4: load q_history and v_history if available ---
    q_rk4 = None
    v_rk4 = None
    t_rk4 = None
    try:
        q_rk4 = np.loadtxt(os.path.join(csv_dir_rk4, 'q_history.csv'), delimiter=',')
        # ensure 2D
        if q_rk4.ndim == 1:
            q_rk4 = q_rk4[:, None]
    except Exception:
        pass

    try:
        v_rk4 = np.loadtxt(os.path.join(csv_dir_rk4, 'v_history.csv'), delimiter=',')
        if v_rk4.ndim == 1:
            v_rk4 = v_rk4[:, None]
    except Exception:
        v_rk4 = None

    try:
        t_rk4 = np.loadtxt(os.path.join(csv_dir_rk4, 'time_history.csv'), delimiter=',')
    except Exception:
        t_rk4 = None

    # --- ETSVI: load q_history and time, v estimated by diff ---
    q_etsvi = None
    t_etsvi = None
    v_etsvi = None
    try:
        q_etsvi = np.loadtxt(os.path.join(csv_dir_etsvi, 'q_history.csv'), delimiter=',')
        if q_etsvi.ndim == 1:
            q_etsvi = q_etsvi[:, None]
    except Exception:
        pass
    try:
        t_etsvi = np.loadtxt(os.path.join(csv_dir_etsvi, 'time_history.csv'), delimiter=',')
    except Exception:
        t_etsvi = None
    if q_etsvi is not None and t_etsvi is not None:
        # finite difference to estimate velocity for joint
        q_col = q_etsvi[:, joint_index] if q_etsvi.shape[1] > joint_index else None
        if q_col is not None:
            v_etsvi = np.gradient(q_col, t_etsvi)

    # --- PyBullet: load q_history and time, v estimated by diff ---
    q_py = None
    t_py = None
    v_py = None
    try:
        q_py = np.loadtxt(os.path.join(csv_dir_py, 'q_history.csv'), delimiter=',')
        if q_py.ndim == 1:
            q_py = q_py[:, None]
    except Exception:
        pass
    try:
        t_py = np.loadtxt(os.path.join(csv_dir_py, 'time_history.csv'), delimiter=',')
    except Exception:
        t_py = None
    if q_py is not None and t_py is not None:
        q_col = q_py[:, joint_index] if q_py.shape[1] > joint_index else None
        if q_col is not None:
            v_py = np.gradient(q_col, t_py)

    # --- Prepare figure ---
    _init_fig(figsize=(8, 6))

    plotted = False
    # plot rk4
    if q_rk4 is not None:
        qcol = q_rk4[:, joint_index] if q_rk4.shape[1] > joint_index else None
        if qcol is not None:
            if v_rk4 is not None and v_rk4.shape[1] > joint_index:
                vcol = v_rk4[:, joint_index]
            elif t_rk4 is not None:
                vcol = np.gradient(qcol, t_rk4)
            else:
                vcol = np.gradient(qcol)
            plt.plot(qcol, vcol, label='RK4', linewidth=1)
            plotted = True

    # plot etsvi
    if q_etsvi is not None and v_etsvi is not None:
        plt.plot(q_etsvi[:, joint_index], v_etsvi, label='ETSVI', linewidth=1)
        plotted = True

    # plot pybullet
    if q_py is not None and v_py is not None:
        plt.plot(q_py[:, joint_index], v_py, label='PyBullet', linewidth=1)
        plotted = True

    if not plotted:
        print('No data available to plot phase plane for joint', joint_index+1)
        return False

    plt.xlabel(f'Joint {joint_index+1} Position [rad]')
    plt.ylabel(f'Joint {joint_index+1} Velocity [rad/s]')
    plt.title(f'Phase Plane - Joint {joint_index+1}')
    plt.grid(True)
    plt.legend()
    filename = f'phase_plane_joint{joint_index+1}_{tag}.png'
    _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=True)

    return True


def plot_poincare_section(tag, dpi_set, joint_index=6, trigger_joint_index=0, surface='q=0', direction='+'):
    """绘制关节的庞加莱截面：用 trigger 关节的 q 过零事件作为截面，提取 target 关节的 (q,v)。
    - joint_index: 目标关节（默认关节7，0-based）
    - trigger_joint_index: 触发过零检测的关节（默认关节1，0-based）
    - direction: '+' 表示从负到正穿越，'-' 表示正到负，其它表示任意过零
    对 ETSVI / PyBullet 无速度记录时用时间序列差分估计速度。
    """
    print(f"Generating Poincaré section for joint {joint_index+1}...")
    base = os.path.expanduser('~/ros2_ws/dynamic_ws/src/vi/csv/')
    csv_dir_etsvi = os.path.join(base, 'etsvi')
    csv_dir_rk4   = os.path.join(base, 'rk4')
    csv_dir_py    = os.path.join(base, 'pybullet')

    def _load_q_v_t(csv_dir):
        q = v = t = None
        try:
            q = np.loadtxt(os.path.join(csv_dir, 'q_history.csv'), delimiter=',')
            if q.ndim == 1:
                q = q[:, None]
        except Exception:
            q = None
        try:
            v = np.loadtxt(os.path.join(csv_dir, 'v_history.csv'), delimiter=',')
            if v.ndim == 1:
                v = v[:, None]
        except Exception:
            v = None
        try:
            t = np.loadtxt(os.path.join(csv_dir, 'time_history.csv'), delimiter=',')
        except Exception:
            t = None
        if q is not None and v is None:
            # fallback: estimate v by gradient if time available
            if t is not None and q.shape[0] == t.shape[0]:
                v_est = np.gradient(q[:, joint_index] if q.shape[1] > joint_index else q[:, 0], t)
                v = np.zeros_like(q)
                v[:, joint_index if q.shape[1] > joint_index else 0] = v_est
            else:
                v = None
        return q, v, t

    def _extract_poincare(q_trigger, q_target, v_target, tarr):
        if q_trigger is None or q_target is None or v_target is None:
            return []
        if tarr is None:
            # assume unit time step if not provided
            tarr = np.arange(len(q_target))
        pts = []
        for i in range(1, len(q_target)):
            qt0, qt1 = q_trigger[i-1], q_trigger[i]
            q0, q1 = q_target[i-1], q_target[i]
            v0, v1 = v_target[i-1], v_target[i]
            if direction == '+':
                crossing = (qt0 <= 0 and qt1 >= 0)
            elif direction == '-':
                crossing = (qt0 >= 0 and qt1 <= 0)
            else:
                crossing = (qt0 * qt1 <= 0)
            if not crossing:
                continue
            dq = qt1 - qt0
            dt = tarr[i] - tarr[i-1]
            if abs(dq) < 1e-12:
                alpha = 0.0
            else:
                alpha = -qt0 / dq
            alpha = np.clip(alpha, 0.0, 1.0)
            v_cross = v0 + alpha * (v1 - v0)
            t_cross = tarr[i-1] + alpha * dt
            # section at q=0
            q_cross = q0 + alpha * (q1 - q0)
            pts.append((q_cross, v_cross, t_cross))
        return pts

    data = {}
    for name, csv_dir in [('RK4', csv_dir_rk4), ('ETSVI', csv_dir_etsvi), ('PyBullet', csv_dir_py)]:
        q, v, t = _load_q_v_t(csv_dir)
        if q is not None:
            col_target = joint_index if q.shape[1] > joint_index else None
            q_target = q[:, col_target] if col_target is not None else None
            col_trig = trigger_joint_index if q.shape[1] > trigger_joint_index else None
            q_trig = q[:, col_trig] if col_trig is not None else None
        else:
            q_target = None
            q_trig = None
        if v is not None:
            col_target_v = joint_index if v.shape[1] > joint_index else None
            v_target = v[:, col_target_v] if col_target_v is not None else None
        else:
            v_target = None
        pts = _extract_poincare(q_trig, q_target, v_target, t)
        data[name] = pts

    _init_fig(figsize=(7, 6))
    cmap = plt.get_cmap('tab10')
    colors = {
        'RK4': cmap(1),
        'ETSVI': cmap(0),
        'PyBullet': cmap(2)
    }
    markers = {
        'RK4': 'o',
        'ETSVI': 's',
        'PyBullet': '^'
    }

    plotted = False
    for name, pts in data.items():
        if pts:
            q_vals = [p[0] for p in pts]
            v_vals = [p[1] for p in pts]
            plt.scatter(q_vals, v_vals, label=name, s=18, marker=markers.get(name, 'o'), alpha=0.8, color=colors.get(name, None))
            plotted = True

    if not plotted:
        print('No Poincaré points found for joint', joint_index+1)
        return False

    plt.xlabel(f'Joint {joint_index+1} Position [rad] (section q=0)')
    plt.ylabel(f'Joint {joint_index+1} Velocity [rad/s]')
    plt.title(f'Poincaré Section - Joint {joint_index+1} (q crossing 0, direction {direction})')
    plt.grid(True, alpha=0.4)
    plt.legend()
    filename = f'poincare_joint{joint_index+1}_{tag}.png'
    _save_fig(tag, filename, dpi_set if dpi_set else DEFAULT_DPI, show=True)
    return True

def main():
    """主函数"""
    # 运行仿真
    # if not run_etsvi():
    #     return 1

    # if not run_rk4():
    #     return 1

    # # 运行 pybullet 对照仿真（内联执行 pybullet_sim.py）
    # if not run_pybullet_inline():
    #     print("Warning: pybullet inline run failed or skipped. Proceeding to plotting with available CSVs.")

    # 绘制结果
    if not plot_results("etsvi_rk4_compare", 1000):
        return 1

    print("All tasks completed successfully!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
