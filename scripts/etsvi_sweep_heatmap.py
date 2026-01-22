#!/usr/bin/env python3
"""
Run ETSVI over an alpha/beta grid and plot heatmap of mean absolute energy error.

Usage:
  python3 etsvi_sweep_heatmap.py [--dry-run] [--dt 0.01] [--T 100]

Ensure workspace is sourced: `source install/setup.bash` before running.
"""
import os
import argparse
import subprocess
import datetime
import numpy as np
import matplotlib.pyplot as plt
try:
    import yaml
except Exception:
    yaml = None


def _find_workspace_root():
    # Find ancestor directory that contains src/vi; fallback to cwd
    p = os.getcwd()
    while True:
        if os.path.isdir(os.path.join(p, 'src', 'vi')):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return os.getcwd()

WORKSPACE_ROOT = _find_workspace_root()
CONFIG_DIR = os.path.join(WORKSPACE_ROOT, 'src', 'vi', 'config')
BASE_CONFIG = os.path.join(CONFIG_DIR, 'vi_params.yaml')
CSV_BASE = os.path.join(WORKSPACE_ROOT, 'src', 'vi', 'csv')
OUT_DIR = os.path.join(WORKSPACE_ROOT, 'src', 'vi', 'fig', 'heatmaps')
LOG_DIR = os.path.join(WORKSPACE_ROOT, 'src', 'vi', 'logs', 'etsvi_sweep')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def _format_param(v):
    try:
        fv = float(v)
        if fv.is_integer():
            return str(int(fv))
        return str(v).replace('.', 'p')
    except Exception:
        return str(v).replace('.', 'p')


def build_label(q_init, dt, T, alpha, beta):
    return f"q{_format_param(q_init)}_dt{_format_param(dt)}_T{_format_param(T)}_a{_format_param(alpha)}_b{_format_param(beta)}"


def write_params(out_path, base_cfg, q_init, dt, T, alpha, beta):
    if yaml is not None:
        try:
            cfg = {}
            with open(base_cfg, 'r') as f:
                cfg = yaml.safe_load(f)
            if '/**' not in cfg or not isinstance(cfg['/**'], dict):
                cfg['/**'] = {'ros__parameters': {}}
            cfg['/**'].setdefault('ros__parameters', {})
            cfg['/**']['ros__parameters']['q_init'] = float(q_init)
            cfg['/**']['ros__parameters']['timestep'] = float(dt)
            cfg['/**']['ros__parameters']['duration'] = float(T)
            cfg.setdefault('etsvi_node', {})
            cfg['etsvi_node'].setdefault('ros__parameters', {})
            cfg['etsvi_node']['ros__parameters']['lyap_alpha'] = float(alpha)
            cfg['etsvi_node']['ros__parameters']['lyap_beta'] = float(beta)
            with open(out_path, 'w') as f:
                yaml.safe_dump(cfg, f, default_flow_style=False)
            return True
        except Exception:
            pass
    # fallback text replace
    try:
        with open(base_cfg, 'r') as f:
            txt = f.read()
        import re
        txt = re.sub(r"q_init:\s*[0-9.eE+-]+", f"q_init: {q_init}", txt)
        txt = re.sub(r"timestep:\s*[0-9.eE+-]+", f"timestep: {dt}", txt)
        txt = re.sub(r"duration:\s*[0-9.eE+-]+", f"duration: {T}", txt)
        txt = re.sub(r"lyap_alpha:\s*[0-9.eE+-]+", f"lyap_alpha: {alpha}", txt)
        txt = re.sub(r"lyap_beta:\s*[0-9.eE+-]+", f"lyap_beta: {beta}", txt)
        with open(out_path, 'w') as f:
            f.write(txt)
        return True
    except Exception:
        return False


def run_etsvi(params_file, timeout=600):
    cmd = ['ros2', 'run', 'vi', 'etsvi_node', '--ros-args', '--params-file', params_file]
    try:
        p = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout)
        return 0, p.stdout
    except subprocess.CalledProcessError as e:
        return 1, e.stdout or str(e)
    except subprocess.TimeoutExpired as e:
        return 2, (e.stdout or '') + '\nTimeout'


def read_mean_abs_energy(params_label):
    path = os.path.join(CSV_BASE, params_label, 'etsvi', 'delta_energy_history.csv')
    if not os.path.exists(path):
        return np.nan
    try:
        arr = np.loadtxt(path, delimiter=',')
        arr = np.atleast_1d(arr).astype(float)
        return float(np.mean(np.abs(arr)))
    except Exception:
        return np.nan


def plot_heatmap(alphas, betas, grid, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    # grid shape: (len(betas), len(alphas)) where rows=beta, cols=alpha
    im = ax.imshow(grid, origin='lower', aspect='auto', cmap='viridis')
    ax.set_xticks(np.arange(len(alphas)))
    ax.set_yticks(np.arange(len(betas)))
    ax.set_xticklabels([str(a) for a in alphas])
    ax.set_yticklabels([str(b) for b in betas])
    ax.set_xlabel('alpha')
    ax.set_ylabel('beta')
    ax.set_title('ETSVI Mean Absolute Energy Error')
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Mean |ΔEnergy| (J)')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--q_init', type=float, default=0.2)
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--T', type=float, default=40.0)
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args()

    alphas = [round(0.1 * i, 2) for i in range(1, 9)]
    betas = [round(0.01 * i, 3) for i in range(1, 9)]

    grid = np.zeros((len(betas), len(alphas))) * np.nan

    for ia, a in enumerate(alphas):
        for ib, b in enumerate(betas):
            label = build_label(args.q_init, args.dt, args.T, a, b)
            params_file = os.path.join(CONFIG_DIR, f'vi_params_{label}.yaml')
            ok = write_params(params_file, BASE_CONFIG, args.q_init, args.dt, args.T, a, b)
            log_file = os.path.join(LOG_DIR, f'etsvi_{label}.log')
            with open(log_file, 'a') as lf:
                lf.write(f"Experiment: {label}\nParams: {params_file}\n")

            if args.dry_run:
                print(f"[dry-run] {label}")
                grid[ib, ia] = np.nan
                continue

            status, out = run_etsvi(params_file, timeout=args.timeout)
            with open(log_file, 'a') as lf:
                lf.write(out + '\n')
            if status != 0:
                print(f"ETSVI failed for {label}, status={status}")
                grid[ib, ia] = np.nan
                continue

            # read result
            mean_abs = read_mean_abs_energy(label)
            grid[ib, ia] = mean_abs
            print(f"alpha={a} beta={b} -> mean_abs={mean_abs}")

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_png = os.path.join(OUT_DIR, f'etsvi_heatmap_{timestamp}.png')
    out_csv = os.path.join(OUT_DIR, f'etsvi_heatmap_{timestamp}.csv')
    plot_heatmap(alphas, betas, grid, out_png)
    # save CSV with betas as rows, alphas as columns
    header = ','.join([str(a) for a in alphas])
    with open(out_csv, 'w') as f:
        f.write(',' + header + '\n')
        for ib, b in enumerate(betas):
            row = ','.join([f"{grid[ib, ia]:.6e}" if not np.isnan(grid[ib, ia]) else '' for ia in range(len(alphas))])
            f.write(f"{b},{row}\n")

    print(f"Saved heatmap: {out_png}")
    print(f"Saved grid CSV: {out_csv}")


if __name__ == '__main__':
    main()
