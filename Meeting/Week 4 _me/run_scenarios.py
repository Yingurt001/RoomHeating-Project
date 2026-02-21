"""
Unified experiment runner for the 6 physical scenarios.

Usage:
    python run_scenarios.py                # run all scenarios
    python run_scenarios.py S1 S2          # run specific scenarios
    python run_scenarios.py --quick        # short sim (t_end=30) for testing

Outputs go to ./results/
"""

import sys, os, argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── project imports ──
# 使用本地 Code 副本（自包含，不依赖上级目录）
CODE_DIR = str(Path(__file__).resolve().parent / "Code")
sys.path.insert(0, CODE_DIR)

from models.pde_2d_model import HeatEquation2D
from controllers.bang_bang import BangBangController
from utils.parameters import T_SET, T_AMBIENT, U_MAX, T_END

# local
from scenarios import (
    make_baseline, make_window, make_window_compare,
    make_door_opening, make_narrow_room, make_L_shape,
)

# ── output directory ──
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)


# =====================================================================
#  Helper: build model from scenario config dict
# =====================================================================
def build_model(cfg):
    """Create HeatEquation2D from a scenario config dict."""
    return HeatEquation2D(
        Lx=cfg['Lx'], Ly=cfg['Ly'],
        nx=cfg.get('nx', 51), ny=cfg.get('ny', 41),
        heater_pos=cfg['heater_pos'],
        thermostat_pos=cfg['thermostat_pos'],
        wall_h=cfg.get('wall_h'),
        domain_mask=cfg.get('domain_mask'),
        h_updater=cfg.get('h_updater'),
    )


def run_one(cfg, t_end=T_END):
    """Run a single scenario, return (model, t, T_field, T_therm, u_arr)."""
    model = build_model(cfg)
    ctrl = BangBangController()

    # record control signal
    u_log = []

    def u_func(t, T_therm):
        u = ctrl.get_u(t, T_therm)
        # update hysteresis state
        if ctrl._on and T_therm > ctrl.T_set + ctrl.delta:
            ctrl.switch(t, T_therm)
        elif not ctrl._on and T_therm < ctrl.T_set - ctrl.delta:
            ctrl.switch(t, T_therm)
        u_log.append((t, u))
        return u

    t_eval = np.arange(0, t_end + 0.5, 0.5)
    t, T_field, T_therm = model.simulate(u_func, t_end=t_end, t_eval=t_eval)

    # reconstruct control signal at t_eval points
    u_arr = np.zeros_like(t)
    if u_log:
        t_log = np.array([x[0] for x in u_log])
        u_log_vals = np.array([x[1] for x in u_log])
        for i, ti in enumerate(t):
            idx = np.searchsorted(t_log, ti, side='right') - 1
            idx = max(0, min(idx, len(u_log_vals) - 1))
            u_arr[i] = u_log_vals[idx]

    return model, t, T_field, T_therm, u_arr


# =====================================================================
#  Plotting functions
# =====================================================================
def plot_field(model, T_snap, cfg, save_path,
               title=None, vmin=5, vmax=25):
    """2D temperature heatmap with heater (★) and thermostat (×) markers."""
    fig, ax = plt.subplots(figsize=(7, 6))

    T_plot = T_snap.copy().T  # transpose: T[x,y] → imshow expects [y,x]
    mask = cfg.get('domain_mask')
    if mask is not None:
        T_plot_masked = np.ma.array(T_snap.copy().T, mask=~mask.T)
    else:
        T_plot_masked = T_snap.T

    im = ax.pcolormesh(model.x, model.y, T_plot_masked,
                       shading='auto', cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Temperature (°C)')

    # mask region in gray
    if mask is not None:
        mask_region = np.ma.array(np.ones_like(T_snap.T), mask=mask.T)
        ax.pcolormesh(model.x, model.y, mask_region,
                      shading='auto', cmap='Greys', vmin=0, vmax=2, alpha=0.5)

    # heater and thermostat markers
    hx, hy = cfg['heater_pos']
    tx, ty = cfg['thermostat_pos']
    ax.plot(hx, hy, 'r*', markersize=15, markeredgecolor='k', label='Heater')
    ax.plot(tx, ty, 'bx', markersize=12, markeredgewidth=3, label='Thermostat')

    # window highlight on south wall
    h_south = cfg.get('wall_h', {}).get('south')
    if h_south is not None:
        from scenarios import H_WALL
        window_mask = h_south > H_WALL + 0.01
        if np.any(window_mask):
            x_arr = np.linspace(0, cfg['Lx'], len(h_south))
            segs = _contiguous_segments(window_mask, x_arr)
            for x0, x1 in segs:
                ax.plot([x0, x1], [0, 0], 'c-', linewidth=5, alpha=0.8,
                        label='Window' if x0 == segs[0][0] else '')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(title or cfg['name'])
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_timeseries(t, T_therm, T_mean, u_arr, cfg, save_path):
    """Dual panel: temperature (mean + sensor) and control signal."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(t, T_therm, 'b-', lw=1.5, label='Thermostat reading')
    ax1.plot(t, T_mean, 'g--', lw=1.2, label='Room mean T')
    ax1.axhline(T_SET, color='r', ls='--', alpha=0.6, label=f'T_set = {T_SET}°C')
    ax1.axhspan(18, 22, color='green', alpha=0.06, label='Comfort zone')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title(cfg['name'])
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(t, u_arr / U_MAX, step='mid', alpha=0.4, color='red')
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Heater')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['OFF', 'ON'])
    ax2.set_ylim(-0.1, 1.3)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_multi_fields(models, T_snaps, cfgs, save_path,
                      title="Scenario Comparison", vmin=5, vmax=25):
    """Side-by-side temperature fields for multiple scenarios."""
    n = len(models)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.ravel() if hasattr(axes, 'ravel') else [axes]

    for i, (model, T_snap, cfg) in enumerate(zip(models, T_snaps, cfgs)):
        ax = axes[i]
        mask = cfg.get('domain_mask')
        if mask is not None:
            T_ma = np.ma.array(T_snap.T, mask=~mask.T)
        else:
            T_ma = T_snap.T
        im = ax.pcolormesh(model.x, model.y, T_ma,
                           shading='auto', cmap='RdYlBu_r', vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(cfg['name'], fontsize=11)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')

        hx, hy = cfg['heater_pos']
        ax.plot(hx, hy, 'r*', markersize=10, markeredgecolor='k')

    # hide unused axes
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_metrics_bar(metrics_dict, save_path):
    """Bar chart comparing RMSE, energy, temperature non-uniformity."""
    names = list(metrics_dict.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    keys = ['rmse', 'energy', 'T_nonuniformity']
    labels = ['RMSE from T_set (°C)', 'Energy (heater·min)', 'Spatial Non-uniformity (°C)']

    for ax, key, lab in zip(axes, keys, labels):
        vals = [metrics_dict[n].get(key, 0) for n in names]
        colors = plt.cm.Set2(np.linspace(0, 0.8, len(names)))
        bars = ax.bar(names, vals, color=colors)
        ax.set_ylabel(lab)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, alpha=0.3, axis='y')
        # add value labels
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{v:.2f}', ha='center', va='bottom', fontsize=9)

    fig.suptitle('Scenario Metrics Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_door_snapshots(model, T_field, t, cfg, save_path,
                        snap_times=[25, 30, 35, 40, 50]):
    """Temperature field snapshots at key moments around door open/close."""
    n = len(snap_times)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4.5))
    if n == 1:
        axes = [axes]

    for ax, t_snap in zip(axes, snap_times):
        idx = np.argmin(np.abs(t - t_snap))
        T_snap = T_field[:, :, idx]
        im = ax.pcolormesh(model.x, model.y, T_snap.T,
                           shading='auto', cmap='RdYlBu_r', vmin=5, vmax=25)
        ax.set_title(f't = {t[idx]:.0f} min', fontsize=11)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # mark door region on west wall
        ax.plot([0, 0], [1.0, 2.5], 'm-', linewidth=4, alpha=0.7)

    fig.suptitle(f'{cfg["name"]} — Door Open/Close Snapshots', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close(fig)


def _contiguous_segments(mask_1d, x_arr):
    """Find contiguous True segments in a 1D bool array, return [(x0,x1),...]."""
    segs = []
    in_seg = False
    for i, v in enumerate(mask_1d):
        if v and not in_seg:
            x0 = x_arr[i]
            in_seg = True
        elif not v and in_seg:
            segs.append((x0, x_arr[i-1]))
            in_seg = False
    if in_seg:
        segs.append((x0, x_arr[-1]))
    return segs


# =====================================================================
#  Compute simple metrics
# =====================================================================
def compute_metrics(t, T_therm, T_field, u_arr, mask=None):
    """Compute RMSE, energy, spatial non-uniformity."""
    rmse = np.sqrt(np.trapz((T_therm - T_SET)**2, t) / (t[-1] - t[0]))
    energy = np.trapz(u_arr, t)

    # spatial non-uniformity: std of final temperature field (active region only)
    T_final = T_field[:, :, -1]
    if mask is not None:
        T_active = T_final[mask]
    else:
        T_active = T_final.ravel()
    nonunif = np.std(T_active)

    return dict(rmse=rmse, energy=energy, T_nonuniformity=nonunif)


# =====================================================================
#  Scenario runners
# =====================================================================
def run_S1(t_end):
    print("\n[S1] Baseline ...")
    cfg = make_baseline()
    model, t, T_field, T_therm, u_arr = run_one(cfg, t_end)
    T_mean = np.mean(T_field, axis=(0, 1))

    plot_field(model, T_field[:,:,-1], cfg,
               RESULTS / "S1_field.png", title="S1 Baseline — Final T field")
    plot_timeseries(t, T_therm, T_mean, u_arr, cfg,
                    RESULTS / "S1_timeseries.png")

    metrics = compute_metrics(t, T_therm, T_field, u_arr)
    print(f"  RMSE={metrics['rmse']:.2f}°C  Energy={metrics['energy']:.1f}  "
          f"NonUnif={metrics['T_nonuniformity']:.2f}°C")
    return cfg, model, t, T_field, T_therm, u_arr, metrics


def run_S2(t_end):
    print("\n[S2] Window ...")
    cfg = make_window()
    model, t, T_field, T_therm, u_arr = run_one(cfg, t_end)
    T_mean = np.mean(T_field, axis=(0, 1))

    plot_field(model, T_field[:,:,-1], cfg,
               RESULTS / "S2_field.png", title="S2 Window — Final T field")
    plot_timeseries(t, T_therm, T_mean, u_arr, cfg,
                    RESULTS / "S2_timeseries.png")

    metrics = compute_metrics(t, T_therm, T_field, u_arr)
    print(f"  RMSE={metrics['rmse']:.2f}°C  Energy={metrics['energy']:.1f}  "
          f"NonUnif={metrics['T_nonuniformity']:.2f}°C")
    return cfg, model, t, T_field, T_therm, u_arr, metrics


def run_S3(t_end):
    print("\n[S3] Window comparison ...")
    cfgs = make_window_compare()
    results = []
    for cfg in cfgs:
        model, t, T_field, T_therm, u_arr = run_one(cfg, t_end)
        metrics = compute_metrics(t, T_therm, T_field, u_arr)
        results.append((cfg, model, t, T_field, T_therm, u_arr, metrics))
        print(f"  {cfg['name']}: RMSE={metrics['rmse']:.2f}  Energy={metrics['energy']:.1f}")

    # multi-panel fields
    plot_multi_fields(
        [r[1] for r in results], [r[3][:,:,-1] for r in results],
        [r[0] for r in results],
        RESULTS / "S3_fields_compare.png",
        title="S3 Window Variants — Final T fields")

    # metrics bar
    m_dict = {r[0]['name']: r[6] for r in results}
    plot_metrics_bar(m_dict, RESULTS / "S3_metrics_bar.png")

    return results


def run_S4(t_end):
    print("\n[S4] Door opening ...")
    cfg = make_door_opening(t_open=30.0, t_close=40.0)
    model, t, T_field, T_therm, u_arr = run_one(cfg, t_end)
    T_mean = np.mean(T_field, axis=(0, 1))

    snap_times = [25, 30, 35, 40, 50]
    snap_times = [s for s in snap_times if s <= t_end]
    plot_door_snapshots(model, T_field, t, cfg,
                        RESULTS / "S4_door_snapshots.png",
                        snap_times=snap_times)
    plot_timeseries(t, T_therm, T_mean, u_arr, cfg,
                    RESULTS / "S4_timeseries.png")

    metrics = compute_metrics(t, T_therm, T_field, u_arr)
    print(f"  RMSE={metrics['rmse']:.2f}°C  Energy={metrics['energy']:.1f}  "
          f"NonUnif={metrics['T_nonuniformity']:.2f}°C")
    return cfg, model, t, T_field, T_therm, u_arr, metrics


def run_S5(t_end):
    print("\n[S5] Narrow room ...")
    cfg = make_narrow_room()
    model, t, T_field, T_therm, u_arr = run_one(cfg, t_end)
    T_mean = np.mean(T_field, axis=(0, 1))

    plot_field(model, T_field[:,:,-1], cfg,
               RESULTS / "S5_field.png", title="S5 Narrow 7.5×2.5 m — Final T field")
    plot_timeseries(t, T_therm, T_mean, u_arr, cfg,
                    RESULTS / "S5_timeseries.png")

    metrics = compute_metrics(t, T_therm, T_field, u_arr)
    print(f"  RMSE={metrics['rmse']:.2f}°C  Energy={metrics['energy']:.1f}  "
          f"NonUnif={metrics['T_nonuniformity']:.2f}°C")
    return cfg, model, t, T_field, T_therm, u_arr, metrics


def run_S6(t_end):
    print("\n[S6] L-shaped room ...")
    cfg = make_L_shape()
    model, t, T_field, T_therm, u_arr = run_one(cfg, t_end)

    mask = cfg['domain_mask']
    T_active_mean = np.array([T_field[:,:,i][mask].mean() for i in range(len(t))])

    plot_field(model, T_field[:,:,-1], cfg,
               RESULTS / "S6_field.png", title="S6 L-Shape — Final T field")
    plot_timeseries(t, T_therm, T_active_mean, u_arr, cfg,
                    RESULTS / "S6_timeseries.png")

    metrics = compute_metrics(t, T_therm, T_field, u_arr, mask=mask)
    print(f"  RMSE={metrics['rmse']:.2f}°C  Energy={metrics['energy']:.1f}  "
          f"NonUnif={metrics['T_nonuniformity']:.2f}°C")
    return cfg, model, t, T_field, T_therm, u_arr, metrics


# =====================================================================
#  All-scenario comparison
# =====================================================================
def run_all_comparison(all_results):
    """Generate cross-scenario comparison plots."""
    print("\n[ALL] Generating comparison plots ...")

    # 2×3 panel of all final fields
    models = [r[1] for r in all_results]
    T_snaps = [r[3][:,:,-1] for r in all_results]
    cfgs = [r[0] for r in all_results]

    plot_multi_fields(models, T_snaps, cfgs,
                      RESULTS / "ALL_6scenarios_fields.png",
                      title="All 6 Scenarios — Final Temperature Fields")

    # metrics comparison bar
    m_dict = {r[0]['name']: r[6] for r in all_results}
    plot_metrics_bar(m_dict, RESULTS / "ALL_metrics_comparison.png")


# =====================================================================
#  Main
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Run heating scenarios")
    parser.add_argument('scenarios', nargs='*', default=[],
                        help='Scenarios to run (S1-S6). Empty = all.')
    parser.add_argument('--quick', action='store_true',
                        help='Short simulation (t_end=30 min) for testing')
    args = parser.parse_args()

    t_end = 30.0 if args.quick else T_END

    run_map = {
        'S1': run_S1, 'S2': run_S2, 'S3': run_S3,
        'S4': run_S4, 'S5': run_S5, 'S6': run_S6,
    }

    to_run = args.scenarios if args.scenarios else ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']

    all_results = []
    s3_results = None

    for s in to_run:
        s = s.upper()
        if s not in run_map:
            print(f"Unknown scenario: {s}, skipping")
            continue
        result = run_map[s](t_end)
        if s == 'S3':
            s3_results = result  # list of 3 results
            all_results.extend(result)
        else:
            all_results.append(result)

    # all-scenario comparison (only if 2+ single scenarios)
    single_results = [r for r in all_results if isinstance(r, tuple) and len(r) == 7]
    if len(single_results) >= 2:
        run_all_comparison(single_results)

    print(f"\n✓ Done! Results saved to {RESULTS}/")


if __name__ == '__main__':
    main()
