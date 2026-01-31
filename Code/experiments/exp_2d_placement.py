"""
Experiment: 2D room thermostat/heater placement study.

1. Tests different heater × thermostat position combinations.
2. Compares PID, LQR, Bang-Bang on the 2D model.
3. Generates temperature field snapshots and performance metrics.
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.pde_2d_model import HeatEquation2D
from controllers.bang_bang import BangBangController
from controllers.pid import PIDController, ziegler_nichols_tuning
from controllers.lqr import LQRController
from utils.parameters import (
    T_AMBIENT, T_INITIAL, T_SET, U_MAX, K_COOL,
    ROOM_LENGTH, ROOM_WIDTH, HYSTERESIS_BAND, DT
)
from utils.metrics import compute_all_metrics
from utils.plotting import plot_2d_snapshots, plot_metrics_bar

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_2d_sim(controller_factory, heater_pos, thermostat_pos,
               t_end=30.0, nx=21, ny=17, dt=0.2, is_bang_bang=False):
    """Run 2D simulation with given positions and controller."""
    model = HeatEquation2D(
        nx=nx, ny=ny,
        heater_pos=heater_pos,
        thermostat_pos=thermostat_pos,
        T0=T_INITIAL
    )

    if is_bang_bang:
        ctrl = controller_factory()
        t_eval = np.arange(0, t_end + 0.5, 0.5)
        t_list, T_therm_list, u_list = [], [], []
        T_field_list = []
        T_current = model.T0.copy().ravel()

        for i, ti in enumerate(t_eval):
            T_2d = T_current.reshape(nx, ny)
            T_therm = model.get_thermostat_temperature(T_2d)
            u_val = ctrl.get_u(ti, T_therm)
            t_list.append(ti)
            T_therm_list.append(T_therm)
            u_list.append(u_val)
            T_field_list.append(T_2d.copy())

            if i < len(t_eval) - 1:
                dt_step = t_eval[i + 1] - ti
                dTdt = model.rhs(ti, T_current, lambda t, T: u_val)
                T_current = T_current + dt_step * dTdt

        t = np.array(t_list)
        T_therm = np.array(T_therm_list)
        u = np.array(u_list)
        T_field = np.stack([f for f in T_field_list], axis=-1)  # (nx, ny, nt)
        m = compute_all_metrics(t, T_therm, u, T_SET,
                                heater=(u > 0.5).astype(float))
        return t, T_field, T_therm, u, m, model
    else:
        ctrl = controller_factory()
        t, T_field, T_therm = model.simulate(ctrl.get_u, t_end=t_end, dt=dt)
        ctrl2 = controller_factory()
        u = np.array([ctrl2.get_u(ti, Ti) for ti, Ti in zip(t, T_therm)])
        m = compute_all_metrics(t, T_therm, u, T_SET)
        return t, T_field, T_therm, u, m, model


def main():
    print("=" * 70)
    print("2D PDE: Heater/Thermostat Placement Experiment")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)

    t_end = 30.0
    nx, ny = 21, 17

    # ==================== 1. Thermostat position study (PID) ====================
    print("\n[1] Thermostat position study — PID controller")
    heater_pos = (0.5, 2.0)

    thermostat_configs = {
        'Near heater':  (1.0, 2.0),
        'Centre':       (2.5, 2.0),
        'Far corner':   (4.5, 3.5),
        'Near wall':    (2.5, 0.3),
        'Opposite':     (4.5, 2.0),
    }

    def make_pid():
        return PIDController(Kp=4.0, Ki=0.5, Kd=0.5,
                             T_set=T_SET, U_max=U_MAX, dt=0.2)

    print(f"\n  Heater at {heater_pos}")
    print(f"  {'Config':<15} {'Therm pos':<15} {'Energy':>8} {'RMSE':>8} {'Settle':>8} {'Overshoot':>10}")
    print("  " + "-" * 68)

    pid_results = {}
    for name, therm_pos in thermostat_configs.items():
        t, T_field, T_therm, u, m, model = run_2d_sim(
            make_pid, heater_pos, therm_pos, t_end=t_end, nx=nx, ny=ny
        )
        pid_results[name] = (t, T_field, T_therm, u, m, model)
        print(f"  {name:<15} {str(therm_pos):<15} {m['energy']:>8.1f} {m['rmse']:>8.3f} "
              f"{m['settling_time']:>8.1f} {m['max_overshoot']:>10.3f}")

    # Temperature traces
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    for name, (t, T_field, T_therm, u, m, model) in pid_results.items():
        ax1.plot(t, T_therm, linewidth=1.5, label=f'{name} (RMSE={m["rmse"]:.3f})')
        ax2.plot(t, u, linewidth=1.0, alpha=0.7)

    ax1.axhline(y=T_SET, color='r', linestyle='--', alpha=0.7,
                label=f'$T_{{set}}$={T_SET}°C')
    ax1.set_ylabel('Thermostat Temperature (°C)', fontsize=12)
    ax1.set_title('2D Room: Thermostat Position Effect (PID)', fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('Time (minutes)', fontsize=12)
    ax2.set_ylabel('Control u(t)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2d_placement_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("\n  Saved: 2d_placement_comparison.png")

    # 2D snapshots for key configs
    for name in ['Near heater', 'Centre', 'Far corner']:
        t, T_field, T_therm, u, m, model = pid_results[name]
        snapshot_times = [0, 5, 15, 30]
        safe_name = name.replace(' ', '_').lower()
        plot_2d_snapshots(
            model.x, model.y, T_field, snapshot_times, t,
            title_prefix=f"2D ({name})",
            save_path=os.path.join(output_dir, f'2d_snap_{safe_name}.png')
        )

    # ==================== 2. Controller comparison at centre ====================
    print("\n[2] Controller comparison — thermostat at centre")
    therm_centre = (2.5, 2.0)

    controllers_2d = {
        'Bang-Bang': {
            'factory': lambda: BangBangController(T_set=T_SET, U_max=U_MAX,
                                                   delta=HYSTERESIS_BAND, initial_on=True),
            'is_bb': True,
        },
        'PID': {
            'factory': make_pid,
            'is_bb': False,
        },
        'LQR': {
            'factory': lambda: LQRController(Q=1.0, R=0.01),
            'is_bb': False,
        },
    }

    print(f"  {'Controller':<15} {'Energy':>8} {'RMSE':>8} {'Settle':>8} {'Cost J':>10}")
    print("  " + "-" * 52)

    ctrl_results = {}
    for ctrl_name, ctrl_info in controllers_2d.items():
        t, T_field, T_therm, u, m, model = run_2d_sim(
            ctrl_info['factory'], heater_pos, therm_centre,
            t_end=t_end, nx=nx, ny=ny, is_bang_bang=ctrl_info['is_bb']
        )
        ctrl_results[ctrl_name] = (t, T_field, T_therm, u, m, model)
        print(f"  {ctrl_name:<15} {m['energy']:>8.1f} {m['rmse']:>8.3f} "
              f"{m['settling_time']:>8.1f} {m['unified_cost']:>10.1f}")

    # Controller comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, (ctrl_name, (t, T_field, T_therm, u, m, model)) in enumerate(ctrl_results.items()):
        ax1.plot(t, T_therm, linewidth=1.5, color=colors[i],
                 label=f'{ctrl_name} (RMSE={m["rmse"]:.3f})')
        ax2.plot(t, u, linewidth=1.0, color=colors[i], alpha=0.8, label=ctrl_name)

    ax1.axhline(y=T_SET, color='k', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Thermostat Temperature (°C)', fontsize=12)
    ax1.set_title('2D Room: Controller Comparison (thermostat at centre)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('Time (minutes)', fontsize=12)
    ax2.set_ylabel('Control u(t)', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2d_controller_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 2d_controller_comparison.png")

    # ==================== 3. Heater position study ====================
    print("\n[3] Heater position study — PID, thermostat at centre")
    heater_configs = {
        'Corner (0.5,0.5)':   (0.5, 0.5),
        'Mid-wall (0.5,2.0)': (0.5, 2.0),
        'Centre (2.5,2.0)':   (2.5, 2.0),
        'Far wall (4.5,2.0)': (4.5, 2.0),
    }

    print(f"  {'Heater pos':<22} {'Energy':>8} {'RMSE':>8} {'Settle':>8}")
    print("  " + "-" * 48)

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, hp in heater_configs.items():
        t, T_field, T_therm, u, m, model = run_2d_sim(
            make_pid, hp, therm_centre, t_end=t_end, nx=nx, ny=ny
        )
        ax.plot(t, T_therm, linewidth=1.5,
                label=f'{name} (RMSE={m["rmse"]:.2f})')
        print(f"  {name:<22} {m['energy']:>8.1f} {m['rmse']:>8.3f} "
              f"{m['settling_time']:>8.1f}")

    ax.axhline(y=T_SET, color='r', linestyle='--', alpha=0.7)
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Thermostat Temperature (°C)', fontsize=12)
    ax.set_title('2D Room: Heater Position Effect (PID)', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2d_heater_position.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 2d_heater_position.png")

    # ==================== 4. Spatial uniformity analysis ====================
    print("\n[4] Spatial uniformity analysis")
    # For the centre-thermostat PID case, compute temperature gradient stats
    t, T_field, T_therm, u, m, model = pid_results['Centre']
    nt = T_field.shape[2]

    T_mean = np.array([T_field[:,:,i].mean() for i in range(nt)])
    T_max = np.array([T_field[:,:,i].max() for i in range(nt)])
    T_min = np.array([T_field[:,:,i].min() for i in range(nt)])
    T_std = np.array([T_field[:,:,i].std() for i in range(nt)])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(t, T_mean, 'b-', linewidth=1.5, label='Mean T')
    ax1.fill_between(t, T_min, T_max, alpha=0.2, color='blue', label='Min-Max range')
    ax1.plot(t, T_therm, 'r--', linewidth=1.0, label='Thermostat reading')
    ax1.axhline(y=T_SET, color='k', linestyle=':', alpha=0.5)
    ax1.set_ylabel('Temperature (°C)', fontsize=12)
    ax1.set_title('2D Room: Spatial Temperature Uniformity (PID, centre)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, T_std, 'g-', linewidth=1.5, label='Spatial std dev')
    ax2.plot(t, T_max - T_min, 'm-', linewidth=1.0, alpha=0.7, label='Max-Min range')
    ax2.set_xlabel('Time (minutes)', fontsize=12)
    ax2.set_ylabel('Temperature Spread (°C)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2d_uniformity.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: 2d_uniformity.png")

    # Metrics bar chart
    all_metrics = {name: m for name, (_, _, _, _, m, _) in pid_results.items()}
    plot_metrics_bar(all_metrics,
                     save_path=os.path.join(output_dir, '2d_placement_metrics.png'))

    print(f"\n{'='*70}")
    print("All 2D PDE experiments complete.")
    print(f"Results in: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
