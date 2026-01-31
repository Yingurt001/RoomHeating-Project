"""
Experiment: Thermostat placement study on 1D PDE model.

1. Varies thermostat position along the room for each controller.
2. Compares spatial temperature distribution.
3. Studies the interplay between sensor location and control performance.
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.pde_1d_model import HeatEquation1D
from controllers.bang_bang import BangBangController
from controllers.pid import PIDController, ziegler_nichols_tuning
from controllers.lqr import LQRController
from utils.parameters import (
    T_AMBIENT, T_INITIAL, T_SET, U_MAX, K_COOL,
    ROOM_LENGTH, HYSTERESIS_BAND, DT
)
from utils.metrics import compute_all_metrics
from utils.plotting import plot_1d_heatmap, plot_metrics_bar

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_1d_with_controller(ctrl_factory, thermostat_pos, heater_pos=0.5,
                           t_end=60.0, dt=0.05, is_bang_bang=False):
    """Run 1D PDE simulation with a given controller and thermostat position."""
    model = HeatEquation1D(
        heater_pos=heater_pos,
        thermostat_pos=thermostat_pos,
        T0=T_INITIAL
    )

    if is_bang_bang:
        # For bang-bang, we need event-based approach — simulate manually
        ctrl = ctrl_factory()
        t_eval = np.arange(0, t_end + dt, dt)
        t_list, T_therm_list, u_list = [], [], []
        T_current = model.T0.copy()

        for i, ti in enumerate(t_eval):
            T_therm = model.get_thermostat_temperature(T_current)
            u_val = ctrl.get_u(ti, T_therm)
            t_list.append(ti)
            T_therm_list.append(T_therm)
            u_list.append(u_val)

            if i < len(t_eval) - 1:
                dt_step = t_eval[i + 1] - ti
                # Forward Euler step for the PDE
                dTdt = model.rhs(ti, T_current, lambda t, T: u_val)
                T_current = T_current + dt_step * dTdt

        t = np.array(t_list)
        T_therm = np.array(T_therm_list)
        u = np.array(u_list)
        m = compute_all_metrics(t, T_therm, u, T_SET,
                                heater=(np.array(u) > 0.5).astype(float))
        return t, None, T_therm, u, m
    else:
        ctrl = ctrl_factory()
        t, T_field, T_therm = model.simulate(ctrl.get_u, t_end=t_end, dt=dt)
        # Recompute u from trajectory
        ctrl2 = ctrl_factory()
        u = np.array([ctrl2.get_u(ti, Ti) for ti, Ti in zip(t, T_therm)])
        m = compute_all_metrics(t, T_therm, u, T_SET)
        return t, T_field, T_therm, u, m


def main():
    print("=" * 70)
    print("1D PDE: Thermostat Placement × Controller Comparison")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)

    thermostat_positions = [0.5, 1.5, 2.5, 3.5, 4.5]
    t_end = 60.0
    heater_pos = 0.5

    # Controller factories
    Kp_zn, Ki_zn, Kd_zn = ziegler_nichols_tuning(K_COOL, U_MAX)

    controllers = {
        'Bang-Bang': {
            'factory': lambda: BangBangController(T_set=T_SET, U_max=U_MAX,
                                                   delta=HYSTERESIS_BAND, initial_on=True),
            'is_bb': True,
        },
        'PID': {
            'factory': lambda: PIDController(Kp=4.0, Ki=0.5, Kd=0.5,
                                              T_set=T_SET, U_max=U_MAX, dt=0.05),
            'is_bb': False,
        },
        'LQR': {
            'factory': lambda: LQRController(Q=1.0, R=0.01),
            'is_bb': False,
        },
    }

    # ==================== 1. Per-controller position sweep ====================
    all_results = {}  # {ctrl_name: {pos: (t, T_field, T_therm, u, m)}}

    for ctrl_name, ctrl_info in controllers.items():
        print(f"\n--- {ctrl_name}: varying thermostat position ---")
        print(f"  {'Position':>10} {'Energy':>10} {'RMSE':>8} {'Settle':>8} {'Overshoot':>10}")
        print("  " + "-" * 50)

        results = {}
        for pos in thermostat_positions:
            t, T_field, T_therm, u, m = run_1d_with_controller(
                ctrl_info['factory'], pos, heater_pos=heater_pos,
                t_end=t_end, is_bang_bang=ctrl_info['is_bb']
            )
            results[pos] = (t, T_field, T_therm, u, m)
            print(f"  {pos:>10.1f} {m['energy']:>10.1f} {m['rmse']:>8.3f} "
                  f"{m['settling_time']:>8.1f} {m['max_overshoot']:>10.3f}")

        all_results[ctrl_name] = results

    # ==================== 2. Thermostat position comparison plots ====================
    for ctrl_name, results in all_results.items():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                        gridspec_kw={'height_ratios': [3, 1]})
        for pos in thermostat_positions:
            t, T_field, T_therm, u, m = results[pos]
            ax1.plot(t, T_therm, linewidth=1.5,
                     label=f'x={pos:.1f}m (RMSE={m["rmse"]:.2f})')
            ax2.plot(t, u, linewidth=1.0, alpha=0.7)

        ax1.axhline(y=T_SET, color='r', linestyle='--', alpha=0.7,
                     label=f'$T_{{set}}$={T_SET}°C')
        ax1.set_ylabel('Thermostat Temperature (°C)', fontsize=12)
        ax1.set_title(f'1D PDE: Thermostat Position Effect ({ctrl_name})', fontsize=14)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel('Time (minutes)', fontsize=12)
        ax2.set_ylabel('Control u(t)', fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f'1d_placement_{ctrl_name.lower().replace("-","_").replace(" ","_")}.png'
        plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nSaved: {fname}")

    # ==================== 3. Heatmaps for PID at each position ====================
    print("\n--- Generating 1D heatmaps ---")
    for pos in [0.5, 2.5, 4.5]:
        if pos in all_results['PID'] and all_results['PID'][pos][1] is not None:
            t, T_field, T_therm, u, m = all_results['PID'][pos]
            model = HeatEquation1D(heater_pos=heater_pos, thermostat_pos=pos)
            plot_1d_heatmap(t, model.x, T_field,
                title=f"1D Temperature Field (PID, thermostat x={pos}m)",
                save_path=os.path.join(output_dir, f'1d_heatmap_x{pos:.0f}.png'))

    # ==================== 4. Cross-controller comparison at fixed position ====================
    print("\n--- Cross-controller comparison at x=2.5m ---")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    mid_pos = 2.5

    print(f"  {'Controller':<15} {'Energy':>10} {'RMSE':>8} {'Settle':>8} {'Cost J':>10}")
    print("  " + "-" * 55)

    for i, (ctrl_name, results) in enumerate(all_results.items()):
        if mid_pos in results:
            t, T_field, T_therm, u, m = results[mid_pos]
            ax1.plot(t, T_therm, linewidth=1.5, color=colors[i],
                     label=f'{ctrl_name} (RMSE={m["rmse"]:.3f})')
            ax2.plot(t, u, linewidth=1.0, color=colors[i], alpha=0.8,
                     label=ctrl_name)
            print(f"  {ctrl_name:<15} {m['energy']:>10.1f} {m['rmse']:>8.3f} "
                  f"{m['settling_time']:>8.1f} {m['unified_cost']:>10.1f}")

    ax1.axhline(y=T_SET, color='k', linestyle='--', alpha=0.5,
                label=f'$T_{{set}}$={T_SET}°C')
    ax1.set_ylabel('Thermostat Temperature (°C)', fontsize=12)
    ax1.set_title(f'1D PDE: Controller Comparison (thermostat at x={mid_pos}m)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Time (minutes)', fontsize=12)
    ax2.set_ylabel('Control u(t)', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1d_controller_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: 1d_controller_comparison.png")

    # ==================== 5. Metrics heatmap: position × controller ====================
    print("\n--- Position × Controller metrics matrix ---")
    metric_keys = ['rmse', 'energy', 'settling_time', 'unified_cost']
    metric_labels = ['RMSE (°C)', 'Energy', 'Settling Time (min)', 'Unified Cost J']
    ctrl_names = list(all_results.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    for idx, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        ax = axes[idx]
        matrix = np.zeros((len(thermostat_positions), len(ctrl_names)))
        for j, ctrl_name in enumerate(ctrl_names):
            for i, pos in enumerate(thermostat_positions):
                if pos in all_results[ctrl_name]:
                    matrix[i, j] = all_results[ctrl_name][pos][4][key]
                    if np.isinf(matrix[i, j]):
                        matrix[i, j] = t_end  # cap inf

        im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(range(len(ctrl_names)))
        ax.set_xticklabels(ctrl_names, fontsize=9)
        ax.set_yticks(range(len(thermostat_positions)))
        ax.set_yticklabels([f'x={p}m' for p in thermostat_positions], fontsize=9)
        ax.set_title(label, fontsize=11)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Annotate values
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f'{matrix[i,j]:.1f}', ha='center', va='center',
                        fontsize=7, color='black')

    plt.suptitle('1D PDE: Position × Controller Performance Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1d_metrics_matrix.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 1d_metrics_matrix.png")

    # ==================== 6. Heater position study ====================
    print("\n--- Heater position study (PID, thermostat at centre) ---")
    heater_positions = [0.5, 1.5, 2.5, 3.5, 4.5]
    therm_pos_fixed = 2.5

    fig, ax = plt.subplots(figsize=(12, 6))
    print(f"  {'Heater pos':>12} {'Energy':>10} {'RMSE':>8} {'Settle':>8}")
    print("  " + "-" * 42)

    for hp in heater_positions:
        ctrl_factory = lambda: PIDController(Kp=4.0, Ki=0.5, Kd=0.5,
                                              T_set=T_SET, U_max=U_MAX, dt=0.05)
        t, T_field, T_therm, u, m = run_1d_with_controller(
            ctrl_factory, therm_pos_fixed, heater_pos=hp, t_end=t_end
        )
        ax.plot(t, T_therm, linewidth=1.5,
                label=f'Heater x={hp}m (RMSE={m["rmse"]:.2f})')
        print(f"  {hp:>12.1f} {m['energy']:>10.1f} {m['rmse']:>8.3f} "
              f"{m['settling_time']:>8.1f}")

    ax.axhline(y=T_SET, color='r', linestyle='--', alpha=0.7)
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Thermostat Temperature (°C)', fontsize=12)
    ax.set_title('1D PDE: Heater Position Effect (PID, thermostat at centre)', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1d_heater_position.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: 1d_heater_position.png")

    print(f"\n{'='*70}")
    print("All 1D PDE experiments complete.")
    print(f"Results in: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
