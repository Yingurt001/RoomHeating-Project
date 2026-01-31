"""
Experiment: 2D room thermostat/heater placement study.

Tests different combinations of heater and thermostat positions
in a 2D rectangular room and compares performance.
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.pde_2d_model import HeatEquation2D
from controllers.pid import PIDController, ziegler_nichols_tuning
from controllers.lqr import LQRController
from utils.parameters import (
    T_AMBIENT, T_INITIAL, T_SET, U_MAX, K_COOL,
    ROOM_LENGTH, ROOM_WIDTH, DT
)
from utils.metrics import compute_all_metrics
from utils.plotting import plot_2d_snapshots, plot_metrics_bar

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_2d_placement(controller_factory, heater_pos, thermostat_pos,
                     t_end=30.0, nx=31, ny=25):
    """
    Run 2D simulation with given heater and thermostat positions.

    Returns
    -------
    t, T_field, T_therm, u, metrics
    """
    model = HeatEquation2D(
        nx=nx, ny=ny,
        heater_pos=heater_pos,
        thermostat_pos=thermostat_pos,
        T0=T_INITIAL
    )

    ctrl = controller_factory()
    t, T_field, T_therm = model.simulate(ctrl.get_u, t_end=t_end, dt=0.1)

    u = np.array([ctrl.get_u(ti, Ti) for ti, Ti in zip(t, T_therm)])
    m = compute_all_metrics(t, T_therm, u, T_SET)

    return t, T_field, T_therm, u, m, model


def main():
    print("=" * 70)
    print("2D PDE: Heater/Thermostat Placement Experiment")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)

    Kp, Ki, Kd = ziegler_nichols_tuning(K_COOL, U_MAX)

    def make_pid():
        return PIDController(Kp=Kp, Ki=Ki, Kd=Kd, T_set=T_SET, U_max=U_MAX,
                             dt=0.1)

    # Heater fixed at corner, vary thermostat
    heater_pos = (0.5, 2.0)
    thermostat_configs = {
        'Near heater': (1.0, 2.0),
        'Centre': (2.5, 2.0),
        'Far corner': (4.5, 2.0),
        'Near wall': (2.5, 0.5),
    }

    print(f"\nHeater at {heater_pos}")
    print(f"\n{'Config':<15} {'Energy':>10} {'RMSE':>8} {'Settle':>8} {'Overshoot':>10}")
    print("-" * 55)

    all_metrics = {}
    for name, therm_pos in thermostat_configs.items():
        t, T_field, T_therm, u, m, model = run_2d_placement(
            make_pid, heater_pos, therm_pos, t_end=30.0
        )
        all_metrics[name] = m
        print(f"{name:<15} {m['energy']:>10.1f} {m['rmse']:>8.3f} "
              f"{m['settling_time']:>8.1f} {m['max_overshoot']:>10.3f}")

        # 2D snapshots for each config
        snapshot_times = [0, 5, 15, 30]
        plot_2d_snapshots(
            model.x, model.y, T_field, snapshot_times, t,
            title_prefix=f"2D Room ({name})",
            save_path=os.path.join(output_dir, f'2d_snap_{name.replace(" ", "_").lower()}.png')
        )

    # --- Thermostat reading comparison ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, therm_pos in thermostat_configs.items():
        t, T_field, T_therm, u, m, model = run_2d_placement(
            make_pid, heater_pos, therm_pos, t_end=30.0
        )
        ax.plot(t, T_therm, linewidth=1.5, label=f'{name} {therm_pos}')

    ax.axhline(y=T_SET, color='r', linestyle='--', alpha=0.7,
               label=f'$T_{{set}}$ = {T_SET}°C')
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Thermostat Temperature (°C)', fontsize=12)
    ax.set_title('2D Room: Thermostat Position Effect', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2d_placement_comparison.png'),
                dpi=150, bbox_inches='tight')
    print(f"\nSaved: 2d_placement_comparison.png")
    plt.close()

    # Metrics bar chart
    plot_metrics_bar(all_metrics,
                     save_path=os.path.join(output_dir, '2d_placement_metrics.png'))

    print("\nAll 2D placement results saved to:", output_dir)


if __name__ == "__main__":
    main()
