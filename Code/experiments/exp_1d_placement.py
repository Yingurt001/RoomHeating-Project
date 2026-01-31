"""
Experiment: Thermostat placement study on 1D PDE model.

Varies the thermostat position along the room and compares
temperature control performance for each controller.
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
from utils.plotting import plot_1d_heatmap, plot_comparison, plot_metrics_bar

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_1d_experiment(controller_factory, thermostat_positions,
                      heater_pos=0.5, t_end=60.0):
    """
    Run 1D PDE simulation for each thermostat position.

    Parameters
    ----------
    controller_factory : callable(T_set, U_max) -> controller
        Factory function that creates a controller.
    thermostat_positions : list of float
        x-coordinates for thermostat.
    heater_pos : float
        Heater x-position.
    t_end : float
        Simulation time.

    Returns
    -------
    results : dict
        {pos: (t, T_field, T_therm, metrics)}
    """
    results = {}
    for pos in thermostat_positions:
        model = HeatEquation1D(
            heater_pos=heater_pos,
            thermostat_pos=pos,
            T0=T_INITIAL
        )
        ctrl = controller_factory()
        t, T_field, T_therm = model.simulate(ctrl.get_u, t_end=t_end, dt=0.05)

        # Compute control signal for metrics
        u = np.array([ctrl.get_u(ti, Ti) for ti, Ti in zip(t, T_therm)])

        m = compute_all_metrics(t, T_therm, u, T_SET)
        results[pos] = (t, T_field, T_therm, u, m)

        # Reset controller state for next run
        if hasattr(ctrl, 'reset'):
            ctrl.reset()

    return results


def main():
    print("=" * 70)
    print("1D PDE: Thermostat Placement Experiment")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)

    thermostat_positions = [0.5, 1.5, 2.5, 3.5, 4.5]
    t_end = 60.0

    # --- PID controller at various positions ---
    print("\n--- PID Controller: varying thermostat position ---")
    Kp, Ki, Kd = ziegler_nichols_tuning(K_COOL, U_MAX)

    def make_pid():
        return PIDController(Kp=Kp, Ki=Ki, Kd=Kd, T_set=T_SET, U_max=U_MAX,
                             dt=0.05)

    pid_results = run_1d_experiment(make_pid, thermostat_positions, t_end=t_end)

    # Print metrics
    print(f"\n{'Position':>10} {'Energy':>10} {'RMSE':>8} {'Settle':>8}")
    print("-" * 40)
    for pos, (t, T_field, T_therm, u, m) in pid_results.items():
        print(f"{pos:>10.1f} {m['energy']:>10.1f} {m['rmse']:>8.3f} "
              f"{m['settling_time']:>8.1f}")

    # --- Plot heatmap for middle position ---
    mid_pos = 2.5
    t, T_field, T_therm, u, m = pid_results[mid_pos]
    model = HeatEquation1D(thermostat_pos=mid_pos)
    plot_1d_heatmap(t, model.x, T_field,
                    title=f"1D Temperature Field (PID, thermostat at x={mid_pos}m)",
                    save_path=os.path.join(output_dir, '1d_heatmap_pid.png'))

    # --- Comparison: thermostat reading at different positions ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for pos in thermostat_positions:
        t, T_field, T_therm, u, m = pid_results[pos]
        ax.plot(t, T_therm, linewidth=1.5, label=f'x = {pos:.1f} m')

    ax.axhline(y=T_SET, color='r', linestyle='--', alpha=0.7,
               label=f'$T_{{set}}$ = {T_SET}°C')
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Thermostat Temperature (°C)', fontsize=12)
    ax.set_title('1D PDE: Thermostat Position Effect (PID Controller)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1d_placement_pid.png'),
                dpi=150, bbox_inches='tight')
    print(f"\nSaved: {os.path.join(output_dir, '1d_placement_pid.png')}")
    plt.close()

    # --- Metrics comparison for different positions ---
    pos_metrics = {f'x={p:.1f}m': m for p, (_, _, _, _, m) in pid_results.items()}
    plot_metrics_bar(pos_metrics,
                     save_path=os.path.join(output_dir, '1d_placement_metrics.png'))

    print("\nAll 1D placement results saved to:", output_dir)


if __name__ == "__main__":
    main()
