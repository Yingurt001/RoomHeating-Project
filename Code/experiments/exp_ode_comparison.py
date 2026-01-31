"""
Experiment: Compare all control strategies on the ODE model.

Runs Bang-Bang, PID, LQR, and Pontryagin controllers on the same
ODE model and produces comparison plots + metrics table.
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.ode_model import simulate_ode, simulate_ode_switching
from controllers.bang_bang import BangBangController
from controllers.pid import PIDController, ziegler_nichols_tuning
from controllers.lqr import LQRController
from controllers.pontryagin import PontryaginController
from utils.parameters import (
    T_AMBIENT, T_INITIAL, T_SET, U_MAX, K_COOL, T_END, DT,
    HYSTERESIS_BAND
)
from utils.metrics import compute_all_metrics
from utils.plotting import (
    plot_temperature_and_heater, plot_temperature_continuous,
    plot_comparison, plot_metrics_bar, plot_pareto
)


def run_bang_bang(t_end=T_END, delta=HYSTERESIS_BAND):
    """Run Bang-Bang with hysteresis."""
    ctrl = BangBangController(T_set=T_SET, U_max=U_MAX, delta=delta,
                              initial_on=True)
    t, T, heater = simulate_ode_switching(ctrl, T0=T_INITIAL, t_end=t_end)
    u = heater * U_MAX
    return t, T, u, heater


def run_pid(t_end=T_END):
    """Run PID controller."""
    Kp, Ki, Kd = ziegler_nichols_tuning(K_COOL, U_MAX)
    ctrl = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, T_set=T_SET, U_max=U_MAX, dt=DT)
    t, T = simulate_ode(ctrl.get_u, T0=T_INITIAL, t_end=t_end)
    u = np.array([ctrl.get_u(ti, Ti) for ti, Ti in zip(t, T)])
    # Re-simulate to get correct u (controller state was consumed)
    ctrl.reset()
    u = np.zeros_like(t)
    for i in range(len(t)):
        u[i] = ctrl.get_u(t[i], T[i])
    return t, T, u, None


def run_lqr(Q=1.0, R=0.01, t_end=T_END):
    """Run LQR controller."""
    ctrl = LQRController(Q=Q, R=R)
    t, T = simulate_ode(ctrl.get_u, T0=T_INITIAL, t_end=t_end)
    u = np.array([ctrl.get_u(ti, Ti) for ti, Ti in zip(t, T)])
    return t, T, u, None


def run_pontryagin(Q=1.0, R=0.01, t_end=T_END):
    """Run Pontryagin optimal controller."""
    ctrl = PontryaginController(Q=Q, R=R, T0=T_INITIAL, t_end=t_end)
    t_opt, T_opt, u_opt, lam_opt = ctrl.get_trajectory()
    return t_opt, T_opt, u_opt, None


def main():
    print("=" * 70)
    print("ODE Model: Controller Comparison Experiment")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)

    # --- Run all controllers ---
    print("\n[1/4] Running Bang-Bang controller...")
    t_bb, T_bb, u_bb, h_bb = run_bang_bang()
    print(f"  Done. Final T = {T_bb[-1]:.2f}°C")

    print("[2/4] Running PID controller...")
    t_pid, T_pid, u_pid, _ = run_pid()
    print(f"  Done. Final T = {T_pid[-1]:.2f}°C")

    print("[3/4] Running LQR controller...")
    t_lqr, T_lqr, u_lqr, _ = run_lqr()
    print(f"  Done. Final T = {T_lqr[-1]:.2f}°C")

    print("[4/4] Running Pontryagin controller...")
    t_pon, T_pon, u_pon, _ = run_pontryagin()
    print(f"  Done. Final T = {T_pon[-1]:.2f}°C")

    # --- Compute metrics ---
    print("\n--- Metrics ---")
    metrics = {}
    metrics['Bang-Bang'] = compute_all_metrics(t_bb, T_bb, u_bb, T_SET,
                                                heater=h_bb)
    metrics['PID'] = compute_all_metrics(t_pid, T_pid, u_pid, T_SET)
    metrics['LQR'] = compute_all_metrics(t_lqr, T_lqr, u_lqr, T_SET)
    metrics['Pontryagin'] = compute_all_metrics(t_pon, T_pon, u_pon, T_SET)

    # Print metrics table
    header = f"{'Strategy':<15} {'Energy':>10} {'RMSE':>8} {'Overshoot':>10} {'Settle':>8} {'Switches':>10} {'Cost J':>10}"
    print(header)
    print("-" * len(header))
    for name, m in metrics.items():
        print(f"{name:<15} {m['energy']:>10.1f} {m['rmse']:>8.3f} "
              f"{m['max_overshoot']:>10.3f} {m['settling_time']:>8.1f} "
              f"{m['switching_count']:>10d} {m['unified_cost']:>10.1f}")

    # --- Generate plots ---
    print("\n--- Generating plots ---")

    plot_temperature_and_heater(
        t_bb, T_bb, h_bb,
        title="Bang-Bang Control (Hysteresis ±0.5°C)",
        delta=HYSTERESIS_BAND,
        save_path=os.path.join(output_dir, 'ode_bangbang.png')
    )

    plot_temperature_continuous(
        t_pid, T_pid, u_pid,
        title="PID Control",
        save_path=os.path.join(output_dir, 'ode_pid.png')
    )

    plot_temperature_continuous(
        t_lqr, T_lqr, u_lqr,
        title="LQR Optimal Control",
        save_path=os.path.join(output_dir, 'ode_lqr.png')
    )

    plot_temperature_continuous(
        t_pon, T_pon, u_pon,
        title="Pontryagin Optimal Control",
        save_path=os.path.join(output_dir, 'ode_pontryagin.png')
    )

    # Comparison plot
    results = {
        'Bang-Bang': (t_bb, T_bb, u_bb),
        'PID': (t_pid, T_pid, u_pid),
        'LQR': (t_lqr, T_lqr, u_lqr),
        'Pontryagin': (t_pon, T_pon, u_pon),
    }
    plot_comparison(results, T_set=T_SET,
                    save_path=os.path.join(output_dir, 'ode_comparison.png'))

    # Metrics bar chart
    plot_metrics_bar(metrics,
                     save_path=os.path.join(output_dir, 'ode_metrics.png'))

    # Pareto front
    plot_pareto(metrics,
                save_path=os.path.join(output_dir, 'ode_pareto.png'))

    print("\nAll results saved to:", output_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
