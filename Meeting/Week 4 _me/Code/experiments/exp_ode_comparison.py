"""
Experiment: Compare all control strategies on the ODE model.

Includes:
- Bang-Bang (hysteresis), PID, LQR, Pontryagin
- Parameter sweeps (PID tuning, LQR Q/R scan)
- Pareto front generation
- Metrics table
"""

import numpy as np
import sys, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.ode_model import simulate_ode, simulate_ode_switching, steady_state_temperature
from controllers.bang_bang import BangBangController
from controllers.pid import PIDController
from controllers.lqr import LQRController
from controllers.pontryagin import pontryagin_bvp
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
    ctrl = BangBangController(T_set=T_SET, U_max=U_MAX, delta=delta, initial_on=True)
    t, T, heater = simulate_ode_switching(ctrl, T0=T_INITIAL, t_end=t_end)
    u = heater * U_MAX
    return t, T, u, heater


def run_pid(Kp=2.0, Ki=0.3, Kd=1.0, t_end=T_END):
    """Run PID controller — simulate and record u simultaneously."""
    ctrl = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, T_set=T_SET, U_max=U_MAX, dt=DT)
    t, T = simulate_ode(ctrl.get_u, T0=T_INITIAL, t_end=t_end)
    # LQR is stateless so we can replay; PID is stateful so we record u via a fresh run
    ctrl2 = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, T_set=T_SET, U_max=U_MAX, dt=DT)
    u = np.zeros_like(t)
    for i in range(len(t)):
        u[i] = ctrl2.get_u(t[i], T[i])
    return t, T, u, None


def run_lqr(Q=1.0, R=0.01, t_end=T_END):
    ctrl = LQRController(Q=Q, R=R)
    t, T = simulate_ode(ctrl.get_u, T0=T_INITIAL, t_end=t_end)
    u = np.array([ctrl.get_u(ti, Ti) for ti, Ti in zip(t, T)])
    return t, T, u, None


def run_pontryagin(Q=1.0, R=0.01, t_end=T_END):
    """Run Pontryagin with continuation method for robust convergence."""
    t, T, u, lam, sol = pontryagin_bvp(
        Q=Q, R=R, T0=T_INITIAL, T_set=T_SET, T_a=T_AMBIENT,
        k=K_COOL, U_max=U_MAX, t_end=t_end, n_nodes=800,
        continuation=True, verbose=True
    )
    if not sol.success:
        print(f"  WARNING: BVP did not converge with R={R}")
        print(f"  Falling back to R=0.1...")
        t, T, u, lam, sol = pontryagin_bvp(
            Q=Q, R=0.1, T0=T_INITIAL, T_set=T_SET, T_a=T_AMBIENT,
            k=K_COOL, U_max=U_MAX, t_end=t_end, n_nodes=800,
            continuation=True
        )
    return t, T, u, None


def pid_parameter_sweep(output_dir):
    """Sweep PID gains and find best combination."""
    print("\n--- PID Parameter Sweep ---")
    Kp_values = [0.5, 1.0, 2.0, 4.0, 8.0]
    Ki_values = [0.05, 0.1, 0.3, 0.5, 1.0]
    Kd_values = [0.0, 0.5, 1.0, 2.0]

    best_cost = float('inf')
    best_params = None
    results_table = []

    # Coarse grid
    for Kp in Kp_values:
        for Ki in Ki_values:
            for Kd in Kd_values:
                try:
                    ctrl = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, T_set=T_SET,
                                         U_max=U_MAX, dt=DT)
                    t, T = simulate_ode(ctrl.get_u, T0=T_INITIAL, t_end=T_END)
                    ctrl2 = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, T_set=T_SET,
                                          U_max=U_MAX, dt=DT)
                    u = np.array([ctrl2.get_u(ti, Ti) for ti, Ti in zip(t, T)])
                    m = compute_all_metrics(t, T, u, T_SET)
                    results_table.append((Kp, Ki, Kd, m))
                    if m['unified_cost'] < best_cost:
                        best_cost = m['unified_cost']
                        best_params = (Kp, Ki, Kd)
                except Exception:
                    pass

    print(f"  Best PID: Kp={best_params[0]}, Ki={best_params[1]}, Kd={best_params[2]}")
    print(f"  Cost J = {best_cost:.2f}")

    # Plot top 5 PID configs
    results_table.sort(key=lambda x: x[3]['unified_cost'])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    for Kp, Ki, Kd, m in results_table[:5]:
        ctrl = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, T_set=T_SET, U_max=U_MAX, dt=DT)
        t, T = simulate_ode(ctrl.get_u, T0=T_INITIAL, t_end=T_END)
        ctrl2 = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, T_set=T_SET, U_max=U_MAX, dt=DT)
        u = np.array([ctrl2.get_u(ti, Ti) for ti, Ti in zip(t, T)])
        label = f'Kp={Kp}, Ki={Ki}, Kd={Kd} (J={m["unified_cost"]:.1f})'
        ax1.plot(t, T, linewidth=1.2, label=label)
        ax2.plot(t, u, linewidth=1.0)

    ax1.axhline(y=T_SET, color='k', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('PID Parameter Sweep — Top 5 Configurations')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('Time (min)')
    ax2.set_ylabel('Control u(t)')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ode_pid_sweep.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: ode_pid_sweep.png")

    return best_params, results_table


def lqr_pareto_sweep(output_dir):
    """Sweep LQR Q/R ratio to generate Pareto front."""
    print("\n--- LQR Pareto Sweep ---")
    Q_values = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    R_values = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

    pareto_points = []
    for Q in Q_values:
        for R in R_values:
            try:
                t, T, u, _ = run_lqr(Q=Q, R=R)
                m = compute_all_metrics(t, T, u, T_SET, Q=Q, R=R)
                pareto_points.append((Q, R, m))
            except Exception:
                pass

    # Plot Pareto front: energy vs RMSE
    fig, ax = plt.subplots(figsize=(10, 7))
    energies = [p[2]['energy'] for p in pareto_points]
    rmses = [p[2]['rmse'] for p in pareto_points]
    qr_ratios = [p[0] / p[1] for p in pareto_points]

    sc = ax.scatter(energies, rmses, c=np.log10(qr_ratios), cmap='viridis',
                    s=60, alpha=0.8, edgecolors='k', linewidths=0.5)
    cb = plt.colorbar(sc, ax=ax, label='log10(Q/R)')
    ax.set_xlabel('Energy Consumption', fontsize=12)
    ax.set_ylabel('Temperature RMSE (°C)', fontsize=12)
    ax.set_title('LQR Pareto Front: Energy vs Comfort (Q/R sweep)', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ode_lqr_pareto_sweep.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"  Saved: ode_lqr_pareto_sweep.png")
    print(f"  Total Pareto points: {len(pareto_points)}")

    return pareto_points


def main():
    print("=" * 70)
    print("ODE Model: Full Experiment Suite")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)

    # ==================== 1. Run all controllers ====================
    print("\n[1/4] Running Bang-Bang controller...")
    t_bb, T_bb, u_bb, h_bb = run_bang_bang()
    print(f"  Final T = {T_bb[-1]:.2f}°C, switches = {int(np.sum(np.abs(np.diff(h_bb)) > 0.5))}")

    print("[2/4] Running PID controller...")
    t_pid, T_pid, u_pid, _ = run_pid(Kp=2.0, Ki=0.3, Kd=1.0)
    print(f"  Final T = {T_pid[-1]:.2f}°C")

    print("[3/4] Running LQR controller...")
    t_lqr, T_lqr, u_lqr, _ = run_lqr(Q=1.0, R=0.01)
    print(f"  Final T = {T_lqr[-1]:.2f}°C")

    print("[4/4] Running Pontryagin controller...")
    t_pon, T_pon, u_pon, _ = run_pontryagin(Q=1.0, R=0.1)
    print(f"  Final T = {T_pon[-1]:.2f}°C")

    # ==================== 2. Metrics ====================
    print("\n--- Metrics Comparison ---")
    metrics = {}
    metrics['Bang-Bang'] = compute_all_metrics(t_bb, T_bb, u_bb, T_SET, heater=h_bb)
    metrics['PID'] = compute_all_metrics(t_pid, T_pid, u_pid, T_SET)
    metrics['LQR'] = compute_all_metrics(t_lqr, T_lqr, u_lqr, T_SET)
    metrics['Pontryagin'] = compute_all_metrics(t_pon, T_pon, u_pon, T_SET)

    header = f"{'Strategy':<15} {'Energy':>10} {'RMSE':>8} {'Overshoot':>10} {'Settle':>8} {'Switches':>10} {'Cost J':>10}"
    print(header)
    print("-" * len(header))
    for name, m in metrics.items():
        print(f"{name:<15} {m['energy']:>10.1f} {m['rmse']:>8.3f} "
              f"{m['max_overshoot']:>10.3f} {m['settling_time']:>8.1f} "
              f"{m['switching_count']:>10d} {m['unified_cost']:>10.1f}")

    # ==================== 3. Individual plots ====================
    print("\n--- Individual Plots ---")
    plot_temperature_and_heater(t_bb, T_bb, h_bb,
        title="Bang-Bang Control (Hysteresis ±0.5°C)", delta=HYSTERESIS_BAND,
        save_path=os.path.join(output_dir, 'ode_bangbang.png'))

    plot_temperature_continuous(t_pid, T_pid, u_pid, title="PID Control",
        save_path=os.path.join(output_dir, 'ode_pid.png'))

    plot_temperature_continuous(t_lqr, T_lqr, u_lqr, title="LQR Optimal Control",
        save_path=os.path.join(output_dir, 'ode_lqr.png'))

    plot_temperature_continuous(t_pon, T_pon, u_pon, title="Pontryagin Optimal Control",
        save_path=os.path.join(output_dir, 'ode_pontryagin.png'))

    # Comparison overlay
    results = {
        'Bang-Bang': (t_bb, T_bb, u_bb),
        'PID': (t_pid, T_pid, u_pid),
        'LQR': (t_lqr, T_lqr, u_lqr),
        'Pontryagin': (t_pon, T_pon, u_pon),
    }
    plot_comparison(results, T_set=T_SET,
        save_path=os.path.join(output_dir, 'ode_comparison.png'))

    plot_metrics_bar(metrics, save_path=os.path.join(output_dir, 'ode_metrics.png'))
    plot_pareto(metrics, save_path=os.path.join(output_dir, 'ode_pareto.png'))

    # ==================== 4. PID parameter sweep ====================
    best_pid, pid_table = pid_parameter_sweep(output_dir)

    # Re-run with best PID
    print(f"\n--- Re-running with best PID: Kp={best_pid[0]}, Ki={best_pid[1]}, Kd={best_pid[2]} ---")
    t_pid_best, T_pid_best, u_pid_best, _ = run_pid(*best_pid)
    metrics['PID (tuned)'] = compute_all_metrics(t_pid_best, T_pid_best, u_pid_best, T_SET)
    results['PID (tuned)'] = (t_pid_best, T_pid_best, u_pid_best)

    # ==================== 5. LQR Pareto sweep ====================
    lqr_pareto = lqr_pareto_sweep(output_dir)

    # ==================== 6. Final comparison with tuned controllers ====================
    print("\n--- Final Comparison (with tuned PID) ---")
    header = f"{'Strategy':<15} {'Energy':>10} {'RMSE':>8} {'Overshoot':>10} {'Settle':>8} {'Switches':>10} {'Cost J':>10}"
    print(header)
    print("-" * len(header))
    for name, m in metrics.items():
        print(f"{name:<15} {m['energy']:>10.1f} {m['rmse']:>8.3f} "
              f"{m['max_overshoot']:>10.3f} {m['settling_time']:>8.1f} "
              f"{m['switching_count']:>10d} {m['unified_cost']:>10.1f}")

    plot_comparison(results, T_set=T_SET,
        save_path=os.path.join(output_dir, 'ode_comparison_final.png'))
    plot_pareto(metrics, save_path=os.path.join(output_dir, 'ode_pareto_final.png'))

    # ==================== 7. Bang-Bang hysteresis sweep ====================
    print("\n--- Bang-Bang Hysteresis Sweep ---")
    delta_values = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    bb_metrics = {}
    for d in delta_values:
        t, T, u, h = run_bang_bang(delta=d)
        m = compute_all_metrics(t, T, u, T_SET, heater=h)
        bb_metrics[f'δ={d}'] = m
        print(f"  δ={d:.2f}: switches={m['switching_count']}, RMSE={m['rmse']:.3f}, energy={m['energy']:.1f}")

    plot_pareto(bb_metrics, save_path=os.path.join(output_dir, 'ode_bb_hysteresis_pareto.png'))

    print(f"\n{'='*70}")
    print("All ODE experiments complete.")
    print(f"Results in: {output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
