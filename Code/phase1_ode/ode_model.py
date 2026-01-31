"""
Phase 1: ODE Model — Newton's Law of Cooling with Bang-Bang Thermostat Control

Model:
    dT/dt = -k * (T - T_a) + u(T)

where:
    T(t)  = room temperature at time t
    T_a   = ambient outside temperature
    k     = cooling constant (heat loss rate)
    u(T)  = heater input, controlled by thermostat

Bang-Bang control (no hysteresis):
    u = U_max   if T < T_set
    u = 0       if T >= T_set

Bang-Bang control (with hysteresis band ±δ):
    u switches ON  when T falls below  T_set - δ
    u switches OFF when T rises above   T_set + δ
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.parameters import (
    T_AMBIENT, T_INITIAL, T_SET, U_MAX, K_COOL,
    HYSTERESIS_BAND, T_END, DT
)


# ============================================================
# 1. Bang-Bang Control (no hysteresis)
# ============================================================

def bang_bang_rhs(t, T, k=K_COOL, T_a=T_AMBIENT, T_set=T_SET, U_max=U_MAX):
    """Right-hand side of the ODE with simple Bang-Bang control."""
    u = U_max if T[0] < T_set else 0.0
    dTdt = -k * (T[0] - T_a) + u
    return [dTdt]


def solve_bang_bang(T0=T_INITIAL, t_end=T_END, k=K_COOL, T_a=T_AMBIENT,
                    T_set=T_SET, U_max=U_MAX, max_switches=500):
    """
    Solve the ODE with Bang-Bang control using event detection.
    We integrate in segments, switching heater state at each crossing of T_set.
    max_switches limits iterations to avoid Zeno-like infinite loops.
    """
    t_all = [0.0]
    T_all = [T0]
    heater_all = [1 if T0 < T_set else 0]

    t_current = 0.0
    T_current = T0
    heater_on = T0 < T_set
    n_switches = 0

    while t_current < t_end and n_switches < max_switches:
        if heater_on:
            # Heater ON: dT/dt = -k(T - T_a) + U_max
            # Event: T crosses T_set from below
            def rhs(t, T):
                return [-k * (T[0] - T_a) + U_max]

            def event_off(t, T):
                return T[0] - T_set
            event_off.terminal = True
            event_off.direction = 1  # crossing upward
        else:
            # Heater OFF: dT/dt = -k(T - T_a)
            # Event: T crosses T_set from above
            def rhs(t, T):
                return [-k * (T[0] - T_a)]

            def event_on(t, T):
                return T[0] - T_set
            event_on.terminal = True
            event_on.direction = -1  # crossing downward

        events = [event_off] if heater_on else [event_on]

        sol = solve_ivp(
            rhs,
            [t_current, t_end],
            [T_current],
            events=events,
            max_step=DT,
            dense_output=True
        )

        # Store results (skip first point to avoid duplicates)
        t_all.extend(sol.t[1:].tolist())
        T_all.extend(sol.y[0, 1:].tolist())
        heater_all.extend([int(heater_on)] * (len(sol.t) - 1))

        # Update state
        t_current = sol.t[-1]
        T_current = sol.y[0, -1]

        # Switch heater
        if sol.t_events[0].size > 0:
            heater_on = not heater_on
            n_switches += 1

    return np.array(t_all), np.array(T_all), np.array(heater_all)


# ============================================================
# 2. Bang-Bang Control WITH hysteresis
# ============================================================

def solve_bang_bang_hysteresis(T0=T_INITIAL, t_end=T_END, k=K_COOL, T_a=T_AMBIENT,
                               T_set=T_SET, U_max=U_MAX, delta=HYSTERESIS_BAND):
    """
    Bang-Bang with hysteresis band [T_set - delta, T_set + delta].
    Heater turns ON  when T drops below T_set - delta.
    Heater turns OFF when T rises above T_set + delta.
    """
    T_low = T_set - delta
    T_high = T_set + delta

    t_all = [0.0]
    T_all = [T0]
    heater_on = T0 < T_low
    heater_all = [int(heater_on)]

    t_current = 0.0
    T_current = T0

    while t_current < t_end:
        if heater_on:
            def rhs(t, T):
                return [-k * (T[0] - T_a) + U_max]

            def event_off(t, T):
                return T[0] - T_high
            event_off.terminal = True
            event_off.direction = 1
            events = [event_off]
        else:
            def rhs(t, T):
                return [-k * (T[0] - T_a)]

            def event_on(t, T):
                return T[0] - T_low
            event_on.terminal = True
            event_on.direction = -1
            events = [event_on]

        sol = solve_ivp(
            rhs,
            [t_current, t_end],
            [T_current],
            events=events,
            max_step=DT,
            dense_output=True
        )

        t_all.extend(sol.t[1:].tolist())
        T_all.extend(sol.y[0, 1:].tolist())
        heater_all.extend([int(heater_on)] * (len(sol.t) - 1))

        t_current = sol.t[-1]
        T_current = sol.y[0, -1]

        if sol.t_events[0].size > 0:
            heater_on = not heater_on

    return np.array(t_all), np.array(T_all), np.array(heater_all)


# ============================================================
# 3. Analytical steady-state analysis
# ============================================================

def steady_state_temperature(k=K_COOL, T_a=T_AMBIENT, U_max=U_MAX):
    """
    Steady state when heater is always ON:
        dT/dt = 0  =>  T_ss = T_a + U_max / k
    This is the maximum temperature the room can reach.
    """
    return T_a + U_max / k


def oscillation_period_estimate(k=K_COOL, T_a=T_AMBIENT, T_set=T_SET,
                                 U_max=U_MAX, delta=HYSTERESIS_BAND):
    """
    Estimate oscillation period for Bang-Bang with hysteresis.

    Heating phase: dT/dt = -k(T - T_a) + U_max = -k*T + (k*T_a + U_max)
        Solution: T(t) = T_ss + (T_low - T_ss) * exp(-k*t), where T_ss = T_a + U_max/k
        Time to go from T_low to T_high: t_heat = (1/k) * ln((T_ss - T_low) / (T_ss - T_high))

    Cooling phase: dT/dt = -k(T - T_a)
        Solution: T(t) = T_a + (T_high - T_a) * exp(-k*t)
        Time to go from T_high to T_low: t_cool = (1/k) * ln((T_high - T_a) / (T_low - T_a))
    """
    T_low = T_set - delta
    T_high = T_set + delta
    T_ss = T_a + U_max / k

    if T_ss <= T_high:
        return float('inf')  # heater cannot reach T_high

    t_heat = (1.0 / k) * np.log((T_ss - T_low) / (T_ss - T_high))
    t_cool = (1.0 / k) * np.log((T_high - T_a) / (T_low - T_a))

    return t_heat + t_cool


# ============================================================
# 4. Plotting
# ============================================================

def plot_results(t, T, heater, title="Bang-Bang Thermostat Control",
                 T_set=T_SET, T_a=T_AMBIENT, delta=None, save_path=None):
    """Plot temperature and heater state over time."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    # Temperature plot
    ax1.plot(t, T, 'b-', linewidth=1.5, label='Room Temperature $T(t)$')
    ax1.axhline(y=T_set, color='r', linestyle='--', alpha=0.7, label=f'$T_{{set}} = {T_set}°C$')
    ax1.axhline(y=T_a, color='cyan', linestyle=':', alpha=0.7, label=f'$T_a = {T_a}°C$ (outside)')

    if delta is not None:
        ax1.axhline(y=T_set + delta, color='orange', linestyle='--', alpha=0.5,
                     label=f'Hysteresis band ±{delta}°C')
        ax1.axhline(y=T_set - delta, color='orange', linestyle='--', alpha=0.5)
        ax1.axhspan(T_set - delta, T_set + delta, alpha=0.08, color='orange')

    ax1.set_ylabel('Temperature (°C)', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Heater state plot
    ax2.fill_between(t, heater, step='post', alpha=0.4, color='red', label='Heater ON')
    ax2.step(t, heater, 'r-', linewidth=1, where='post')
    ax2.set_xlabel('Time (minutes)', fontsize=12)
    ax2.set_ylabel('Heater State', fontsize=12)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['OFF', 'ON'])
    ax2.set_ylim(-0.1, 1.3)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_parameter_sensitivity(param_name, param_values, results_list,
                                T_set=T_SET, save_path=None):
    """Plot multiple runs varying one parameter."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for val, (t, T, _) in zip(param_values, results_list):
        ax.plot(t, T, linewidth=1.2, label=f'{param_name} = {val}')

    ax.axhline(y=T_set, color='r', linestyle='--', alpha=0.7, label=f'$T_{{set}} = {T_set}°C$')
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.set_title(f'Parameter Sensitivity: {param_name}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# 5. Main — run all experiments
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1: ODE Model — Newton's Law of Cooling")
    print("=" * 60)

    # --- Steady state analysis ---
    T_ss = steady_state_temperature()
    print(f"\nSteady-state temperature (heater always ON): {T_ss:.1f} °C")
    print(f"  (T_a={T_AMBIENT}, U_max={U_MAX}, k={K_COOL})")

    if T_ss < T_SET:
        print(f"  WARNING: Heater cannot reach T_set={T_SET}°C. Need stronger heater or better insulation.")

    # --- Experiment 1: Bang-Bang without hysteresis ---
    print(f"\n--- Experiment 1: Bang-Bang (no hysteresis) ---")
    t1, T1, h1 = solve_bang_bang()
    n_switches = np.sum(np.abs(np.diff(h1)))
    print(f"  Number of heater switches: {n_switches}")
    print(f"  Final temperature: {T1[-1]:.2f} °C")

    plot_results(t1, T1, h1,
                 title="Experiment 1: Bang-Bang Control (No Hysteresis)",
                 save_path="exp1_bang_bang.png")

    # --- Experiment 2: Bang-Bang with hysteresis ---
    print(f"\n--- Experiment 2: Bang-Bang (hysteresis ±{HYSTERESIS_BAND}°C) ---")
    t2, T2, h2 = solve_bang_bang_hysteresis()
    n_switches_h = np.sum(np.abs(np.diff(h2)))
    period_est = oscillation_period_estimate()
    print(f"  Number of heater switches: {n_switches_h}")
    print(f"  Estimated oscillation period: {period_est:.2f} min")
    print(f"  Final temperature: {T2[-1]:.2f} °C")

    plot_results(t2, T2, h2,
                 title=f"Experiment 2: Bang-Bang Control (Hysteresis ±{HYSTERESIS_BAND}°C)",
                 delta=HYSTERESIS_BAND,
                 save_path="exp2_hysteresis.png")

    # --- Experiment 3: Parameter sensitivity — varying k ---
    print(f"\n--- Experiment 3: Sensitivity to cooling constant k ---")
    k_values = [0.05, 0.1, 0.15, 0.2, 0.3]
    results_k = []
    for k in k_values:
        res = solve_bang_bang_hysteresis(k=k)
        results_k.append(res)
        n_sw = np.sum(np.abs(np.diff(res[2])))
        print(f"  k={k:.2f}: switches={n_sw}, final T={res[1][-1]:.2f}°C")

    plot_parameter_sensitivity('k (cooling constant)', k_values, results_k,
                                save_path="exp3_sensitivity_k.png")

    # --- Experiment 4: Parameter sensitivity — varying U_max ---
    print(f"\n--- Experiment 4: Sensitivity to heater power U_max ---")
    u_values = [5.0, 10.0, 15.0, 20.0, 30.0]
    results_u = []
    for u in u_values:
        res = solve_bang_bang_hysteresis(U_max=u)
        results_u.append(res)
        n_sw = np.sum(np.abs(np.diff(res[2])))
        T_ss_u = steady_state_temperature(U_max=u)
        print(f"  U_max={u:.1f}: switches={n_sw}, T_steady={T_ss_u:.1f}°C, final T={res[1][-1]:.2f}°C")

    plot_parameter_sensitivity('$U_{max}$ (heater power)', u_values, results_u,
                                save_path="exp4_sensitivity_umax.png")

    # --- Experiment 5: Varying hysteresis band ---
    print(f"\n--- Experiment 5: Effect of hysteresis band width ---")
    delta_values = [0.0, 0.25, 0.5, 1.0, 2.0]
    results_delta = []
    for d in delta_values:
        if d == 0:
            res = solve_bang_bang()
        else:
            res = solve_bang_bang_hysteresis(delta=d)
        results_delta.append(res)
        n_sw = np.sum(np.abs(np.diff(res[2])))
        print(f"  delta={d:.2f}°C: switches={n_sw}")

    plot_parameter_sensitivity('δ (hysteresis band)', delta_values, results_delta,
                                save_path="exp5_sensitivity_delta.png")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("Summary of key findings:")
    print(f"  - Room reaches ~{T_SET}°C from {T_INITIAL}°C with outside temp {T_AMBIENT}°C")
    print(f"  - Without hysteresis: rapid switching (chattering) near T_set")
    print(f"  - With hysteresis ±{HYSTERESIS_BAND}°C: stable oscillation, period ≈ {period_est:.1f} min")
    print(f"  - Higher k (worse insulation) → faster cooling → more frequent switching")
    print(f"  - Higher U_max (stronger heater) → faster heating → shorter ON periods")
    print(f"{'=' * 60}")
