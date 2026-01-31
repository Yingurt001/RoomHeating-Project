"""
Zeno Effect Analysis for Bang-Bang Thermostat Control.

This experiment rigorously investigates the Zeno phenomenon:
    As hysteresis band delta -> 0, the switching frequency -> infinity,
    potentially producing infinitely many switches in finite time.

We provide:
1. Analytical derivation of switching intervals as a function of delta
2. Numerical verification across a range of delta values
3. Formal hybrid systems representation (Goebel, Sanfelice & Teel, 2012)
4. Convergence analysis: does the total switching time series converge?

References:
    - Zhang J, Johansson KH, Lygeros J, Sastry SS. Zeno hybrid systems.
      Int. J. Robust Nonlinear Control. 2001;11(5):435-451.
    - Goebel R, Sanfelice RG, Teel AR. Hybrid Dynamical Systems.
      Princeton University Press, 2012.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.parameters import (T_AMBIENT, T_INITIAL, T_SET, U_MAX, K_COOL,
                               T_END, DT)
from controllers.bang_bang import BangBangController
from models.ode_model import simulate_ode_switching

# Ensure results directory exists
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# =====================================================================
# Part 1: Analytical switching intervals
# =====================================================================

def analytical_switching_times(delta, k=K_COOL, T_a=T_AMBIENT, T_set=T_SET,
                                U_max=U_MAX):
    """
    Exact switching half-periods for the Bang-Bang thermostat ODE.

    ODE (heater ON):   dT/dt = -k(T - T_a) + U_max
                        => T(t) = T_ss + (T_start - T_ss)*exp(-k*t)
                        where T_ss = T_a + U_max/k

    ODE (heater OFF):  dT/dt = -k(T - T_a)
                        => T(t) = T_a + (T_start - T_a)*exp(-k*t)

    Heating phase: from T_low = T_set - delta to T_high = T_set + delta
        t_heat = (1/k) * ln((T_ss - T_low) / (T_ss - T_high))

    Cooling phase: from T_high to T_low
        t_cool = (1/k) * ln((T_high - T_a) / (T_low - T_a))

    As delta -> 0 (first-order Taylor expansion):
        t_heat ~ 2*delta / (k * (T_ss - T_set))
        t_cool ~ 2*delta / (k * (T_set - T_a))

    Both are O(delta), so the period P = t_heat + t_cool = O(delta).
    Number of switches N ~ T_end / P = O(1/delta) -> infinity.

    Returns
    -------
    t_heat, t_cool : float
        Heating and cooling half-periods (minutes).
    t_heat_approx, t_cool_approx : float
        First-order approximations.
    """
    T_low = T_set - delta
    T_high = T_set + delta
    T_ss = T_a + U_max / k

    if T_ss <= T_high:
        return np.inf, np.inf, np.inf, np.inf

    # Exact
    t_heat = (1.0 / k) * np.log((T_ss - T_low) / (T_ss - T_high))
    t_cool = (1.0 / k) * np.log((T_high - T_a) / (T_low - T_a))

    # First-order approximation (valid for small delta)
    t_heat_approx = 2 * delta / (k * (T_ss - T_set))
    t_cool_approx = 2 * delta / (k * (T_set - T_a))

    return t_heat, t_cool, t_heat_approx, t_cool_approx


def zeno_convergence_analysis(deltas=None, k=K_COOL, T_a=T_AMBIENT,
                               T_set=T_SET, U_max=U_MAX, T_end=T_END):
    """
    For each delta, compute:
    - Exact period P(delta)
    - Number of switches N(delta) = T_end / P(delta)
    - Total accumulated switching time sum

    Key question: does N -> infinity as delta -> 0?
    If P ~ C*delta, then N ~ T_end/(C*delta) -> infinity.
    This is "chattering Zeno" — not true Zeno (finite accumulation point),
    but the switching count diverges.
    """
    if deltas is None:
        deltas = np.logspace(-3, 0, 200)

    results = {
        'deltas': deltas,
        'periods': [],
        'n_switches': [],
        't_heat': [],
        't_cool': [],
        't_heat_approx': [],
        't_cool_approx': [],
    }

    T_ss = T_a + U_max / k

    for d in deltas:
        th, tc, th_a, tc_a = analytical_switching_times(d, k, T_a, T_set, U_max)
        P = th + tc
        N = T_end / P if P > 0 else np.inf
        results['periods'].append(P)
        results['n_switches'].append(N)
        results['t_heat'].append(th)
        results['t_cool'].append(tc)
        results['t_heat_approx'].append(th_a)
        results['t_cool_approx'].append(tc_a)

    for key in results:
        if key != 'deltas':
            results[key] = np.array(results[key])

    # Compute the linear coefficient: P ≈ C * delta
    # C = 2/(k*(T_ss - T_set)) + 2/(k*(T_set - T_a))
    C_exact = 2.0 / (k * (T_ss - T_set)) + 2.0 / (k * (T_set - T_a))
    results['C_linear'] = C_exact

    return results


# =====================================================================
# Part 2: Numerical verification via simulation
# =====================================================================

def simulate_zeno_numerical(delta_values=None, t_end=60.0, max_sw=50000):
    """
    Run Bang-Bang simulations for various delta values.
    Record actual switching times and intervals.
    """
    if delta_values is None:
        delta_values = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

    results = []

    for delta in delta_values:
        ctrl = BangBangController(T_set=T_SET, U_max=U_MAX, delta=delta,
                                   initial_on=True)
        t, T, heater = simulate_ode_switching(
            ctrl, T0=T_INITIAL, t_end=t_end, k=K_COOL,
            T_a=T_AMBIENT, U_max=U_MAX, dt=DT, max_switches=max_sw
        )

        # Find switch times from heater state changes
        switch_indices = np.where(np.abs(np.diff(heater)) > 0.5)[0]
        switch_times = t[switch_indices + 1]

        # Compute intervals between switches
        if len(switch_times) > 1:
            intervals = np.diff(switch_times)
        else:
            intervals = np.array([])

        # Only count switches after initial transient (after first reaching T_set)
        # to measure steady-state switching
        idx_settled = np.where(np.abs(T - T_SET) < delta + 1.0)[0]
        if len(idx_settled) > 0:
            t_settled = t[idx_settled[0]]
            ss_switches = switch_times[switch_times > t_settled]
            if len(ss_switches) > 1:
                ss_intervals = np.diff(ss_switches)
                mean_interval = np.mean(ss_intervals)
            else:
                mean_interval = np.inf
        else:
            ss_switches = switch_times
            mean_interval = np.inf

        results.append({
            'delta': delta,
            't': t,
            'T': T,
            'heater': heater,
            'n_switches': len(switch_times),
            'switch_times': switch_times,
            'intervals': intervals,
            'mean_ss_interval': mean_interval,
        })

        print(f"  delta={delta:.4f}: {len(switch_times)} switches, "
              f"mean interval={mean_interval:.4f} min")

    return results


# =====================================================================
# Part 3: Formal Zeno characterisation
# =====================================================================

def check_zeno_type(k=K_COOL, T_a=T_AMBIENT, T_set=T_SET, U_max=U_MAX):
    """
    Classify the Zeno behaviour of the Bang-Bang thermostat.

    Following Zhang et al. (2001), a Zeno execution is one where
    infinitely many discrete transitions occur in finite (hybrid) time.

    For Bang-Bang thermostat with delta > 0:
        - The switching intervals are CONSTANT (geometric ratio = 1)
        - Sum of intervals = N * P(delta) -> infinity as N -> infinity
        - Therefore: NO Zeno execution for any delta > 0

    For delta = 0 (ideal relay):
        - The switching surface is T = T_set (a single point)
        - Once T reaches T_set, the system enters a "sliding mode"
        - Practically: the system chatters with dt-limited switching
        - Mathematically: this is a FIRST-ORDER Zeno point
          (the switching times accumulate at a finite time)

    Classification:
        delta > 0: Non-Zeno, periodic limit cycle with period P(delta)
        delta -> 0+: Chattering Zeno (switching frequency diverges)
        delta = 0: Zeno point at t* where T(t*) = T_set for the first time
    """
    T_ss = T_a + U_max / k

    report = {}
    report['T_ss'] = T_ss
    report['T_ss_minus_Tset'] = T_ss - T_set
    report['Tset_minus_Ta'] = T_set - T_a

    # The equilibrium temperature with heater ON
    # T_ss = 155°C >> T_set = 20°C, so the system can always heat past T_set
    report['can_reach_Tset'] = T_ss > T_set

    # For delta > 0, compute period
    for delta in [0.5, 0.1, 0.01, 0.001]:
        th, tc, _, _ = analytical_switching_times(delta)
        P = th + tc
        report[f'period_delta_{delta}'] = P

    # Time to first reach T_set from T_initial
    # dT/dt = -k(T - T_a) + U_max = -kT + kT_a + U_max
    # T(t) = T_ss + (T0 - T_ss)*exp(-kt)
    # T(t*) = T_set => t* = -(1/k)*ln((T_set - T_ss)/(T_INITIAL - T_ss))
    t_reach = -(1.0 / k) * np.log((T_set - T_ss) / (T_INITIAL - T_ss))
    report['t_reach_Tset'] = t_reach

    # For delta = 0, at t = t_reach, the system hits the switching surface
    # and enters Zeno (chattering) because:
    #   dT/dt|_{ON} = -k(T_set - T_a) + U_max > 0 (wants to go up)
    #   dT/dt|_{OFF} = -k(T_set - T_a) < 0        (wants to go down)
    # Both vector fields point "through" the switching surface
    report['dTdt_on_at_Tset'] = -k * (T_set - T_a) + U_max
    report['dTdt_off_at_Tset'] = -k * (T_set - T_a)
    report['is_zeno_point'] = (report['dTdt_on_at_Tset'] > 0 and
                                report['dTdt_off_at_Tset'] < 0)

    # Filippov sliding mode: the equivalent control at T = T_set is
    # u_eq = k*(T_set - T_a), which keeps T constant at T_set
    report['u_eq_filippov'] = k * (T_set - T_a)
    report['u_eq_fraction'] = report['u_eq_filippov'] / U_max

    return report


# =====================================================================
# Part 4: Visualization
# =====================================================================

def plot_zeno_analysis(analytical_results, numerical_results, zeno_report):
    """Generate comprehensive Zeno analysis figure."""
    fig = plt.figure(figsize=(16, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    deltas = analytical_results['deltas']
    periods = analytical_results['periods']
    n_switches = analytical_results['n_switches']
    C = analytical_results['C_linear']

    # --- Panel (a): Period vs delta (log-log) ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.loglog(deltas, periods, 'b-', linewidth=2, label='Exact $P(\\delta)$')
    ax1.loglog(deltas, C * deltas, 'r--', linewidth=1.5,
               label=f'Linear approx $P \\approx {C:.2f}\\delta$')
    ax1.set_xlabel('Hysteresis band $\\delta$ (°C)', fontsize=11)
    ax1.set_ylabel('Oscillation period $P$ (min)', fontsize=11)
    ax1.set_title('(a) Switching Period vs Hysteresis Band', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- Panel (b): Number of switches vs delta ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.loglog(deltas, n_switches, 'b-', linewidth=2, label='Analytical')

    # Overlay numerical results
    num_deltas = [r['delta'] for r in numerical_results]
    num_nsw = [r['n_switches'] for r in numerical_results]
    ax2.loglog(num_deltas, num_nsw, 'ro', markersize=8, label='Numerical', zorder=5)

    ax2.set_xlabel('Hysteresis band $\\delta$ (°C)', fontsize=11)
    ax2.set_ylabel('Number of switches $N$', fontsize=11)
    ax2.set_title('(b) Switch Count Divergence as $\\delta \\to 0$', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # --- Panel (c): Exact vs approximate switching times ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.loglog(deltas, analytical_results['t_heat'], 'b-', linewidth=2,
               label='$t_{heat}$ (exact)')
    ax3.loglog(deltas, analytical_results['t_heat_approx'], 'b--', linewidth=1.5,
               label='$t_{heat}$ (approx)')
    ax3.loglog(deltas, analytical_results['t_cool'], 'r-', linewidth=2,
               label='$t_{cool}$ (exact)')
    ax3.loglog(deltas, analytical_results['t_cool_approx'], 'r--', linewidth=1.5,
               label='$t_{cool}$ (approx)')
    ax3.set_xlabel('Hysteresis band $\\delta$ (°C)', fontsize=11)
    ax3.set_ylabel('Half-period (min)', fontsize=11)
    ax3.set_title('(c) Heating and Cooling Half-Periods', fontsize=12)
    ax3.legend(fontsize=10, loc='upper left')
    ax3.grid(True, alpha=0.3)

    # --- Panel (d): Trajectories for selected delta values ---
    ax4 = fig.add_subplot(gs[1, 1])
    selected = [r for r in numerical_results if r['delta'] in [2.0, 0.5, 0.1, 0.02]]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, r in enumerate(selected):
        # Only plot after initial transient, near setpoint
        mask = r['t'] > 10
        ax4.plot(r['t'][mask], r['T'][mask], color=colors[i], linewidth=1.2,
                 label=f"$\\delta = {r['delta']}$°C")
    ax4.axhline(y=T_SET, color='k', linestyle=':', alpha=0.5, label='$T_{set}$')
    ax4.set_xlabel('Time (min)', fontsize=11)
    ax4.set_ylabel('Temperature (°C)', fontsize=11)
    ax4.set_title('(d) Temperature Trajectories (Steady State)', fontsize=12)
    ax4.set_xlim(20, 50)
    ax4.set_ylim(T_SET - 3, T_SET + 3)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # --- Panel (e): Switching interval time series ---
    ax5 = fig.add_subplot(gs[2, 0])
    for i, r in enumerate(selected):
        if len(r['intervals']) > 2:
            # Show last N intervals (steady state)
            ss_int = r['intervals'][-min(50, len(r['intervals'])):]
            ax5.plot(range(len(ss_int)), ss_int, 'o-', color=colors[i],
                     markersize=3, linewidth=1,
                     label=f"$\\delta = {r['delta']}$°C")
    ax5.set_xlabel('Switch index (last 50)', fontsize=11)
    ax5.set_ylabel('Interval between switches (min)', fontsize=11)
    ax5.set_title('(e) Steady-State Switching Intervals', fontsize=12)
    ax5.legend(fontsize=9)
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)

    # --- Panel (f): Hybrid automaton phase portrait ---
    ax6 = fig.add_subplot(gs[2, 1])
    # Show phase portrait: T on x-axis, dT/dt on y-axis, for both modes
    T_range = np.linspace(T_AMBIENT, T_SET + 5, 200)

    dTdt_on = -K_COOL * (T_range - T_AMBIENT) + U_MAX
    dTdt_off = -K_COOL * (T_range - T_AMBIENT)

    ax6.plot(T_range, dTdt_on, 'r-', linewidth=2, label='Heater ON: $f_1(T)$')
    ax6.plot(T_range, dTdt_off, 'b-', linewidth=2, label='Heater OFF: $f_0(T)$')
    ax6.axhline(y=0, color='k', linewidth=0.5)
    ax6.axvline(x=T_SET, color='gray', linestyle=':', alpha=0.7)

    # Mark the Zeno point
    ax6.plot(T_SET, zeno_report['dTdt_on_at_Tset'], 'rv', markersize=12,
             label=f"$f_1(T_{{set}}) = {zeno_report['dTdt_on_at_Tset']:.1f}$")
    ax6.plot(T_SET, zeno_report['dTdt_off_at_Tset'], 'b^', markersize=12,
             label=f"$f_0(T_{{set}}) = {zeno_report['dTdt_off_at_Tset']:.1f}$")

    # Shade the region where Zeno occurs (both flows cross the surface)
    ax6.fill_between([T_SET - 0.5, T_SET + 0.5],
                     [min(dTdt_off) - 1, min(dTdt_off) - 1],
                     [max(dTdt_on) + 1, max(dTdt_on) + 1],
                     alpha=0.1, color='purple', label='Switching surface')

    # Filippov equivalent
    u_eq = zeno_report['u_eq_filippov']
    dTdt_eq = -K_COOL * (T_SET - T_AMBIENT) + u_eq
    ax6.plot(T_SET, dTdt_eq, 'g*', markersize=15, zorder=10,
             label=f'Filippov: $u_{{eq}} = {u_eq:.2f}$')

    ax6.set_xlabel('Temperature $T$ (°C)', fontsize=11)
    ax6.set_ylabel('$dT/dt$ (°C/min)', fontsize=11)
    ax6.set_title('(f) Phase Portrait & Zeno Point Analysis', fontsize=12)
    ax6.legend(fontsize=8, loc='upper right')
    ax6.grid(True, alpha=0.3)

    plt.savefig(os.path.join(RESULTS_DIR, 'zeno_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: zeno_analysis.png")


def plot_hybrid_automaton_diagram():
    """
    Create a schematic diagram of the hybrid automaton.

    The Bang-Bang thermostat as a hybrid dynamical system:

    Hybrid automaton H = (Q, X, f, Init, Dom, E, G, R) where:
        Q = {q_ON, q_OFF}              — discrete states
        X = R (temperature)             — continuous state
        f(q_ON, T) = -k(T-T_a) + U     — flow in ON mode
        f(q_OFF,T) = -k(T-T_a)         — flow in OFF mode
        Dom(q_ON) = {T : T <= T_set+δ} — domain (flow set)
        Dom(q_OFF)= {T : T >= T_set-δ} — domain (flow set)
        G(q_ON->q_OFF) = {T : T = T_set+δ} — guard (jump condition)
        G(q_OFF->q_ON) = {T : T = T_set-δ} — guard (jump condition)
        R = identity                    — reset map (T unchanged at switch)
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Draw two state boxes
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    box_on = FancyBboxPatch((1, 2), 3, 2, boxstyle="round,pad=0.2",
                             facecolor='#ffcccc', edgecolor='red', linewidth=2)
    box_off = FancyBboxPatch((6, 2), 3, 2, boxstyle="round,pad=0.2",
                              facecolor='#cce5ff', edgecolor='blue', linewidth=2)
    ax.add_patch(box_on)
    ax.add_patch(box_off)

    # State labels
    ax.text(2.5, 3.3, '$q_{ON}$', fontsize=16, ha='center', va='center',
            fontweight='bold', color='red')
    ax.text(2.5, 2.6, '$\\dot{T} = -k(T-T_a) + U_{max}$', fontsize=10,
            ha='center', va='center', color='darkred')

    ax.text(7.5, 3.3, '$q_{OFF}$', fontsize=16, ha='center', va='center',
            fontweight='bold', color='blue')
    ax.text(7.5, 2.6, '$\\dot{T} = -k(T-T_a)$', fontsize=10,
            ha='center', va='center', color='darkblue')

    # Transition arrows
    # ON -> OFF (top arrow)
    ax.annotate('', xy=(6, 4.3), xytext=(4, 4.3),
                arrowprops=dict(arrowstyle='->', color='purple', lw=2))
    ax.text(5, 4.7, '$T = T_{set} + \\delta$\n(switch OFF)',
            fontsize=10, ha='center', va='center', color='purple')

    # OFF -> ON (bottom arrow)
    ax.annotate('', xy=(4, 1.7), xytext=(6, 1.7),
                arrowprops=dict(arrowstyle='->', color='purple', lw=2))
    ax.text(5, 1.1, '$T = T_{set} - \\delta$\n(switch ON)',
            fontsize=10, ha='center', va='center', color='purple')

    # Domain labels
    ax.text(2.5, 0.3, 'Dom: $T \\leq T_{set} + \\delta$',
            fontsize=10, ha='center', color='darkred',
            bbox=dict(boxstyle='round', facecolor='#fff0f0', alpha=0.8))
    ax.text(7.5, 0.3, 'Dom: $T \\geq T_{set} - \\delta$',
            fontsize=10, ha='center', color='darkblue',
            bbox=dict(boxstyle='round', facecolor='#f0f5ff', alpha=0.8))

    # Title
    ax.set_title('Hybrid Automaton: Bang-Bang Thermostat with Hysteresis $\\delta$',
                 fontsize=14, fontweight='bold')

    # Zeno note
    ax.text(5, -0.5,
            'As $\\delta \\to 0$: Domain overlap shrinks to $\\{T_{set}\\}$, '
            'guard sets collapse,\n'
            'producing Zeno execution at $T = T_{set}$ '
            '(Filippov sliding mode: $u_{eq} = k(T_{set} - T_a)$)',
            fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='#ffffcc', alpha=0.8))

    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.5, 5.5)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.savefig(os.path.join(RESULTS_DIR, 'hybrid_automaton.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: hybrid_automaton.png")


def plot_delta_convergence_table(numerical_results, analytical_results):
    """
    Create a comparison figure: analytical vs numerical periods.
    Also show the Zeno convergence argument.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel (a): Mean interval comparison
    ax = axes[0]
    num_deltas = [r['delta'] for r in numerical_results]
    num_intervals = [r['mean_ss_interval'] for r in numerical_results]

    # Analytical prediction
    ana_intervals = []
    for d in num_deltas:
        th, tc, _, _ = analytical_switching_times(d)
        ana_intervals.append((th + tc) / 2)  # half-period = interval between switches

    ax.loglog(num_deltas, num_intervals, 'ro-', markersize=8, linewidth=2,
              label='Numerical (mean)')
    ax.loglog(num_deltas, ana_intervals, 'bs--', markersize=8, linewidth=1.5,
              label='Analytical (half-period)')

    # Reference slope
    d_ref = np.array(num_deltas)
    ax.loglog(d_ref, 0.15 * d_ref, 'k:', linewidth=1, alpha=0.5,
              label='Slope 1 reference')

    ax.set_xlabel('Hysteresis band $\\delta$ (°C)', fontsize=12)
    ax.set_ylabel('Switching interval (min)', fontsize=12)
    ax.set_title('(a) Analytical vs Numerical Switching Intervals', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel (b): Cumulative energy from switching
    ax = axes[1]
    c_s_values = [0.0, 0.1, 0.5, 1.0]  # switching cost per event
    for c_s in c_s_values:
        energies = []
        for r in numerical_results:
            # Base energy: approximate as duty_cycle * U_max * t_end
            # Duty cycle = u_eq / U_max for ideal relay
            duty = K_COOL * (T_SET - T_AMBIENT) / U_MAX
            E_base = duty * U_MAX * 60.0  # 60 min simulation
            E_switch = c_s * r['n_switches']
            energies.append(E_base + E_switch)
        ax.semilogx(num_deltas, energies, 'o-', markersize=6, linewidth=1.5,
                     label=f'$c_s = {c_s}$')

    ax.set_xlabel('Hysteresis band $\\delta$ (°C)', fontsize=12)
    ax.set_ylabel('Total energy (°C·min)', fontsize=12)
    ax.set_title('(b) Energy Cost with Switching Penalty', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'zeno_convergence.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: zeno_convergence.png")


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 60)
    print("Zeno Effect Analysis for Bang-Bang Thermostat")
    print("=" * 60)

    # --- Analytical ---
    print("\n1. Analytical convergence analysis...")
    ana = zeno_convergence_analysis()
    C = ana['C_linear']
    print(f"   Linear coefficient: P ≈ {C:.4f} * delta")
    print(f"   For delta=0.5: P = {ana['periods'][np.argmin(np.abs(ana['deltas'] - 0.5))]:.4f} min")
    print(f"   For delta=0.01: P = {ana['periods'][np.argmin(np.abs(ana['deltas'] - 0.01))]:.6f} min")

    # --- Zeno classification ---
    print("\n2. Zeno point classification...")
    zeno = check_zeno_type()
    print(f"   T_ss (heater ON steady state) = {zeno['T_ss']:.1f} °C")
    print(f"   Time to reach T_set = {zeno['t_reach_Tset']:.2f} min")
    print(f"   dT/dt at T_set (ON)  = {zeno['dTdt_on_at_Tset']:.2f} °C/min > 0")
    print(f"   dT/dt at T_set (OFF) = {zeno['dTdt_off_at_Tset']:.2f} °C/min < 0")
    print(f"   Zeno point at T = T_set? {zeno['is_zeno_point']}")
    print(f"   Filippov equivalent control: u_eq = {zeno['u_eq_filippov']:.2f} "
          f"({zeno['u_eq_fraction']:.1%} of U_max)")

    # --- Numerical verification ---
    print("\n3. Numerical simulation verification...")
    delta_values = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    num = simulate_zeno_numerical(delta_values, t_end=60.0)

    # --- Summary table ---
    print("\n" + "=" * 75)
    print("Summary: Analytical vs Numerical")
    print("-" * 75)
    print(f"{'delta':>8s} | {'P_ana (min)':>12s} | {'P_num (min)':>12s} | "
          f"{'N_switches':>10s} | {'Error %':>8s}")
    print("-" * 75)
    for r in num:
        th, tc, _, _ = analytical_switching_times(r['delta'])
        P_ana = th + tc
        P_num = 2.0 * r['mean_ss_interval'] if r['mean_ss_interval'] < np.inf else np.inf
        if P_ana > 0 and P_num < np.inf:
            err = abs(P_num - P_ana) / P_ana * 100
        else:
            err = np.nan
        print(f"{r['delta']:>8.4f} | {P_ana:>12.6f} | {P_num:>12.6f} | "
              f"{r['n_switches']:>10d} | {err:>7.2f}%")
    print("=" * 75)

    # --- Plots ---
    print("\n4. Generating figures...")
    plot_zeno_analysis(ana, num, zeno)
    plot_hybrid_automaton_diagram()
    plot_delta_convergence_table(num, ana)

    print("\n✓ Zeno analysis complete.")
    print("  Key finding: As δ → 0, switching count N ~ 1/δ → ∞ (chattering Zeno).")
    print("  For δ = 0, the system has a Zeno point at T = T_set.")
    print("  Filippov regularisation gives sliding mode with u_eq = k(T_set - T_a).")

    return ana, num, zeno


if __name__ == '__main__':
    main()
