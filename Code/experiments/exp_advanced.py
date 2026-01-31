"""
Advanced experiments:
1. Convection effect study (α parameter sensitivity on PDE models)
2. Joint optimal placement (heater + thermostat position optimization)
3. Multi-room building control strategies
"""

import numpy as np
import sys, os
import itertools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.pde_1d_model import HeatEquation1D
from models.pde_2d_model import HeatEquation2D
from models.building_model import BuildingModel
from controllers.bang_bang import BangBangController
from controllers.pid import PIDController
from controllers.lqr import LQRController
from utils.parameters import (
    T_AMBIENT, T_INITIAL, T_SET, U_MAX, K_COOL,
    ROOM_LENGTH, ROOM_WIDTH, HYSTERESIS_BAND
)
from utils.metrics import compute_all_metrics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# Part 1: Convection Effect Study — How α changes PDE control difficulty
# =============================================================================
def experiment_convection_1d(output_dir):
    """Study how effective thermal diffusivity affects 1D PDE control."""
    print("=" * 70)
    print("Experiment 1: Convection Effect on 1D PDE (α sensitivity)")
    print("=" * 70)

    alpha_values = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    alpha_labels = {
        0.005: 'Pure diffusion (stagnant)',
        0.01:  'Default (weak convection)',
        0.02:  'Mild convection',
        0.05:  'Moderate convection',
        0.1:   'Strong convection (fan)',
        0.2:   'Forced ventilation',
        0.5:   'Strong forced air',
    }

    heater_pos, therm_pos = 0.5, 2.5
    t_end = 120.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    print(f"  Heater: x={heater_pos}m, Thermostat: x={therm_pos}m")
    print(f"  {'α (m²/min)':<15} {'Label':<25} {'RMSE':>8} {'Settle':>10} {'Energy':>10}")
    print("  " + "-" * 72)

    results_alpha = []
    for alpha in alpha_values:
        model = HeatEquation1D(alpha=alpha, heater_pos=heater_pos,
                               thermostat_pos=therm_pos, T0=T_INITIAL)
        ctrl = PIDController(Kp=4.0, Ki=0.5, Kd=0.5,
                             T_set=T_SET, U_max=U_MAX, dt=0.05)
        t, T_field, T_therm = model.simulate(ctrl.get_u, t_end=t_end, dt=0.05)

        ctrl2 = PIDController(Kp=4.0, Ki=0.5, Kd=0.5,
                              T_set=T_SET, U_max=U_MAX, dt=0.05)
        u = np.array([ctrl2.get_u(ti, Ti) for ti, Ti in zip(t, T_therm)])
        m = compute_all_metrics(t, T_therm, u, T_SET)

        label_short = alpha_labels.get(alpha, '')
        settle = f'{m["settling_time"]:.1f}' if m['settling_time'] < 999 else 'unstable'
        print(f"  {alpha:<15.3f} {label_short:<25} {m['rmse']:>8.3f} "
              f"{settle:>10} {m['energy']:>10.1f}")

        ax1.plot(t, T_therm, linewidth=1.5,
                 label=f'α={alpha} (RMSE={m["rmse"]:.2f})')
        ax2.plot(t, u, linewidth=0.8, alpha=0.7)
        results_alpha.append((alpha, m))

    ax1.axhline(y=T_SET, color='r', linestyle='--', alpha=0.7)
    ax1.set_ylabel('Thermostat Temperature (°C)', fontsize=12)
    ax1.set_title('1D PDE: Effect of Thermal Diffusivity α (heater x=0.5, therm x=2.5)',
                  fontsize=13)
    ax1.legend(fontsize=8, loc='lower right')
    ax2.set_xlabel('Time (min)', fontsize=12)
    ax2.set_ylabel('Control u(t)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adv_convection_1d.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: adv_convection_1d.png")

    # Summary: α vs RMSE and settling time
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    alphas = [r[0] for r in results_alpha]
    rmses = [r[1]['rmse'] for r in results_alpha]
    settles = [min(r[1]['settling_time'], t_end) for r in results_alpha]
    energies = [r[1]['energy'] for r in results_alpha]

    ax1.semilogx(alphas, rmses, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('α (m²/min)', fontsize=12)
    ax1.set_ylabel('RMSE (°C)', fontsize=12)
    ax1.set_title('Temperature Error vs Diffusivity')
    ax1.axhline(y=0.5, color='g', linestyle='--', alpha=0.5, label='Comfort threshold')
    ax1.legend()

    ax2.semilogx(alphas, settles, 'rs-', linewidth=2, markersize=8)
    ax2.set_xlabel('α (m²/min)', fontsize=12)
    ax2.set_ylabel('Settling Time (min)', fontsize=12)
    ax2.set_title('Settling Time vs Diffusivity')
    ax2.axhline(y=t_end, color='gray', linestyle=':', alpha=0.5, label='Simulation end')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adv_alpha_sensitivity.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: adv_alpha_sensitivity.png")

    return results_alpha


# =============================================================================
# Part 2: Joint Optimal Placement (heater + thermostat)
# =============================================================================
def experiment_optimal_placement_1d(output_dir):
    """Find the optimal heater and thermostat positions jointly."""
    print("\n" + "=" * 70)
    print("Experiment 2: Joint Optimal Placement (1D PDE)")
    print("=" * 70)

    alpha = 0.1  # Realistic with convection
    t_end = 60.0
    positions = np.arange(0.25, 5.0, 0.5)  # 10 positions

    # Grid search over heater × thermostat positions
    cost_matrix = np.full((len(positions), len(positions)), np.nan)
    rmse_matrix = np.full_like(cost_matrix, np.nan)
    settle_matrix = np.full_like(cost_matrix, np.nan)

    print(f"  α = {alpha} m²/min (with convection)")
    print(f"  Grid: {len(positions)} × {len(positions)} = {len(positions)**2} configurations")

    best_cost = float('inf')
    best_config = None

    for i, h_pos in enumerate(positions):
        for j, t_pos in enumerate(positions):
            model = HeatEquation1D(alpha=alpha, heater_pos=h_pos,
                                   thermostat_pos=t_pos, T0=T_INITIAL)
            ctrl = PIDController(Kp=4.0, Ki=0.5, Kd=0.5,
                                 T_set=T_SET, U_max=U_MAX, dt=0.05)
            t, T_field, T_therm = model.simulate(ctrl.get_u, t_end=t_end, dt=0.05)

            ctrl2 = PIDController(Kp=4.0, Ki=0.5, Kd=0.5,
                                  T_set=T_SET, U_max=U_MAX, dt=0.05)
            u = np.array([ctrl2.get_u(ti, Ti) for ti, Ti in zip(t, T_therm)])
            m = compute_all_metrics(t, T_therm, u, T_SET)

            cost_matrix[i, j] = m['unified_cost']
            rmse_matrix[i, j] = m['rmse']
            settle_matrix[i, j] = min(m['settling_time'], t_end)

            if m['unified_cost'] < best_cost:
                best_cost = m['unified_cost']
                best_config = (h_pos, t_pos, m)

    print(f"\n  Optimal: heater={best_config[0]:.1f}m, therm={best_config[1]:.1f}m")
    print(f"  RMSE={best_config[2]['rmse']:.3f}, Cost J={best_config[2]['unified_cost']:.1f}")

    # Heatmaps
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for ax, matrix, title, cmap in [
        (axes[0], rmse_matrix, 'RMSE (°C)', 'YlOrRd'),
        (axes[1], cost_matrix, 'Unified Cost J', 'YlOrRd'),
        (axes[2], settle_matrix, 'Settling Time (min)', 'YlOrRd'),
    ]:
        im = ax.imshow(matrix, origin='lower', aspect='auto', cmap=cmap,
                        extent=[positions[0]-0.25, positions[-1]+0.25,
                                positions[0]-0.25, positions[-1]+0.25])
        ax.set_xlabel('Thermostat Position (m)', fontsize=11)
        ax.set_ylabel('Heater Position (m)', fontsize=11)
        ax.set_title(title, fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Mark optimal
        ax.plot(best_config[1], best_config[0], 'w*', markersize=15,
                markeredgecolor='k', markeredgewidth=1.5)

    plt.suptitle(f'1D PDE: Joint Heater × Thermostat Placement Optimization (α={alpha})',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adv_optimal_placement_1d.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: adv_optimal_placement_1d.png")

    return cost_matrix, rmse_matrix, positions, best_config


def experiment_optimal_placement_2d(output_dir):
    """Joint optimal placement in 2D room."""
    print("\n" + "=" * 70)
    print("Experiment 3: Joint Optimal Placement (2D PDE)")
    print("=" * 70)

    alpha = 0.1
    t_end = 30.0
    nx, ny = 16, 13

    # Heater and thermostat position grids
    x_positions = [0.5, 1.5, 2.5, 3.5, 4.5]
    y_positions = [0.5, 2.0, 3.5]

    # All (x,y) positions for both heater and thermostat
    all_positions = [(x, y) for x in x_positions for y in y_positions]

    print(f"  α = {alpha} m²/min, grid: {nx}×{ny}")
    print(f"  Testing {len(all_positions)} positions for heater and thermostat each")

    best_cost = float('inf')
    best_config = None
    results_2d = []

    for h_pos in all_positions:
        for t_pos in all_positions:
            if h_pos == t_pos:
                continue  # skip co-located (trivial case)
            model = HeatEquation2D(nx=nx, ny=ny, alpha=alpha,
                                   heater_pos=h_pos, thermostat_pos=t_pos,
                                   T0=T_INITIAL)
            ctrl = PIDController(Kp=4.0, Ki=0.5, Kd=0.5,
                                 T_set=T_SET, U_max=U_MAX, dt=0.2)
            t, T_field, T_therm = model.simulate(ctrl.get_u, t_end=t_end, dt=0.2)

            ctrl2 = PIDController(Kp=4.0, Ki=0.5, Kd=0.5,
                                  T_set=T_SET, U_max=U_MAX, dt=0.2)
            u = np.array([ctrl2.get_u(ti, Ti) for ti, Ti in zip(t, T_therm)])
            m = compute_all_metrics(t, T_therm, u, T_SET)

            # Also compute spatial uniformity at final time
            T_final = T_field[:, :, -1]
            spatial_std = T_final.std()
            m['spatial_std'] = spatial_std

            results_2d.append((h_pos, t_pos, m))

            if m['unified_cost'] < best_cost:
                best_cost = m['unified_cost']
                best_config = (h_pos, t_pos, m)

    print(f"\n  Best: heater={best_config[0]}, therm={best_config[1]}")
    print(f"  RMSE={best_config[2]['rmse']:.3f}, Cost={best_config[2]['unified_cost']:.1f}, "
          f"Spatial σ={best_config[2]['spatial_std']:.2f}")

    # Find top 10 configurations
    results_2d.sort(key=lambda x: x[2]['unified_cost'])
    print(f"\n  Top 10 configurations:")
    print(f"  {'Rank':>4} {'Heater':>14} {'Therm':>14} {'RMSE':>8} {'Cost':>8} {'σ_spatial':>10}")
    print("  " + "-" * 62)
    for rank, (hp, tp, m) in enumerate(results_2d[:10], 1):
        print(f"  {rank:>4} {str(hp):>14} {str(tp):>14} {m['rmse']:>8.3f} "
              f"{m['unified_cost']:>8.1f} {m['spatial_std']:>10.2f}")

    # Pareto: RMSE vs spatial uniformity
    fig, ax = plt.subplots(figsize=(10, 7))
    rmses_2d = [r[2]['rmse'] for r in results_2d]
    stds_2d = [r[2]['spatial_std'] for r in results_2d]
    costs_2d = [r[2]['unified_cost'] for r in results_2d]

    sc = ax.scatter(rmses_2d, stds_2d, c=costs_2d, cmap='viridis',
                    s=40, alpha=0.7, edgecolors='k', linewidths=0.3)
    plt.colorbar(sc, ax=ax, label='Unified Cost J')

    # Mark top 3
    for rank, (hp, tp, m) in enumerate(results_2d[:3], 1):
        ax.scatter(m['rmse'], m['spatial_std'], s=200, marker='*',
                   edgecolors='red', facecolors='yellow', linewidths=2, zorder=10)
        ax.annotate(f'#{rank}\nH={hp}\nT={tp}',
                    (m['rmse'], m['spatial_std']),
                    fontsize=7, ha='center', va='bottom',
                    xytext=(0, 10), textcoords='offset points')

    ax.set_xlabel('Thermostat RMSE (°C)', fontsize=12)
    ax.set_ylabel('Spatial Temperature σ (°C)', fontsize=12)
    ax.set_title('2D Room: RMSE vs Spatial Uniformity (all H×T configurations)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adv_optimal_placement_2d.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: adv_optimal_placement_2d.png")

    return results_2d, best_config


# =============================================================================
# Part 3: Multi-Room Building Control
# =============================================================================
def experiment_building(output_dir):
    """Multi-room building: independent vs coordinated control."""
    print("\n" + "=" * 70)
    print("Experiment 4: Multi-Room Building Control")
    print("=" * 70)

    n_rooms = 5
    k_ext = 0.1
    k_int_values = [0.0, 0.02, 0.05, 0.1, 0.2]
    t_end = 120.0

    # Strategy 1: Independent PID — same parameters for all rooms
    # Strategy 2: Independent PID — tuned per room position
    # Strategy 3: Selective heating — only heat some rooms

    print(f"\n  Building: {n_rooms} rooms in a row")
    print(f"  Rooms 0 and {n_rooms-1} have exterior walls")
    print(f"  k_ext = {k_ext}, testing k_int from {k_int_values[0]} to {k_int_values[-1]}")

    # ===== Part A: Effect of k_int (wall coupling) =====
    print(f"\n  --- Part A: Effect of internal wall coupling ---")

    fig, axes = plt.subplots(2, len(k_int_values), figsize=(4*len(k_int_values), 8),
                              sharey='row')
    if len(k_int_values) == 1:
        axes = axes.reshape(2, 1)

    for col, k_int in enumerate(k_int_values):
        building = BuildingModel(n_rooms=n_rooms, k_ext=k_ext, k_int=k_int,
                                 T0=T_INITIAL)

        def make_pid(room_idx):
            return lambda: PIDController(Kp=4.0, Ki=0.5, Kd=0.5,
                                         T_set=T_SET, U_max=U_MAX, dt=0.01)

        controllers = [make_pid(i)() for i in range(n_rooms)]
        u_funcs = [c.get_u for c in controllers]
        t, T = building.simulate(u_funcs, t_end=t_end)

        # Plot temperatures
        ax1 = axes[0, col]
        for i in range(n_rooms):
            style = '-' if i in [0, n_rooms-1] else '--'
            ax1.plot(t, T[i], style, linewidth=1.2,
                     label=f'Room {i}' + (' (ext)' if i in [0, n_rooms-1] else ''))
        ax1.axhline(y=T_SET, color='r', linestyle=':', alpha=0.5)
        ax1.set_title(f'k_int = {k_int}')
        if col == 0:
            ax1.set_ylabel('Temperature (°C)')
        ax1.legend(fontsize=6)

        # Compute per-room metrics
        ax2 = axes[1, col]
        room_rmse = []
        for i in range(n_rooms):
            ctrl2 = PIDController(Kp=4.0, Ki=0.5, Kd=0.5,
                                  T_set=T_SET, U_max=U_MAX, dt=0.01)
            u = np.array([ctrl2.get_u(tj, T[i, j]) for j, tj in enumerate(t)])
            m = compute_all_metrics(t, T[i], u, T_SET)
            room_rmse.append(m['rmse'])

        colors = ['red' if i in [0, n_rooms-1] else 'blue' for i in range(n_rooms)]
        ax2.bar(range(n_rooms), room_rmse, color=colors, alpha=0.7)
        ax2.set_xlabel('Room #')
        if col == 0:
            ax2.set_ylabel('RMSE (°C)')
        ax2.set_xticks(range(n_rooms))

    axes[0, 0].set_ylabel('Temperature (°C)')
    axes[1, 0].set_ylabel('RMSE (°C)')
    fig.suptitle(f'Building ({n_rooms} rooms): Effect of Internal Wall Coupling k_int',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adv_building_coupling.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: adv_building_coupling.png")

    # ===== Part B: Control strategy comparison =====
    print(f"\n  --- Part B: Control strategies for building ---")
    k_int = 0.05

    strategies_results = {}

    # Strategy 1: Uniform PID
    building = BuildingModel(n_rooms=n_rooms, k_ext=k_ext, k_int=k_int,
                             T0=T_INITIAL)
    ctrls_uniform = [PIDController(Kp=4.0, Ki=0.5, Kd=0.5,
                                    T_set=T_SET, U_max=U_MAX, dt=0.01)
                     for _ in range(n_rooms)]
    t, T = building.simulate([c.get_u for c in ctrls_uniform], t_end=t_end)
    strategies_results['Uniform PID'] = (t, T)

    # Strategy 2: Adaptive PID — exterior rooms get higher gains
    building2 = BuildingModel(n_rooms=n_rooms, k_ext=k_ext, k_int=k_int,
                              T0=T_INITIAL)
    ctrls_adaptive = []
    for i in range(n_rooms):
        if i in [0, n_rooms-1]:
            # Exterior rooms: higher gains to compensate for extra heat loss
            ctrls_adaptive.append(
                PIDController(Kp=8.0, Ki=1.0, Kd=0.0,
                              T_set=T_SET, U_max=U_MAX, dt=0.01))
        else:
            ctrls_adaptive.append(
                PIDController(Kp=4.0, Ki=0.5, Kd=0.5,
                              T_set=T_SET, U_max=U_MAX, dt=0.01))
    t2, T2 = building2.simulate([c.get_u for c in ctrls_adaptive], t_end=t_end)
    strategies_results['Adaptive PID'] = (t2, T2)

    # Strategy 3: LQR for all rooms
    building3 = BuildingModel(n_rooms=n_rooms, k_ext=k_ext, k_int=k_int,
                              T0=T_INITIAL)
    ctrls_lqr = [LQRController(Q=1.0, R=0.01) for _ in range(n_rooms)]
    t3, T3 = building3.simulate([c.get_u for c in ctrls_lqr], t_end=t_end)
    strategies_results['LQR'] = (t3, T3)

    # Strategy 4: Only heat exterior rooms (energy saving)
    building4 = BuildingModel(n_rooms=n_rooms, k_ext=k_ext, k_int=k_int,
                              T0=T_INITIAL)
    ctrls_selective = []
    for i in range(n_rooms):
        if i in [0, n_rooms-1]:
            ctrls_selective.append(
                PIDController(Kp=8.0, Ki=1.0, Kd=0.0,
                              T_set=T_SET, U_max=U_MAX, dt=0.01))
        else:
            ctrls_selective.append(
                PIDController(Kp=0.0, Ki=0.0, Kd=0.0,
                              T_set=T_SET, U_max=0.0, dt=0.01))
    t4, T4 = building4.simulate([c.get_u for c in ctrls_selective], t_end=t_end)
    strategies_results['Exterior-only'] = (t4, T4)

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    strat_list = list(strategies_results.items())

    print(f"\n  {'Strategy':<20} {'Avg RMSE':>10} {'Max RMSE':>10} {'Total Energy':>12}")
    print("  " + "-" * 55)

    for idx, (name, (t, T)) in enumerate(strat_list):
        ax = axes[idx // 2, idx % 2]
        rmses = []
        total_energy = 0
        for i in range(n_rooms):
            style = '-' if i in [0, n_rooms-1] else '--'
            ax.plot(t, T[i], style, linewidth=1.2, label=f'Room {i}')
            # Simple RMSE calc
            rmse_i = np.sqrt(np.mean((T[i] - T_SET) ** 2))
            rmses.append(rmse_i)
        ax.axhline(y=T_SET, color='r', linestyle=':', alpha=0.5)
        ax.set_title(f'{name}')
        ax.legend(fontsize=7, ncol=3)

        avg_rmse = np.mean(rmses)
        max_rmse = np.max(rmses)
        print(f"  {name:<20} {avg_rmse:>10.3f} {max_rmse:>10.3f}")

    axes[1, 0].set_xlabel('Time (min)')
    axes[1, 1].set_xlabel('Time (min)')
    axes[0, 0].set_ylabel('Temperature (°C)')
    axes[1, 0].set_ylabel('Temperature (°C)')

    fig.suptitle(f'Building Control Strategies ({n_rooms} rooms, k_int={k_int})',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adv_building_strategies.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: adv_building_strategies.png")

    # ===== Part C: Optimal number of heated rooms =====
    print(f"\n  --- Part C: Energy vs comfort — how many rooms to heat? ---")
    k_int = 0.1  # Good coupling

    configs = {
        'All rooms heated': list(range(n_rooms)),
        'Exterior + centre': [0, n_rooms//2, n_rooms-1],
        'Exterior only': [0, n_rooms-1],
        'Centre only': [n_rooms//2],
        'Alternating': list(range(0, n_rooms, 2)),
    }

    print(f"  {'Config':<25} {'Heated rooms':>15} {'Avg RMSE':>10} {'Total Energy':>12}")
    print("  " + "-" * 65)

    pareto_data = []
    for name, heated_rooms in configs.items():
        building = BuildingModel(n_rooms=n_rooms, k_ext=k_ext, k_int=k_int,
                                 T0=T_INITIAL)
        ctrls = []
        for i in range(n_rooms):
            if i in heated_rooms:
                ctrls.append(PIDController(Kp=8.0, Ki=1.0, Kd=0.0,
                                           T_set=T_SET, U_max=U_MAX, dt=0.01))
            else:
                ctrls.append(PIDController(Kp=0.0, Ki=0.0, Kd=0.0,
                                           T_set=T_SET, U_max=0.0, dt=0.01))

        t, T = building.simulate([c.get_u for c in ctrls], t_end=t_end)

        rmses = [np.sqrt(np.mean((T[i] - T_SET)**2)) for i in range(n_rooms)]
        avg_rmse = np.mean(rmses)

        # Total energy
        total_energy = 0
        for i in range(n_rooms):
            if i in heated_rooms:
                ctrl2 = PIDController(Kp=8.0, Ki=1.0, Kd=0.0,
                                      T_set=T_SET, U_max=U_MAX, dt=0.01)
                u = np.array([ctrl2.get_u(tj, T[i, j]) for j, tj in enumerate(t)])
                total_energy += np.trapz(u, t)

        print(f"  {name:<25} {str(heated_rooms):>15} {avg_rmse:>10.3f} {total_energy:>12.1f}")
        pareto_data.append((name, len(heated_rooms), avg_rmse, total_energy))

    # Pareto plot: energy vs avg RMSE
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, n_heated, rmse, energy in pareto_data:
        ax.scatter(energy, rmse, s=150, zorder=5)
        ax.annotate(f'{name}\n({n_heated} rooms)',
                    (energy, rmse), fontsize=9,
                    xytext=(8, 8), textcoords='offset points')

    ax.set_xlabel('Total Energy Consumption', fontsize=12)
    ax.set_ylabel('Average RMSE (°C)', fontsize=12)
    ax.set_title(f'Building Pareto: Energy vs Comfort ({n_rooms} rooms, k_int={k_int})',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adv_building_pareto.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: adv_building_pareto.png")

    return pareto_data


# =============================================================================
# Part 4: Convection-enhanced 1D PDE with all controllers
# =============================================================================
def experiment_1d_convection_controllers(output_dir):
    """With realistic α, compare all controllers on 1D PDE."""
    print("\n" + "=" * 70)
    print("Experiment 5: 1D PDE with Convection — Controller Comparison")
    print("=" * 70)

    alpha = 0.1  # Realistic with convection
    heater_pos = 0.5
    therm_pos = 2.5
    t_end = 60.0

    controller_configs = {
        'Bang-Bang': lambda: BangBangController(T_set=T_SET, U_max=U_MAX,
                                                 delta=HYSTERESIS_BAND, initial_on=True),
        'PID': lambda: PIDController(Kp=4.0, Ki=0.5, Kd=0.5,
                                      T_set=T_SET, U_max=U_MAX, dt=0.05),
        'PID (tuned)': lambda: PIDController(Kp=8.0, Ki=1.0, Kd=0.0,
                                              T_set=T_SET, U_max=U_MAX, dt=0.05),
        'LQR': lambda: LQRController(Q=1.0, R=0.01),
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    print(f"  α = {alpha} m²/min, heater x={heater_pos}, therm x={therm_pos}")
    print(f"  {'Controller':<20} {'RMSE':>8} {'Settle':>10} {'Energy':>10} {'Cost J':>10}")
    print("  " + "-" * 62)

    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']
    for idx, (name, factory) in enumerate(controller_configs.items()):
        model = HeatEquation1D(alpha=alpha, heater_pos=heater_pos,
                               thermostat_pos=therm_pos, T0=T_INITIAL)
        ctrl = factory()
        t, T_field, T_therm = model.simulate(ctrl.get_u, t_end=t_end, dt=0.05)

        ctrl2 = factory()
        u = np.array([ctrl2.get_u(ti, Ti) for ti, Ti in zip(t, T_therm)])
        m = compute_all_metrics(t, T_therm, u, T_SET)

        settle = f'{m["settling_time"]:.1f}' if m['settling_time'] < 999 else 'unstable'
        print(f"  {name:<20} {m['rmse']:>8.3f} {settle:>10} "
              f"{m['energy']:>10.1f} {m['unified_cost']:>10.1f}")

        ax1.plot(t, T_therm, linewidth=1.5, color=colors[idx],
                 label=f'{name} (RMSE={m["rmse"]:.3f})')
        ax2.plot(t, u, linewidth=1.0, color=colors[idx], alpha=0.8)

    ax1.axhline(y=T_SET, color='k', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Temperature (°C)', fontsize=12)
    ax1.set_title(f'1D PDE with Convection (α={alpha}): Controller Comparison', fontsize=14)
    ax1.legend(fontsize=10)

    ax2.set_xlabel('Time (min)', fontsize=12)
    ax2.set_ylabel('Control u(t)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adv_1d_convection_controllers.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: adv_1d_convection_controllers.png")


def main():
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(output_dir, exist_ok=True)

    # Run all experiments
    experiment_convection_1d(output_dir)
    experiment_optimal_placement_1d(output_dir)
    experiment_1d_convection_controllers(output_dir)
    experiment_optimal_placement_2d(output_dir)
    experiment_building(output_dir)

    print("\n" + "=" * 70)
    print("All advanced experiments complete.")
    print(f"Results in: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
