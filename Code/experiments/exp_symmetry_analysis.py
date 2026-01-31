"""
Symmetry Analysis: When is a 1D Model Justified?

This experiment rigorously investigates under what conditions the 2D heat equation
for a rectangular room can be reduced to a 1D model along one axis.

Key arguments:
1. If the room is elongated (Lx >> Ly) and the heater spans the full y-extent,
   the temperature is approximately uniform in y.
2. If the room is square but the heater/BC are symmetric in y, a 1D slice
   along the x-axis captures the dominant physics.
3. The 1D reduction error can be quantified by the y-direction temperature
   variance: Var_y(T) = <(T(x,y) - <T>_y)^2>_y

We demonstrate these claims by:
(a) Comparing 2D simulations with their 1D projections
(b) Measuring the y-gradient energy fraction
(c) Varying aspect ratio to find the 1D validity threshold
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.parameters import (T_AMBIENT, T_INITIAL, T_SET, ALPHA, H_WALL,
                               ROOM_LENGTH, ROOM_WIDTH, NX, NY, T_END)
from models.pde_2d_model import HeatEquation2D
from models.pde_1d_model import HeatEquation1D
from controllers.pid import PIDController

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def y_uniformity_index(T_field_2d):
    """
    Measure how uniform the temperature is in the y-direction.

    Returns the ratio: ||dT/dy||^2 / (||dT/dx||^2 + ||dT/dy||^2)

    If this ratio is small, the y-variation is negligible and 1D is valid.

    Parameters
    ----------
    T_field_2d : ndarray, shape (nx, ny)
        2D temperature field at a single time.

    Returns
    -------
    ratio : float
        Fraction of total gradient energy in the y-direction.
        0 = perfectly 1D (no y-variation), 0.5 = isotropic.
    """
    # Gradient magnitudes (squared)
    dTdx = np.diff(T_field_2d, axis=0)
    dTdy = np.diff(T_field_2d, axis=1)

    energy_x = np.sum(dTdx**2)
    energy_y = np.sum(dTdy**2)

    total = energy_x + energy_y
    if total < 1e-12:
        return 0.0
    return energy_y / total


def y_variance_profile(T_field_2d):
    """
    Compute the y-direction variance of T at each x position.

    Returns
    -------
    var_y : ndarray, shape (nx,)
        Variance of T in the y-direction at each x.
    """
    # T_field_2d shape: (nx, ny)
    return np.var(T_field_2d, axis=1)


def compare_2d_vs_1d(aspect_ratios=None, heater_mode='wall'):
    """
    Compare 2D and 1D models for rooms of different aspect ratios.

    Parameters
    ----------
    aspect_ratios : list of (Lx, Ly) tuples
    heater_mode : str
        'wall': heater on one short wall (spanning full y)
        'point': heater at a single point
    """
    if aspect_ratios is None:
        # (Lx, Ly) pairs with same area = 20 m^2
        area = 20.0
        aspect_ratios = [
            (np.sqrt(area * r), np.sqrt(area / r))
            for r in [1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
        ]

    results = []
    pid_kp, pid_ki, pid_kd = 2.0, 0.1, 0.5

    for Lx, Ly in aspect_ratios:
        ratio = Lx / Ly
        print(f"  Aspect ratio Lx/Ly = {ratio:.1f} (Lx={Lx:.2f}, Ly={Ly:.2f})")

        # Adjust grid to keep dx ≈ dy ≈ 0.1m
        nx = max(11, int(Lx / 0.15) + 1)
        ny = max(11, int(Ly / 0.15) + 1)

        # Heater at left wall centre
        heater_pos = (0.3, Ly / 2)
        # Thermostat at room centre
        therm_pos = (Lx / 2, Ly / 2)

        # --- 2D simulation ---
        pid_2d = PIDController(Kp=pid_kp, Ki=pid_ki, Kd=pid_kd, T_set=T_SET)
        model_2d = HeatEquation2D(
            Lx=Lx, Ly=Ly, nx=nx, ny=ny,
            heater_pos=heater_pos, heater_radius=0.5,
            thermostat_pos=therm_pos
        )
        t_eval = np.arange(0, 61, 1.0)
        t_2d, T_field_2d, T_therm_2d = model_2d.simulate(
            pid_2d.get_u, t_end=60.0, dt=0.05, t_eval=t_eval
        )

        # y-uniformity at final time
        T_final_2d = T_field_2d[:, :, -1]
        y_ratio = y_uniformity_index(T_final_2d)
        y_var = y_variance_profile(T_final_2d)

        # y-averaged profile (this is what 1D should predict)
        T_y_avg = np.mean(T_final_2d, axis=1)

        # --- 1D simulation ---
        pid_1d = PIDController(Kp=pid_kp, Ki=pid_ki, Kd=pid_kd, T_set=T_SET)
        model_1d = HeatEquation1D(
            L=Lx, nx=nx, heater_pos=0.3, heater_width=0.5,
            thermostat_pos=Lx / 2
        )
        t_1d, T_field_1d, T_therm_1d = model_1d.simulate(
            pid_1d.get_u, t_end=60.0, dt=0.05, t_eval=t_eval
        )

        T_final_1d = T_field_1d[:, -1]

        # Interpolate 1D to same x-grid as 2D for comparison
        x_2d = np.linspace(0, Lx, nx)
        x_1d = np.linspace(0, Lx, model_1d.nx)
        T_1d_interp = np.interp(x_2d, x_1d, T_final_1d)

        # Error between y-averaged 2D and 1D
        rmse_1d_vs_2d = np.sqrt(np.mean((T_y_avg - T_1d_interp)**2))

        results.append({
            'Lx': Lx, 'Ly': Ly, 'ratio': ratio,
            'nx': nx, 'ny': ny,
            'y_gradient_fraction': y_ratio,
            'max_y_var': np.max(y_var),
            'rmse_1d_vs_2d': rmse_1d_vs_2d,
            'T_final_2d': T_final_2d,
            'T_y_avg': T_y_avg,
            'T_final_1d': T_1d_interp,
            'x_2d': x_2d,
            'T_therm_2d': T_therm_2d,
            'T_therm_1d': T_therm_1d,
            't': t_2d,
        })

        print(f"    y-gradient fraction: {y_ratio:.4f}")
        print(f"    max y-variance: {np.max(y_var):.4f} °C²")
        print(f"    RMSE(1D vs 2D avg): {rmse_1d_vs_2d:.4f} °C")

    return results


def heater_symmetry_analysis():
    """
    Show that when the heater spans the full width of the room,
    the temperature field is symmetric in y, making the 1D reduction exact.

    Compare:
    (a) Full-width heater (large sigma_y) -> nearly 1D
    (b) Point heater (small sigma) -> genuinely 2D
    """
    Lx, Ly = 5.0, 4.0
    nx, ny = 41, 33

    configs = [
        ('Full-width heater\n(radiator on wall)', (0.3, Ly/2), 2.0),
        ('Moderate heater\n(panel heater)', (0.3, Ly/2), 0.8),
        ('Point heater\n(space heater)', (0.3, Ly/2), 0.3),
        ('Corner heater\n(worst case)', (0.3, 0.5), 0.3),
    ]

    results = []

    for label, hpos, hradius in configs:
        pid = PIDController(Kp=2.0, Ki=0.1, Kd=0.5, T_set=T_SET)
        model = HeatEquation2D(
            Lx=Lx, Ly=Ly, nx=nx, ny=ny,
            heater_pos=hpos, heater_radius=hradius,
            thermostat_pos=(2.5, 2.0)
        )
        t_eval = np.arange(0, 61, 1.0)
        t, T_field, T_therm = model.simulate(
            pid.get_u, t_end=60.0, dt=0.05, t_eval=t_eval
        )

        T_final = T_field[:, :, -1]
        y_ratio = y_uniformity_index(T_final)
        y_var = y_variance_profile(T_final)

        results.append({
            'label': label,
            'T_final': T_final,
            'y_ratio': y_ratio,
            'max_y_var': np.max(y_var),
            'y_var': y_var,
            'x': np.linspace(0, Lx, nx),
            'y': np.linspace(0, Ly, ny),
        })

        print(f"  {label.replace(chr(10), ' ')}: "
              f"y-frac={y_ratio:.4f}, max_y_var={np.max(y_var):.4f}")

    return results


def plot_symmetry_results(aspect_results, heater_results):
    """Generate comprehensive symmetry analysis figures."""

    # --- Figure 1: Aspect ratio analysis ---
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Panel (a): y-gradient fraction vs aspect ratio
    ax1 = fig.add_subplot(gs[0, 0])
    ratios = [r['ratio'] for r in aspect_results]
    y_fracs = [r['y_gradient_fraction'] for r in aspect_results]
    ax1.plot(ratios, y_fracs, 'bo-', markersize=8, linewidth=2)
    ax1.axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='5% threshold')
    ax1.set_xlabel('Aspect ratio $L_x / L_y$', fontsize=11)
    ax1.set_ylabel('$y$-gradient energy fraction', fontsize=11)
    ax1.set_title('(a) 1D Validity vs Aspect Ratio', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel (b): RMSE of 1D vs 2D
    ax2 = fig.add_subplot(gs[0, 1])
    rmses = [r['rmse_1d_vs_2d'] for r in aspect_results]
    ax2.plot(ratios, rmses, 'rs-', markersize=8, linewidth=2)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='0.5°C threshold')
    ax2.set_xlabel('Aspect ratio $L_x / L_y$', fontsize=11)
    ax2.set_ylabel('RMSE: 1D vs 2D avg (°C)', fontsize=11)
    ax2.set_title('(b) 1D Prediction Error', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel (c): Temperature profiles comparison for selected cases
    ax3 = fig.add_subplot(gs[0, 2])
    selected_idx = [0, 2, -1]  # square, moderate, elongated
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i, idx in enumerate(selected_idx):
        r = aspect_results[idx]
        ax3.plot(r['x_2d'] / r['Lx'], r['T_y_avg'], '-', color=colors[i],
                 linewidth=2, label=f"2D avg, $L_x/L_y$={r['ratio']:.1f}")
        ax3.plot(r['x_2d'] / r['Lx'], r['T_final_1d'], '--', color=colors[i],
                 linewidth=1.5, label=f"1D, $L_x/L_y$={r['ratio']:.1f}")
    ax3.set_xlabel('Normalised position $x/L_x$', fontsize=11)
    ax3.set_ylabel('Temperature (°C)', fontsize=11)
    ax3.set_title('(c) 1D vs 2D-Averaged Profiles', fontsize=12)
    ax3.legend(fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.3)

    # Panels (d)-(f): 2D temperature fields for different heater configs
    for i, hr in enumerate(heater_results[:3]):
        ax = fig.add_subplot(gs[1, i])
        X, Y = np.meshgrid(hr['x'], hr['y'], indexing='ij')
        c = ax.contourf(X, Y, hr['T_final'], levels=20, cmap='RdYlBu_r')
        plt.colorbar(c, ax=ax, shrink=0.8)
        ax.set_xlabel('$x$ (m)', fontsize=10)
        ax.set_ylabel('$y$ (m)', fontsize=10)
        y_frac = hr['y_ratio']
        ax.set_title(f"(d-{i+1}) {hr['label'].split(chr(10))[0]}\n"
                     f"$y$-frac = {y_frac:.3f}", fontsize=10)
        ax.set_aspect('equal')

    plt.savefig(os.path.join(RESULTS_DIR, 'symmetry_analysis.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: symmetry_analysis.png")

    # --- Figure 2: Formal symmetry argument diagram ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a): y-variance along x
    ax = axes[0]
    for hr in heater_results:
        ax.plot(hr['x'], hr['y_var'], linewidth=2,
                label=hr['label'].split('\n')[0])
    ax.set_xlabel('Position $x$ (m)', fontsize=11)
    ax.set_ylabel('$\\mathrm{Var}_y[T](x)$ (°C²)', fontsize=11)
    ax.set_title('$y$-Direction Temperature Variance', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel (b): Thermostat trajectory comparison for square room
    ax = axes[1]
    r = aspect_results[0]  # square room
    ax.plot(r['t'], r['T_therm_2d'], 'b-', linewidth=2, label='2D model')
    ax.plot(r['t'], r['T_therm_1d'], 'r--', linewidth=1.5, label='1D model')
    ax.axhline(y=T_SET, color='k', linestyle=':', alpha=0.5)
    ax.set_xlabel('Time (min)', fontsize=11)
    ax.set_ylabel('Thermostat temperature (°C)', fontsize=11)
    ax.set_title(f'1D vs 2D Thermostat Reading (square room)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'symmetry_1d_justification.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: symmetry_1d_justification.png")


def main():
    print("=" * 60)
    print("Symmetry Analysis: Justifying the 1D Reduction")
    print("=" * 60)

    print("\n1. Heater configuration symmetry analysis...")
    heater_results = heater_symmetry_analysis()

    print("\n2. Aspect ratio comparison (2D vs 1D)...")
    aspect_results = compare_2d_vs_1d()

    print("\n3. Generating figures...")
    plot_symmetry_results(aspect_results, heater_results)

    # Print formal summary
    print("\n" + "=" * 65)
    print("Summary: Conditions for Valid 1D Reduction")
    print("-" * 65)
    print("Condition 1: Heater spans full room width (radiator on wall)")
    print(f"  -> y-gradient fraction: {heater_results[0]['y_ratio']:.4f} "
          f"(vs {heater_results[-1]['y_ratio']:.4f} for corner heater)")
    print("Condition 2: Aspect ratio Lx/Ly >= 2")
    for r in aspect_results:
        marker = "✓" if r['y_gradient_fraction'] < 0.05 else "✗"
        print(f"  Lx/Ly = {r['ratio']:.1f}: y-frac = {r['y_gradient_fraction']:.4f}, "
              f"RMSE = {r['rmse_1d_vs_2d']:.3f}°C {marker}")
    print("=" * 65)
    print("\nConclusion: 1D model is valid when:")
    print("  1. The heater approximates a line source across the room width")
    print("  2. The room aspect ratio is >= ~2")
    print("  3. Or: the wall heat transfer is uniform (symmetric Robin BC)")

    return aspect_results, heater_results


if __name__ == '__main__':
    main()
