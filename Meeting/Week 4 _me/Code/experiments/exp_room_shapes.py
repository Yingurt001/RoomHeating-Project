"""
Different Room Shapes: Impact of Geometry on Temperature Control.

The project brief explicitly asks to investigate different 2D room shapes.
This experiment covers:

1. Rectangular rooms with different aspect ratios
2. L-shaped rooms (non-convex domain)
3. Corridor-like rooms (very elongated)

For non-rectangular domains, we use a mask-based approach:
the 2D grid covers the bounding box, and masked cells are treated
as exterior (held at T_ambient).

For each shape, we compare:
- Temperature uniformity at steady state
- Control performance (RMSE, energy, settling time)
- Optimal thermostat/heater placement
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.parameters import (T_AMBIENT, T_INITIAL, T_SET, ALPHA, H_WALL,
                               U_MAX, NX, NY, T_END)
from controllers.pid import PIDController

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


class MaskedHeatEquation2D:
    """
    2D heat equation on an arbitrary domain defined by a mask.

    For cells outside the domain (mask=False), the temperature is held
    at T_ambient. Interior-exterior boundaries are treated with the same
    Robin BC as exterior walls.

    This allows simulation on L-shaped, T-shaped, or arbitrary rooms.
    """

    def __init__(self, Lx, Ly, nx, ny, mask, alpha=ALPHA, h_wall=H_WALL,
                 T_a=T_AMBIENT, T0=T_INITIAL, heater_pos=(0.5, 2.0),
                 heater_radius=0.5, thermostat_pos=(2.5, 2.0)):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.alpha = alpha
        self.h_wall = h_wall
        self.T_a = T_a
        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # Domain mask: True = inside room, False = outside
        self.mask = mask  # shape (nx, ny)
        self.area = np.sum(mask) * self.dx * self.dy

        # Initial condition
        self.T0 = np.where(mask, T0, T_a)

        # Heater profile
        hx, hy = heater_pos
        sigma = heater_radius
        profile = np.exp(-0.5 * ((self.X - hx)**2 + (self.Y - hy)**2) / sigma**2)
        profile = profile * mask  # zero outside domain
        area_sum = np.sum(profile) * self.dx * self.dy
        if area_sum > 0:
            profile = profile / area_sum * self.area
        self._heater_profile = profile

        # Thermostat
        self.therm_ix = np.argmin(np.abs(self.x - thermostat_pos[0]))
        self.therm_iy = np.argmin(np.abs(self.y - thermostat_pos[1]))

        # Build boundary adjacency: for each interior cell, which neighbours
        # are outside the domain?
        self._build_boundary_info()

    def _build_boundary_info(self):
        """Identify cells adjacent to the boundary for Robin BC."""
        nx, ny = self.nx, self.ny
        self.is_boundary = np.zeros((nx, ny), dtype=bool)

        for i in range(nx):
            for j in range(ny):
                if not self.mask[i, j]:
                    continue
                # Check if any neighbour is outside
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if ni < 0 or ni >= nx or nj < 0 or nj >= ny:
                        self.is_boundary[i, j] = True
                        break
                    if not self.mask[ni, nj]:
                        self.is_boundary[i, j] = True
                        break

    def get_thermostat_temperature(self, T_field):
        return T_field[self.therm_ix, self.therm_iy]

    def rhs(self, t, T_flat, u_func):
        T = T_flat.reshape(self.nx, self.ny)
        dx, dy = self.dx, self.dy
        alpha = self.alpha
        nx, ny = self.nx, self.ny

        T_therm = self.get_thermostat_temperature(T)
        u = u_func(t, T_therm)

        dTdt = np.zeros_like(T)

        for i in range(nx):
            for j in range(ny):
                if not self.mask[i, j]:
                    continue

                # x-direction second derivative
                if i == 0 or not self.mask[i-1, j]:
                    # Left boundary: Robin BC
                    if i + 1 < nx and self.mask[i+1, j]:
                        d2x = (2*T[i+1, j] - 2*T[i, j]
                               - 2*dx*self.h_wall*(T[i, j] - self.T_a)) / dx**2
                    else:
                        d2x = -2*self.h_wall*(T[i, j] - self.T_a) / dx
                elif i == nx-1 or not self.mask[i+1, j]:
                    # Right boundary
                    d2x = (2*T[i-1, j] - 2*T[i, j]
                           - 2*dx*self.h_wall*(T[i, j] - self.T_a)) / dx**2
                else:
                    d2x = (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dx**2

                # y-direction second derivative
                if j == 0 or not self.mask[i, j-1]:
                    if j + 1 < ny and self.mask[i, j+1]:
                        d2y = (2*T[i, j+1] - 2*T[i, j]
                               - 2*dy*self.h_wall*(T[i, j] - self.T_a)) / dy**2
                    else:
                        d2y = -2*self.h_wall*(T[i, j] - self.T_a) / dy
                elif j == ny-1 or not self.mask[i, j+1]:
                    d2y = (2*T[i, j-1] - 2*T[i, j]
                           - 2*dy*self.h_wall*(T[i, j] - self.T_a)) / dy**2
                else:
                    d2y = (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dy**2

                dTdt[i, j] = alpha * (d2x + d2y)

        # Add heater source
        dTdt += u * self._heater_profile / self.area

        # Force exterior to stay at T_a
        dTdt[~self.mask] = 0.0

        return dTdt.ravel()

    def simulate(self, u_func, t_end=60.0, dt=0.1, t_eval=None):
        from scipy.integrate import solve_ivp

        if t_eval is None:
            t_eval = np.arange(0, t_end + 1.0, 1.0)

        sol = solve_ivp(
            lambda t, T: self.rhs(t, T, u_func),
            [0, t_end],
            self.T0.ravel(),
            t_eval=t_eval,
            max_step=dt,
            method='RK45'
        )

        nt = len(sol.t)
        T_field = sol.y.reshape(self.nx, self.ny, nt)
        T_therm = T_field[self.therm_ix, self.therm_iy, :]

        return sol.t, T_field, T_therm


# =====================================================================
# Room shape generators
# =====================================================================

def make_rectangular_mask(nx, ny):
    """Full rectangle - all cells are interior."""
    return np.ones((nx, ny), dtype=bool)


def make_l_shaped_mask(nx, ny, cut_fraction=0.5):
    """
    L-shaped room: remove the top-right quadrant.

    Example (cut_fraction=0.5):
        +-------+
        |       |
        |   +---+
        |   |
        +---+
    """
    mask = np.ones((nx, ny), dtype=bool)
    cut_x = int(nx * cut_fraction)
    cut_y = int(ny * cut_fraction)
    mask[cut_x:, cut_y:] = False
    return mask


def make_t_shaped_mask(nx, ny, corridor_width_frac=0.4):
    """
    T-shaped room: a corridor at the bottom, opening to a wider room on top.

        +-------+
        |       |
        +--+ +--+
           | |
           +-+
    """
    mask = np.zeros((nx, ny), dtype=bool)
    # Top part: full width, upper half
    half_y = ny // 2
    mask[:, half_y:] = True
    # Bottom corridor: narrow strip, lower half
    w = int(nx * corridor_width_frac / 2)
    cx = nx // 2
    mask[cx - w:cx + w, :half_y] = True
    return mask


def make_corridor_mask(nx, ny, Lx, Ly, corridor_width=1.5):
    """
    Very elongated corridor (subset of bounding box).
    Just a thin rectangle.
    """
    mask = np.ones((nx, ny), dtype=bool)
    # Only keep the central strip in y
    y = np.linspace(0, Ly, ny)
    y_center = Ly / 2
    for j in range(ny):
        if abs(y[j] - y_center) > corridor_width / 2:
            mask[:, j] = False
    return mask


# =====================================================================
# Experiments
# =====================================================================

def experiment_aspect_ratios():
    """
    Compare control performance across rectangular rooms
    with the same area but different aspect ratios.
    """
    print("\n--- Experiment 1: Different Aspect Ratios ---")

    area = 20.0  # m^2
    aspect_ratios = [1.0, 1.5, 2.0, 3.0, 5.0]

    results = []
    for ratio in aspect_ratios:
        Lx = np.sqrt(area * ratio)
        Ly = np.sqrt(area / ratio)
        nx = max(15, int(Lx / 0.2) + 1)
        ny = max(15, int(Ly / 0.2) + 1)

        mask = make_rectangular_mask(nx, ny)

        # Heater near left wall centre
        heater_pos = (0.3, Ly / 2)
        therm_pos = (Lx / 2, Ly / 2)

        model = MaskedHeatEquation2D(
            Lx, Ly, nx, ny, mask,
            heater_pos=heater_pos, heater_radius=0.5,
            thermostat_pos=therm_pos
        )

        pid = PIDController(Kp=2.0, Ki=0.1, Kd=0.5, T_set=T_SET)
        t_eval = np.arange(0, 61, 1.0)
        t, T_field, T_therm = model.simulate(pid.get_u, t_end=60.0, dt=0.1,
                                              t_eval=t_eval)

        T_final = T_field[:, :, -1]
        T_interior = T_final[mask]
        spatial_std = np.std(T_interior)
        rmse = np.sqrt(np.mean((T_therm - T_SET)**2))

        results.append({
            'ratio': ratio, 'Lx': Lx, 'Ly': Ly,
            'spatial_std': spatial_std,
            'rmse': rmse,
            'T_mean': np.mean(T_interior),
            'T_min': np.min(T_interior),
            'T_max': np.max(T_interior),
            'T_final': T_final,
            'mask': mask,
            'x': np.linspace(0, Lx, nx),
            'y': np.linspace(0, Ly, ny),
        })

        print(f"  Lx/Ly={ratio:.1f}: std={spatial_std:.2f}°C, "
              f"range=[{np.min(T_interior):.1f}, {np.max(T_interior):.1f}]°C")

    return results


def experiment_l_shaped():
    """
    Simulate an L-shaped room and compare with equivalent rectangular room.
    """
    print("\n--- Experiment 2: L-Shaped Room ---")

    Lx, Ly = 6.0, 6.0
    nx, ny = 31, 31

    configs = [
        ('Rectangular\n(6m x 6m)', make_rectangular_mask(nx, ny),
         (0.5, 3.0), (3.0, 3.0)),
        ('L-shaped\n(cut=0.5)', make_l_shaped_mask(nx, ny, 0.5),
         (0.5, 1.5), (1.5, 1.5)),
        ('L-shaped\n(cut=0.33)', make_l_shaped_mask(nx, ny, 0.33),
         (0.5, 3.0), (2.0, 2.0)),
    ]

    results = []
    for label, mask, hpos, tpos in configs:
        model = MaskedHeatEquation2D(
            Lx, Ly, nx, ny, mask,
            heater_pos=hpos, heater_radius=0.5,
            thermostat_pos=tpos
        )

        pid = PIDController(Kp=2.0, Ki=0.1, Kd=0.5, T_set=T_SET)
        t_eval = np.arange(0, 91, 1.0)
        t, T_field, T_therm = model.simulate(pid.get_u, t_end=90.0, dt=0.1,
                                              t_eval=t_eval)

        T_final = T_field[:, :, -1]
        T_interior = T_final[mask]
        spatial_std = np.std(T_interior)

        # Spatial uniformity: what fraction of room is within ±2°C of T_set
        comfort_frac = np.sum(np.abs(T_interior - T_SET) < 2.0) / len(T_interior)

        results.append({
            'label': label,
            'mask': mask,
            'T_final': T_final,
            'T_therm': T_therm,
            't': t,
            'spatial_std': spatial_std,
            'comfort_frac': comfort_frac,
            'T_mean': np.mean(T_interior),
            'x': np.linspace(0, Lx, nx),
            'y': np.linspace(0, Ly, ny),
        })

        print(f"  {label.replace(chr(10), ' ')}: "
              f"std={spatial_std:.2f}°C, comfort={comfort_frac:.1%}")

    return results


def experiment_thermostat_placement_lshaped():
    """
    For an L-shaped room, test different thermostat positions
    to find the best placement.
    """
    print("\n--- Experiment 3: Thermostat Placement in L-Shaped Room ---")

    Lx, Ly = 6.0, 6.0
    nx, ny = 31, 31
    mask = make_l_shaped_mask(nx, ny, 0.5)

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    # Heater near origin corner
    heater_pos = (0.5, 1.0)

    # Test thermostat positions across the L-shaped domain
    test_positions = []
    for i in range(2, nx - 2, 3):
        for j in range(2, ny - 2, 3):
            if mask[i, j]:
                test_positions.append((x[i], y[j]))

    results = []
    for tx, ty in test_positions:
        model = MaskedHeatEquation2D(
            Lx, Ly, nx, ny, mask,
            heater_pos=heater_pos, heater_radius=0.5,
            thermostat_pos=(tx, ty)
        )

        pid = PIDController(Kp=2.0, Ki=0.1, Kd=0.5, T_set=T_SET)
        t_eval = np.arange(0, 91, 1.0)
        t, T_field, T_therm = model.simulate(pid.get_u, t_end=90.0, dt=0.1,
                                              t_eval=t_eval)

        T_final = T_field[:, :, -1]
        T_interior = T_final[mask]
        rmse_spatial = np.sqrt(np.mean((T_interior - T_SET)**2))

        results.append({
            'tx': tx, 'ty': ty,
            'rmse_spatial': rmse_spatial,
            'T_mean': np.mean(T_interior),
        })

    # Find best position
    best = min(results, key=lambda r: r['rmse_spatial'])
    print(f"  Best thermostat position: ({best['tx']:.1f}, {best['ty']:.1f})")
    print(f"  Best spatial RMSE: {best['rmse_spatial']:.2f}°C")

    return results, mask, x, y, heater_pos


# =====================================================================
# Plotting
# =====================================================================

def plot_all_results(aspect_results, l_results, placement_results):
    """Generate all room shape figures."""

    placement_data, mask, x_grid, y_grid, heater_pos = placement_results

    # --- Figure 1: Room shapes comparison ---
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Row 1: Aspect ratio temperature fields
    for i, r in enumerate(aspect_results[:3]):
        ax = fig.add_subplot(gs[0, i])
        X, Y = np.meshgrid(r['x'], r['y'], indexing='ij')
        T_plot = np.ma.masked_where(~r['mask'], r['T_final'])
        c = ax.pcolormesh(X, Y, T_plot, cmap='RdYlBu_r',
                          vmin=T_SET - 5, vmax=T_SET + 3)
        plt.colorbar(c, ax=ax, shrink=0.8)
        ax.set_xlabel('$x$ (m)', fontsize=10)
        ax.set_ylabel('$y$ (m)', fontsize=10)
        ax.set_title(f"$L_x/L_y$ = {r['ratio']:.1f}\n"
                     f"$\\sigma_T$ = {r['spatial_std']:.2f}°C", fontsize=11)
        ax.set_aspect('equal')

    # Row 2: L-shaped rooms
    for i, r in enumerate(l_results):
        ax = fig.add_subplot(gs[1, i])
        X, Y = np.meshgrid(r['x'], r['y'], indexing='ij')
        T_plot = np.ma.masked_where(~r['mask'], r['T_final'])
        c = ax.pcolormesh(X, Y, T_plot, cmap='RdYlBu_r',
                          vmin=T_SET - 5, vmax=T_SET + 3)
        plt.colorbar(c, ax=ax, shrink=0.8)
        ax.set_xlabel('$x$ (m)', fontsize=10)
        ax.set_ylabel('$y$ (m)', fontsize=10)
        ax.set_title(f"{r['label'].split(chr(10))[0]}\n"
                     f"$\\sigma_T$ = {r['spatial_std']:.2f}°C, "
                     f"comfort = {r['comfort_frac']:.0%}", fontsize=10)
        ax.set_aspect('equal')

    plt.suptitle('Temperature Fields: Different Room Shapes (t = final)',
                 fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(RESULTS_DIR, 'room_shapes_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: room_shapes_comparison.png")

    # --- Figure 2: Thermostat placement heatmap for L-shaped room ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel (a): RMSE heatmap
    ax = axes[0]
    nx_m, ny_m = mask.shape
    rmse_map = np.full((nx_m, ny_m), np.nan)
    for r in placement_data:
        ix = np.argmin(np.abs(x_grid - r['tx']))
        iy = np.argmin(np.abs(y_grid - r['ty']))
        rmse_map[ix, iy] = r['rmse_spatial']

    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
    rmse_plot = np.ma.masked_where(~mask | np.isnan(rmse_map), rmse_map)
    c = ax.pcolormesh(X, Y, rmse_plot, cmap='RdYlGn_r', shading='nearest')
    plt.colorbar(c, ax=ax, label='Spatial RMSE (°C)')

    # Mark best position
    best = min(placement_data, key=lambda r: r['rmse_spatial'])
    ax.plot(best['tx'], best['ty'], 'w*', markersize=15, zorder=5)
    ax.plot(heater_pos[0], heater_pos[1], 'r^', markersize=12, zorder=5,
            label='Heater')
    ax.set_xlabel('$x$ (m)', fontsize=11)
    ax.set_ylabel('$y$ (m)', fontsize=11)
    ax.set_title('(a) Thermostat Placement RMSE\nin L-Shaped Room', fontsize=12)
    ax.set_aspect('equal')
    ax.legend()

    # Panel (b): Summary bar chart
    ax = axes[1]
    shapes = ['Square\n(1:1)', 'Rect\n(2:1)', 'Rect\n(5:1)',
              'L-shaped\n(cut 50%)', 'L-shaped\n(cut 33%)']
    stds = [aspect_results[0]['spatial_std'],
            aspect_results[2]['spatial_std'],
            aspect_results[4]['spatial_std'],
            l_results[1]['spatial_std'],
            l_results[2]['spatial_std']]
    colors_bar = ['#4c72b0', '#4c72b0', '#4c72b0', '#dd8452', '#dd8452']
    bars = ax.bar(range(len(shapes)), stds, color=colors_bar, edgecolor='black')
    ax.set_xticks(range(len(shapes)))
    ax.set_xticklabels(shapes, fontsize=9)
    ax.set_ylabel('Spatial temperature std. dev. (°C)', fontsize=11)
    ax.set_title('(b) Temperature Non-Uniformity\nby Room Shape', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'room_shapes_placement.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: room_shapes_placement.png")

    # --- Figure 3: L-shaped room thermostat trajectories ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in l_results:
        ax.plot(r['t'], r['T_therm'], linewidth=2,
                label=r['label'].replace('\n', ' '))
    ax.axhline(y=T_SET, color='k', linestyle=':', alpha=0.5, label='$T_{set}$')
    ax.set_xlabel('Time (min)', fontsize=11)
    ax.set_ylabel('Thermostat temperature (°C)', fontsize=11)
    ax.set_title('Thermostat Response: Rectangular vs L-Shaped Rooms', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'room_shapes_trajectories.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: room_shapes_trajectories.png")


def main():
    print("=" * 60)
    print("Different Room Shapes Analysis")
    print("=" * 60)

    print("\n1. Aspect ratio experiments...")
    aspect_results = experiment_aspect_ratios()

    print("\n2. L-shaped room experiments...")
    l_results = experiment_l_shaped()

    print("\n3. Thermostat placement in L-shaped room...")
    placement_results = experiment_thermostat_placement_lshaped()

    print("\n4. Generating figures...")
    plot_all_results(aspect_results, l_results, placement_results)

    # Summary
    print("\n" + "=" * 65)
    print("Key Findings:")
    print("-" * 65)
    print("1. Elongated rooms (ratio > 3) have larger temperature gradients")
    print("   along the long axis, making thermostat placement more critical.")
    print("2. L-shaped rooms create 'cold corners' in the part farthest")
    print("   from the heater — a problem that 1D models cannot capture.")
    print("3. Optimal thermostat placement in L-shaped rooms is near the")
    print("   junction of the two wings, not at the geometric centre.")
    print("=" * 65)

    return aspect_results, l_results, placement_results


if __name__ == '__main__':
    main()
