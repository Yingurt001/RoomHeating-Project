"""
2D PDE Model: Heat equation for a rectangular room.

    dT/dt = alpha * (d^2T/dx^2 + d^2T/dy^2) + S(x, y, t)

Boundary Conditions (Robin, per-wall segmented):
    -alpha * dT/dn = h(s) * (T - T_a)
    where n is the outward normal and h(s) can vary along each wall.

Supports:
    - Segmented h arrays: different h per grid point on each wall
      (windows, doors, insulated segments)
    - Domain masking: irregular room shapes (e.g. L-shaped)
    - Time-varying BC: callback to update h arrays each timestep
      (e.g. door opening/closing)

Method: Method of Lines with 2D finite differences, vectorised.

References:
- Strikwerda JC. Finite Difference Schemes and Partial Differential
  Equations. 2nd ed. SIAM, 2004.
  (Chapter 9: Multi-dimensional problems)
"""

import numpy as np
from scipy.integrate import solve_ivp
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.parameters import (
    T_AMBIENT, T_INITIAL, ALPHA, H_WALL,
    ROOM_LENGTH, ROOM_WIDTH, NX, NY, T_END, DT
)


class HeatEquation2D:
    """
    2D heat equation solver for a rectangular room.

    Parameters
    ----------
    Lx, Ly : float
        Room dimensions (m).
    nx, ny : int
        Number of grid points in x, y.
    alpha : float
        Thermal diffusivity (m^2/min).
    h_wall : float
        Default wall heat transfer coefficient (1/m).
    T_a : float
        Ambient temperature.
    T0 : float or ndarray
        Initial temperature.
    heater_pos : tuple (x, y)
        Centre of heater.
    heater_radius : float
        Heater influence radius (m).
    thermostat_pos : tuple (x, y)
        Position of thermostat sensor.
    wall_h : dict, optional
        Per-wall h arrays: {'south': array(nx), 'north': array(nx),
        'west': array(ny), 'east': array(ny)}.
        Missing keys default to np.full(n, h_wall).
    domain_mask : ndarray of bool (nx, ny), optional
        True = active region, False = excluded (e.g. L-shape cutout).
    h_updater : callable(t, model), optional
        Called at each RHS evaluation to update h arrays (time-varying BC).
    """

    def __init__(self, Lx=ROOM_LENGTH, Ly=ROOM_WIDTH, nx=NX, ny=NY,
                 alpha=ALPHA, h_wall=H_WALL, T_a=T_AMBIENT, T0=T_INITIAL,
                 heater_pos=(0.5, 2.0), heater_radius=0.5,
                 thermostat_pos=(2.5, 2.0),
                 wall_h=None, domain_mask=None, h_updater=None):
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

        # Per-wall h arrays (segmented Robin BC)
        if wall_h is None:
            self.h_south = np.full(nx, h_wall)
            self.h_north = np.full(nx, h_wall)
            self.h_west = np.full(ny, h_wall)
            self.h_east = np.full(ny, h_wall)
        else:
            self.h_south = wall_h.get('south', np.full(nx, h_wall))
            self.h_north = wall_h.get('north', np.full(nx, h_wall))
            self.h_west = wall_h.get('west', np.full(ny, h_wall))
            self.h_east = wall_h.get('east', np.full(ny, h_wall))

        # Domain mask (for irregular shapes)
        if domain_mask is not None:
            self.mask = domain_mask
        else:
            self.mask = np.ones((nx, ny), dtype=bool)

        # Time-varying BC callback
        self._h_updater = h_updater

        # Initial condition
        if np.isscalar(T0):
            self.T0 = np.full((nx, ny), T0)
        else:
            self.T0 = np.array(T0).reshape(nx, ny)

        # Heater profile
        self.heater_pos = heater_pos
        self.heater_radius = heater_radius
        self._heater_profile = self._make_heater_profile()

        # Thermostat position
        self.thermostat_pos = thermostat_pos
        self.therm_ix = np.argmin(np.abs(self.x - thermostat_pos[0]))
        self.therm_iy = np.argmin(np.abs(self.y - thermostat_pos[1]))

    def _make_heater_profile(self):
        """Create 2D Gaussian heater profile."""
        hx, hy = self.heater_pos
        sigma = self.heater_radius
        profile = np.exp(-0.5 * ((self.X - hx)**2 + (self.Y - hy)**2) / sigma**2)
        # Normalise so integral over area = Lx * Ly
        area = np.sum(profile) * self.dx * self.dy
        if area > 0:
            profile = profile / area * self.Lx * self.Ly
        return profile

    def get_thermostat_temperature(self, T_field):
        """Read temperature at thermostat position."""
        return T_field[self.therm_ix, self.therm_iy]

    def rhs(self, t, T_flat, u_func):
        """
        Right-hand side of the semi-discrete 2D system.

        T_flat is of shape (nx*ny,), reshaped to (nx, ny).
        """
        # Time-varying BC: update h arrays if callback is set
        if self._h_updater is not None:
            self._h_updater(t, self)

        T = T_flat.reshape(self.nx, self.ny)
        dx, dy = self.dx, self.dy
        alpha = self.alpha

        T_therm = self.get_thermostat_temperature(T)
        u = u_func(t, T_therm)

        dTdt = np.zeros_like(T)

        # Interior: central differences
        dTdt[1:-1, 1:-1] = alpha * (
            (T[2:, 1:-1] - 2*T[1:-1, 1:-1] + T[:-2, 1:-1]) / dx**2 +
            (T[1:-1, 2:] - 2*T[1:-1, 1:-1] + T[1:-1, :-2]) / dy**2
        )

        # Robin BC: ghost-point approach with per-wall h arrays

        # West wall (x=0), interior y
        dTdt[0, 1:-1] = alpha * (
            (2*T[1, 1:-1] - 2*T[0, 1:-1] - 2*dx*self.h_west[1:-1]*(T[0, 1:-1] - self.T_a)) / dx**2 +
            (T[0, 2:] - 2*T[0, 1:-1] + T[0, :-2]) / dy**2
        )

        # East wall (x=Lx), interior y
        dTdt[-1, 1:-1] = alpha * (
            (2*T[-2, 1:-1] - 2*T[-1, 1:-1] - 2*dx*self.h_east[1:-1]*(T[-1, 1:-1] - self.T_a)) / dx**2 +
            (T[-1, 2:] - 2*T[-1, 1:-1] + T[-1, :-2]) / dy**2
        )

        # South wall (y=0), interior x
        dTdt[1:-1, 0] = alpha * (
            (T[2:, 0] - 2*T[1:-1, 0] + T[:-2, 0]) / dx**2 +
            (2*T[1:-1, 1] - 2*T[1:-1, 0] - 2*dy*self.h_south[1:-1]*(T[1:-1, 0] - self.T_a)) / dy**2
        )

        # North wall (y=Ly), interior x
        dTdt[1:-1, -1] = alpha * (
            (T[2:, -1] - 2*T[1:-1, -1] + T[:-2, -1]) / dx**2 +
            (2*T[1:-1, -2] - 2*T[1:-1, -1] - 2*dy*self.h_north[1:-1]*(T[1:-1, -1] - self.T_a)) / dy**2
        )

        # Corners: use both Robin BCs
        # h_x = west or east, h_y = south or north
        for (ix, iy) in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:
            # x-direction: west (ix=0) or east (ix=-1)
            h_x = self.h_west[iy] if ix == 0 else self.h_east[iy]
            if ix == 0:
                d2x = (2*T[1, iy] - 2*T[0, iy] - 2*dx*h_x*(T[0, iy] - self.T_a)) / dx**2
            else:
                d2x = (2*T[-2, iy] - 2*T[-1, iy] - 2*dx*h_x*(T[-1, iy] - self.T_a)) / dx**2
            # y-direction: south (iy=0) or north (iy=-1)
            h_y = self.h_south[ix] if iy == 0 else self.h_north[ix]
            if iy == 0:
                d2y = (2*T[ix, 1] - 2*T[ix, 0] - 2*dy*h_y*(T[ix, 0] - self.T_a)) / dy**2
            else:
                d2y = (2*T[ix, -2] - 2*T[ix, -1] - 2*dy*h_y*(T[ix, -1] - self.T_a)) / dy**2
            dTdt[ix, iy] = alpha * (d2x + d2y)

        # Add heater source
        dTdt += u * self._heater_profile / (self.Lx * self.Ly)

        # Domain mask: freeze excluded regions
        dTdt[~self.mask] = 0.0

        return dTdt.ravel()

    def simulate(self, u_func, t_end=T_END, dt=DT, t_eval=None):
        """
        Run the 2D heat equation simulation.

        Parameters
        ----------
        u_func : callable(t, T_thermostat) -> float
            Controller.
        t_end : float
            Simulation end time.
        dt : float
            Maximum time step for solver.
        t_eval : ndarray, optional
            Times at which to store solution.

        Returns
        -------
        t : ndarray of shape (nt,)
        T_field : ndarray of shape (nx, ny, nt)
        T_therm : ndarray of shape (nt,)
        """
        if t_eval is None:
            t_eval = np.arange(0, t_end + 0.5, 0.5)

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
