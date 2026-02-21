"""
1D PDE Model: Heat equation along a room cross-section.

    dT/dt = alpha * d^2T/dx^2 + S(x, t)

where S(x, t) is a localised heat source (heater) and boundary conditions
model heat loss through walls.

Boundary Conditions (Robin / convective):
    Left wall  (x=0): -alpha * dT/dx = h * (T - T_a)   (heat loss to outside)
    Right wall (x=L): -alpha * dT/dx = -h * (T - T_a)   (heat loss to outside)

    Note: sign convention follows outward normal direction.

Method: Method of Lines (MOL) — discretise space with finite differences,
then solve the resulting ODE system with scipy.solve_ivp.

References:
- Strikwerda JC. Finite Difference Schemes and Partial Differential
  Equations. 2nd ed. SIAM, 2004.
  (Chapters 7-8: Heat equation discretisation, stability analysis)
"""

import numpy as np
from scipy.integrate import solve_ivp
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.parameters import (
    T_AMBIENT, T_INITIAL, T_SET, ALPHA, K_COOL, H_WALL,
    ROOM_LENGTH, NX, T_END, DT
)


class HeatEquation1D:
    """
    1D heat equation solver using Method of Lines.

    Parameters
    ----------
    L : float
        Room length (m).
    nx : int
        Number of spatial grid points.
    alpha : float
        Thermal diffusivity (m^2/min).
    h_wall : float
        Wall heat transfer coefficient (1/m) for Robin BC.
    T_a : float
        Ambient temperature outside.
    T0 : float or ndarray
        Initial temperature (scalar for uniform, array for spatial).
    heater_pos : float
        Position of heater centre along x (m).
    heater_width : float
        Width of heater along x (m).
    thermostat_pos : float
        Position of thermostat along x (m). Temperature at this point
        is used for control feedback.
    """

    def __init__(self, L=ROOM_LENGTH, nx=NX, alpha=ALPHA, h_wall=H_WALL,
                 T_a=T_AMBIENT, T0=T_INITIAL, heater_pos=0.5,
                 heater_width=0.5, thermostat_pos=2.5):
        self.L = L
        self.nx = nx
        self.alpha = alpha
        self.h_wall = h_wall
        self.T_a = T_a
        self.dx = L / (nx - 1)
        self.x = np.linspace(0, L, nx)

        # Initial condition
        if np.isscalar(T0):
            self.T0 = np.full(nx, T0)
        else:
            self.T0 = np.array(T0)

        # Heater spatial profile: Gaussian-like localisation
        self.heater_pos = heater_pos
        self.heater_width = heater_width
        self._heater_profile = self._make_heater_profile()

        # Thermostat: nearest grid index
        self.thermostat_pos = thermostat_pos
        self.thermostat_idx = np.argmin(np.abs(self.x - thermostat_pos))

    def _make_heater_profile(self):
        """Create spatial heater profile (normalised so integral ~ 1)."""
        sigma = self.heater_width / 2.0
        profile = np.exp(-0.5 * ((self.x - self.heater_pos) / sigma) ** 2)
        # Normalise so total heating input equals u(t) * L
        profile = profile / (np.sum(profile) * self.dx) * self.L
        return profile

    def get_thermostat_temperature(self, T_field):
        """Read temperature at the thermostat position."""
        return T_field[self.thermostat_idx]

    def rhs(self, t, T_flat, u_func):
        """
        Right-hand side of the semi-discrete system.

        Parameters
        ----------
        t : float
            Current time.
        T_flat : ndarray of shape (nx,)
            Temperature at each grid point.
        u_func : callable(t, T_thermostat) -> float
            Controller function.
        """
        T = T_flat
        dx = self.dx
        alpha = self.alpha
        nx = self.nx

        # Read thermostat
        T_therm = self.get_thermostat_temperature(T)

        # Get control input
        u = u_func(t, T_therm)

        # Allocate dT/dt
        dTdt = np.zeros(nx)

        # Interior points: central difference for d^2T/dx^2
        dTdt[1:-1] = alpha * (T[2:] - 2 * T[1:-1] + T[:-2]) / dx**2

        # Robin BC at x=0 (left wall):
        # -alpha * dT/dx|_0 = h_wall * (T[0] - T_a)
        # Ghost point: T[-1] = T[1] - 2*dx*h_wall/alpha * (T[0] - T_a)
        # => d2T/dx2|_0 = (T[1] - T[0] - dx*h_wall/alpha*(T[0]-T_a)) * 2/(dx^2)
        # Simplified using ghost point approach:
        dTdt[0] = alpha * (2 * T[1] - 2 * T[0] - 2 * dx * self.h_wall * (T[0] - self.T_a)) / dx**2

        # Robin BC at x=L (right wall):
        # alpha * dT/dx|_L = h_wall * (T[-1] - T_a)  (outward flux positive to right)
        dTdt[-1] = alpha * (2 * T[-2] - 2 * T[-1] - 2 * dx * self.h_wall * (T[-1] - self.T_a)) / dx**2

        # Add heater source
        dTdt += u * self._heater_profile / self.L

        return dTdt

    def simulate(self, u_func, t_end=T_END, dt=DT, t_eval=None):
        """
        Run the 1D heat equation simulation.

        Parameters
        ----------
        u_func : callable(t, T_thermostat) -> float
            Controller. Receives thermostat temperature reading.
        t_end : float
            Simulation end time.
        dt : float
            Maximum time step for solver.
        t_eval : ndarray, optional
            Times at which to store solution.

        Returns
        -------
        t : ndarray of shape (nt,)
            Time array.
        T_field : ndarray of shape (nx, nt)
            Temperature field. T_field[i, j] = T(x_i, t_j).
        T_therm : ndarray of shape (nt,)
            Temperature at thermostat position.
        """
        if t_eval is None:
            # Use coarser output for storage efficiency
            t_eval = np.arange(0, t_end + 0.1, 0.1)

        sol = solve_ivp(
            lambda t, T: self.rhs(t, T, u_func),
            [0, t_end],
            self.T0,
            t_eval=t_eval,
            max_step=dt,
            method='RK45'
        )

        T_field = sol.y          # shape (nx, nt)
        t = sol.t
        T_therm = T_field[self.thermostat_idx, :]

        return t, T_field, T_therm
