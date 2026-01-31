"""
Multi-room building model: N rooms in a row sharing internal walls.

Each room has its own heater and thermostat. Internal walls have lower
thermal resistance than external walls, allowing heat exchange between
adjacent rooms.

Model for room i:
    dT_i/dt = -k_ext * (T_i - T_a)                     # external wall loss
              - k_int * (T_i - T_{i-1})                  # left neighbour
              - k_int * (T_i - T_{i+1})                  # right neighbour
              + u_i(t)                                    # heater input

where k_ext applies to rooms with exterior walls and k_int governs
inter-room heat exchange.

This is a coupled ODE system:
    dT/dt = A*T + B*u + c

References:
- Blasco C et al. (2012). Modelling and PID control of HVAC system.
- Astrom KJ, Murray RM (2021). Feedback Systems, Ch. 11.
"""

import numpy as np
from scipy.integrate import solve_ivp
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.parameters import T_AMBIENT, T_INITIAL, T_SET, U_MAX, K_COOL


class BuildingModel:
    """
    Multi-room building with N rooms in a linear arrangement.

    Room layout: [ext | room 0 | int | room 1 | ... | room N-1 | ext]

    Parameters
    ----------
    n_rooms : int
        Number of rooms.
    k_ext : float
        External wall cooling constant (1/min). Higher = worse insulation.
    k_int : float
        Internal wall coupling constant (1/min). Higher = more heat exchange.
    T_a : float
        Ambient (outdoor) temperature.
    T0 : float or array
        Initial temperature (scalar for uniform, array per room).
    exterior_walls : list of list
        For each room, which sides are exterior. Default: room 0 has left
        exterior, room N-1 has right exterior.
    """

    def __init__(self, n_rooms=5, k_ext=0.1, k_int=0.05,
                 T_a=T_AMBIENT, T0=T_INITIAL, U_max=U_MAX):
        self.n_rooms = n_rooms
        self.k_ext = k_ext
        self.k_int = k_int
        self.T_a = T_a
        self.U_max = U_max

        if np.isscalar(T0):
            self.T0 = np.full(n_rooms, T0)
        else:
            self.T0 = np.array(T0)

        # Build system matrix A
        self._build_system_matrix()

    def _build_system_matrix(self):
        """Build the coupled ODE system matrix."""
        n = self.n_rooms
        A = np.zeros((n, n))

        for i in range(n):
            # External wall losses
            if i == 0 or i == n - 1:
                # End rooms have one external wall
                A[i, i] -= self.k_ext
            if i == 0 and i == n - 1:
                # Single room: two external walls
                A[i, i] -= self.k_ext

            # Internal wall coupling
            if i > 0:
                A[i, i] -= self.k_int
                A[i, i-1] += self.k_int
            if i < n - 1:
                A[i, i] -= self.k_int
                A[i, i+1] += self.k_int

        self.A = A

        # Constant term from ambient: c = -k_ext * T_a for exterior rooms
        self.c = np.zeros(n)
        if n > 1:
            self.c[0] = self.k_ext * self.T_a
            self.c[-1] = self.k_ext * self.T_a
        else:
            self.c[0] = 2 * self.k_ext * self.T_a

    def rhs(self, t, T, u_funcs):
        """
        Right-hand side: dT/dt = A*T + u + c

        Parameters
        ----------
        t : float
        T : ndarray of shape (n_rooms,)
        u_funcs : list of callable(t, T_i) -> float
            One controller per room.
        """
        u = np.array([u_funcs[i](t, T[i]) for i in range(self.n_rooms)])
        u = np.clip(u, 0, self.U_max)
        return self.A @ T + u + self.c

    def simulate(self, u_funcs, t_end=120.0, dt=0.01, t_eval=None):
        """
        Simulate the building.

        Parameters
        ----------
        u_funcs : list of callable(t, T_i)
            Controller for each room.
        t_end : float
        dt : float

        Returns
        -------
        t : ndarray of shape (nt,)
        T : ndarray of shape (n_rooms, nt)
        """
        if t_eval is None:
            t_eval = np.arange(0, t_end + 0.1, 0.1)

        sol = solve_ivp(
            lambda t, T: self.rhs(t, T, u_funcs),
            [0, t_end],
            self.T0,
            t_eval=t_eval,
            max_step=dt,
            method='RK45'
        )
        return sol.t, sol.y

    def simulate_with_strategies(self, strategies, t_end=120.0):
        """
        Run building with different control strategies.

        Parameters
        ----------
        strategies : dict
            {name: list of controller factories}

        Returns
        -------
        results : dict
            {name: (t, T_array, u_array, metrics_per_room)}
        """
        from utils.metrics import compute_all_metrics

        results = {}
        for name, ctrl_factories in strategies.items():
            controllers = [f() for f in ctrl_factories]
            u_funcs = [c.get_u for c in controllers]

            t, T = self.simulate(u_funcs, t_end=t_end)

            # Compute u for each room
            u_array = np.zeros_like(T)
            for i in range(self.n_rooms):
                ctrl_replay = ctrl_factories[i]()
                u_array[i] = [ctrl_replay.get_u(tj, T[i, j])
                              for j, tj in enumerate(t)]

            # Metrics per room
            metrics = []
            for i in range(self.n_rooms):
                m = compute_all_metrics(t, T[i], u_array[i], T_SET)
                metrics.append(m)

            results[name] = (t, T, u_array, metrics)

        return results
