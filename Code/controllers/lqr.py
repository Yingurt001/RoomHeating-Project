"""
Linear Quadratic Regulator (LQR) for room heating.

State-space model:
    dx/dt = A*x + B*u
where x = T - T_set (temperature deviation from setpoint).

    dx/dt = -k*x - k*(T_set - T_a) + u

Rewrite as: dx/dt = A*x + B*u + f
    A = -k
    B = 1
    f = -k*(T_set - T_a)  (constant disturbance from ambient)

The LQR minimises the quadratic cost:
    J = integral_0^inf [ Q*x^2 + R*u^2 ] dt

The optimal gain K is found by solving the algebraic Riccati equation:
    A'P + PA - PBR^{-1}B'P + Q = 0
    K = R^{-1} B' P

For scalar systems: P satisfies -k*P + P*(-k) - P^2/R + Q = 0
    => P = R*(-k + sqrt(k^2 + Q/R))

References:
- Anderson BDO, Moore JB. Optimal Control: Linear Quadratic Methods.
  Prentice-Hall, 1990. (Reprint: Dover, 2007.)
  (Definitive textbook on LQR theory and Riccati equations)
- Astrom KJ, Murray RM. Feedback Systems. 2nd ed. Princeton, 2021.
  Chapter 7: State Feedback. (LQR derivation for linear systems)
"""

import numpy as np
from scipy.linalg import solve_continuous_are
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.parameters import T_SET, T_AMBIENT, U_MAX, K_COOL


class LQRController:
    """
    LQR controller for the room heating ODE.

    Parameters
    ----------
    Q : float
        State cost weight (penalises temperature deviation).
    R : float
        Control cost weight (penalises energy usage).
    k : float
        Cooling constant.
    T_set : float
        Temperature setpoint.
    T_a : float
        Ambient temperature.
    U_max : float
        Maximum control output.
    """

    def __init__(self, Q=1.0, R=0.01, k=K_COOL, T_set=T_SET, T_a=T_AMBIENT,
                 U_max=U_MAX):
        self.Q = Q
        self.R = R
        self.k = k
        self.T_set = T_set
        self.T_a = T_a
        self.U_max = U_max

        # Compute LQR gain
        self.K, self.P = self._solve_riccati()

        # Steady-state feedforward: to maintain T_set, need u_ss = k*(T_set - T_a)
        self.u_ss = k * (T_set - T_a)

    def _solve_riccati(self):
        """
        Solve the algebraic Riccati equation for the scalar system.

        For scalar case A=-k, B=1:
            P = R * (-k + sqrt(k^2 + Q/R))
            K = P / R = -k + sqrt(k^2 + Q/R)
        """
        A = np.array([[-self.k]])
        B = np.array([[1.0]])
        Q_mat = np.array([[self.Q]])
        R_mat = np.array([[self.R]])

        P = solve_continuous_are(A, B, Q_mat, R_mat)
        K = (1.0 / self.R) * B.T @ P

        return K[0, 0], P[0, 0]

    def get_u(self, t, T):
        """
        Compute LQR control output.

        u = -K * (T - T_set) + u_ss

        Clamped to [0, U_max].
        """
        x = T - self.T_set  # deviation from setpoint
        u = -self.K * x + self.u_ss
        return np.clip(u, 0.0, self.U_max)

    def get_cost_weights(self):
        """Return (Q, R) for external analysis."""
        return self.Q, self.R

    def get_gain(self):
        """Return the optimal feedback gain K."""
        return self.K

    def reset(self):
        """LQR is stateless (no integrator), nothing to reset."""
        pass


def pareto_scan(Q_values, R_values, k=K_COOL, T_set=T_SET, T_a=T_AMBIENT,
                U_max=U_MAX):
    """
    Scan Q/R parameter space and return LQR controllers for Pareto analysis.

    Parameters
    ----------
    Q_values : array-like
        State cost weights to scan.
    R_values : array-like
        Control cost weights to scan.

    Returns
    -------
    controllers : list of LQRController
        One controller per (Q, R) combination.
    params : list of (Q, R) tuples
    """
    controllers = []
    params = []
    for Q in Q_values:
        for R in R_values:
            ctrl = LQRController(Q=Q, R=R, k=k, T_set=T_set, T_a=T_a,
                                 U_max=U_max)
            controllers.append(ctrl)
            params.append((Q, R))
    return controllers, params
