"""
Pontryagin Minimum Principle Controller for room heating.

We seek to minimise:
    J = integral_0^T_f [ Q*(T - T_set)^2 + R*u^2 ] dt

Subject to:
    dT/dt = -k*(T - T_a) + u,   u in [0, U_max]

The Hamiltonian is:
    H = Q*(T - T_set)^2 + R*u^2 + lambda * (-k*(T - T_a) + u)

Necessary conditions (Pontryagin):
1. State equation:    dT/dt = dH/d_lambda = -k*(T - T_a) + u*
2. Costate equation:  d_lambda/dt = -dH/dT = -2*Q*(T - T_set) + k*lambda
3. Optimality:        dH/du = 0 => u* = -lambda / (2R), clamped to [0, U_max]

Boundary conditions:
    T(0) = T_0       (initial temperature)
    lambda(T_f) = 0  (free terminal state, transversality condition)

This is a Two-Point Boundary Value Problem (TPBVP), solved with scipy.solve_bvp.

References:
- Liberzon D. Calculus of Variations and Optimal Control Theory: A Concise
  Introduction. Princeton University Press, 2012.
  (Chapters 4-5: Pontryagin minimum principle with constraints)
- Kirk DE. Optimal Control Theory: An Introduction. Dover, 2004.
  (Chapter 5: The minimum principle; Chapter 6: constrained controls)
"""

import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.parameters import (
    T_SET, T_AMBIENT, T_INITIAL, U_MAX, K_COOL, T_END
)


def optimal_control_from_costate(lam, R, U_max):
    """
    Compute optimal control from costate variable.

    u* = -lambda / (2R), clamped to [0, U_max].
    """
    u = -lam / (2.0 * R)
    return np.clip(u, 0.0, U_max)


def pontryagin_bvp(Q=1.0, R=0.01, T0=T_INITIAL, T_set=T_SET, T_a=T_AMBIENT,
                    k=K_COOL, U_max=U_MAX, t_end=T_END, n_nodes=500,
                    continuation=True, verbose=False):
    """
    Solve the optimal control problem using Pontryagin's minimum principle.

    Sets up and solves the TPBVP:
        dT/dt = -k*(T - T_a) + u*(lambda)
        d_lambda/dt = -2*Q*(T - T_set) + k*lambda
        T(0) = T0,  lambda(t_end) = 0

    Uses a continuation method for robust convergence: starts with large R
    (easy problem), then gradually decreases R toward the target value,
    using each solution as the initial guess for the next.

    Parameters
    ----------
    Q : float
        State cost weight.
    R : float
        Control cost weight.
    T0 : float
        Initial temperature.
    T_set : float
        Desired temperature.
    T_a : float
        Ambient temperature.
    k : float
        Cooling constant.
    U_max : float
        Maximum control input.
    t_end : float
        Final time.
    n_nodes : int
        Number of mesh nodes for BVP solver.
    continuation : bool
        If True, use continuation method for small R values.
    verbose : bool
        Print progress information.

    Returns
    -------
    t : ndarray
        Time array.
    T : ndarray
        Optimal temperature trajectory.
    u : ndarray
        Optimal control trajectory.
    lam : ndarray
        Costate trajectory.
    sol : OdeSolution
        The full BVP solution object.
    """

    def _solve_single(Q_val, R_val, t_mesh, y_guess):
        """Solve BVP for a single (Q, R) pair."""
        def ode(t, y):
            T_val = y[0]
            lam = y[1]
            u = optimal_control_from_costate(lam, R_val, U_max)
            dTdt = -k * (T_val - T_a) + u
            dldt = -2.0 * Q_val * (T_val - T_set) + k * lam
            return np.vstack([dTdt, dldt])

        def bc(ya, yb):
            return np.array([
                ya[0] - T0,
                yb[1] - 0.0
            ])

        sol = solve_bvp(ode, bc, t_mesh, y_guess, tol=1e-4, max_nodes=10000)
        return sol

    t_mesh = np.linspace(0, t_end, n_nodes)

    # Initial guess: linear interpolation for T, zero for lambda
    T_guess = T0 + (T_set - T0) * np.minimum(t_mesh / (t_end * 0.3), 1.0)
    lam_guess = -2.0 * R * (T_guess - T_set)  # heuristic costate guess
    y_guess = np.vstack([T_guess, lam_guess])

    if continuation and R < 0.5:
        # Continuation: solve easy problems first, then refine
        R_sequence = []
        R_cur = max(R, 0.001)
        R_start = 2.0
        while R_start > R_cur * 1.5:
            R_sequence.append(R_start)
            R_start /= 3.0
        R_sequence.append(R_cur)

        if verbose:
            print(f"  Continuation: R sequence = {[f'{r:.4f}' for r in R_sequence]}")

        current_mesh = t_mesh
        current_guess = y_guess

        for i, R_step in enumerate(R_sequence):
            sol = _solve_single(Q, R_step, current_mesh, current_guess)
            if sol.success:
                current_mesh = sol.x
                current_guess = sol.y
                if verbose:
                    print(f"    R={R_step:.4f}: converged ({len(sol.x)} nodes)")
            else:
                if verbose:
                    print(f"    R={R_step:.4f}: FAILED, using last good solution")
                if i > 0:
                    break
                # Even the easiest problem failed — fall through to direct solve
                sol = _solve_single(Q, R, t_mesh, y_guess)
                break
    else:
        sol = _solve_single(Q, R, t_mesh, y_guess)

    if not sol.success:
        print(f"Warning: BVP solver did not converge: {sol.message}")

    t = sol.x
    T_opt = sol.y[0]
    lam_opt = sol.y[1]
    u_opt = optimal_control_from_costate(lam_opt, R, U_max)

    return t, T_opt, u_opt, lam_opt, sol


class PontryaginController:
    """
    Pre-computed optimal controller from Pontryagin's principle.

    Solves the TPBVP once, then interpolates the optimal control trajectory.
    This is an open-loop controller (pre-computed, not feedback).
    """

    def __init__(self, Q=1.0, R=0.01, T0=T_INITIAL, T_set=T_SET,
                 T_a=T_AMBIENT, k=K_COOL, U_max=U_MAX, t_end=T_END):
        self.Q = Q
        self.R = R
        self.T_set = T_set
        self.U_max = U_max

        # Solve BVP
        t, T, u, lam, sol = pontryagin_bvp(
            Q=Q, R=R, T0=T0, T_set=T_set, T_a=T_a,
            k=k, U_max=U_max, t_end=t_end
        )
        self._t = t
        self._T = T
        self._u = u
        self._lam = lam
        self._sol = sol

    def get_u(self, t, T):
        """Interpolate pre-computed optimal control at time t."""
        return float(np.interp(t, self._t, self._u))

    def get_trajectory(self):
        """Return pre-computed (t, T, u, lambda)."""
        return self._t, self._T, self._u, self._lam

    def reset(self):
        """Open-loop controller, nothing to reset."""
        pass
