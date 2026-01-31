"""
ODE Model: Newton's Law of Cooling for a spatially uniform room.

    dT/dt = -k * (T - T_a) + u(t)

Given a control input function u(t), this module solves for T(t).
The model is decoupled from any specific control strategy.
"""

import numpy as np
from scipy.integrate import solve_ivp
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.parameters import T_AMBIENT, T_INITIAL, K_COOL, T_END, DT, U_MAX


def ode_rhs(t, T, k, T_a, u_func):
    """Right-hand side: dT/dt = -k*(T - T_a) + u(t, T)."""
    u = u_func(t, T[0])
    return [-k * (T[0] - T_a) + u]


def simulate_ode(u_func, T0=T_INITIAL, t_end=T_END, k=K_COOL, T_a=T_AMBIENT,
                 dt=DT, t_eval=None):
    """
    Simulate the ODE model with a given control function.

    Parameters
    ----------
    u_func : callable(t, T) -> float
        Control input function. Takes time and temperature, returns heating rate.
    T0 : float
        Initial temperature.
    t_end : float
        Simulation end time (minutes).
    k : float
        Cooling constant.
    T_a : float
        Ambient temperature.
    dt : float
        Maximum time step.
    t_eval : array-like, optional
        Times at which to store the solution.

    Returns
    -------
    t : ndarray
        Time array.
    T : ndarray
        Temperature array.
    """
    if t_eval is None:
        t_eval = np.arange(0, t_end + dt, dt)

    sol = solve_ivp(
        lambda t, T: ode_rhs(t, T, k, T_a, u_func),
        [0, t_end],
        [T0],
        t_eval=t_eval,
        max_step=dt,
        method='RK45'
    )

    return sol.t, sol.y[0]


def simulate_ode_switching(controller, T0=T_INITIAL, t_end=T_END, k=K_COOL,
                           T_a=T_AMBIENT, U_max=U_MAX, dt=DT,
                           max_switches=2000):
    """
    Simulate ODE with a switching controller using event detection.

    This is designed for controllers like Bang-Bang that have discrete
    switching events. It integrates piecewise between switch points.

    Parameters
    ----------
    controller : object
        Must have methods:
        - get_u(t, T) -> float : current control input
        - get_switch_event(t, T) -> float : event function (zero-crossing triggers switch)
        - get_switch_direction() -> int : +1, -1, or 0
        - switch(t, T) : update internal state at switch
        - is_on() -> bool : current heater state
    T0 : float
        Initial temperature.
    t_end : float
        Simulation end time.
    k : float
        Cooling constant.
    T_a : float
        Ambient temperature.
    U_max : float
        Maximum heating rate.
    dt : float
        Maximum time step.
    max_switches : int
        Safety limit on number of switches.

    Returns
    -------
    t : ndarray
        Time array.
    T : ndarray
        Temperature array.
    heater : ndarray
        Heater state array (0 or 1).
    """
    t_all = [0.0]
    T_all = [T0]
    heater_all = [1 if controller.is_on() else 0]

    t_current = 0.0
    T_current = T0
    n_switches = 0

    while t_current < t_end and n_switches < max_switches:
        def rhs(t, T, _ctrl=controller, _k=k, _Ta=T_a):
            u = _ctrl.get_u(t, T[0])
            return [-_k * (T[0] - _Ta) + u]

        direction = controller.get_switch_direction()

        def event_func(t, T, _ctrl=controller):
            return _ctrl.get_switch_event(t, T[0])
        event_func.terminal = True
        event_func.direction = direction

        sol = solve_ivp(
            rhs,
            [t_current, t_end],
            [T_current],
            events=[event_func],
            max_step=dt,
            dense_output=True
        )

        t_all.extend(sol.t[1:].tolist())
        T_all.extend(sol.y[0, 1:].tolist())
        heater_all.extend([int(controller.is_on())] * (len(sol.t) - 1))

        t_current = sol.t[-1]
        T_current = sol.y[0, -1]

        if sol.t_events[0].size > 0:
            controller.switch(t_current, T_current)
            n_switches += 1

    return np.array(t_all), np.array(T_all), np.array(heater_all)


def steady_state_temperature(k=K_COOL, T_a=T_AMBIENT, U_max=U_MAX):
    """Steady state when heater is always ON: T_ss = T_a + U_max / k."""
    return T_a + U_max / k


def oscillation_period_estimate(k=K_COOL, T_a=T_AMBIENT, T_set=20.0,
                                U_max=U_MAX, delta=0.5):
    """
    Estimate oscillation period for Bang-Bang with hysteresis.

    t_heat = (1/k) * ln((T_ss - T_low) / (T_ss - T_high))
    t_cool = (1/k) * ln((T_high - T_a) / (T_low - T_a))
    """
    T_low = T_set - delta
    T_high = T_set + delta
    T_ss = T_a + U_max / k

    if T_ss <= T_high:
        return float('inf')

    t_heat = (1.0 / k) * np.log((T_ss - T_low) / (T_ss - T_high))
    t_cool = (1.0 / k) * np.log((T_high - T_a) / (T_low - T_a))

    return t_heat + t_cool
