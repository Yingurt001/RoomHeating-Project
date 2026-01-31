"""
Unified evaluation metrics for comparing control strategies.

All metrics take time series data and return scalar values,
enabling consistent comparison across strategies and models.
"""

import numpy as np


def energy_consumption(t, u):
    """
    Total energy consumption: E = integral u(t) dt.

    Parameters
    ----------
    t : ndarray
        Time array.
    u : ndarray
        Control input array (heater power at each time step).

    Returns
    -------
    E : float
        Total energy consumed.
    """
    return np.trapz(u, t)


def temperature_rmse(t, T, T_set):
    """
    Root mean square error of temperature from setpoint.

    RMSE = sqrt(1/T_f * integral (T(t) - T_set)^2 dt)

    Parameters
    ----------
    t : ndarray
        Time array.
    T : ndarray
        Temperature array.
    T_set : float
        Desired temperature.

    Returns
    -------
    rmse : float
    """
    duration = t[-1] - t[0]
    if duration <= 0:
        return 0.0
    integrand = (T - T_set) ** 2
    return np.sqrt(np.trapz(integrand, t) / duration)


def max_overshoot(T, T_set):
    """
    Maximum overshoot above setpoint.

    Parameters
    ----------
    T : ndarray
        Temperature array.
    T_set : float
        Desired temperature.

    Returns
    -------
    overshoot : float
        Maximum T - T_set (0 if never exceeded).
    """
    return max(0.0, np.max(T) - T_set)


def settling_time(t, T, T_set, band=0.5):
    """
    Time at which T first enters and stays within T_set +/- band.

    Parameters
    ----------
    t : ndarray
        Time array.
    T : ndarray
        Temperature array.
    T_set : float
        Desired temperature.
    band : float
        Half-width of acceptable band (default 0.5 deg C).

    Returns
    -------
    t_settle : float
        Settling time, or np.inf if never settles.
    """
    within_band = np.abs(T - T_set) <= band

    # Find last time outside band
    outside_indices = np.where(~within_band)[0]
    if len(outside_indices) == 0:
        return t[0]  # always within band

    last_outside = outside_indices[-1]
    if last_outside >= len(t) - 1:
        return np.inf  # never settles

    return t[last_outside + 1]


def switching_count(heater):
    """
    Count the number of heater state switches (ON->OFF or OFF->ON).

    Parameters
    ----------
    heater : ndarray
        Heater state array (binary 0/1).

    Returns
    -------
    n_switches : int
    """
    return int(np.sum(np.abs(np.diff(heater)) > 0.5))


def unified_cost(t, T, u, T_set, Q=1.0, R=0.01):
    """
    Unified cost functional: J = integral [Q*(T-T_set)^2 + R*u^2] dt.

    Parameters
    ----------
    t : ndarray
        Time array.
    T : ndarray
        Temperature array.
    u : ndarray
        Control input array.
    T_set : float
        Desired temperature.
    Q : float
        State cost weight.
    R : float
        Control cost weight.

    Returns
    -------
    J : float
        Total cost.
    """
    integrand = Q * (T - T_set)**2 + R * u**2
    return np.trapz(integrand, t)


def compute_all_metrics(t, T, u, T_set, heater=None, Q=1.0, R=0.01):
    """
    Compute all metrics in one call.

    Parameters
    ----------
    t : ndarray
        Time array.
    T : ndarray
        Temperature array.
    u : ndarray
        Control input (continuous).
    T_set : float
        Desired temperature.
    heater : ndarray, optional
        Binary heater state (for switching count).
        If None, switches are estimated from u.
    Q, R : float
        Cost weights.

    Returns
    -------
    metrics : dict
        Dictionary of all computed metrics.
    """
    if heater is None:
        heater = (u > 0.01 * np.max(u)).astype(float)

    return {
        'energy': energy_consumption(t, u),
        'rmse': temperature_rmse(t, T, T_set),
        'max_overshoot': max_overshoot(T, T_set),
        'settling_time': settling_time(t, T, T_set),
        'switching_count': switching_count(heater),
        'unified_cost': unified_cost(t, T, u, T_set, Q, R),
    }
