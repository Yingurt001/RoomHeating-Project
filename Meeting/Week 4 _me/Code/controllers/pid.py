"""
PID Controller for room heating.

    u(t) = Kp * e(t) + Ki * integral(e) + Kd * de/dt

where e(t) = T_set - T(t) is the tracking error.

The output is clamped to [0, U_max] (heater cannot cool or exceed max power).

References:
- Astrom KJ, Murray RM. Feedback Systems: An Introduction for Scientists
  and Engineers. 2nd ed. Princeton University Press, 2021.
  Chapter 11: PID Control.
  Freely available: https://fbswiki.org/
- Blasco C, Monreal J, Benitez I, Lluna A. Modelling and PID control of
  HVAC system according to energy efficiency and comfort criteria.
  Springer, 2012. DOI: 10.1007/978-3-642-27509-8_31
"""

import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.parameters import T_SET, U_MAX, DT


class PIDController:
    """
    Discrete PID controller with anti-windup clamping.

    Parameters
    ----------
    Kp : float
        Proportional gain.
    Ki : float
        Integral gain.
    Kd : float
        Derivative gain.
    T_set : float
        Temperature setpoint.
    U_max : float
        Maximum control output (heater power).
    dt : float
        Time step for discrete integration.
    """

    def __init__(self, Kp=2.0, Ki=0.1, Kd=0.5, T_set=T_SET, U_max=U_MAX,
                 dt=DT):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.T_set = T_set
        self.U_max = U_max
        self.dt = dt

        # Internal state
        self._integral = 0.0
        self._prev_error = None

    def get_u(self, t, T):
        """
        Compute PID control output.

        Parameters
        ----------
        t : float
            Current time.
        T : float
            Current temperature.

        Returns
        -------
        u : float
            Control input, clamped to [0, U_max].
        """
        error = self.T_set - T

        # Proportional term
        P = self.Kp * error

        # Integral term (trapezoidal rule with anti-windup)
        self._integral += error * self.dt
        I = self.Ki * self._integral

        # Derivative term
        if self._prev_error is None:
            D = 0.0
        else:
            D = self.Kd * (error - self._prev_error) / self.dt
        self._prev_error = error

        # Total output, clamped
        u = P + I + D
        u_clamped = np.clip(u, 0.0, self.U_max)

        # Anti-windup: if output is saturated, stop integrating
        if u != u_clamped:
            self._integral -= error * self.dt

        return u_clamped

    def reset(self):
        """Reset controller internal state."""
        self._integral = 0.0
        self._prev_error = None


def ziegler_nichols_tuning(k_cool, U_max=U_MAX):
    """
    Estimate PID gains using Ziegler-Nichols-inspired heuristics
    for the first-order room heating system.

    The room ODE dT/dt = -k*(T - T_a) + u has time constant tau = 1/k.
    For a first-order system, classical Z-N open-loop tuning gives:
        Kp = 1.2 * tau / (K * L)
    where K is the static gain and L is the apparent dead time.

    For our system with no dead time, we use a simplified approach based
    on the system time constant.

    Parameters
    ----------
    k_cool : float
        Cooling constant (1/min).
    U_max : float
        Maximum heater power.

    Returns
    -------
    Kp, Ki, Kd : float
        Suggested PID gains.
    """
    tau = 1.0 / k_cool  # system time constant

    # Heuristic tuning for first-order system
    Kp = 0.6 * U_max / 15.0  # scale with heater power
    Ki = Kp / (0.5 * tau)
    Kd = Kp * 0.125 * tau

    return Kp, Ki, Kd
