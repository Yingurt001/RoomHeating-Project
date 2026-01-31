"""
Bang-Bang Controller (with optional hysteresis).

References:
- Goebel R, Sanfelice RG, Teel AR. Hybrid Dynamical Systems: Modeling,
  Stability, and Robustness. Princeton University Press, 2012.
  (Thermostat as canonical hybrid system example)
- Zhang J, Johansson KH, Lygeros J, Sastry SS. Zeno hybrid systems.
  Int. J. Robust Nonlinear Control. 2001;11(5):435-451.
  (Zeno effect analysis for bang-bang switching)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.parameters import T_SET, U_MAX, HYSTERESIS_BAND


class BangBangController:
    """
    Bang-Bang thermostat with optional hysteresis band.

    Without hysteresis (delta=0):
        u = U_max if T < T_set, else 0

    With hysteresis (delta > 0):
        u switches ON  when T < T_set - delta
        u switches OFF when T > T_set + delta
    """

    def __init__(self, T_set=T_SET, U_max=U_MAX, delta=HYSTERESIS_BAND,
                 initial_on=True):
        self.T_set = T_set
        self.U_max = U_max
        self.delta = delta
        self._on = initial_on

    def get_u(self, t, T):
        """Return current control input."""
        return self.U_max if self._on else 0.0

    def get_switch_event(self, t, T):
        """Event function: zero-crossing triggers a switch."""
        if self._on:
            return T - (self.T_set + self.delta)  # switch OFF when T > T_high
        else:
            return T - (self.T_set - self.delta)  # switch ON when T < T_low

    def get_switch_direction(self):
        """Direction of zero-crossing: +1 (rising) or -1 (falling)."""
        return 1 if self._on else -1

    def switch(self, t, T):
        """Toggle heater state."""
        self._on = not self._on

    def is_on(self):
        """Return current heater state."""
        return self._on

    def reset(self, initial_on=True):
        """Reset controller state."""
        self._on = initial_on


class BangBangNoHysteresis(BangBangController):
    """Bang-Bang with zero hysteresis (delta=0). Demonstrates Zeno effect."""

    def __init__(self, T_set=T_SET, U_max=U_MAX, initial_on=True):
        super().__init__(T_set=T_set, U_max=U_max, delta=0.0,
                         initial_on=initial_on)
