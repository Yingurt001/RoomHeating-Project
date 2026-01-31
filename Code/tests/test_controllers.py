"""Tests for control strategies."""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from controllers.bang_bang import BangBangController, BangBangNoHysteresis
from controllers.pid import PIDController, ziegler_nichols_tuning
from controllers.lqr import LQRController, pareto_scan
from controllers.pontryagin import pontryagin_bvp, optimal_control_from_costate
from utils.parameters import T_SET, U_MAX, K_COOL, T_AMBIENT
from utils.metrics import compute_all_metrics


# ===================== Bang-Bang Tests =====================

class TestBangBang:
    def test_on_below_setpoint(self):
        ctrl = BangBangController(T_set=20.0, U_max=15.0, delta=0.5,
                                  initial_on=True)
        assert ctrl.get_u(0, 15.0) == 15.0

    def test_off_above_setpoint(self):
        ctrl = BangBangController(T_set=20.0, U_max=15.0, delta=0.5,
                                  initial_on=False)
        assert ctrl.get_u(0, 25.0) == 0.0

    def test_hysteresis_event(self):
        ctrl = BangBangController(T_set=20.0, delta=0.5, initial_on=True)
        # When ON, event triggers at T = T_set + delta = 20.5
        assert ctrl.get_switch_event(0, 20.5) == pytest.approx(0.0, abs=1e-10)

    def test_switch_toggles(self):
        ctrl = BangBangController(initial_on=True)
        assert ctrl.is_on() is True
        ctrl.switch(0, 20.0)
        assert ctrl.is_on() is False
        ctrl.switch(0, 19.0)
        assert ctrl.is_on() is True

    def test_no_hysteresis_variant(self):
        ctrl = BangBangNoHysteresis()
        assert ctrl.delta == 0.0

    def test_reset(self):
        ctrl = BangBangController(initial_on=True)
        ctrl.switch(0, 20.0)
        assert ctrl.is_on() is False
        ctrl.reset()
        assert ctrl.is_on() is True


# ===================== PID Tests =====================

class TestPID:
    def test_positive_error_gives_positive_output(self):
        """When T < T_set, error is positive, output should be > 0."""
        ctrl = PIDController(Kp=2.0, Ki=0.0, Kd=0.0, T_set=20.0)
        u = ctrl.get_u(0, 15.0)
        assert u > 0

    def test_zero_error_zero_output(self):
        """At setpoint with no integral/derivative, output should be near 0."""
        ctrl = PIDController(Kp=2.0, Ki=0.0, Kd=0.0, T_set=20.0)
        u = ctrl.get_u(0, 20.0)
        assert abs(u) < 0.01

    def test_output_clamped_above(self):
        """Output should not exceed U_max."""
        ctrl = PIDController(Kp=100.0, Ki=0.0, Kd=0.0, T_set=20.0, U_max=15.0)
        u = ctrl.get_u(0, 0.0)
        assert u == 15.0

    def test_output_clamped_below(self):
        """Output should not go below 0."""
        ctrl = PIDController(Kp=2.0, Ki=0.0, Kd=0.0, T_set=20.0)
        u = ctrl.get_u(0, 30.0)
        assert u == 0.0

    def test_integral_accumulates(self):
        """Integral term should accumulate over time."""
        ctrl = PIDController(Kp=0.0, Ki=1.0, Kd=0.0, T_set=20.0, dt=1.0)
        ctrl.get_u(0, 19.0)  # error = 1.0
        u = ctrl.get_u(1, 19.0)  # error = 1.0, integral = 2.0
        assert u > 0

    def test_ziegler_nichols_returns_positive(self):
        Kp, Ki, Kd = ziegler_nichols_tuning(0.1)
        assert Kp > 0
        assert Ki > 0
        assert Kd > 0

    def test_reset(self):
        ctrl = PIDController(Kp=1.0, Ki=1.0, Kd=1.0)
        ctrl.get_u(0, 15.0)
        ctrl.reset()
        assert ctrl._integral == 0.0
        assert ctrl._prev_error is None


# ===================== LQR Tests =====================

class TestLQR:
    def test_gain_positive(self):
        """LQR gain should be positive for the heating problem."""
        ctrl = LQRController(Q=1.0, R=0.01)
        assert ctrl.K > 0

    def test_at_setpoint_gives_feedforward(self):
        """At T=T_set, output should be the feedforward term u_ss."""
        ctrl = LQRController(Q=1.0, R=0.01)
        u = ctrl.get_u(0, T_SET)
        expected = K_COOL * (T_SET - T_AMBIENT)
        assert abs(u - expected) < 0.5

    def test_below_setpoint_heats_more(self):
        """Below setpoint, should heat more than feedforward."""
        ctrl = LQRController(Q=1.0, R=0.01)
        u_at = ctrl.get_u(0, T_SET)
        u_below = ctrl.get_u(0, T_SET - 5.0)
        assert u_below > u_at

    def test_output_clamped(self):
        """Output should stay within [0, U_max]."""
        ctrl = LQRController(Q=100.0, R=0.001)
        u = ctrl.get_u(0, 0.0)  # very cold, high demand
        assert 0 <= u <= U_MAX

    def test_higher_Q_gives_higher_gain(self):
        """Higher Q (more weight on tracking) should give higher gain."""
        ctrl_low = LQRController(Q=0.1, R=0.01)
        ctrl_high = LQRController(Q=10.0, R=0.01)
        assert ctrl_high.K > ctrl_low.K

    def test_pareto_scan(self):
        """Pareto scan should return correct number of controllers."""
        controllers, params = pareto_scan([0.1, 1.0], [0.01, 0.1])
        assert len(controllers) == 4
        assert len(params) == 4


# ===================== Pontryagin Tests =====================

class TestPontryagin:
    def test_optimal_control_clamping(self):
        """Optimal control should be clamped to [0, U_max]."""
        u = optimal_control_from_costate(-100.0, R=0.01, U_max=15.0)
        assert u == 15.0

        u = optimal_control_from_costate(100.0, R=0.01, U_max=15.0)
        assert u == 0.0

    def test_bvp_converges(self):
        """BVP solver should converge for reasonable parameters."""
        t, T, u, lam, sol = pontryagin_bvp(
            Q=1.0, R=1.0, T0=15.0, t_end=30.0, n_nodes=200
        )
        assert sol.success, f"BVP did not converge: {sol.message}"

    def test_initial_condition_satisfied(self):
        """T(0) should equal T0."""
        t, T, u, lam, sol = pontryagin_bvp(T0=10.0, t_end=60.0, n_nodes=200)
        assert abs(T[0] - 10.0) < 0.01

    def test_transversality_condition(self):
        """lambda(t_end) should be approximately 0."""
        t, T, u, lam, sol = pontryagin_bvp(t_end=60.0, n_nodes=200)
        assert abs(lam[-1]) < 0.1, f"lambda(t_end) = {lam[-1]}, expected ~0"

    def test_temperature_approaches_setpoint(self):
        """With moderate R, T should approach T_set."""
        t, T, u, lam, sol = pontryagin_bvp(
            Q=1.0, R=0.1, T0=10.0, t_end=60.0, n_nodes=300
        )
        # Final temperature should be closer to T_set than initial
        assert abs(T[-1] - T_SET) < abs(T[0] - T_SET), \
            f"T should approach T_set: T(0)={T[0]:.1f}, T(end)={T[-1]:.1f}"


# ===================== Metrics Tests =====================

class TestMetrics:
    def test_compute_all_metrics(self):
        t = np.linspace(0, 100, 1000)
        T = np.full_like(t, T_SET)
        u = np.full_like(t, 1.5)
        m = compute_all_metrics(t, T, u, T_SET)
        assert m['rmse'] == pytest.approx(0.0, abs=1e-10)
        assert m['max_overshoot'] == pytest.approx(0.0, abs=1e-10)
        assert m['energy'] > 0
