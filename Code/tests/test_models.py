"""Tests for physical models (ODE, 1D PDE, 2D PDE)."""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.ode_model import (
    simulate_ode, simulate_ode_switching,
    steady_state_temperature, oscillation_period_estimate
)
from models.pde_1d_model import HeatEquation1D
from models.pde_2d_model import HeatEquation2D
from controllers.bang_bang import BangBangController
from utils.parameters import T_AMBIENT, T_INITIAL, T_SET, U_MAX, K_COOL


# ===================== ODE Model Tests =====================

class TestODEModel:
    def test_constant_heating_reaches_steady_state(self):
        """With constant u=U_max, T should approach T_a + U_max/k."""
        T_ss_expected = steady_state_temperature()

        def u_const(t, T):
            return U_MAX

        t, T = simulate_ode(u_const, T0=T_INITIAL, t_end=200.0)
        assert abs(T[-1] - T_ss_expected) < 0.5, \
            f"Expected T_ss ~ {T_ss_expected:.1f}, got {T[-1]:.1f}"

    def test_no_heating_cools_to_ambient(self):
        """With u=0, T should decay to T_ambient."""
        def u_zero(t, T):
            return 0.0

        t, T = simulate_ode(u_zero, T0=T_SET, t_end=200.0)
        assert abs(T[-1] - T_AMBIENT) < 0.5, \
            f"Expected T ~ {T_AMBIENT}, got {T[-1]:.1f}"

    def test_temperature_monotone_heating(self):
        """When heating from below, temperature should increase monotonically."""
        def u_const(t, T):
            return U_MAX

        t, T = simulate_ode(u_const, T0=T_INITIAL, t_end=50.0)
        # Check monotonically increasing (allowing small numerical noise)
        diffs = np.diff(T)
        assert np.all(diffs >= -1e-10), "Temperature should increase with constant heating"

    def test_bang_bang_switching(self):
        """Bang-Bang controller should produce switches."""
        ctrl = BangBangController(delta=0.5)
        t, T, heater = simulate_ode_switching(ctrl, t_end=60.0)
        n_switches = np.sum(np.abs(np.diff(heater)) > 0.5)
        assert n_switches > 5, f"Expected many switches, got {n_switches}"

    def test_bang_bang_temperature_in_range(self):
        """With hysteresis, temperature should stay near T_set after settling."""
        ctrl = BangBangController(delta=0.5)
        t, T, heater = simulate_ode_switching(ctrl, t_end=120.0)
        # After settling (last 50% of time), T should be within wider band
        T_late = T[len(T)//2:]
        assert np.all(T_late > T_SET - 2.0), \
            f"Temperature too low: min = {T_late.min():.2f}"
        assert np.all(T_late < T_SET + 2.0), \
            f"Temperature too high: max = {T_late.max():.2f}"

    def test_steady_state_formula(self):
        """Verify steady-state formula T_ss = T_a + U_max/k."""
        T_ss = steady_state_temperature(k=0.1, T_a=5.0, U_max=15.0)
        assert abs(T_ss - 155.0) < 0.01

    def test_oscillation_period_positive(self):
        """Period estimate should be positive and finite."""
        period = oscillation_period_estimate(delta=0.5)
        assert 0 < period < 100, f"Period = {period}, expected finite positive"


# ===================== 1D PDE Tests =====================

class TestPDE1D:
    def test_1d_uniform_initial_condition(self):
        """Starting uniform, with no heating, should cool uniformly."""
        model = HeatEquation1D(nx=21, T0=T_SET)

        def u_zero(t, T):
            return 0.0

        t, T_field, T_therm = model.simulate(u_zero, t_end=10.0, dt=0.1)
        # All temperatures should decrease
        assert T_field[:, -1].max() < T_SET

    def test_1d_heater_warms_nearby(self):
        """With heating, points near heater should be warmer."""
        model = HeatEquation1D(nx=21, heater_pos=0.5, thermostat_pos=2.5)

        def u_const(t, T):
            return U_MAX

        t, T_field, T_therm = model.simulate(u_const, t_end=20.0, dt=0.1)
        # Near heater (x~0.5) should be warmer than far end (x~4.5)
        T_near = T_field[2, -1]   # near x=0.5
        T_far = T_field[-2, -1]   # near x=4.5
        assert T_near > T_far, \
            f"Near heater T={T_near:.1f} should be > far T={T_far:.1f}"

    def test_1d_thermostat_reads_correct_position(self):
        """Thermostat should read temperature at its grid position."""
        model = HeatEquation1D(nx=51, thermostat_pos=2.5)
        T_field = np.linspace(10, 30, 51)
        T_read = model.get_thermostat_temperature(T_field)
        expected_idx = model.thermostat_idx
        assert abs(T_read - T_field[expected_idx]) < 1e-10


# ===================== 2D PDE Tests =====================

class TestPDE2D:
    def test_2d_initial_condition(self):
        """2D model should start at uniform T0."""
        model = HeatEquation2D(nx=11, ny=11, T0=T_INITIAL)
        assert model.T0.shape == (11, 11)
        assert np.allclose(model.T0, T_INITIAL)

    def test_2d_cooling_without_heating(self):
        """Without heating, room should cool toward ambient."""
        model = HeatEquation2D(nx=11, ny=11, T0=T_SET)

        def u_zero(t, T):
            return 0.0

        t, T_field, T_therm = model.simulate(u_zero, t_end=10.0, dt=0.2)
        # Average temperature should decrease
        T_avg_start = T_field[:, :, 0].mean()
        T_avg_end = T_field[:, :, -1].mean()
        assert T_avg_end < T_avg_start

    def test_2d_thermostat_position(self):
        """Thermostat reads correct grid position."""
        model = HeatEquation2D(nx=11, ny=11, thermostat_pos=(2.5, 2.0))
        T_field = np.random.rand(11, 11)
        T_read = model.get_thermostat_temperature(T_field)
        assert abs(T_read - T_field[model.therm_ix, model.therm_iy]) < 1e-10
