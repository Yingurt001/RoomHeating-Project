"""
Scenario configurations for the physical scenarios.

Each make_*() returns a dict (or list of dicts) with keys:
    name, Lx, Ly, wall_h, heater_pos, thermostat_pos,
    [nx, ny, domain_mask, h_updater, thermostat_agg]  (optional)

S1–S6 : original 6 scenarios (single heater, single thermostat)
S7    : multi-heater comparison (distributed heating)
S8    : multi-thermostat comparison (spatial sensing)
"""

import numpy as np

# ── defaults (mirror Code/utils/parameters.py) ──
H_WALL = 0.5
NX = 51
NY = 41


def make_baseline():
    """S1: 5×5 square room, south wall Robin (exterior), rest Neumann (insulated)."""
    wall_h = {
        'south': np.full(NX, 0.0),
        'north': np.full(NX, H_WALL),
        'west':  np.full(NY, 0.0),
        'east':  np.full(NY, 0.0),
    }
    return dict(name="S1_Baseline", Lx=5, Ly=5, nx=NX, ny=NY,
                wall_h=wall_h,
                heater_pos=(2.5, 0.5), thermostat_pos=(2.5, 2.5))


def make_window():
    """S2: south wall with a 2 m single-glazed window (h=2.5) at centre."""
    h_south = np.full(NX, H_WALL)
    # h_east = np.full(NY, H_WALL)
    x = np.linspace(0, 5, NX)
    h_south[(x >= 0.5) & (x <= 3.5)] = 2.5 # window area
    # h_east[(y >= 1.0) & (y <= 3.0)] = 2.5
    wall_h = {
        'south': h_south,
        'north': np.full(NX, 0.0),
        'west':  np.full(NY, 0.0),
        'east':  np.full(NY, 0.0),
    }
    return dict(name="S2_Window", Lx=5, Ly=5, nx=NX, ny=NY,
                wall_h=wall_h,
                heater_pos=(2.5, 0.5), thermostat_pos=(2.5, 2.5))


def make_window_east_south():
    """S2: south wall with a 2 m single-glazed window (h=2.5) at centre."""
    h_south = np.full(NX, H_WALL)
    h_east = np.full(NY, H_WALL)
    x = np.linspace(0, 5, NX)
    y = np.linspace(0, 5, NY)
    h_south[(x >= 0.5) & (x <= 3.5)] = 2.5 # window area
    h_east[(y >= 1.0) & (y <= 3.0)] = 2.5
    wall_h = {
        'south': h_south,
        'north': np.full(NX, 0.0),
        'west':  np.full(NY, 0.0),
        'east':  h_east,
    }
    return dict(name="S2_Window", Lx=5, Ly=5, nx=NX, ny=NY,
                wall_h=wall_h,
                heater_pos=(2.5, 0.5), thermostat_pos=(2.5, 2.5))


def make_window_compare():
    """S3: three window variants — small / large / double-glazed."""
    configs = []
    x = np.linspace(0, 5, NX)
    for label, width, h_win in [
        ("Small_1m",    1.0, 2.5),
        ("Large_3m",    3.0, 2.5),
        ("DoubleGlaze", 2.0, 1.0),
    ]:
        h_south = np.full(NX, H_WALL)
        h_south[(x >= 2.5 - width/2) & (x <= 2.5 + width/2)] = h_win
        wall_h = {
            'south': h_south,
            'north': np.full(NX, 0.0),
            'west':  np.full(NY, 0.0),
            'east':  np.full(NY, 0.0),
        }
        configs.append(dict(
            name=f"S3_{label}", Lx=5, Ly=5, nx=NX, ny=NY,
            wall_h=wall_h,
            heater_pos=(2.5, 0.5), thermostat_pos=(2.5, 2.5)))
    return configs


def make_door_opening(t_open=30.0, t_close=40.0):
    """S4: west wall door (y ∈ [1, 2.5]) opens at t_open, closes at t_close."""
    def h_updater(t, model):
        h_west = np.full(model.ny, 0.0)
        if t_open <= t <= t_close:
            y = np.linspace(0, model.Ly, model.ny)
            h_west[(y >= 1.0) & (y <= 2.5)] = 10.0
        model.h_west = h_west

    wall_h = {
        'south': np.full(NX, H_WALL),
        'north': np.full(NX, 0.0),
        'west':  np.full(NY, 0.0),
        'east':  np.full(NY, 0.0),
    }
    return dict(name="S4_Door", Lx=5, Ly=5, nx=NX, ny=NY,
                wall_h=wall_h,
                heater_pos=(2.5, 0.5), thermostat_pos=(2.5, 2.5),
                h_updater=h_updater)


def make_narrow_room():
    """S5: narrow room 7.5×2.5 m (aspect ratio 3:1, same area ~18.75 m²)."""
    nx, ny = 76, 26   # keep grid density ≈ 0.1 m
    wall_h = {
        'south': np.full(nx, H_WALL),
        'north': np.full(nx, 0.0),
        'west':  np.full(ny, 0.0),
        'east':  np.full(ny, 0.0),
    }
    return dict(name="S5_Narrow", Lx=7.5, Ly=2.5, nx=nx, ny=ny,
                wall_h=wall_h,
                heater_pos=(3.75, 0.5), thermostat_pos=(3.75, 1.25))


def make_L_shape():
    """S6: L-shaped room — 5×5 minus top-right quadrant [2.5,5]×[2.5,5]."""
    mask = np.ones((NX, NY), dtype=bool)
    x = np.linspace(0, 5, NX)
    y = np.linspace(0, 5, NY)
    X, Y = np.meshgrid(x, y, indexing='ij')
    mask[(X >= 1.5) & (Y >= 1.5)] = False

    wall_h = {
        'south': np.full(NX, H_WALL),
        'north': np.full(NX, 0.0),
        'west':  np.full(NY, 0.0),
        'east':  np.full(NY, 0.0),
    }
    return dict(name="S6_L_Shape", Lx=5, Ly=5, nx=NX, ny=NY,
                wall_h=wall_h, domain_mask=mask,
                heater_pos=(1.25, 0.5), thermostat_pos=(1.25, 2.0))


# =====================================================================
#  S7: Multi-heater comparison
# =====================================================================

def _baseline_wall_h():
    """South wall Robin, rest Neumann (shared by S7/S8)."""
    return {
        'south': np.full(NX, H_WALL),
        'north': np.full(NX, 0.0),
        'west':  np.full(NY, 0.0),
        'east':  np.full(NY, 0.0),
    }


def make_multi_heater():
    """S7: multi-heater comparison — 1 / 2 / 3 heaters (same total power)."""
    configs = [
        dict(name="S7_1Heater",
             heater_pos=(2.5, 0.5),
             thermostat_pos=(2.5, 2.5)),
        dict(name="S7_2Heater_SouthWall",
             heater_pos=[(1.25, 0.5), (3.75, 0.5)],
             thermostat_pos=(2.5, 2.5)),
        dict(name="S7_3Heater_Distributed",
             heater_pos=[(1.25, 0.5), (2.5, 2.5), (3.75, 0.5)],
             thermostat_pos=(2.5, 2.5)),
    ]
    return [dict(Lx=5, Ly=5, nx=NX, ny=NY, wall_h=_baseline_wall_h(), **c)
            for c in configs]


# =====================================================================
#  S8: Multi-thermostat comparison
# =====================================================================

def make_multi_thermostat():
    """S8: multi-thermostat comparison — 1 / 3-mean / 3-min sensors."""
    configs = [
        dict(name="S8_1Therm",
             thermostat_pos=(2.5, 2.5)),
        dict(name="S8_3Therm_Mean",
             thermostat_pos=[(1.25, 1.25), (2.5, 2.5), (3.75, 3.75)],
             thermostat_agg='mean'),
        dict(name="S8_3Therm_Min",
             thermostat_pos=[(1.25, 1.25), (2.5, 2.5), (3.75, 3.75)],
             thermostat_agg='min'),
    ]
    return [dict(Lx=5, Ly=5, nx=NX, ny=NY, wall_h=_baseline_wall_h(),
                 heater_pos=(2.5, 0.5), **c)
            for c in configs]
