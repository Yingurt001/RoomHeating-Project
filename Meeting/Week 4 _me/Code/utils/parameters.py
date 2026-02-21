"""
Shared physical parameters for the room heating project.

All units are SI unless otherwise noted.
"""

# --- Room & Environment ---
T_AMBIENT = 5.0        # Outside temperature (deg C), typical UK winter
T_INITIAL = 10.0       # Initial room temperature (deg C)
T_SET = 20.0           # Thermostat set-point (deg C)

# --- Room geometry (for PDE models) ---
ROOM_LENGTH = 5.0      # Room length (m)
ROOM_WIDTH = 4.0       # Room width (m), for 2D model

# --- Heater ---
U_MAX = 15.0           # Maximum heating rate (deg C/min equivalent, normalised)
                       # In ODE model: dT/dt contribution when heater is ON

# --- Cooling ---
K_COOL = 0.1           # Newton's cooling constant (1/min)
                       # Higher k = room loses heat faster (poor insulation)

# --- Thermal diffusivity (for PDE models) ---
ALPHA = 0.01           # Thermal diffusivity (m^2/min), for spatial models

# --- Heat transfer coefficient at walls (for Robin BC) ---
H_WALL = 0.5           # Wall heat transfer coefficient (1/m)

# --- Hysteresis band (optional, to avoid Zeno-like chattering) ---
HYSTERESIS_BAND = 0.5  # Half-width of dead band (deg C)
                       # Heater ON below T_set - delta, OFF above T_set + delta

# --- Simulation ---
T_END = 120.0          # Total simulation time (minutes)
DT = 0.01              # Time step (minutes)

# --- PDE discretisation ---
NX = 51                # Number of spatial grid points in x
NY = 41                # Number of spatial grid points in y (for 2D)
