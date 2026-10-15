# active_grid/config.py

# --- Grid Physical Limits ---
V_NOM_KV = 0.4
V_MIN_PU = 0.95
V_MAX_PU = 1.05
BUS_COUNT = 15

# --- Simulation Settings ---
N_SAMPLES_TRAIN = 3000
N_SAMPLES_TEST = 500
SENSOR_NOISE_STD = 0.005

# --- DER Locations ---
DER_BUSES = [5, 8, 14] 

# --- Control Limits ---
# Kept at 50kW as per your request.
MAX_P_MW = 0.05   
MAX_Q_MVAR = 0.05