# active_grid/utils.py
import numpy as np

def get_standard_profiles(n_samples):
    """
    Generates synthetic 'Standard Load Profiles' and 'PV Profiles'.
    Instead of random noise, we create realistic daily shapes.
    """
    
    # 1. Create a Time Vector (0 to 24 hours) repeated for n_samples
    # We assume each sample is a snapshot taken at a random time of day
    time_of_day = np.random.uniform(0, 24, n_samples)
    
    # --- PV Profile (Bell Curve peaking at 13:00) ---
    # Formula: exp( - (t - mu)^2 / (2sigma^2) )
    pv_shape = np.exp(-((time_of_day - 13)**2) / (2 * 2.5**2))
    # Clip small values to 0 (night time)
    pv_shape[pv_shape < 0.01] = 0
    
    # --- Residential Load Profile (H0-like) ---
    # Two peaks: Morning (8am) and Evening (7pm)
    morning_peak = np.exp(-((time_of_day - 8)**2) / (2 * 1.5**2))
    evening_peak = np.exp(-((time_of_day - 19)**2) / (2 * 2.5**2))
    base_load = 0.2 # Always some fridge/standby load
    
    load_shape = 0.4*morning_peak + 0.8*evening_peak + base_load
    
    return pv_shape, load_shape

def generate_scenarios(n_samples, n_buses):
    """
    Generates Grid Inputs (P_load, Q_load, P_pv_potential)
    based on the standard profiles.
    """
    pv_profile, load_profile = get_standard_profiles(n_samples)
    
    # Initialize Arrays
    load_p = np.zeros((n_samples, n_buses))
    load_q = np.zeros((n_samples, n_buses))
    pv_gen = np.zeros((n_samples, n_buses))
    
    for i in range(n_samples):
        # Scale profiles with randomness for each bus (Heterogeneity)
        # Avg House Load: 4kW peak (0.004 MW)
        # Avg PV System: 5kW peak (0.005 MW)
        
        # Random scaling factors for each bus
        scale_load = np.random.uniform(0.002, 0.008, n_buses)
        scale_pv = np.random.uniform(0.000, 0.010, n_buses) # Some houses have no PV (0.0)
        
        # Apply Profile + Random Variation
        # We add random noise so the curve isn't perfect
        noise = np.random.normal(1.0, 0.1, n_buses)
        
        load_p[i, :] = load_profile[i] * scale_load * noise
        
        # Assume Power Factor 0.95 Lagging for loads
        load_q[i, :] = load_p[i, :] * 0.33 
        
        pv_gen[i, :] = pv_profile[i] * scale_pv * noise

    return load_p, load_q, pv_gen