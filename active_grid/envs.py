# active_grid/envs.py
import pandapower as pp
import pandapower.plotting as plot
import numpy as np
import pandas as pd
from . import config

class DistributionGridEnv:
    def __init__(self):
        self.net = self._build_branched_grid()
        self.mesh_closed = False

    def _build_branched_grid(self):
        net = pp.create_empty_network()
        
        # 1. Buses
        pp.create_bus(net, vn_kv=20.0, name="MV_Grid")
        pp.create_bus(net, vn_kv=config.V_NOM_KV, name="Substation")
        
        for i in range(2, config.BUS_COUNT):
            pp.create_bus(net, vn_kv=config.V_NOM_KV, name=f"Bus_{i}")

        # 2. Connection
        pp.create_ext_grid(net, bus=0)
        pp.create_transformer(net, hv_bus=0, lv_bus=1, std_type="0.4 MVA 20/0.4 kV")

        # 3. BALANCED GRID TOPOLOGY
        # 'NAYY 4x150' is too strong (no drop). 'NAYY 4x50' is too weak (collapse).
        # We use 'NAYY 4x120 SE' as the Goldilocks cable.
        cable_type = "NAYY 4x120 SE" 
        
        # Feeder A (Residential): 1 -> ... -> 8
        prev_bus = 1
        for i in range(2, 9): 
            # Reduced max length slightly to 400m to prevent collapse
            length = np.random.uniform(0.2, 0.4) 
            pp.create_line(net, from_bus=prev_bus, to_bus=i, length_km=length, std_type=cable_type)
            prev_bus = i
            
        # Feeder B (Commercial): 1 -> ... -> 14
        prev_bus = 1
        for i in range(9, 15): 
            length = np.random.uniform(0.3, 0.5)
            pp.create_line(net, from_bus=prev_bus, to_bus=i, length_km=length, std_type=cable_type)
            prev_bus = i

        # 4. Loads & PVs
        for i in range(2, config.BUS_COUNT):
            pp.create_load(net, bus=i, p_mw=0.0, q_mvar=0.0, name=f"Load_{i}")
            # Installed capacity 100 kVA (Moderate)
            pp.create_sgen(net, bus=i, p_mw=0.0, q_mvar=0.0, sn_mva=0.10, name=f"PV_{i}")

        return net

    def step(self, load_p, load_q, der_p_sets, der_q_sets):
        # 1. Update Loads
        for i in range(2, config.BUS_COUNT):
            load_idx = self.net.load[self.net.load.bus == i].index
            self.net.load.loc[load_idx, 'p_mw'] = load_p[i]
            self.net.load.loc[load_idx, 'q_mvar'] = load_q[i]

            # 2. Update DERs
            p_val = der_p_sets.get(i, 0.0) 
            q_val = der_q_sets.get(i, 0.0)
            
            sgen_idx = self.net.sgen[self.net.sgen.bus == i].index
            self.net.sgen.loc[sgen_idx, 'p_mw'] = p_val
            self.net.sgen.loc[sgen_idx, 'q_mvar'] = q_val

        # 3. Power Flow
        try:
            pp.runpp(self.net, algorithm='nr')
        except:
            return None 

        v_pu = self.net.res_bus.vm_pu.values.copy()
        noise = np.random.normal(0, config.SENSOR_NOISE_STD, size=len(v_pu))
        return v_pu + noise