"""The original ADG_datadriven plant, with a neutral-preserving OLTC retrofit."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import os
import numpy as np
import pandapower as pp
from active_grid import config, utils
from active_grid.envs import DistributionGridEnv
from active_grid.randomness import reference_random

REFERENCE_REPOSITORY = 'anuragmohapatra90/ADG_datadriven'
REFERENCE_COMMIT = '55a6fa8e80aa810912ebbfb182c6db11e165fac0'


@dataclass(frozen=True)
class Config:
    seed: int = 42
    v_min: float = config.V_MIN_PU
    v_max: float = config.V_MAX_PU


class GridLab:
    def __init__(self, cfg=Config()):
        self.cfg = cfg
        with reference_random(cfg.seed):
            self.env = DistributionGridEnv()
        self.net = self.env.net
        # Same NR equations, without optional JIT warnings on every training step.
        pp.set_user_pf_options(self.net, numba=False)
        self.tid = int(self.net.trafo.index[0])
        self.lv_bus = 1
        self.bus_ids = np.arange(1, config.BUS_COUNT)
        self.nv = len(self.bus_ids)
        self.tap0, self.tap_min, self.tap_max = 0, -4, 4
        self.tap_step_percent = 1.25
        # A secondary-reference ratio trim: exactly 0.95–1.05 at ideal no load.
        # Positive taps raise LV voltage; neutral preserves the reference AC solution.
        for key, value in dict(tap_side='lv', tap_neutral=self.tap0, tap_pos=self.tap0, tap_min=self.tap_min,
                tap_max=self.tap_max, tap_step_percent=self.tap_step_percent, tap_step_degree=0, oltc=True,
                tap_changer_type='Ratio').items():
            self.net.trafo.loc[self.tid, key] = value
        self.voltage_controller = None
        self.training_report = None

    def new_net(self):
        return deepcopy(self.net)

    def generate_scenario(self, seed):
        # Exact reference generator. Its first draw is the synthetic time of day.
        with reference_random(seed):
            state = np.random.get_state()
            hour = float(np.random.uniform(0, 24, 1)[0])
            np.random.set_state(state)
            p, q, pv = utils.generate_scenarios(1, config.BUS_COUNT)
        # The reference plant has no load/PV elements at buses 0 and 1.
        for values in (p, q, pv):
            values[:, :2] = 0.
        return {'load_p': p[0], 'load_q': q[0], 'pv': pv[0], 'hour': hour,
                'seed': int(seed)}

    def set_snapshot(self, net, scenario, load_scale=1., pv_scale=1.):
        p = scenario['load_p'] * float(load_scale) - scenario['pv'] * float(pv_scale)
        q = scenario['load_q'] * float(load_scale)
        buses = net.load.bus.to_numpy(int)
        net.load.loc[:, 'p_mw'] = p[buses]
        net.load.loc[:, 'q_mvar'] = q[buses]
        net.sgen.loc[:, ['p_mw', 'q_mvar']] = 0.
        return p, q

    def solve(self, net):
        tap=float(net.trafo.loc[self.tid,'tap_pos'])
        if not np.isfinite(tap) or not self.tap_min<=tap<=self.tap_max or not tap.is_integer():
            raise ValueError(f'OLTC tap must be an integer in [{self.tap_min}, {self.tap_max}].')
        pp.runpp(net, algorithm='nr', numba=False, max_iteration=10)
        v = net.res_bus.loc[self.bus_ids, 'vm_pu'].to_numpy(float)
        if not np.isfinite(v).all():
            raise ValueError('Nonfinite AC voltages; this snapshot is invalid.')
        return v

    def violation(self, y):
        v = np.asarray(y)[:self.nv]
        return max(float(np.max(self.cfg.v_min-v)), float(np.max(v-self.cfg.v_max)), 0.)

    def grid_fingerprint(self):
        line = self.net.line[['from_bus', 'to_bus', 'length_km', 'r_ohm_per_km', 'x_ohm_per_km']]
        return hashlib.sha256(line.to_json().encode()).hexdigest()

    def metadata(self):
        return {'reference_repository': REFERENCE_REPOSITORY, 'reference_commit': REFERENCE_COMMIT,
                'source_repository': os.getenv('GRID_LAB_SOURCE_REPOSITORY', ''),
                'source_commit': os.getenv('GRID_LAB_COMMIT', ''), 'grid_seed': self.cfg.seed,
                'grid_fingerprint': self.grid_fingerprint(), 'der_buses': list(config.DER_BUSES)}
