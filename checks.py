"""Upstream identity and independent physical parity checks for the ADG plant."""
from pathlib import Path
import hashlib
import numpy as np
from pandas.testing import assert_frame_equal
from active_grid import config, utils
from active_grid.envs import DistributionGridEnv
from active_grid.randomness import reference_random
from grid_lab import GridLab

REFERENCE_BLOBS = {
    'active_grid/config.py': 'ec13bcc49282cc7cbbfd47a415b7228349a7002f',
    'active_grid/envs.py': 'b1c07316030cb627df5925230ff9bbb0cfff3b02',
    'active_grid/utils.py': 'aa131dfc3af920818f03f176b12fbaaa5a719feb',
    'active_grid/mlp_model.py': '7c8fdc3d7c02da3ce1a77adfcf4468853df48b7c',
}


def run_checks(lab=None):
    lab = lab or GridLab()
    root = Path(__file__).resolve().parent
    for path, expected in REFERENCE_BLOBS.items():
        content = (root/path).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest() == expected, path
    with reference_random(lab.cfg.seed):
        original = DistributionGridEnv()
    for name in ['bus', 'line', 'load', 'sgen', 'ext_grid']:
        assert_frame_equal(original.net[name], lab.net[name])
    assert len(lab.net.bus) == 15 and len(lab.net.line) == 13
    assert config.DER_BUSES == [5, 8, 14]
    assert np.all(lab.net.sgen.sn_mva == .10)
    scenario = lab.generate_scenario(6)
    with reference_random(6):
        p, q, pv = utils.generate_scenarios(1, config.BUS_COUNT)
        measured = original.step(p[0]-pv[0], q[0], {}, {})
    assert measured is not None
    assert np.array_equal(scenario['load_p'][2:], p[0, 2:])
    assert np.array_equal(scenario['load_q'][2:], q[0, 2:])
    assert np.array_equal(scenario['pv'][2:], pv[0, 2:])
    trial = lab.new_net()
    lab.set_snapshot(trial, scenario)
    y = lab.solve(trial)
    assert np.allclose(trial.res_bus.vm_pu, original.net.res_bus.vm_pu, atol=1e-10)
    trial.trafo.loc[lab.tid, 'tap_pos'] = 1
    moved = lab.solve(trial)
    assert np.max(np.abs(moved-y)) > .005
    assert GridLab(lab.cfg).grid_fingerprint() == lab.grid_fingerprint()
    return {'upstream_files': 'byte-identical', 'grid_profiles': 'reference parity passed',
            'neutral_AC': 'reference parity passed', 'OLTC_voltage_response': 'passed',
            'control_sites': config.DER_BUSES}
