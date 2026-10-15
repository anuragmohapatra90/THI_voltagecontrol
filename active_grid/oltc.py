"""Best settled OLTC tap, evaluated by AC power flow at identical injections."""
from copy import deepcopy
import numpy as np
import pandapower as pp


def oltc_score(lab, y, tap):
    v = y[:lab.nv]
    return (lab.violation(y), float(np.mean((v-1.)**2)), abs(tap-lab.tap0))


def best_oltc(lab, baseline):
    net = deepcopy(baseline)
    candidates = []
    for tap in range(lab.tap_min, lab.tap_max+1):
        net.trafo.loc[lab.tid, 'tap_pos'] = tap
        try:
            y = lab.solve(net)
            candidates.append((oltc_score(lab, y, tap), tap))
        except (pp.LoadflowNotConverged, ValueError):
            pass
    if not candidates:
        raise ValueError('No OLTC tap produced a converged AC solution.')
    _, tap = min(candidates)
    net.trafo.loc[lab.tid, 'tap_pos'] = tap
    return net, lab.solve(net), tap
