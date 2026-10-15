"""Tap-aware voltage estimates and an OLTC policy with no online AC oracle."""
from copy import deepcopy
import numpy as np
import pandapower as pp
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from . import config, utils
from .mlp_model import VoltageEstimator
from .randomness import reference_random
from .controller import net_bus_pq


class TapVoltageEstimator:
    """Inputs are known net P/Q and tap; outputs are estimates, not measurements."""
    def __init__(self, model, sx, sy):
        self.model, self.sx, self.sy = model.eval(), sx, sy

    def predict_profiles(self, p, q, taps):
        taps = np.asarray(taps, dtype=float).reshape(-1)
        features = np.column_stack([np.tile(np.r_[p, q], (len(taps), 1)), taps])
        with torch.no_grad():
            y = self.model(torch.tensor(self.sx.transform(features), dtype=torch.float32)).numpy()
        return self.sy.inverse_transform(y)


def _campaign(lab, count, seed, name):
    net = lab.new_net()
    xs, ys = [], []
    with reference_random(seed):
        p, q, pv = utils.generate_scenarios(count, config.BUS_COUNT)
        loads = np.random.uniform(.5, 5., count)
        solar = np.random.uniform(0., 5., count)
        taps = np.random.randint(lab.tap_min, lab.tap_max+1, count)
        for i in range(count):
            bp, bq = lab.set_snapshot(net, {'load_p':p[i], 'load_q':q[i], 'pv':pv[i]}, loads[i], solar[i])
            # The original grid has no load or generation at buses 0 and 1.
            bp[:2] = 0.; bq[:2] = 0.
            net.trafo.loc[lab.tid, 'tap_pos'] = int(taps[i])
            try:
                lab.solve(net)
            except (pp.LoadflowNotConverged, ValueError):
                pass
            else:
                xs.append(np.r_[bp, bq, taps[i]])
                ys.append(net.res_bus.loc[range(config.BUS_COUNT), 'vm_pu'].to_numpy().copy())
            if (i+1)%500==0:
                print(f'Tap MLP {name}: {i+1}/{count}, {len(xs)} converged', flush=True)
    if len(xs)<50:
        raise ValueError('Too few converged tap-estimator training samples.')
    return np.asarray(xs), np.asarray(ys)


def prepare_tap_estimator(lab, samples=3000, test_samples=500):
    if getattr(lab, 'tap_estimator', None) is not None:
        return lab.tap_estimator
    print('Preparing tap-aware OLTC MLP (same ADG plant, all nine taps)…', flush=True)
    torch.set_num_threads(1)
    # fork_rng also preserves the original MPC model/training RNG stream.
    with torch.random.fork_rng():
        torch.manual_seed(lab.cfg.seed+100)
        x, y = _campaign(lab, samples, lab.cfg.seed+101, 'training')
        xt, yt = _campaign(lab, test_samples, lab.cfg.seed+102, 'test')
        xa, xv, ya, yv = train_test_split(x, y, test_size=.2, random_state=42)
        sx, sy = StandardScaler().fit(xa), StandardScaler().fit(ya)
        data = TensorDataset(torch.tensor(sx.transform(xa), dtype=torch.float32),
                             torch.tensor(sy.transform(ya), dtype=torch.float32))
        loader = DataLoader(data, batch_size=32, shuffle=True,
                            generator=torch.Generator().manual_seed(lab.cfg.seed+100))
        model = VoltageEstimator(2*config.BUS_COUNT+1, config.BUS_COUNT)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        for _ in range(100):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(model(xb), yb)
                loss.backward(); optimizer.step()
        estimator = TapVoltageEstimator(model, sx, sy)
        def predict(features):
            with torch.no_grad():
                return sy.inverse_transform(model(torch.tensor(sx.transform(features), dtype=torch.float32)).numpy())
        error = np.abs(predict(xt)-yt)
        lab.tap_training_report = {'architecture':'31 -> 64 ReLU -> 32 ReLU -> 15',
            'train_samples':len(xa), 'validation_samples':len(xv), 'test_samples':len(xt),
            'failed_training_power_flows':samples-len(x), 'failed_test_power_flows':test_samples-len(xt),
            'epochs':100, 'batch_size':32, 'voltage_test_clean_mae_pu':float(error.mean()),
            'voltage_test_max_error_pu':float(error.max()),
            'validation_mae_pu':float(np.abs(predict(xv)-yv).mean()),
            'tap_range':[lab.tap_min,lab.tap_max], 'load_range':[.5,5.], 'pv_range':[0.,5.],
            'targets':'clean AC voltages; zero DER commands', 'grid_fingerprint':lab.grid_fingerprint()}
    lab.tap_estimator = estimator
    print('Tap-aware MLP held-out AC voltage MAE:', round(float(error.mean()),6), 'pu', flush=True)
    return estimator


def choose_tap(lab, predictions, taps, initial_tap):
    """Pure prediction-only decision. No network, power flow or actual voltages."""
    predictions = np.asarray(predictions, dtype=float)
    taps = list(map(int, taps))
    if predictions.shape!=(len(taps), config.BUS_COUNT) or not np.isfinite(predictions).all():
        raise ValueError('Invalid tap-aware MLP voltage estimates.')
    v = predictions[:, lab.bus_ids]
    violation = np.maximum(0., lab.cfg.v_min-v)+np.maximum(0., v-lab.cfg.v_max)
    current = taps.index(int(initial_tap))
    if violation[current].max()==0:
        return int(initial_tap), 'estimated voltages in band'
    # Worst bus first; then aggregate violation; then minimum switching distance.
    scores = [(float(d.max()), float(np.sum(d*d)), abs(t-int(initial_tap)), abs(t))
              for d,t in zip(violation,taps)]
    target = taps[min(range(len(taps)), key=lambda i:scores[i])]
    return target, 'estimated feeder violation' if target!=initial_tap else 'no predicted improvement'


def informed_oltc(lab, baseline, initial_tap=0):
    """Choose using MLP only; AC evaluates the chosen tap, with explicit rollback."""
    if int(initial_tap)!=initial_tap or not lab.tap_min<=initial_tap<=lab.tap_max:
        raise ValueError('Invalid initial informed OLTC tap.')
    estimator = prepare_tap_estimator(lab)
    p,q = net_bus_pq(baseline, range(config.BUS_COUNT))
    taps = list(range(lab.tap_min,lab.tap_max+1))
    predictions = estimator.predict_profiles(p,q,taps)
    selected, reason = choose_tap(lab,predictions,taps,int(initial_tap))
    net = deepcopy(baseline)
    net.trafo.loc[lab.tid,'tap_pos'] = selected
    applied = selected
    try:
        y = lab.solve(net)
    except (pp.LoadflowNotConverged, ValueError):
        reason = 'selected tap power-flow failure; rolled back'
        applied = int(initial_tap)
        net = deepcopy(baseline)
        net.trafo.loc[lab.tid,'tap_pos'] = applied
        try:
            y = lab.solve(net)
        except (pp.LoadflowNotConverged, ValueError):
            # Baseline is already converged: no AC sweep or hidden better-tap search.
            net = deepcopy(baseline); applied = 0
            y = net.res_bus.loc[lab.bus_ids,'vm_pu'].to_numpy().copy()
            reason += ' to neutral'
    flags = {'reason':reason, 'requested_tap':selected, 'initial_tap':int(initial_tap),
        'steps':abs(applied-int(initial_tap)), 'at_limit':applied in (lab.tap_min,lab.tap_max),
        'predicted_voltages':predictions[taps.index(applied)].copy(),
        'candidate_taps':taps, 'candidate_predictions':predictions}
    return net,y,applied,flags
