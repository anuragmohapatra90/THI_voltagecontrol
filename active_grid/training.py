"""Original ADG profiles, random three-DER actions and Adam/MSE MLP training."""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from . import config, utils
from .randomness import reference_random
from .mlp_model import VoltageEstimator
from .controller import MLP_MPC_Controller, net_bus_pq


def _campaign(lab, count, seed, name):
    # One fixed reference plant is shared by train, test and the live experiment.
    env = deepcopy(lab.env)
    xs, ys, physical = [], [], []
    with reference_random(seed):
        p, q, pv = utils.generate_scenarios(count, config.BUS_COUNT)
        for i in range(count):
            action_p, action_q = {}, {}
            for bus in config.DER_BUSES:
                action_p[bus] = np.random.uniform(-config.MAX_P_MW, config.MAX_P_MW)
                action_q[bus] = np.random.uniform(-config.MAX_Q_MVAR, config.MAX_Q_MVAR)
            measured = env.step(p[i]-pv[i], q[i], action_p, action_q)
            if measured is not None and np.isfinite(measured).all():
                bp, bq = net_bus_pq(env.net, range(config.BUS_COUNT))
                xs.append(np.r_[bp, bq])
                ys.append(measured)
                physical.append(env.net.res_bus.vm_pu.to_numpy().copy())
            if (i+1) % 500 == 0:
                print(f'ADG {name}: {i+1}/{count} snapshots, {len(xs)} converged', flush=True)
    if len(xs) < 50:
        raise ValueError(f'Too few converged {name} samples on the reference grid.')
    return np.asarray(xs), np.asarray(ys), np.asarray(physical)


def train_controller(lab, samples=config.N_SAMPLES_TRAIN, test_samples=config.N_SAMPLES_TEST):
    torch.set_num_threads(1)
    torch.manual_seed(lab.cfg.seed)
    x, y, _ = _campaign(lab, samples, lab.cfg.seed+1, 'training')
    x_test, y_test, y_clean = _campaign(lab, test_samples, lab.cfg.seed+2, 'test')
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.2, random_state=42)
    sx, sy = StandardScaler().fit(x_train), StandardScaler().fit(y_train)
    dataset = TensorDataset(torch.tensor(sx.transform(x_train), dtype=torch.float32),
                            torch.tensor(sy.transform(y_train), dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=32, shuffle=True,
                        generator=torch.Generator().manual_seed(lab.cfg.seed))
    model = VoltageEstimator(2*config.BUS_COUNT, config.BUS_COUNT)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(100):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
    controller = MLP_MPC_Controller(model, sx, sy, range(config.BUS_COUNT), config.DER_BUSES,
                                    lab.cfg.v_min, lab.cfg.v_max)

    def predict(features):
        with torch.no_grad():
            return controller.model(torch.tensor(sx.transform(features), dtype=torch.float64)).numpy()*sy.scale_+sy.mean_

    predicted = predict(x_test)
    report = {'architecture': '30 -> 64 ReLU -> 32 ReLU -> 15',
              'train_samples': len(x_train), 'validation_samples': len(x_val), 'test_samples': len(x_test),
              'failed_training_power_flows': samples-len(x), 'failed_test_power_flows': test_samples-len(x_test),
              'epochs': 100, 'batch_size': 32, 'learning_rate': .001,
              'sensor_noise_std_pu': config.SENSOR_NOISE_STD,
              'voltage_test_mae_pu': float(np.abs(predicted-y_test).mean()),
              'voltage_test_clean_mae_pu': float(np.abs(predicted-y_clean).mean()),
              'voltage_test_max_error_pu': float(np.abs(predicted-y_clean).max()),
              'validation_mae_pu': float(np.abs(predict(x_val)-y_val).mean()),
              'grid_fingerprint': lab.grid_fingerprint()}
    controller.training_report = report
    lab.voltage_controller, lab.training_report = controller, report
    print('ADG held-out AC voltage MAE:', round(report['voltage_test_clean_mae_pu'], 6), 'pu', flush=True)
    return report
