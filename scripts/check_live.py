"""Original ADG injections, reference MPC, exhaustive taps and live Gradio checks."""
from copy import deepcopy
from pathlib import Path
import os, sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grid_lab import GridLab
from dashboard import ScenarioExperiment, comparison_plot
from active_grid import config
from active_grid.oltc import best_oltc, oltc_score
from live_control import prepare_controller, net_bus_pq


def check_live(lab=None):
    lab = lab or GridLab()
    c = prepare_controller(lab)
    assert c.der_buses == [5, 8, 14] and c.n_der == 3
    assert c.training_report['grid_fingerprint'] == lab.grid_fingerprint()
    assert c.training_report['voltage_test_clean_mae_pu'] < .01
    session = ScenarioExperiment(lab, rng=np.random.default_rng(11))
    first, second = session.sample(1, 1), session.sample(2, 1)
    assert first['scenario_seed'] == second['scenario_seed']
    assert first['scenario_id'] == second['scenario_id']
    assert np.isclose(second['gross_load_p_kw'], 2*first['gross_load_p_kw'])
    assert np.isclose(second['load_q_kvar'], 2*first['load_q_kvar'])
    assert not np.allclose(first['y_base'], second['y_base'])
    for key in ['oltc', 'mpc']:
        assert second['baseline'].load.equals(second[key].load)
        assert np.array_equal(second['baseline'].sgen.bus, second[key].sgen.bus)
        assert len(second[key].sgen) == 13
    assert second['mpc'].trafo.loc[lab.tid, 'tap_pos'] == 0
    assert second['baseline'].trafo.loc[lab.tid, 'tap_pos'] == 0
    assert (second['baseline'].sgen[['p_mw', 'q_mvar']] == 0).all().all()
    assert (second['oltc'].sgen[['p_mw', 'q_mvar']] == 0).all().all()
    trial = deepcopy(second['baseline'])
    selected = oltc_score(lab, second['y_oltc'], second['tap_pos'])
    for tap in range(lab.tap_min, lab.tap_max+1):
        trial.trafo.loc[lab.tid, 'tap_pos'] = tap
        assert selected <= oltc_score(lab, lab.solve(trial), tap)
    # Physical sign and existing-element check, independently of the MLP.
    trial = deepcopy(second['baseline'])
    p0, q0 = net_bus_pq(trial, c.bus_ids)
    c.apply_setpoints(trial, {8: .001}, {8: .002})
    p1, q1 = net_bus_pq(trial, c.bus_ids)
    assert np.isclose(p1[8], p0[8]-.001) and np.isclose(q1[8], q0[8]-.002)
    assert len(trial.sgen) == 13
    lab.solve(trial)
    assert trial.res_bus.loc[8, 'vm_pu'] > second['baseline'].res_bus.loc[8, 'vm_pu']
    # Verify the unchanged voltage/action objective and gain response.
    p, q = net_bus_pq(second['baseline'], c.bus_ids)
    x = np.full(6, .001)
    _, jac = c.objective_with_gradient(x, p, q, (1, .1, 1000))
    for i in range(6):
        d = np.zeros(6); d[i] = 1e-7
        fa = c.objective_with_gradient(x+d, p, q, (1, .1, 1000))[0]
        fb = c.objective_with_gradient(x-d, p, q, (1, .1, 1000))[0]
        assert np.isclose((fa-fb)/2e-7, jac[i], atol=1e-5, rtol=1e-5)
    ap, aq = c.get_optimal_setpoints(p, q, 1, .1, 0)
    assert np.allclose(list(ap.values()), 0) and np.allclose(list(aq.values()), 0)
    before, after, taps, saturation = [], [], [], []
    for seed, lf, pf in [(0,1,1), (3,1,1), (6,1,1), (8,1,1), (6,2,0), (8,2,0)]:
        session.base = lab.generate_scenario(seed)
        f = session.sample(lf, pf)
        before.append(lab.violation(f['y_base']))
        after.append(lab.violation(f['y_mpc']))
        taps.append(lab.violation(f['y_oltc']))
        assert f['flags']['optimizer_action_applied']
        commands = f['actions'][['P (kW)', 'Q (kVar)']].to_numpy()
        assert np.max(np.abs(commands)) <= 50.00001
        saturation.append(float(np.mean(np.abs(commands) >= 49.99)))
        actual_p, actual_q = net_bus_pq(f['mpc'], c.bus_ids)
        expected_p, expected_q = net_bus_pq(f['baseline'], c.bus_ids)
        for _, row in f['actions'].iterrows():
            b = int(row['Bus']); expected_p[b] -= row['P (kW)']/1000; expected_q[b] -= row['Q (kVar)']/1000
        assert np.allclose(actual_p, expected_p, atol=6e-6)
        assert np.allclose(actual_q, expected_q, atol=6e-6)
    assert np.mean(after) < .6*np.mean(before), (before, after)
    assert np.mean(saturation) < .5, saturation
    print('Native ADG six-case AC violations (uncontrolled / best OLTC / MPC):', list(zip(before, taps, after)))
    print('Mean saturated command fraction:', float(np.mean(saturation)))
    previous = session.base['seed']; session.new_scenario()
    assert session.base['seed'] != previous
    f = session.sample(1, 1)
    plot = comparison_plot(f, lab).to_plotly_json()
    assert [t['name'] for t in plot['data'][:3]] == ['Uncontrolled', 'Best OLTC', 'MLP + MPC']
    assert plot['data'][0]['y'].count(None) == 2
    assert set(plot['data'][0]['customdata']) == {str(b) for b in range(15)} | {''}
    print('Original ADG configuration, three-DER signs/bounds, reactive PQ, controller and exhaustive tap checks: PASS')


def check_server(lab):
    import json
    from urllib.request import build_opener, ProxyHandler
    from gradio_client import Client
    from dashboard import build_dashboard, launch_dashboard
    app = build_dashboard(lab)
    try:
        url, credentials = launch_dashboard(app, share=False)
        assert credentials is None and app.auth is None
        with build_opener(ProxyHandler({})).open(url+'config') as response:
            assert 'dependencies' in json.load(response) and '/login' not in response.url
        client = Client(url, verbose=False)
        initial = client.predict(1, 1, 1, .1, 1000, api_name='/initial_snapshot')
        scaled = client.predict(2, 1, 1, .1, 1000, api_name='/react')
        new = client.predict(2, 1, 1, .1, 1000, api_name='/new_scenario')
        assert initial[1] == scaled[1] and initial[0] != scaled[0]
        assert new[1] != scaled[1]
        assert 'Synthetic daily profile' in new[1] and 'OLTC tap position:' in new[2]
        plot = json.loads(initial[0]['plot'])
        assert [t['name'] for t in plot['data'][:3]] == ['Uncontrolled', 'Best OLTC', 'MLP + MPC']
        types = [c['type'] for c in app.config['components']]
        assert types.count('button') == 1 and types.count('plot') == 1
        assert not {'dropdown', 'tabs', 'tabitem', 'timer', 'radio', 'checkbox'} & set(types)
        sliders = [c['props'] for c in app.config['components'] if c['type'] == 'slider']
        assert [(c['minimum'], c['maximum'], c['value']) for c in sliders] == [(.5, 2., 1.), (0., 2., 1.)]
        assert app.config['title'] == 'Voltage control for ADG'
        print('No-login Gradio, original slider ranges, new synthetic scenarios and three overlaid controls: PASS')
        if os.getenv('GRID_LAB_BROWSER_CHECK') == '1':
            from check_browser import check_browser
            check_browser(url)
    finally:
        app.close()


if __name__ == '__main__':
    lab = GridLab(); check_live(lab); check_server(lab)
