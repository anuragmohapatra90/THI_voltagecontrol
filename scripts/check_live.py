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
from active_grid.local_oltc import settle_oltc
from live_control import prepare_controller, net_bus_pq
from dashboard_diagnostics import control_deltas, diagnostics_html


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
    _,oracle_y,oracle_tap=best_oltc(lab,second['baseline'])
    selected = oltc_score(lab, oracle_y, oracle_tap)
    for tap in range(lab.tap_min, lab.tap_max+1):
        trial.trafo.loc[lab.tid, 'tap_pos'] = tap
        assert selected <= oltc_score(lab, lab.solve(trial), tap)
    # Both feeder heads already contain load and PV; scaling moves the local secondary voltage.
    assert set(lab.net.load.bus)==set(lab.net.sgen.bus)==set(range(2,15))
    assert 1 not in set(lab.net.load.bus) and 1 not in set(lab.net.sgen.bus)
    daytime=lab.generate_scenario(0)
    for b in [2,9]:assert daytime['load_p'][b]>0 and daytime['pv'][b]>0
    local_voltage=[];local_taps=[]
    for multiplier in [1,2,3]:
        n=lab.new_net();lab.set_snapshot(n,lab.generate_scenario(6),multiplier,0);lab.solve(n)
        local_voltage.append(float(n.res_bus.loc[1,'vm_pu']))
        _,_,tap,_=settle_oltc(lab,n)
        local_taps.append(tap)
    assert all(a>b for a,b in zip(local_voltage,local_voltage[1:])),local_voltage
    assert local_taps==[0,0,1],(local_voltage,local_taps)
    print('Mixed feeder-head sites; loading 1/2/3x gives V1/taps:',list(zip(local_voltage,local_taps)))
    # A local relay cannot see the unsafe feeder end if the secondary bus remains in band.
    neutral=lab.new_net();lab.set_snapshot(neutral,lab.generate_scenario(6))
    neutral_y=lab.solve(neutral)
    assert lab.oltc_v_min<=neutral.res_bus.loc[1,'vm_pu']<=lab.oltc_v_max
    assert lab.violation(neutral_y)>0
    local_net,local_y,local_tap,local_flags=settle_oltc(lab,neutral)
    assert local_tap==0 and np.allclose(local_y,neutral_y)
    assert local_flags['measured_bus']==1 and local_flags['steps']==0
    # A stronger real load draw moves the existing secondary bus enough to trigger a tap.
    heavy=lab.new_net();lab.set_snapshot(heavy,lab.generate_scenario(6),3,0)
    lab.solve(heavy)
    assert heavy.res_bus.loc[1,'vm_pu']<lab.oltc_v_min
    result,_,tap,flags=settle_oltc(lab,heavy)
    assert tap>0 and flags['steps']>0
    assert lab.oltc_v_min<=flags['secondary_voltage_pu']<=lab.oltc_v_max
    # Secondary voltage moves independently of the MV slack; one-step actions have the right sign.
    unloaded=lab.new_net()
    for source,direction in [(.97,1),(1.04,-1)]:
        unloaded.ext_grid.loc[:,'vm_pu']=source
        result,y,tap,flags=settle_oltc(lab,unloaded)
        assert direction*tap>0 and lab.oltc_v_min<=flags['secondary_voltage_pu']<=lab.oltc_v_max
        assert len(flags['history'])<=9
        assert all(b['tap']-a['tap']==direction for a,b in zip(flags['history'],flags['history'][1:]))
        again,_,held,held_flags=settle_oltc(lab,unloaded,tap)
        assert held==tap and held_flags['steps']==0
        assert result.res_bus.loc[0,'vm_pu']==source and not np.isclose(result.res_bus.loc[1,'vm_pu'],source)
    unloaded.ext_grid.loc[:,'vm_pu']=.9
    result,_,tap,flags=settle_oltc(lab,unloaded)
    assert tap==4 and flags['reason']=='tap limit' and flags['at_limit']
    # A failed trial tap rolls back to its last valid local solution.
    from unittest.mock import patch
    from pandapower import LoadflowNotConverged
    unloaded.ext_grid.loc[:,'vm_pu']=.97
    original_solve=lab.solve
    def fail_next_tap(net):
        if net.trafo.loc[lab.tid,'tap_pos']==1:raise LoadflowNotConverged('trial tap regression')
        return original_solve(net)
    with patch.object(lab,'solve',side_effect=fail_next_tap):
        result,y,tap,flags=settle_oltc(lab,unloaded)
    assert tap==0 and flags['reason']=='next tap power-flow failure' and np.isfinite(y).all()
    print('Bus-1-only relay, no downstream feedback, direction, deadband memory, bounds and trial-tap rollback: PASS')
    # Match the displayed MLP prediction to these applied commands and this AC solve.
    estimates=second['voltage_estimates']
    assert estimates['Bus'].tolist()==c.bus_ids
    bp,bq=net_bus_pq(second['baseline'],c.bus_ids)
    dp=second['actions']['ΔP (kW)'].to_numpy()/1000
    dq=second['actions']['ΔQ (kVar)'].to_numpy()/1000
    predicted=c.predict_voltages(bp,bq,dp,dq)
    actual=second['mpc'].res_bus.loc[c.bus_ids,'vm_pu'].to_numpy()
    assert np.allclose(estimates['Estimate (pu)'],predicted)
    assert np.allclose(estimates['Actual (pu)'],actual)
    assert np.allclose(estimates['Error (pu)'],predicted-actual)
    expected_cost=np.sum(dp**2)+.1*np.sum(dq**2)+1000*np.sum(np.maximum(0,.95-predicted)**2+np.maximum(0,predicted-1.05)**2)
    assert np.isclose(expected_cost,second['flags']['objective'])
    # Both signed bounds must be marked; ordinary commands must stay unmarked.
    saturated=deepcopy(second['baseline'])
    c.apply_setpoints(saturated,{5:.05,8:-.05,14:.01},{5:-.05,8:.01,14:.05})
    actions,limits=control_deltas(second['baseline'],saturated,c.der_buses)
    assert limits[['ΔP','ΔQ']].to_numpy().sum()==4
    html=diagnostics_html({**second,'actions':actions,'at_limit':limits},lab)
    assert html.count('class="max-label"')==4 and '4/6 commands' in html
    assert 'ΔP (kW)' in html and 'ΔQ (kVar)' in html
    assert 'Full MPC cost function' in html and '0.95' in html and '1.05' in html
    old_seed=session.base['seed'];old_id=session.scenario_id
    from dashboard import new_scenario
    invalid=new_scenario(6,1,1,.1,1000,session)
    assert invalid[0] is session and all(x=={'__type__':'update'} for x in invalid[1:5])
    assert 'data-error="true"' in invalid[5]
    assert session.base['seed']==old_seed and session.scenario_id==old_id
    # Upper endpoints are real profile multipliers, with finite AC results when feasible.
    session.base=lab.generate_scenario(0)
    for lf,pf in [(5,0),(.5,5),(5,5)]:
        stressed=session.sample(lf,pf)
        assert np.isclose(stressed['gross_load_p_kw'],session.base['load_p'].sum()*lf*1000)
        assert np.isclose(stressed['load_q_kvar'],session.base['load_q'].sum()*lf*1000)
        assert np.isclose(stressed['pv_kw'],session.base['pv'].sum()*pf*1000)
        assert np.isfinite(stressed['y_mpc']).all()
        assert np.abs(stressed['actions'][['ΔP (kW)','ΔQ (kVar)']].to_numpy()).max()<=50.00001
        for key in ['oltc','mpc']:assert stressed['baseline'].load.equals(stressed[key].load)
        if (lf,pf)==(5,0):assert min(stressed['y_base'])<.95
        if (lf,pf)==(.5,5):assert max(stressed['y_base'])>1.05
    # A physically non-convergent 5x load snapshot keeps the last display and reports why.
    from dashboard import update_configuration
    session.base=lab.generate_scenario(6)
    previous_frame=session.last
    failed=update_configuration(5,0,1,.1,1000,session)
    assert failed[0] is session and all(x=={'__type__':'update'} for x in failed[1:5])
    assert 'Power flow failed' in failed[5] and 'Previous result kept' in failed[5]
    assert session.last is previous_frame
    for lf,pf in [(float('nan'),1),(1,float('inf')),(1,-.1),(5.1,1),(1,5.1)]:
        try:session.sample(lf,pf)
        except ValueError:pass
        else:raise AssertionError('Invalid profile multipliers must be rejected')
    print('5x profile multipliers, fixed DER limits and non-convergent-result retention: PASS')
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
        commands = f['actions'][['ΔP (kW)', 'ΔQ (kVar)']].to_numpy()
        assert np.max(np.abs(commands)) <= 50.00001
        saturation.append(float(np.mean(np.abs(commands) >= 49.99)))
        actual_p, actual_q = net_bus_pq(f['mpc'], c.bus_ids)
        expected_p, expected_q = net_bus_pq(f['baseline'], c.bus_ids)
        for _, row in f['actions'].iterrows():
            b = int(row['Bus']); expected_p[b] -= row['ΔP (kW)']/1000; expected_q[b] -= row['ΔQ (kVar)']/1000
        assert np.allclose(actual_p, expected_p, atol=6e-6)
        assert np.allclose(actual_q, expected_q, atol=6e-6)
    assert np.mean(after) < .6*np.mean(before), (before, after)
    assert np.mean(saturation) < .5, saturation
    print('Native ADG six-case AC violations (uncontrolled / local OLTC / MPC):', list(zip(before, taps, after)))
    print('Mean saturated command fraction:', float(np.mean(saturation)))
    previous = session.base['seed']; session.new_scenario()
    assert session.base['seed'] != previous
    f = session.sample(1, 1)
    plot = comparison_plot(f, lab).to_plotly_json()
    assert [t['name'] for t in plot['data'][:3]] == ['Uncontrolled', 'OLTC', 'MLP + MPC']
    assert plot['data'][0]['y'].count(None) == 2
    maps=[t for t in plot['data'] if t.get('mode')=='markers+text']
    assert len(maps)==3
    for trace in maps:
        symbols=trace['marker']['symbol']
        assert symbols[:2]==['circle','circle']
        for b in range(2,15):
            assert symbols[b]==('diamond' if b in config.DER_BUSES else 'triangle-up')
        assert np.allclose([r[4] for r in trace['customdata']], [f['pv_bus_kw'][b] for b in range(15)])
        assert np.allclose([r[9] for r in trace['customdata']], [f['load_bus_kw'][b] for b in range(15)])
        for row in trace['customdata']:
            assert np.isclose(row[9]-row[4],row[2]),row
        for b in [2,9]:assert trace['customdata'][b][7]=='Load + PV'
        assert 'Gross load P' in trace['hovertemplate'] and 'Load Q' in trace['hovertemplate']
    assert {t.get('name') for t in plot['data']} >= {'Load + PV','Load + PV + DER','At limit'}
    assert {row[0] for row in plot['data'][0]['customdata']} == {str(b) for b in range(15)} | {''}
    for trace in plot['data'][:3]:
        for row in trace['customdata']:
            if row[0]=='':continue
            b=int(row[0])
            assert np.allclose(row[1:],[f[key].res_bus.loc[b,'vm_pu'] for key in ['baseline','oltc','mpc']])
        assert 'local band' in trace['hovertemplate']
        assert all(name in trace['hovertemplate'] for name in ['Uncontrolled:','OLTC:','MLP + MPC:'])
    # All cases share the source voltage: hovering this coincident point must retain every value.
    assert plot['data'][0]['customdata'][0]==['0',1.,1.,1.]
    print('Every voltage tooltip includes all three bus-aligned AC results, including coincident curves: PASS')
    print('Original ADG configuration, three-DER signs/bounds, reactive PQ, controller and exhaustive tap checks: PASS')


def check_server(lab):
    import json
    from urllib.request import build_opener, ProxyHandler
    from gradio_client import Client
    from dashboard import build_dashboard, launch_dashboard
    app = build_dashboard(lab,scenario_seed=0)
    try:
        url, credentials = launch_dashboard(app, share=False)
        assert credentials is None and app.auth is None
        with build_opener(ProxyHandler({})).open(url+'config') as response:
            assert 'dependencies' in json.load(response) and '/login' not in response.url
        client = Client(url, verbose=False)
        initial = client.predict(1, 1, 1, .1, 1000, api_name='/initial_snapshot')
        scaled = client.predict(2, 1, 1, .1, 1000, api_name='/react')
        higher = client.predict(.5, 5, 1, .1, 1000, api_name='/react')
        assert higher[2]==scaled[2] and higher[0]!=scaled[0]
        failed = client.predict(5,0,1,.1,1000,api_name='/react')
        assert 'Power flow failed' in failed[4] and 'data-error="true"' in failed[4]
        recovered = client.predict(1,1,1,.1,1000,api_name='/react')
        assert recovered[2]==higher[2] and 'data-error="false"' in recovered[4]
        new = client.predict(2, 1, 1, .1, 1000, api_name='/new_scenario')
        assert initial[2] == scaled[2] and initial[0] != scaled[0]
        assert len(initial)==5 and initial[1].count('<tr data-bus=')==18
        assert new[2] != scaled[2]
        assert 'Synthetic daily profile' in new[2] and 'OLTC tap position:' in new[3]
        assert 'MLP estimate vs actual' in new[1] and 'Full MPC cost function' in new[1]
        assert 'ΔP (kW)' in new[1] and 'ΔQ (kVar)' in new[1]
        plot = json.loads(initial[0]['plot'])
        assert [t['name'] for t in plot['data'][:3]] == ['Uncontrolled', 'OLTC', 'MLP + MPC']
        types = [c['type'] for c in app.config['components']]
        assert types.count('button') == 1 and types.count('plot') == 1
        assert not {'dropdown', 'tabs', 'tabitem', 'timer', 'radio', 'checkbox'} & set(types)
        sliders = [c['props'] for c in app.config['components'] if c['type'] == 'slider']
        assert [(c['minimum'], c['maximum'], c['value']) for c in sliders] == [(.5, 5., 1.), (0., 5., 1.)]
        assert app.config['title'] == 'Voltage control for ADG'
        gains=[c['props'] for c in app.config['components'] if c['type']=='number']
        assert [g['info'] for g in gains]==['Higher: use less ΔP','Higher: use less ΔQ','Higher: stronger voltage correction']
        print('Recoverable power-flow error response and next-request recovery: PASS')
        print('No-login Gradio, extended 5x slider ranges, new synthetic scenarios and three overlaid controls: PASS')
        if os.getenv('GRID_LAB_BROWSER_CHECK') == '1':
            from check_browser import check_browser
            check_browser(url)
    finally:
        app.close()


if __name__ == '__main__':
    lab = GridLab(); check_live(lab); check_server(lab)
