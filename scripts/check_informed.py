"""Tap-aware training, prediction-only decisions and synchronized tab state."""
from copy import deepcopy
from unittest.mock import patch
import numpy as np
from pandapower import LoadflowNotConverged
from active_grid import config
from active_grid.informed_oltc import prepare_tap_estimator,choose_tap,informed_oltc
from active_grid.controller import net_bus_pq


def check_informed(lab):
    estimator=prepare_tap_estimator(lab)
    report=lab.tap_training_report
    assert report['grid_fingerprint']==lab.grid_fingerprint()
    assert report['tap_range']==[-4,4] and report['voltage_test_clean_mae_pu']<.025,report
    taps=list(range(-4,5))
    # Pure policy: a safe held tap stays put; low/high feeders call for opposite taps.
    profiles=np.ones((9,15))
    assert choose_tap(lab,profiles,taps,2)==(2,'estimated voltages in band')
    profiles[:,0]=1.2
    assert choose_tap(lab,profiles,taps,2)==(2,'estimated voltages in band')
    for sign in [1,-1]:
        profiles=np.ones((9,15))
        profiles[:,1:]=1-sign*.08+np.asarray(taps)[:,None]*.0125
        chosen,reason=choose_tap(lab,profiles,taps,0)
        assert chosen==sign*3 and reason=='estimated feeder violation'
    # No AC oracle: synthetic predictions deliberately disagree with baseline AC.
    baseline=lab.new_net();lab.set_snapshot(baseline,lab.generate_scenario(0));lab.solve(baseline)
    profiles=np.ones((9,15));profiles[:,1:]=.9+np.asarray(taps)[:,None]*.0125
    physical_solve=lab.solve
    seen=[]
    def record(net):
        seen.append(int(net.trafo.loc[lab.tid,'tap_pos']))
        return physical_solve(net)
    with patch.object(estimator,'predict_profiles',return_value=profiles),patch.object(lab,'solve',side_effect=record):
        controlled,y,tap,flags=informed_oltc(lab,baseline)
    assert tap==4 and seen==[4],(tap,seen)
    assert flags['requested_tap']==4 and flags['at_limit'] and np.isfinite(y).all()
    assert (controlled.sgen[['p_mw','q_mvar']]==0).all().all()
    assert baseline.load.equals(controlled.load)
    # Failed selected tap must roll back explicitly, without scanning real AC candidates.
    seen=[]
    def fail_selected(net):
        t=int(net.trafo.loc[lab.tid,'tap_pos']);seen.append(t)
        if t==4:raise LoadflowNotConverged('test selected tap failure')
        return physical_solve(net)
    with patch.object(estimator,'predict_profiles',return_value=profiles),patch.object(lab,'solve',side_effect=fail_selected):
        controlled,y,tap,flags=informed_oltc(lab,baseline,1)
    assert seen==[4,1] and tap==1 and 'rolled back' in flags['reason']
    assert np.array_equal(flags['predicted_voltages'],profiles[taps.index(1)])
    # Both selected and held taps fail: use the already-converged neutral snapshot.
    with patch.object(estimator,'predict_profiles',return_value=profiles),patch.object(lab,'solve',side_effect=LoadflowNotConverged('all fail')):
        controlled,y,tap,flags=informed_oltc(lab,baseline,1)
    assert tap==0 and np.allclose(y,baseline.res_bus.loc[lab.bus_ids,'vm_pu'])
    assert np.array_equal(flags['predicted_voltages'],profiles[taps.index(0)])
    # The learned tap input must distinguish physically different profiles.
    p,q=net_bus_pq(baseline,range(config.BUS_COUNT))
    predicted=estimator.predict_profiles(p,q,taps)
    assert predicted.shape==(9,15) and np.isfinite(predicted).all()
    assert predicted[-1,1]-predicted[0,1]>.06
    errors=[]
    for t in [-4,0,4]:
        n=deepcopy(baseline);n.trafo.loc[lab.tid,'tap_pos']=t;lab.solve(n)
        errors.append(np.mean(np.abs(predicted[taps.index(t)]-n.res_bus.vm_pu.to_numpy())))
    assert max(errors)<.025,errors
    from dashboard import ScenarioExperiment,comparison_plot,select_mode,update_configuration
    from dashboard_diagnostics import diagnostics_html
    session=ScenarioExperiment(lab,rng=np.random.default_rng(0))
    f=session.sample()
    estimates=f['informed_voltage_estimates']
    assert estimates.Bus.tolist()==list(range(15))
    assert np.allclose(estimates['Actual (pu)'],f['informed_oltc'].res_bus.vm_pu)
    assert np.allclose(estimates['Estimate (pu)'],f['informed_flags']['predicted_voltages'])
    assert np.allclose(estimates['Error (pu)'],estimates['Estimate (pu)']-estimates['Actual (pu)'])
    local=comparison_plot(f,lab).to_plotly_json();informed=comparison_plot(f,lab,'mlp').to_plotly_json()
    assert local['data'][0]['y']==informed['data'][0]['y']
    assert local['data'][2]['y']==informed['data'][2]['y']
    assert 'MLP-informed OLTC' in informed['layout']['annotations'][2]['text']
    for trace in informed['data'][:3]:
        for row in trace['customdata']:
            if row[0]:assert np.isclose(row[2],f['informed_oltc'].res_bus.loc[int(row[0]),'vm_pu'])
    # Tab selection is cached rendering only, preserving scenario/RNG and both tap memories.
    before=(session.scenario_id,session.base['seed'],session.tap_pos,session.informed_tap_pos,deepcopy(session.rng.bit_generator.state))
    with patch.object(lab,'solve',side_effect=AssertionError('Tab switch must not solve')):
        for mode in ['mlp','local','mlp']:
            state,html,status=select_mode(session,mode)
            assert state==mode and f'data-oltc-mode="{mode}"' in html
            assert 'OLTC tap position' in status
    after=(session.scenario_id,session.base['seed'],session.tap_pos,session.informed_tap_pos,session.rng.bit_generator.state)
    assert before==after
    assert 'At applied OLTC tap' in diagnostics_html(f,lab,'mlp')
    assert 'After MPC' in diagnostics_html(f,lab)
    previous=session.last
    failed=update_configuration(5,0,1,.1,1000,session,'mlp')
    assert failed[0] is session and session.last is previous
    assert len(failed)==7 and all(v=={'__type__':'update'} for v in failed[1:6])
    from active_grid.local_oltc import settle_oltc
    comparisons=[]
    for seed,lf,pf in [(6,1,1),(0,.5,5),(6,3,0)]:
        b=lab.new_net();lab.set_snapshot(b,lab.generate_scenario(seed),lf,pf);yb=lab.solve(b)
        _,yl,tl,_=settle_oltc(lab,b)
        _,yi,ti,_=informed_oltc(lab,b)
        comparisons.append({'seed':seed,'load':lf,'pv':pf,'local_tap':tl,'informed_tap':ti,
            'actual_violations':[lab.violation(yb),lab.violation(yl),lab.violation(yi)]})
    print('Actual AC comparison (uncontrolled/local/informed OLTC):',comparisons)
    print('Tap-aware clean held-out AC MAE:',report['voltage_test_clean_mae_pu'])
    print('Prediction-only tap selection, no AC search, rollback, matched per-bus diagnostics and cached synchronized tabs: PASS')
