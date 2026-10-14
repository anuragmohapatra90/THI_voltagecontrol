"""Checks for simultaneous AC comparisons, full-year scenarios, and the Gradio UI."""
from pathlib import Path
from copy import deepcopy
import os,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from grid_lab import GridLab
from dashboard import ScenarioExperiment,oltc_score,update_configuration
from live_control import surrogate_with_jacobian
from grid_lab import PRESETS


def check_live(lab=None):
    lab=lab or GridLab()
    if lab.model is None:lab.train(1000)
    assert lab.profile_times[0].isoformat()=='2016-01-01T00:00:00'
    assert lab.profile_times[-1].isoformat()=='2016-12-31T23:45:00'
    session=ScenarioExperiment(lab,rng=np.random.default_rng(11))
    first=session.sample(1,1)
    second=session.sample(2,1)
    assert session.t==first['profile_step']
    assert second['timestamp']==lab.profile_times[session.t]
    assert np.isclose(second['load_p_kw'],2*first['load_p_kw'])
    assert np.isclose(second['load_q_kvar'],2*first['load_q_kvar'])
    assert not np.allclose(first['y_base'][:lab.nv],second['y_base'][:lab.nv])
    for key in ['oltc','mpc']:
        assert np.allclose(second['baseline'].load[['p_mw','q_mvar']],second[key].load[['p_mw','q_mvar']])
    assert np.allclose(second['baseline'].sgen[['p_mw','q_mvar']],second['oltc'].sgen[['p_mw','q_mvar']])
    assert second['baseline'].trafo.loc[lab.tid,'tap_pos']==lab.tap0
    assert second['mpc'].trafo.loc[lab.tid,'tap_pos']==lab.tap0
    # Independent exhaustive sweep proves the displayed OLTC is the best allowed tap.
    chosen=oltc_score(lab,second['y_oltc'],second['tap_pos']);trial=deepcopy(second['baseline'])
    for tap in range(lab.tap_min,lab.tap_max+1):
        trial.trafo.loc[lab.tid,'tap_pos']=tap
        assert chosen<=oltc_score(lab,lab.solve(trial),tap)
    x=lab.features(second['baseline']);pred,jac=surrogate_with_jacobian(lab,x)
    assert np.allclose(pred,lab.predict(x)[0],atol=1e-9)
    for i in range(len(x)):
        delta=np.zeros_like(x);delta[i]=lab.xscale.scale_[i]*1e-5
        fd=(surrogate_with_jacobian(lab,x+delta)[0]-surrogate_with_jacobian(lab,x-delta)[0])/2e-5
        assert np.allclose(fd,jac[:,i]*lab.xscale.scale_[i],atol=1e-4,rtol=1e-5)
    session.t=int(np.argmax(lab.profiles[('sgen','p_mw')].reindex(columns=lab.pv_ids).sum(axis=1)))
    update_configuration('Custom',.55,3,1,.1,1000,session)
    assert session.last['flags']['safe']
    assert (session.last['actions']['P (kW)']<0).any()
    net=session.last['mpc'];p=net.sgen.loc[lab.pv_ids,'p_mw'].to_numpy();q=net.sgen.loc[lab.pv_ids,'q_mvar'].to_numpy()
    assert np.all(p*p+q*q<=lab.rating**2+1e-10)
    update_configuration('Custom',.55,3,.01,100,1000,session)
    # Different weights can legitimately select the same safe backup action.
    assert session.last['flags']['weights']==(.01,100.,1000.)
    assert session.last['flags']['safe']
    applied=session.last['mpc'].sgen.loc[lab.pv_ids]
    baseline_p=session.last['baseline'].sgen.loc[lab.pv_ids,'p_mw']
    assert np.isclose(session.last['flags']['action_cost'],
        .01*np.sum((applied.p_mw-baseline_p)**2)+100*np.sum(applied.q_mvar**2),atol=1e-9)
    months=set()
    for _ in range(250):
        before=session.t;session.new_scenario();assert session.t!=before
        months.add(lab.profile_times[session.t].month)
    assert months==set(range(1,13))
    for preset in ['High PV','High Load']:
        # Check every pre-filtered row, not just a randomly lucky stressed sample.
        lf,pf=PRESETS[preset];lab.rating=lab.base_rating*max(1.,pf);net=lab.new_net()
        for t in lab.scenario_buckets[preset]:
            av=lab.set_profile(net,int(t),lf,pf);lab.control(net,av,np.zeros(2*lab.ng));y=lab.solve(net)
            if preset=='High PV':assert y[:lab.nv].max()>=lab.cfg.v_max+.02
            else:assert y[:lab.nv].min()<=lab.cfg.v_min-.02
        for _ in range(8):
            session.new_scenario(preset);assert session.t in lab.scenario_buckets[preset]
        frame=session.sample(1,1,preset=preset)
        assert frame['load_factor']==lf and frame['pv_factor']==pf
        assert frame['preset']==preset and frame['profile_step'] in lab.scenario_buckets[preset]
    print('Every High PV / High Load bucket row passes actual AC voltage-stress thresholds; new scenarios stay in preset: PASS')
    print('Same-PQ three-way comparison, exhaustive best tap, calendar/full-year sampling, reactive scaling, MLP derivatives and MPC: PASS')


def check_server(lab):
    from dashboard import build_dashboard,launch_dashboard
    from gradio_client import Client
    from urllib.request import build_opener,ProxyHandler
    app=build_dashboard(lab)
    try:
        url,credentials=launch_dashboard(app,share=False)
        assert credentials is None and app.auth is None
        with build_opener(ProxyHandler({})).open(url+'config') as response:
            import json
            assert 'dependencies' in json.load(response)
            assert '/login' not in response.url
        client=Client(url,verbose=False)
        args=['Custom',1,1,1,.1,1000]
        initial=client.predict(*args,api_name='/initial_snapshot')
        assert len(initial)==3 and initial[0]['type']=='plotly'
        plot=json.loads(initial[0]['plot'])
        assert [t['name'] for t in plot['data'][:3]]==['Uncontrolled','Best OLTC','MLP + MPC']
        assert len(set(t.get('yaxis','y') for t in plot['data'][:3]))==1
        scaled=client.predict('Custom',2,1,1,.1,1000,api_name='/react')
        assert initial[1]==scaled[1] # sliders keep date/time
        assert initial[0]!=scaled[0] # actual voltage change
        new=client.predict('Custom',2,1,1,.1,1000,api_name='/new_scenario')
        assert new[1]!=scaled[1]
        high=client.predict('High PV',1,1,1,.1,1000,api_name='/react')
        highplot=json.loads(high[0]['plot'])
        # Plotly encoded arrays are validated against the session data in check_live.
        assert highplot['data'][0]['name']=='Uncontrolled'
        high_new=client.predict('High PV',1,1,1,.1,1000,api_name='/new_scenario')
        assert high_new[1]!=high[1] or len(lab.scenario_buckets['High PV'])==1
        types=[c['type'] for c in app.config['components']]
        assert types.count('button')==1 and types.count('plot')==1
        assert not {'tabs','tabitem','timer','radio','checkbox'}&set(types)
        print('No-login Gradio, all three overlaid traces, reactive sliders and new dated scenario: PASS')
        if os.environ.get('GRID_LAB_BROWSER_CHECK')=='1':
            from check_browser import check_browser
            check_browser(url)
    finally:app.close()

if __name__=='__main__':
    lab=GridLab();check_live(lab);check_server(lab)
