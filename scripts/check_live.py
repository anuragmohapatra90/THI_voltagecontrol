"""Behavioral checks for the ADG reference controller and three-way dashboard."""
from pathlib import Path
from copy import deepcopy
import os,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from grid_lab import GridLab,PRESETS
from dashboard import ScenarioExperiment,oltc_score,update_configuration
from live_control import prepare_controller,net_bus_pq,voltage_violation
from active_grid.controller import MAX_P_MW,MAX_Q_MVAR
from scenarios import apply_layout,has_voltage_valley,feeder_paths


def check_live(lab=None):
    lab=lab or GridLab();controller=prepare_controller(lab)
    assert lab.profile_times[0].isoformat()=='2016-01-01T00:00:00'
    assert lab.profile_times[-1].isoformat()=='2016-12-31T23:45:00'
    assert controller.training_report['voltage_test_mae_pu']<.006
    session=ScenarioExperiment(lab,rng=np.random.default_rng(11))
    first=session.sample(1,1);second=session.sample(2,1)
    assert session.t==first['profile_step']
    assert second['timestamp']==lab.profile_times[session.t]
    assert first['layout']==second['layout']
    raw_p=lab.profiles[('load','p_mw')].iloc[session.t].reindex(lab.net.load.index).to_numpy()
    raw_q=lab.profiles[('load','q_mvar')].iloc[session.t].reindex(lab.net.load.index).to_numpy()
    assert np.allclose(second['baseline'].load.p_mw,2*raw_p)
    assert np.allclose(second['baseline'].load.q_mvar,2*raw_q)
    assert np.isclose(second['load_p_kw'],2*first['load_p_kw'])
    assert np.isclose(second['load_q_kvar'],2*first['load_q_kvar'])
    assert not np.allclose(first['y_base'][:lab.nv],second['y_base'][:lab.nv])
    for key in ['oltc','mpc']:
        assert np.allclose(second['baseline'].load[['bus','p_mw','q_mvar']],second[key].load[['bus','p_mw','q_mvar']])
        assert np.array_equal(second['baseline'].sgen.bus,second[key].sgen.bus)
    assert np.allclose(second['baseline'].sgen[['p_mw','q_mvar']],second['oltc'].sgen[['p_mw','q_mvar']])
    assert np.allclose(second['baseline'].sgen.loc[lab.pv_ids,['p_mw','q_mvar']],second['mpc'].sgen.loc[lab.pv_ids,['p_mw','q_mvar']])
    assert second['baseline'].trafo.loc[lab.tid,'tap_pos']==lab.tap0
    assert second['mpc'].trafo.loc[lab.tid,'tap_pos']==lab.tap0
    chosen=oltc_score(lab,second['y_oltc'],second['tap_pos']);trial=deepcopy(second['baseline'])
    for tap in range(lab.tap_min,lab.tap_max+1):
        trial.trafo.loc[lab.tid,'tap_pos']=tap
        assert chosen<=oltc_score(lab,lab.solve(trial),tap)
    # Independent feature/sign check: +DER P/Q must subtract from net bus consumption.
    p0,q0=net_bus_pq(second['baseline'],controller.bus_ids)
    trial=deepcopy(second['baseline']);bus=controller.der_buses[0];position=controller.bus_ids.index(bus)
    controller.apply_setpoints(trial,{bus:.01},{bus:.02})
    p1,q1=net_bus_pq(trial,controller.bus_ids)
    dp=np.zeros(len(p0));dq=dp.copy();dp[position]=.01;dq[position]=.02
    assert np.allclose(p1,p0-dp) and np.allclose(q1,q0-dq)
    lab.solve(trial)
    assert trial.res_bus.loc[bus,'vm_pu']>second['baseline'].res_bus.loc[bus,'vm_pu']
    session.new_scenario('High PV');frame=session.sample(1,1,preset='High PV')
    p,q=net_bus_pq(frame['baseline'],controller.bus_ids)
    x=np.full(2*controller.n_der,.001);_,jac=controller.objective_with_gradient(x,p,q,(1,.1,1000))
    for i in range(len(x)):
        d=np.zeros_like(x);d[i]=1e-7
        a=controller.objective_with_gradient(x+d,p,q,(1,.1,1000))[0]
        b=controller.objective_with_gradient(x-d,p,q,(1,.1,1000))[0]
        assert np.isclose((a-b)/2e-7,jac[i],atol=1e-5,rtol=1e-5)
    # Disabling the voltage penalty must produce zero minimum-cost action.
    zero_p,zero_q=controller.get_optimal_setpoints(p,q,1,.1,0)
    assert np.allclose(list(zero_p.values()),0) and np.allclose(list(zero_q.values()),0)
    update_configuration('High PV',1,1,1,.1,1000,session)
    assert np.linalg.norm(session.last['actions'][['P (kW)','Q (kVar)']])>1
    assert session.last['flags']['optimizer_action_applied']
    # Evaluate actual controlled AC outcomes, not only predicted voltage or solver success.
    for preset in ['High PV','High Load']:
        lf,pf=PRESETS[preset];lab.rating=lab.base_rating*max(1.,pf);net=lab.new_net()
        for t in lab.scenario_buckets[preset]:
            av=lab.set_profile(net,int(t),lf,pf);lab.control(net,av,np.zeros(2*lab.ng))
            layout=lab.spatial_layouts[preset][int(t)];apply_layout(lab,net,layout);y=lab.solve(net)
            assert np.ptp(y[:lab.nv])>=.115
            if preset=='High PV':
                assert y[:lab.nv].max()>=lab.cfg.v_max+.02
                assert has_voltage_valley(net,layout['path'])
            else:assert y[:lab.nv].min()<=lab.cfg.v_min-.02
        before=[];after=[];oltc_after=[];solver_ok=[]
        bucket=lab.scenario_buckets[preset]
        session.preset=preset
        for t in bucket[np.unique(np.linspace(0,len(bucket)-1,6,dtype=int))]:
            session.t=int(t);session.layout=lab.spatial_layouts[preset][int(t)]
            frame=session.sample(1,1,preset=preset)
            assert frame['load_factor']==lf and frame['pv_factor']==pf
            before.append(voltage_violation(lab,frame['y_base']));after.append(voltage_violation(lab,frame['y_mpc']))
            oltc_after.append(voltage_violation(lab,frame['y_oltc']))
            solver_ok.append(frame['flags']['solver_success'])
            assert frame['flags']['optimizer_action_applied']
            assert np.abs(frame['actions']['P (kW)']).max()<=MAX_P_MW*1000+1e-8
            assert np.abs(frame['actions']['Q (kVar)']).max()<=MAX_Q_MVAR*1000+1e-8
            control_rows=frame['mpc'].sgen[frame['mpc'].sgen.type=='adg_control_der']
            assert np.all(control_rows.p_mw**2+control_rows.q_mvar**2<=control_rows.sn_mva**2+1e-10)
            actual_p,actual_q=net_bus_pq(frame['mpc'],controller.bus_ids)
            expected_p,expected_q=net_bus_pq(frame['baseline'],controller.bus_ids)
            for _,row in frame['actions'].iterrows():
                j=controller.bus_ids.index(int(row['Bus']))
                expected_p[j]-=row['P (kW)']/1000;expected_q[j]-=row['Q (kVar)']/1000
            assert np.allclose(actual_p,expected_p,atol=6e-6) and np.allclose(actual_q,expected_q,atol=6e-6)
        assert np.mean(after)<.8*np.mean(before),(preset,before,after)
        assert sum(solver_ok)>=len(solver_ok)*.75,(preset,solver_ok)
        print(f'{preset}: AC voltage violation uncontrolled {np.mean(before):.5f}, best OLTC {np.mean(oltc_after):.5f}, MPC {np.mean(after):.5f} pu; optimizer {sum(solver_ok)}/{len(solver_ok)} converged: PASS')
        print('Paired violations:',list(zip(before,oltc_after,after)))
        if preset=='High PV':
            assert np.mean(after)<.5*np.mean(oltc_after),(after,oltc_after)
            assert sum(a<o-1e-3 for a,o in zip(after,oltc_after))>=len(after)//2,(after,oltc_after)
        for _ in range(8):
            session.new_scenario(preset);assert session.t in bucket
    months=set()
    for _ in range(250):
        previous=session.t;session.new_scenario('Custom');assert session.t!=previous
        months.add(lab.profile_times[session.t].month)
    assert months==set(range(1,13))
    print('Reference MLP/MPC, net-PQ and actuator signs, autograd, gain response, actual AC improvement, tap benchmark and preset/calendar checks: PASS')


def check_server(lab):
    from dashboard import build_dashboard,launch_dashboard
    from gradio_client import Client
    from urllib.request import build_opener,ProxyHandler
    import json
    app=build_dashboard(lab)
    try:
        url,credentials=launch_dashboard(app,share=False)
        assert credentials is None and app.auth is None
        with build_opener(ProxyHandler({})).open(url+'config') as response:
            assert 'dependencies' in json.load(response) and '/login' not in response.url
        client=Client(url,verbose=False)
        args=['Custom',1,1,1,.1,1000]
        initial=client.predict(*args,api_name='/initial_snapshot')
        assert len(initial)==3 and initial[0]['type']=='plotly'
        assert 'OLTC tap position:' in initial[2]
        plot=json.loads(initial[0]['plot'])
        assert [t['name'] for t in plot['data'][:3]]==['Uncontrolled','Best OLTC','MLP + MPC']
        assert len(set(t.get('yaxis','y') for t in plot['data'][:3]))==1
        paths=feeder_paths(lab)
        labels=[str(b) for path in paths for b in path]+['']
        assert set(plot['data'][0]['customdata'])==set(labels)
        assert plot['data'][0]['y'].count(None)==len(paths)
        scaled=client.predict('Custom',2,1,1,.1,1000,api_name='/react')
        assert initial[1]==scaled[1] and initial[0]!=scaled[0]
        new=client.predict('Custom',2,1,1,.1,1000,api_name='/new_scenario')
        assert new[1]!=scaled[1]
        high=client.predict('High PV',1,1,1,.1,1000,api_name='/react')
        high_new=client.predict('High PV',1,1,1,.1,1000,api_name='/new_scenario')
        assert high_new[1]!=high[1] or len(lab.scenario_buckets['High PV'])==1
        assert app.config['title']=='Voltage control for ADG'
        types=[c['type'] for c in app.config['components']]
        assert types.count('button')==1 and types.count('plot')==1
        assert not {'tabs','tabitem','timer','radio','checkbox'}&set(types)
        print('No-login Gradio, renamed dashboard, explicit OLTC tap, overlaid traces and reactive scenarios: PASS')
        if os.environ.get('GRID_LAB_BROWSER_CHECK')=='1':
            from check_browser import check_browser
            check_browser(url)
    finally:app.close()

if __name__=='__main__':
    lab=GridLab();check_live(lab);check_server(lab)
