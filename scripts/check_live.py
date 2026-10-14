"""Behavioral checks for reactive sliders, evolving PQ, pause, OLTC, and MLP-MPC."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from grid_lab import GridLab
from dashboard import LiveExperiment,update_configuration,tick,toggle_live
from live_control import surrogate_with_jacobian


def check_live(lab=None):
    lab=lab or GridLab()
    if lab.model is None:lab.train(1000)
    session=LiveExperiment(lab)
    first=session.sample(1,1,False,'OLTC')
    first_p=first['baseline'].load.p_mw.sum()
    second=session.sample(2,1,False,'OLTC')
    assert np.isclose(second['baseline'].load.p_mw.sum(),2*first_p)
    assert np.isclose(second['baseline'].load.q_mvar.sum(),2*first['baseline'].load.q_mvar.sum())
    assert not np.allclose(first['y_base'][:lab.nv],second['y_base'][:lab.nv])
    # Derivatives of the actual fitted network are needed for a responsive optimizer.
    x=lab.features(second['baseline']);pred,jac=surrogate_with_jacobian(lab,x)
    assert np.allclose(pred,lab.predict(x)[0],atol=1e-9)
    for i in range(len(x)):
        delta=np.zeros_like(x);delta[i]=lab.xscale.scale_[i]*1e-5
        fd=(surrogate_with_jacobian(lab,x+delta)[0]-surrogate_with_jacobian(lab,x-delta)[0])/2e-5
        assert np.allclose(fd,jac[:,i]*lab.xscale.scale_[i],atol=1e-4,rtol=1e-5)
    update_configuration(.55,3,True,'MLP + MPC',1,.1,1000,session)
    assert session.last['flags']['safe']
    assert (session.last['actions']['P (kW)']<0).any()
    net=session.last['controlled']
    p=net.sgen.loc[lab.pv_ids,'p_mw'].to_numpy();q=net.sgen.loc[lab.pv_ids,'q_mvar'].to_numpy()
    assert np.all(p*p+q*q<=lab.rating**2+1e-10)
    qp=session.last['actions']['Q (kVar)'].to_numpy()
    update_configuration(.55,3,True,'MLP + MPC',.01,100,1000,session)
    assert session.last['flags']['safe']
    assert not np.allclose(qp,session.last['actions']['Q (kVar)'].to_numpy(),atol=.1)
    t=session.t;tick(session);assert session.t==t # paused does not advance
    toggle_live(session);assert session.running
    for _ in range(3):
        before=session.t;tick(session);assert session.t==(before+1)%lab.steps
        actual=session.last['baseline'].load.p_mw.to_numpy()
        expected=lab.profiles[('load','p_mw')].iloc[session.t].reindex(lab.net.load.index).to_numpy()*.55
        assert np.allclose(actual,expected)
    toggle_live(session);assert not session.running
    t=session.t;tick(session);assert session.t==t
    session.sample(1,1,True,'OLTC')
    before=session.last['tap_pos'];session.sample(1,1,True,'OLTC')
    assert abs(session.last['tap_pos']-before)<=1
    session.sample(1,1,False,'OLTC');assert session.last['tap_pos']==lab.tap0
    assert np.allclose(session.last['y_base'],session.last['y_controlled'],atol=1e-9)
    print('Reactive PQ, network Jacobian, safe live MPC, gains, timer, pause, and OLTC state: PASS')

if __name__=='__main__':check_live()


def check_server(lab):
    """Exercise the real Gradio event/state transport without a public tunnel."""
    from dashboard import build_dashboard,launch_dashboard
    from gradio_client import Client
    app=build_dashboard(lab)
    try:
        url,auth=launch_dashboard(app,share=False,auth=('ci','local-test-only'))
        client=Client(url,auth=auth,verbose=False)
        args=[1,1,True,'MLP + MPC',1,.1,1000]
        initial=client.predict(*args,api_name='/initial_snapshot')
        assert len(initial)==6 and initial[0]['type']=='plotly'
        scaled=client.predict(2,1,True,'MLP + MPC',1,.1,1000,api_name='/react')
        assert initial[4]!=scaled[4]
        caption=scaled[4]
        client.predict(api_name='/play_pause')
        advanced=client.predict(api_name='/tick')
        assert advanced[4]!=caption
        client.predict(api_name='/play_pause')
        client.predict(api_name='/tick')
        held=client.predict(2,1,True,'MLP + MPC',1,.1,1000,api_name='/react')
        assert held[4]==advanced[4]
        types=[c['type'] for c in app.config['components']]
        assert types.count('plot')==3 and types.count('dataframe')==1
        assert 'tabs' not in types and 'tabitem' not in types
        print('Authenticated Gradio initial render, slider reaction, live event transport, pause, and simple layout: PASS')
    finally:app.close()
