"""Physics and control checks; no fabricated grids or forecasts."""
import numpy as np
from grid_lab import GridLab, Config

def run_checks(lab=None):
    lab=lab or GridLab()
    net=lab.new_net(); available=lab.set_profile(net,48,1,1)
    lab.rating=lab.base_rating.copy()
    lab.control(net,available,np.zeros(2*lab.ng)); y=lab.solve(net)
    net.trafo.loc[lab.tid,'tap_pos']=lab.tap0+1
    y1=lab.solve(net)
    delta=float(np.max(np.abs(y1[:lab.nv]-y[:lab.nv])))
    assert delta>1e-4, 'Tap positions must affect voltage.'
    for a in [0,.5,1]:
        for q in [-1,0,1]:
            u=np.r_[np.full(lab.ng,a),np.full(lab.ng,q)]
            info=lab.control(net,available,u)
            p=net.sgen.loc[lab.pv_ids,'p_mw'].to_numpy()
            reactive=net.sgen.loc[lab.pv_ids,'q_mvar'].to_numpy()
            assert np.all(p*p+reactive*reactive<=lab.rating**2+1e-10)
            assert np.all(p>=0)
    for mode in ['Uncontrolled','Automatic OLTC']:
        d,v,a=lab.run(mode,steps=4,start=48)
        assert d.converged.all()
        assert (d.tap_moves<=1).all()
        assert d.tap_pos.between(lab.tap_min,lab.tap_max).all()
        assert v.shape==(4,lab.nv)
        expected=(d.v_min_pu>=lab.cfg.v_min)&(d.v_max_pu<=lab.cfg.v_max)&(d.max_loading_percent<=lab.cfg.thermal_limit)
        assert (d.safe==expected).all()
    if lab.model is not None:
        d,_,a=lab.run('MLP + MPC',steps=2,start=48)
        assert d.converged.all()
        assert (d.tap_pos==lab.tap0).all()
        assert (a.iloc[:,:lab.ng].to_numpy()>=-1e-8).all()
        assert (a.iloc[:,:lab.ng].to_numpy()<=lab.cfg.max_curtailment+1e-8).all()
        assert (np.abs(a.iloc[:,lab.ng:].to_numpy())<=1+1e-8).all()
    return {'tap_voltage_change_pu':delta,'inverter_capability':'passed','OLTC_step_and_bounds':'passed','PQ_power_flow':'passed','MLP_MPC':'passed' if lab.model is not None else 'train first'}

if __name__=='__main__': print(run_checks())
