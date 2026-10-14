"""Full-year SimBench profile buckets, AC-screened for substantial voltage stress."""
import numpy as np
import pandapower as pp
from grid_lab import PRESETS


def prepare_buckets(lab,candidates=256):
    if hasattr(lab,'scenario_buckets'):return lab.scenario_buckets
    pv=lab.profiles[('sgen','p_mw')].reindex(columns=lab.pv_ids).sum(axis=1).to_numpy()
    load=lab.profiles[('load','p_mw')].sum(axis=1).to_numpy()
    pools={
        'High PV':np.flatnonzero(pv>=np.quantile(pv,.90)),
        'High Load':np.flatnonzero((load>=np.quantile(load,.90)) & (pv<=.1*pv.max())),
    }
    buckets={};original_rating=lab.rating.copy()
    try:
        for name,pool in pools.items():
            if not len(pool):raise ValueError(f'No full-year profiles qualify for {name}.')
            # An evenly distributed candidate subset keeps Colab startup lightweight.
            selected=pool[np.unique(np.linspace(0,len(pool)-1,min(candidates,len(pool)),dtype=int))]
            # Include the annual profile extremes even if temporal spacing misses them.
            signal=pv if name=='High PV' else load
            selected=np.unique(np.r_[selected,pool[np.argsort(signal[pool])[-16:]]])
            lf,pf=PRESETS[name];lab.rating=lab.base_rating*max(1.,pf)
            net=lab.new_net();valid=[]
            for t in selected:
                av=lab.set_profile(net,int(t),lf,pf)
                lab.control(net,av,np.zeros(2*lab.ng))
                try:
                    y=lab.solve(net);v=y[:lab.nv]
                    stressed=(v.max()>=lab.cfg.v_max+.02) if name=='High PV' else (v.min()<=lab.cfg.v_min-.02)
                    if stressed:valid.append(int(t))
                except (pp.LoadflowNotConverged,ValueError):pass
            if not valid:
                raise ValueError(f'No converged voltage-stressed {name} scenarios. Adjust PRESETS for this grid.')
            buckets[name]=np.asarray(valid,dtype=int)
    finally:lab.rating=original_rating
    buckets['Custom']=np.arange(lab.steps)
    lab.scenario_buckets=buckets
    return buckets
