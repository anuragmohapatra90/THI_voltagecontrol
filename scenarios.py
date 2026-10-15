"""Dated SimBench PQ, random spatial layouts, and AC-screened stress buckets."""
import networkx as nx
import numpy as np
import pandapower as pp
from grid_lab import PRESETS


def feeder_paths(lab):
    """Actual LV root-to-leaf paths; bus-number order is not electrical distance."""
    if hasattr(lab,'feeder_paths'):return lab.feeder_paths
    graph=nx.Graph()
    for _,r in lab.net.line[lab.net.line.in_service].iterrows():
        graph.add_edge(int(r.from_bus),int(r.to_bus),weight=float(r.length_km))
    graph.add_nodes_from(map(int,lab.bus_ids))
    if not nx.is_tree(graph):raise ValueError('Spatial scenarios require a connected radial LV feeder.')
    root=lab.lv_bus
    paths=[nx.shortest_path(graph,root,b) for b in graph if graph.degree(b)==1 and b!=root]
    paths.sort(key=lambda p:sum(graph[a][b]['weight'] for a,b in zip(p,p[1:])),reverse=True)
    if not paths or len(paths[0])<5:raise ValueError('Choose a feeder with an interior and downstream injection location.')
    lab.feeder_paths=paths
    return paths


def control_locations(lab):
    """Four fixed sites: long feeder interior/end and two other feeder ends."""
    paths=feeder_paths(lab);long=paths[0]
    return sorted(set([long[-3],long[-1]]+[p[-1] for p in paths[1:3]]))


def random_layout(lab,rng,preset='Custom'):
    """Relocate existing PQ elements; preserve each element's dated P, Q and rating."""
    paths=feeder_paths(lab);focus=paths[0]
    middle=int(rng.choice(focus[1:-2]));tail=int(rng.choice(focus[-2:]))
    other=[p[-1] for p in paths[1:]]
    # Concentrate demand at an interior site plus another branch. Local PV at the
    # far end produces reverse flow on the distal section, not a global V offset.
    load_other=int(rng.choice(other[:2]))
    load_buses=rng.choice([middle,load_other],size=len(lab.net.load),p=[.65,.35]).astype(int)
    if preset=='High Load':
        # Little sunlight: concentrate demand downstream, stressing the voltage
        # spread across branches rather than inventing night-time solar power.
        load_buses=rng.choice([tail,middle,load_other],size=len(lab.net.load),p=[.55,.25,.20]).astype(int)
    pv_other=[b for b in other if b!=load_other] or other
    pv_buses=rng.choice(pv_other,size=len(lab.pv_ids)).astype(int)
    largest=int(np.argmax(lab.net.sgen.loc[lab.pv_ids,'p_mw'].to_numpy()))
    pv_buses[largest]=tail
    return {'load_buses':load_buses.tolist(),'pv_buses':pv_buses.tolist(),
            'path':list(focus),'load_site':middle,'pv_site':tail}


def apply_layout(lab,net,layout):
    net.load.loc[lab.net.load.index,'bus']=layout['load_buses']
    net.sgen.loc[lab.pv_ids,'bus']=layout['pv_buses']
    return net


def has_voltage_valley(net,path,depth=.003,recovery=.02):
    v=net.res_bus.loc[path,'vm_pu'].to_numpy()
    j=int(np.argmin(v))
    return 0<j<len(v)-1 and v[0]-v[j]>=depth and v[-1]-v[j]>=recovery


def prepare_buckets(lab,candidates=256):
    if hasattr(lab,'scenario_buckets'):return lab.scenario_buckets
    feeder_paths(lab)
    pv=lab.profiles[('sgen','p_mw')].reindex(columns=lab.pv_ids).sum(axis=1).to_numpy()
    load=lab.profiles[('load','p_mw')].sum(axis=1).to_numpy()
    pools={'High PV':np.flatnonzero(pv>=np.quantile(pv,.90)),
           'High Load':np.flatnonzero((load>=np.quantile(load,.90)) & (pv<=.1*pv.max()))}
    buckets={};layouts={};original_rating=lab.rating.copy()
    rng=np.random.default_rng(lab.cfg.seed+101)
    try:
        for name,pool in pools.items():
            if not len(pool):raise ValueError(f'No full-year profiles qualify for {name}.')
            selected=pool[np.unique(np.linspace(0,len(pool)-1,min(candidates,len(pool)),dtype=int))]
            signal=pv if name=='High PV' else load
            selected=np.unique(np.r_[selected,pool[np.argsort(signal[pool])[-16:]]])
            lf,pf=PRESETS[name];lab.rating=lab.base_rating*max(1.,pf)
            net=lab.new_net();valid=[];layouts[name]={}
            for t in selected:
                av=lab.set_profile(net,int(t),lf,pf);lab.control(net,av,np.zeros(2*lab.ng))
                for _ in range(4):
                    layout=random_layout(lab,rng,name);apply_layout(lab,net,layout)
                    try:
                        y=lab.solve(net);v=y[:lab.nv]
                        stressed=(v.max()>=lab.cfg.v_max+.02 and has_voltage_valley(net,layout['path'])) if name=='High PV' else (v.min()<=lab.cfg.v_min-.02)
                        # Screen using neutral AC voltages only, never MPC performance.
                        # Keep a spatial spread, avoiding near-collapse examples.
                        if stressed and v.max()-v.min()>=.115 and v.min()>=.78:
                            valid.append(int(t));layouts[name][int(t)]=layout;break
                    except (pp.LoadflowNotConverged,ValueError):pass
            if not valid:raise ValueError(f'No converged spatially stressed {name} scenarios. Adjust PRESETS for this grid.')
            buckets[name]=np.asarray(valid,dtype=int)
    finally:lab.rating=original_rating
    buckets['Custom']=np.arange(lab.steps)
    lab.spatial_layouts=layouts;lab.scenario_buckets=buckets
    return buckets
