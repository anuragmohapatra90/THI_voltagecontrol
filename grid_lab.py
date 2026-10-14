"""Balanced AC SimBench experiment. MLP surrogate + receding-horizon inverter MPC."""
from dataclasses import dataclass, asdict
from copy import deepcopy
import warnings
import os
import numpy as np
import pandas as pd
import pandapower as pp
import simbench as sb
from scipy.optimize import minimize
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

PRESETS = {
    'High PV': (0.55, 8.0),
    'High Load': (7.0, 0.10),
    'Custom': (1.0, 1.0),
}
MODES = ['Uncontrolled', 'Automatic OLTC', 'MLP + MPC']

@dataclass(frozen=True)
class Config:
    grid_code: str = '1-LV-rural1--0-sw'
    retrofit_oltc: bool = True
    groups: int = 3
    inverter_oversize: float = 1.10
    v_min: float = 0.95
    v_max: float = 1.05
    thermal_limit: float = 100.0
    oltc_low: float = 0.99
    oltc_high: float = 1.01
    oltc_delay_steps: int = 1
    horizon: int = 4
    max_curtailment: float = 1.0
    slew: float = 1.0
    forecast: str = 'Perfect'
    seed: int = 42
    load_scale_max: float = 8.0
    pv_scale_max: float = 10.0

class GridLab:
    def __init__(self, cfg=Config()):
        self.cfg = cfg
        self.net = sb.get_simbench_net(cfg.grid_code)
        if len(self.net.trafo) != 1 or len(self.net.trafo3w):
            raise ValueError('This first project supports one two-winding transformer: choose an LV grid.')
        if len(self.net.gen) or len(self.net.storage):
            raise ValueError('Choose an LV grid without gen/storage elements; their controls are not implemented.')
        if sb.profiles_are_missing(self.net):
            raise ValueError('SimBench profile data are missing.')
        self.profiles = sb.get_absolute_values(self.net, profiles_instead_of_study_cases=True)
        self.profile_times = pd.DatetimeIndex(pd.to_datetime(
            self.net.profiles['load']['time'], format='%d.%m.%Y %H:%M'))
        if len(self.profile_times) != len(self.profiles[('load','p_mw')]):
            raise ValueError('SimBench calendar and absolute profiles do not align.')
        # Retain absolute profiles separately; do not copy a full-year dataset per power flow.
        self.net.pop('profiles', None)
        self.tid = self.net.trafo.index[0]
        tr = self.net.trafo.loc[self.tid]
        self.original_tap = {k: str(tr.get(k)) for k in ['tap_side','tap_pos','tap_min','tap_max','tap_step_percent','oltc']}
        self.retrofit = False
        missing_tap = any(pd.isna(tr.get(k)) for k in ['tap_side','tap_neutral','tap_min','tap_max','tap_step_percent'])
        fixed_tap = missing_tap or float(tr.get('tap_step_percent',0) or 0) <= 0 or not bool(tr.get('oltc',False))
        if fixed_tap and not cfg.retrofit_oltc:
            raise ValueError('This transformer has no declared OLTC. Enable the explicit simulated retrofit.')
        if fixed_tap:
            self.retrofit = True
            for k,v in dict(tap_side='hv',tap_neutral=0,tap_min=-8,tap_max=8,tap_step_percent=1.25,tap_step_degree=0,oltc=True).items():
                self.net.trafo.loc[self.tid,k] = v
        self.net.trafo.loc[self.tid,'tap_changer_type'] = 'Ratio'
        self.net.trafo.loc[self.tid,'tap_pos'] = self.net.trafo.loc[self.tid,'tap_neutral']
        self.tap0 = int(self.net.trafo.loc[self.tid,'tap_pos'])
        self.tap_min = int(self.net.trafo.loc[self.tid,'tap_min'])
        self.tap_max = int(self.net.trafo.loc[self.tid,'tap_max'])
        self.lv_bus = int(self.net.trafo.loc[self.tid,'lv_bus'])
        self.bus_ids = self.net.bus.index[self.net.bus.in_service & (self.net.bus.vn_kv < 1)].to_numpy()
        self.line_ids = self.net.line.index[self.net.line.in_service].to_numpy()
        self.trafo_ids = self.net.trafo.index[self.net.trafo.in_service].to_numpy()
        types = self.net.sgen.type.fillna('').astype(str).str.lower()
        self.pv_ids = self.net.sgen.index[self.net.sgen.in_service & types.str.contains('pv|solar')].to_numpy()
        if not len(self.pv_ids):
            raise ValueError('No PV-labelled sgens in this grid. Choose another LV grid.')
        self.ng = min(cfg.groups,len(self.pv_ids))
        self.group = np.arange(len(self.pv_ids)) % self.ng
        # Ratings are fixed across controllers in each scenario. PV factors >1 represent added installed PV/inverter capacity.
        rated = self.net.sgen.loc[self.pv_ids,'p_mw'].to_numpy(float)
        sn = self.net.sgen.loc[self.pv_ids,'sn_mva'].to_numpy(float)
        self.base_rating = np.maximum(np.nan_to_num(sn), rated*cfg.inverter_oversize)
        self.rating = self.base_rating.copy()
        for column in ['const_z_p_percent','const_i_p_percent','const_z_q_percent','const_i_q_percent']:
            if column in self.net.load: self.net.load[column] = self.net.load[column].fillna(0.0)
        self.net.load.loc[:,'scaling'] = 1.0
        self.net.sgen.loc[:,'scaling'] = 1.0
        self.steps = len(self.profiles[('load','p_mw')])
        self.nv = len(self.bus_ids)
        self.model = None
        self.training_report = None

    def new_net(self):
        return deepcopy(self.net)

    def set_profile(self, net, t, load_factor, pv_factor):
        for (elm,col), frame in self.profiles.items():
            if elm not in ['load','sgen'] or col not in ['p_mw','q_mvar'] or not len(net[elm]):
                continue
            row = frame.iloc[int(t) % self.steps].reindex(net[elm].index).to_numpy(float)
            if not np.isfinite(row).all():
                raise ValueError(f'Nonfinite {elm}.{col} profile at {t}.')
            net[elm].loc[:,col] = row * (load_factor if elm == 'load' else 1.0)
        # Scenario changes available PV; its inverter rating is fixed across compared controllers.
        net.sgen.loc[self.pv_ids,'p_mw'] *= pv_factor
        net.sgen.loc[self.pv_ids,'q_mvar'] = 0.0
        available = net.sgen.loc[self.pv_ids,'p_mw'].to_numpy(float)
        return available

    def control(self, net, available, u):
        u = np.asarray(u,float)
        a, r = u[:self.ng][self.group], u[self.ng:][self.group]
        # Available irradiance power may exceed inverter rating; clipping is not commanded curtailment.
        feasible_p = np.minimum(available,self.rating)
        p = feasible_p*(1-a)
        qcap = np.sqrt(np.maximum(self.rating**2-p**2,0))
        q = r*qcap  # pandapower sgen: positive q injects reactive power.
        net.sgen.loc[self.pv_ids,'p_mw'] = p
        net.sgen.loc[self.pv_ids,'q_mvar'] = q
        return dict(curtailed_mw=float(np.sum(feasible_p-p)), clipped_mw=float(np.sum(available-feasible_p)), q_mvar=float(q.sum()))

    def features(self, net):
        return np.r_[net.load[['p_mw','q_mvar']].to_numpy().ravel(),
                     net.sgen[['p_mw','q_mvar']].to_numpy().ravel(),
                     net.trafo.loc[self.tid,'tap_pos']]

    def solve(self, net):
        pp.runpp(net, algorithm='nr', init='dc', calculate_voltage_angles=True, voltage_depend_loads=False,
                 max_iteration=30, numba=False, check_connectivity=True)
        y = np.r_[net.res_bus.loc[self.bus_ids,'vm_pu'].to_numpy(),
                  net.res_line.loc[self.line_ids,'loading_percent'].to_numpy(),
                  net.res_trafo.loc[self.trafo_ids,'loading_percent'].to_numpy()]
        if not np.isfinite(y).all():
            raise ValueError('Disconnected or nonfinite monitored results; this run is invalid.')
        return y

    def violation(self,y):
        v,thermal = y[:self.nv],y[self.nv:]
        return max(float(np.max(self.cfg.v_min-v)),float(np.max(v-self.cfg.v_max)),
                   float(np.max(thermal/self.cfg.thermal_limit-1)),0.)

    def train(self, samples=1000, progress=None):
        if samples < 100:
            raise ValueError('Use at least 100 successful samples.')
        rng=np.random.default_rng(self.cfg.seed)
        net=self.new_net(); xs=[]; ys=[]; failed=0
        # Enrich rare PV/load peaks so stressful daytime operation is represented.
        pvtotal=self.profiles[('sgen','p_mw')].reindex(columns=self.pv_ids).sum(axis=1).to_numpy()
        loadtotal=self.profiles[('load','p_mw')].sum(axis=1).to_numpy()
        pvpool=np.flatnonzero(pvtotal>=np.quantile(pvtotal,.90))
        loadpool=np.flatnonzero(loadtotal>=np.quantile(loadtotal,.90))
        for i in range(samples):
            t=int(rng.choice(pvpool if i%3==0 else loadpool)) if i%3!=2 else int(rng.integers(self.steps))
            pvfactor=rng.uniform(.0,self.cfg.pv_scale_max)
            self.rating=self.base_rating*max(1.0,pvfactor)
            available=self.set_profile(net,t,rng.uniform(.3,self.cfg.load_scale_max),pvfactor)
            net.trafo.loc[self.tid,'tap_pos']=rng.integers(self.tap_min,self.tap_max+1)
            a=rng.uniform(0,1,self.ng)
            if i%5==0: a[:]=0
            if i%7==0: a[:]=1
            u=np.r_[a,rng.uniform(-1,1,self.ng)]
            self.control(net,available,u)
            try:
                y=self.solve(net)
                xs.append(self.features(net)); ys.append(y)
            except (pp.LoadflowNotConverged,ValueError):
                failed+=1
            if progress and (i%50==0 or i==samples-1): progress((i+1)/samples)
        self.rating=self.base_rating.copy()
        if len(xs)<100: raise ValueError('Too few converged training samples.')
        x=np.asarray(xs); y=np.asarray(ys)
        # Separate train / calibration / final test: test errors never set the guard margin.
        xtr,xother,ytr,yother=train_test_split(x,y,test_size=.30,random_state=self.cfg.seed)
        xc,xt,yc,yt=train_test_split(xother,yother,test_size=.5,random_state=self.cfg.seed+1)
        self.xscale=StandardScaler().fit(xtr); self.yscale=StandardScaler().fit(ytr)
        self.model=MLPRegressor(hidden_layer_sizes=(64,64),activation='tanh',solver='lbfgs',
                               max_iter=800,alpha=.001,random_state=self.cfg.seed,tol=1e-7)
        with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            sx=self.xscale.transform(xtr); sy=self.yscale.transform(ytr)
            self.linear=Ridge(alpha=.01).fit(sx,sy)
            self.model.fit(sx, sy-self.linear.predict(sx))
        self.xlo=xtr.min(axis=0); self.xhi=xtr.max(axis=0)
        ec=np.abs(self.predict(xc)-yc); et=np.abs(self.predict(xt)-yt)
        self.margin_v=max(.002,float(np.quantile(ec[:,:self.nv].max(axis=1),.95)))
        self.margin_t=max(2.,float(np.quantile(ec[:,self.nv:].max(axis=1),.95)))
        self.training_report={
            'successful_samples':len(x),'failed_power_flows':failed,
            'train_samples':len(xtr),'calibration_samples':len(xc),'test_samples':len(xt),
            'voltage_test_mae_pu':float(et[:,:self.nv].mean()),
            'voltage_test_max_error_pu':float(et[:,:self.nv].max()),
            'thermal_test_mae_percentage_points':float(et[:,self.nv:].mean()),
            'guard_voltage_pu':self.margin_v,'guard_thermal_percentage_points':self.margin_t,
            'training_convergence_warning':any(isinstance(w.message,ConvergenceWarning) for w in caught),
        }
        return self.training_report

    def predict(self,x):
        with threadpool_limits(limits=1):
            sx=self.xscale.transform(np.atleast_2d(x))
            return self.yscale.inverse_transform(self.linear.predict(sx)+self.model.predict(sx))

    def oltc(self,net,counter):
        y=self.solve(net)
        v=float(net.res_bus.loc[self.lv_bus,'vm_pu'])
        if self.cfg.oltc_low<=v<=self.cfg.oltc_high: return y,0,0,'Within OLTC deadband'
        counter+=1
        if counter<self.cfg.oltc_delay_steps: return y,counter,0,'OLTC delay'
        tap=int(net.trafo.loc[self.tid,'tap_pos']); target=(self.cfg.oltc_low+self.cfg.oltc_high)/2
        best=abs(v-target); newtap=tap
        # Test directions physically: works with either HV-side or LV-side tap convention.
        for trial in [tap-1,tap+1]:
            if not self.tap_min<=trial<=self.tap_max: continue
            net.trafo.loc[self.tid,'tap_pos']=trial
            try:
                self.solve(net); score=abs(float(net.res_bus.loc[self.lv_bus,'vm_pu'])-target)
                if score<best: best=score; newtap=trial
            except (pp.LoadflowNotConverged,ValueError): pass
        net.trafo.loc[self.tid,'tap_pos']=newtap
        return self.solve(net),0,abs(newtap-tap),'Tap moved' if newtap!=tap else 'Tap limit / no improving move'

    def mpc(self,net,t,lf,pf,previous):
        if self.model is None: raise ValueError('Train the MLP first.')
        h=self.cfg.horizon; dim=2*self.ng
        fnet=self.new_net(); fnet.trafo.loc[self.tid,'tap_pos']=net.trafo.loc[self.tid,'tap_pos']
        forecasts=[]
        for k in range(h):
            ft=t+k if self.cfg.forecast=='Perfect' else t
            av=self.set_profile(fnet,ft,lf,pf)
            forecasts.append((deepcopy(fnet),av))
        def outputs(z):
            features=[]
            for k,(fn,av) in enumerate(forecasts):
                self.control(fn,av,z.reshape(h,dim)[k]); features.append(self.features(fn))
            return self.predict(np.asarray(features))
        def objective(z):
            u=z.reshape(h,dim); y=outputs(z); v=y[:,:self.nv]; th=y[:,self.nv:]
            viol=np.maximum(self.cfg.v_min+self.margin_v-v,0)**2+np.maximum(v-self.cfg.v_max+self.margin_v,0)**2
            dv=np.diff(np.vstack([previous,u]),axis=0)
            return float(np.mean((v-1)**2)+5000*np.sum(viol)
                         +10*np.sum(np.maximum(th/self.cfg.thermal_limit-1,0)**2)
                         +.06*np.mean(u[:,:self.ng])+ .01*np.mean(u[:,self.ng:]**2)+.005*np.mean(dv**2)
                         +.003*np.mean((th/100)**2))
        def constraints(z):
            u=z.reshape(h,dim); y=outputs(z); v=y[:,:self.nv]; th=y[:,self.nv:]
            du=np.diff(np.vstack([previous,u]),axis=0)
            return np.r_[(v-self.cfg.v_min-self.margin_v).ravel(),
                         (self.cfg.v_max-self.margin_v-v).ravel(),
                         ((self.cfg.thermal_limit-self.margin_t-th)/100).ravel(),
                         (self.cfg.slew-np.abs(du)).ravel()]
        bounds=([(0,self.cfg.max_curtailment)]*self.ng+[(-1,1)]*self.ng)*h
        starts=[np.tile(previous,h),np.tile(np.r_[np.full(self.ng,min(.5,self.cfg.max_curtailment)),np.zeros(self.ng)],h)]
        initial=min(starts,key=lambda z:float(np.sum(np.minimum(constraints(z),0)**2)))
        result=minimize(objective,initial,method='SLSQP',bounds=bounds,
                        constraints=[{'type':'ineq','fun':constraints}],
                        options={'maxiter':100,'ftol':1e-5})
        proposal=result.x.reshape(h,dim)[0] if np.isfinite(result.x).all() else previous
        fn,av=forecasts[0]; self.control(fn,av,proposal); fx=self.features(fn)
        span=np.maximum(self.xhi-self.xlo,1e-8)
        ood=bool(np.any(fx<self.xlo-.1*span) or np.any(fx>self.xhi+.1*span))
        # First-action AC screening; bounded backup candidates when the MLP/solver fails.
        candidates=[proposal,previous,np.zeros(dim)] if not ood and result.success else [previous,np.zeros(dim)]
        for a in [0,.25,.5,self.cfg.max_curtailment]:
            for r in [-1,0,1]: candidates.append(np.r_[np.full(self.ng,min(a,self.cfg.max_curtailment)),np.full(self.ng,r)])
        current_av=self.set_profile(net,t,lf,pf)
        evaluated=[]
        for idx,u in enumerate(candidates):
            u=np.clip(u,np.maximum(np.r_[np.zeros(self.ng),-np.ones(self.ng)],previous-self.cfg.slew),
                       np.minimum(np.r_[np.full(self.ng,self.cfg.max_curtailment),np.ones(self.ng)],previous+self.cfg.slew))
            info=self.control(net,current_av,u)
            try:
                y=self.solve(net); viol=self.violation(y)
                cost=float(np.mean(u[:self.ng])+.02*np.mean(u[self.ng:]**2))
                evaluated.append((viol,cost,u.copy(),y.copy(),info,idx))
                if idx==0 and result.success and not ood and viol<=1e-7: break
            except (pp.LoadflowNotConverged,ValueError): pass
        if not evaluated: raise ValueError('All AC-screened actions failed to converge.')
        chosen=min(evaluated,key=lambda e:(e[0]>1e-7,e[0] if e[0]>1e-7 else e[1]))
        viol,_,u,y,info,idx=chosen
        self.control(net,current_av,u); self.solve(net)
        accepted=(idx==0 and result.success and not ood)
        status='MPC / AC verified' if accepted else 'AC-screened backup'
        if viol>1e-7: status+=' / residual violation'
        return y,u,info,status,dict(solver_success=bool(result.success),solver_message=str(result.message),out_of_domain=ood)

    def run(self,mode='Automatic OLTC',preset='Custom',start=0,steps=24,custom=None,progress=None):
        if mode not in MODES: raise ValueError(mode)
        lf,pf=custom if custom is not None else PRESETS[preset]
        self.rating=self.base_rating*max(1.0,pf)
        net=self.new_net(); rows=[]; voltages=[]; actions=[]; previous=np.zeros(2*self.ng); counter=0
        for i in range(steps):
            t=(start+i)%self.steps; available=self.set_profile(net,t,lf,pf)
            extra=dict(solver_success=None,solver_message='',out_of_domain=False)
            moves=0; u=np.zeros(2*self.ng)
            try:
                info=self.control(net,available,u)
                if mode=='MLP + MPC': y,u,info,status,extra=self.mpc(net,t,lf,pf,previous)
                elif mode=='Automatic OLTC': y,counter,moves,status=self.oltc(net,counter)
                else: y=self.solve(net); status='Fixed neutral tap'
                previous=u
                v=y[:self.nv]; th=y[self.nv:]; viol=self.violation(y)
                rows.append(dict(step=i,profile_step=t,hours=i*.25,controller=mode,v_min_pu=float(v.min()),v_max_pu=float(v.max()),
                                 max_loading_percent=float(th.max()),safe=viol<=1e-7,converged=True,
                                 voltage_violating_buses=int(np.sum((v<self.cfg.v_min)|(v>self.cfg.v_max))),
                                 overloaded_elements=int(np.sum(th>self.cfg.thermal_limit)),
                                 tap_pos=int(net.trafo.loc[self.tid,'tap_pos']),tap_moves=moves,status=status,**info,**extra))
                voltages.append(v.copy()); actions.append(u.copy())
            except (pp.LoadflowNotConverged,ValueError) as exc:
                rows.append(dict(step=i,profile_step=t,hours=i*.25,controller=mode,safe=False,converged=False,status=str(exc)))
                voltages.append(np.full(self.nv,np.nan)); actions.append(np.full(2*self.ng,np.nan))
            if progress: progress((i+1)/steps)
        return pd.DataFrame(rows),pd.DataFrame(voltages,columns=[str(b) for b in self.bus_ids]),pd.DataFrame(actions,columns=[f'curtail_g{i}' for i in range(self.ng)]+[f'q_fraction_g{i}' for i in range(self.ng)])

    def metadata(self):
        return {'source_repository':os.environ.get('GRID_LAB_SOURCE_REPOSITORY'),
                'source_commit':os.environ.get('GRID_LAB_COMMIT'),
                'config':asdict(self.cfg),'oltc_retrofit':self.retrofit,'original_tap':self.original_tap,
                'pv_sgen_indices':self.pv_ids.tolist(),'pv_group':self.group.tolist(),'inverter_rating_mva':self.rating.tolist(),
                'monitored_bus_indices':self.bus_ids.tolist(),'training':self.training_report,
                'pv_capacity_rule':'factor > 1 adds installed PV/inverter capacity; factor <= 1 changes availability only; ratings are identical across controllers within a scenario',
                'profile_step_minutes':15,'forecast_note':'Perfect forecast uses future SimBench values; Persistence holds the current injections.'}
