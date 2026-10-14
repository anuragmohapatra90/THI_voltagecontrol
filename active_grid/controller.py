"""ADG_datadriven's direct-PQ, voltage-penalty MPC on the actual SimBench buses."""
import numpy as np
import torch
import pandapower as pp
from scipy.optimize import minimize

MAX_P_MW=.05
MAX_Q_MVAR=.05
DER_RATING_MVA=.10


def attach_der(net,der_buses):
    existing=net.sgen[net.sgen.type.fillna('')=='adg_control_der']
    ids={int(r.bus):i for i,r in existing.iterrows()}
    for b in der_buses:
        if b not in ids:
            ids[b]=pp.create_sgen(net,bus=b,p_mw=0,q_mvar=0,sn_mva=DER_RATING_MVA,
                name=f'Control DER {b}',type='adg_control_der')
    return ids


def apply_der_setpoints(net,der_buses,p_set,q_set):
    ids=attach_der(net,der_buses)
    for b,i in ids.items():net.sgen.loc[i,['p_mw','q_mvar']]=[p_set.get(b,0.),q_set.get(b,0.)]
    return ids


def net_bus_pq(net,bus_ids):
    """Consumption positive: load minus generation, with explicit bus-ID alignment."""
    load=net.load[net.load.in_service].groupby('bus')[['p_mw','q_mvar']].sum()
    gen=net.sgen[net.sgen.in_service].groupby('bus')[['p_mw','q_mvar']].sum()
    pq=load.reindex(bus_ids,fill_value=0)-gen.reindex(bus_ids,fill_value=0)
    return pq.p_mw.to_numpy(float),pq.q_mvar.to_numpy(float)


def voltage_violation(lab,y):
    v=y[:lab.nv]
    return max(float(np.max(lab.cfg.v_min-v)),float(np.max(v-lab.cfg.v_max)),0.)


class MLP_MPC_Controller:
    """Same estimator, net-PQ input, signed DER actions and objective as the reference."""
    def __init__(self,model,scaler_X,scaler_y,bus_ids,der_buses,v_min=.95,v_max=1.05):
        self.model=model.eval().double()
        self.scaler_X=scaler_X;self.scaler_y=scaler_y
        self.bus_ids=list(map(int,bus_ids));self.der_buses=list(map(int,der_buses))
        self.n_der=len(self.der_buses);self.n_bus=len(self.bus_ids)
        self.positions=np.asarray([self.bus_ids.index(b) for b in self.der_buses])
        self.v_min=v_min;self.v_max=v_max
        self.x_mean=torch.tensor(scaler_X.mean_,dtype=torch.float64)
        self.x_scale=torch.tensor(scaler_X.scale_,dtype=torch.float64)
        self.y_mean=torch.tensor(scaler_y.mean_,dtype=torch.float64)
        self.y_scale=torch.tensor(scaler_y.scale_,dtype=torch.float64)
        self.action_map=torch.zeros((2*self.n_bus,2*self.n_der),dtype=torch.float64)
        for i,pos in enumerate(self.positions):
            self.action_map[pos,i]=-1.
            self.action_map[self.n_bus+pos,self.n_der+i]=-1.
        self.last_result=None

    def _predict(self,features):
        return self.model((features-self.x_mean)/self.x_scale)*self.y_scale+self.y_mean

    def predict_voltages(self,load_p,load_q,der_p_setpoints,der_q_setpoints):
        base=torch.tensor(np.r_[load_p,load_q],dtype=torch.float64)
        action=torch.tensor(np.r_[der_p_setpoints,der_q_setpoints],dtype=torch.float64)
        with torch.no_grad():return self._predict(base+self.action_map@action).numpy().flatten()

    def objective_with_gradient(self,x,load_p,load_q,weights):
        """Same objective; autograd avoids float32 finite-difference zero gradients."""
        wp,wq,wv=map(float,weights)
        action=torch.tensor(x,dtype=torch.float64,requires_grad=True)
        base=torch.tensor(np.r_[load_p,load_q],dtype=torch.float64)
        predicted=self._predict(base+self.action_map@action)
        low=torch.clamp(self.v_min-predicted,min=0)
        high=torch.clamp(predicted-self.v_max,min=0)
        cost=wp*torch.sum(action[:self.n_der]**2)+wq*torch.sum(action[self.n_der:]**2)
        cost+=wv*torch.sum(low**2+high**2)
        gradient=torch.autograd.grad(cost,action)[0]
        return float(cost.detach()),gradient.detach().numpy()

    def get_optimal_setpoints(self,current_load_p,current_load_q,weight_p=1.,weight_q=.1,weight_v=1000.):
        weights=(weight_p,weight_q,weight_v)
        x0=np.zeros(2*self.n_der)
        bounds=[(-MAX_P_MW,MAX_P_MW)]*self.n_der+[(-MAX_Q_MVAR,MAX_Q_MVAR)]*self.n_der
        self.last_result=minimize(self.objective_with_gradient,x0,
            args=(current_load_p,current_load_q,weights),jac=True,method='SLSQP',bounds=bounds,
            options={'ftol':1e-4,'maxiter':100,'disp':False})
        if not np.isfinite(self.last_result.x).all():raise ValueError('MPC returned nonfinite P/Q actions.')
        action=np.clip(self.last_result.x,np.asarray(bounds)[:,0],np.asarray(bounds)[:,1])
        return ({b:float(v) for b,v in zip(self.der_buses,action[:self.n_der])},
                {b:float(v) for b,v in zip(self.der_buses,action[self.n_der:])})

    def attach_der(self,net):
        """Reference's ±50 kW/kVar flexible DERs, separately from profiled PV."""
        return attach_der(net,self.der_buses)

    def apply_setpoints(self,net,p_set,q_set):
        return apply_der_setpoints(net,self.der_buses,p_set,q_set)

    def run_snapshot(self,lab,net,weights=(1.,.1,1000.)):
        p,q=net_bus_pq(net,self.bus_ids)
        p_set,q_set=self.get_optimal_setpoints(p,q,*weights)
        self.apply_setpoints(net,p_set,q_set)
        applied=True
        try:y=lab.solve(net)
        except (pp.LoadflowNotConverged,ValueError):
            # No substitute search: a failed AC evaluation retains the zero-action baseline.
            p_set={b:0. for b in self.der_buses};q_set=p_set.copy()
            self.apply_setpoints(net,p_set,q_set);y=lab.solve(net);applied=False
        p_action=np.asarray(list(p_set.values()));q_action=np.asarray(list(q_set.values()))
        predicted=self.predict_voltages(p,q,p_action,q_action)
        violation=voltage_violation(lab,y)
        action=np.r_[p_action,q_action]
        objective,_=self.objective_with_gradient(action,p,q,weights)
        flags={'voltage_safe':violation<=1e-7,'safe':lab.violation(y)<=1e-7,
            'solver_success':bool(self.last_result.success),'solver_message':str(self.last_result.message),
            'optimizer_action_applied':applied,'weights':tuple(map(float,weights)),
            'action_cost':float(weights[0]*np.sum(p_action**2)+weights[1]*np.sum(q_action**2)),
            'objective':objective,'predicted_voltages':predicted}
        return y,p_set,q_set,flags
