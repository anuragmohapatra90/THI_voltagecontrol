"""Fast single-snapshot MLP-MPC, matching the original dashboard's P/Q cost."""
import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
import pandapower as pp


def surrogate_with_jacobian(lab,x):
    """Predict in physical units and differentiate the trained tanh/linear-skip MLP."""
    sx=(x-lab.xscale.mean_)/lab.xscale.scale_
    value=sx.copy(); jac=np.eye(len(x))
    for i,(w,b) in enumerate(zip(lab.model.coefs_,lab.model.intercepts_)):
        value=value@w+b; jac=jac@w
        if i<len(lab.model.coefs_)-1:
            value=np.tanh(value); jac*=1-value[None,:]**2
    scaled=lab.linear.coef_@sx+lab.linear.intercept_+value
    y=scaled*lab.yscale.scale_+lab.yscale.mean_
    derivative=(lab.linear.coef_.T+jac)/lab.xscale.scale_[:,None]*lab.yscale.scale_[None,:]
    return y,derivative.T


def snapshot_mpc(lab,net,available,previous,weights=(1.,.1,1000.)):
    """One receding control sample, warm-started and checked with actual AC power flow."""
    if lab.model is None:raise ValueError('Prepare the MLP before starting the dashboard.')
    n=lab.ng; dim=2*n; wp,wq,wv=map(float,weights)
    base=lab.features(net)
    pvpos=net.sgen.index.get_indexer(lab.pv_ids)
    pslots=2*len(net.load)+2*pvpos; qslots=pslots+1
    feasible=np.minimum(available,lab.rating); group=lab.group
    cache={}
    def evaluate(u):
        if 'u' in cache and np.array_equal(u,cache['u']):return cache['data']
        p=feasible*(1-u[:n][group])
        cap=np.sqrt(np.maximum(lab.rating**2-p*p,0))
        q=u[n:][group]*cap
        dx=np.zeros((len(base),dim)); x=base.copy();x[pslots]=p;x[qslots]=q
        dpdu=np.zeros((len(p),dim));dpdu[np.arange(len(p)),group]=-feasible
        dqdu=np.zeros_like(dpdu)
        dqdu[np.arange(len(p)),group]=u[n:][group]*p*feasible/np.maximum(cap,1e-8)
        dqdu[np.arange(len(p)),n+group]=cap
        dx[pslots]=dpdu;dx[qslots]=dqdu
        y,dy=surrogate_with_jacobian(lab,x); dy=dy@dx
        curtailed=feasible-p
        low=np.maximum(lab.cfg.v_min-y[:lab.nv],0)
        high=np.maximum(y[:lab.nv]-lab.cfg.v_max,0)
        thermal=np.maximum(y[lab.nv:]/lab.cfg.thermal_limit-1,0)
        cost=wp*np.sum(curtailed**2)+wq*np.sum(q*q)+wv*np.sum(low*low+high*high+thermal*thermal)
        grad=-2*wp*curtailed@dpdu+2*wq*q@dqdu
        grad+=2*wv*(high-low)@dy[:lab.nv]
        grad+=2*wv*thermal@dy[lab.nv:]/lab.cfg.thermal_limit
        cache['u']=u.copy();cache['data']=(cost,grad,y,dy,x)
        return cache['data']
    def objective(u):
        cost,grad,*_=evaluate(u);return cost,grad
    def constraint(u):
        _,_,y,_,_=evaluate(u)
        return np.r_[y[:lab.nv]-lab.cfg.v_min-lab.margin_v,
                     lab.cfg.v_max-lab.margin_v-y[:lab.nv],
                     (lab.cfg.thermal_limit-lab.margin_t-y[lab.nv:])/100]
    def cjac(u):
        _,_,_,dy,_=evaluate(u)
        return np.vstack([dy[:lab.nv],-dy[:lab.nv],-dy[lab.nv:]/100])
    lower=np.maximum(np.r_[np.zeros(n),-np.ones(n)],previous-lab.cfg.slew)
    upper=np.minimum(np.r_[np.full(n,lab.cfg.max_curtailment),np.ones(n)],previous+lab.cfg.slew)
    starts=[np.clip(previous,lower,upper),np.clip(np.r_[np.full(n,.5),np.zeros(n)],lower,upper)]
    with threadpool_limits(limits=1):
        initial=min(starts,key=lambda u:np.sum(np.minimum(constraint(u),0)**2))
        result=minimize(objective,initial,jac=True,method='SLSQP',bounds=list(zip(lower,upper)),
                        constraints=[{'type':'ineq','fun':constraint,'jac':cjac}],
                        options={'maxiter':60,'ftol':1e-7})
        proposal=result.x if np.isfinite(result.x).all() else previous
        _,_,_,_,fx=evaluate(proposal)
    span=np.maximum(lab.xhi-lab.xlo,1e-8)
    ood=bool(np.any(fx<lab.xlo-.1*span) or np.any(fx>lab.xhi+.1*span))
    candidates=[proposal] if result.success and not ood else []
    candidates.extend([previous,np.zeros(dim)])
    for a in [.25,.5,lab.cfg.max_curtailment]:
        for r in [-1,0,1]:candidates.append(np.r_[np.full(n,min(a,lab.cfg.max_curtailment)),np.full(n,r)])
    evaluated=[]
    for index,u in enumerate(candidates):
        u=np.clip(u,lower,upper)
        info=lab.control(net,available,u)
        try:
            y=lab.solve(net);violation=lab.violation(y)
            p=net.sgen.loc[lab.pv_ids,'p_mw'].to_numpy()
            q=net.sgen.loc[lab.pv_ids,'q_mvar'].to_numpy()
            action_cost=wp*np.sum((feasible-p)**2)+wq*np.sum(q*q)
            evaluated.append((violation,action_cost,u.copy(),y.copy(),info,index))
            if index==0 and result.success and not ood and violation<=1e-7:break
        except (pp.LoadflowNotConverged,ValueError):pass
    if not evaluated:raise ValueError('No control candidate produced a converged AC solution.')
    chosen=min(evaluated,key=lambda e:(e[0]>1e-7,e[0] if e[0]>1e-7 else e[1]))
    violation,action_cost,u,y,info,index=chosen
    lab.control(net,available,u);lab.solve(net)
    accepted=index==0 and result.success and not ood
    return y,u,info,{'safe':violation<=1e-7,'backup':not accepted,
                    'solver_success':bool(result.success),'solver_message':str(result.message),'out_of_domain':ood,
                    'action_cost':float(action_cost),'weights':(wp,wq,wv)}
