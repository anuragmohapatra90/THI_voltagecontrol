"""Reference Adam/MSE/standard-scalers training, using actual SimBench AC samples."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandapower as pp
from grid_lab import PRESETS
from scenarios import prepare_buckets
from .mlp_model import VoltageEstimator
from .controller import MLP_MPC_Controller,net_bus_pq,MAX_P_MW,MAX_Q_MVAR,apply_der_setpoints


def train_controller(lab,samples=3000,test_samples=500):
    if samples<100 or test_samples<50:raise ValueError('Use at least 100 training and 50 test samples.')
    torch.set_num_threads(1);torch.manual_seed(lab.cfg.seed)
    buckets=prepare_buckets(lab);rng=np.random.default_rng(lab.cfg.seed)
    bus_ids=lab.net.bus.index.tolist()
    der_buses=sorted(set(lab.net.sgen.loc[lab.pv_ids,'bus'].astype(int)))
    xs=[];ys=[];failed=0;original_rating=lab.rating.copy()
    try:
        for i in range(samples+test_samples):
            if i%3==0:
                mode='High PV';t=int(rng.choice(buckets[mode]));lf,pf=PRESETS[mode]
                lf*=rng.uniform(.8,1.2);pf*=rng.uniform(.85,1.15)
            elif i%3==1:
                mode='High Load';t=int(rng.choice(buckets[mode]));lf,pf=PRESETS[mode]
                lf=min(lab.cfg.load_scale_max,lf*rng.uniform(.85,1.15));pf*=rng.uniform(.5,1.5)
            else:
                t=int(rng.integers(lab.steps));lf=rng.uniform(.3,lab.cfg.load_scale_max);pf=rng.uniform(0,lab.cfg.pv_scale_max)
            net=lab.new_net();lab.rating=lab.base_rating*max(1.,pf)
            av=lab.set_profile(net,t,lf,pf);lab.control(net,av,np.zeros(2*lab.ng))
            p=rng.uniform(-MAX_P_MW,MAX_P_MW,len(der_buses));q=rng.uniform(-MAX_Q_MVAR,MAX_Q_MVAR,len(der_buses))
            if i%5==0:p[:]=0.;q[:]=0.
            apply_der_setpoints(net,der_buses,dict(zip(der_buses,p)),dict(zip(der_buses,q)))
            try:
                lab.solve(net);bp,bq=net_bus_pq(net,bus_ids)
                xs.append(np.r_[bp,bq]);ys.append(net.res_bus.loc[bus_ids,'vm_pu'].to_numpy())
            except (pp.LoadflowNotConverged,ValueError):failed+=1
    finally:lab.rating=original_rating
    x=np.asarray(xs);y=np.asarray(ys)
    if len(x)<samples*.8:raise ValueError('Too few converged AC training samples.')
    # A distinct held-out test portion is never used by Adam or the scalers.
    test_count=min(test_samples,len(x)//5)
    x_full,x_test=x[:-test_count],x[-test_count:];y_full,y_test=y[:-test_count],y[-test_count:]
    x_train,x_val,y_train,y_val=train_test_split(x_full,y_full,test_size=.2,random_state=lab.cfg.seed)
    sx=StandardScaler().fit(x_train);sy=StandardScaler().fit(y_train)
    xt=torch.tensor(sx.transform(x_train),dtype=torch.float32)
    yt=torch.tensor(sy.transform(y_train),dtype=torch.float32)
    loader=DataLoader(TensorDataset(xt,yt),batch_size=32,shuffle=True,
        generator=torch.Generator().manual_seed(lab.cfg.seed))
    model=VoltageEstimator(2*len(bus_ids),len(bus_ids))
    criterion=nn.MSELoss();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    model.train()
    for epoch in range(100):
        for xb,yb in loader:
            optimizer.zero_grad();loss=criterion(model(xb),yb);loss.backward();optimizer.step()
    controller=MLP_MPC_Controller(model,sx,sy,bus_ids,der_buses,lab.cfg.v_min,lab.cfg.v_max)
    def predict(x):
        with torch.no_grad():
            return controller.model(torch.tensor(sx.transform(x),dtype=torch.float64)).numpy()*sy.scale_+sy.mean_
    error=np.abs(predict(x_test)-y_test)
    report={'architecture':'net bus P/Q -> 64 ReLU -> 32 ReLU -> bus voltage',
        'train_samples':len(x_train),'validation_samples':len(x_val),'test_samples':len(x_test),
        'failed_power_flows':failed,'epochs':100,'batch_size':32,'learning_rate':.001,
        'voltage_test_mae_pu':float(error.mean()),'voltage_test_max_error_pu':float(error.max()),
        'validation_mae_pu':float(np.abs(predict(x_val)-y_val).mean())}
    controller.training_report=report
    lab.voltage_controller=controller
    lab.training_report=report
    return report
