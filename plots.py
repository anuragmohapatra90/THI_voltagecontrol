"""Static scientific figures and comparison summaries."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS={'Uncontrolled':'#77869c','Automatic OLTC':'#df9a2d','MLP + MPC':'#067f89'}

def summary_table(runs):
    result=[]
    for mode,(d,_,_) in runs.items():
        def col(name): return d.get(name,pd.Series(dtype=float))
        result.append({'Controller':mode,'Safe steps':int(d.safe.sum()),'Steps':len(d),
            'PF failures':int((~d.converged).sum()),'Min V (pu)':col('v_min_pu').min(),
            'Max V (pu)':col('v_max_pu').max(),'Max loading (%)':col('max_loading_percent').max(),
            'Tap moves':int(col('tap_moves').sum()),'Curtailment (kWh)':col('curtailed_mw').sum()*250,
            'Clipping (kWh)':col('clipped_mw').sum()*250,'Backup steps':int(d.status.str.contains('backup').sum())})
    return pd.DataFrame(result)

def comparison_figure(runs,cfg):
    fig,axs=plt.subplots(2,2,figsize=(13,7.5),layout='constrained')
    for mode,(d,_,_) in runs.items():
        if 'v_min_pu' not in d: continue
        c=COLORS[mode]; t=d.hours
        axs[0,0].plot(t,d.v_min_pu,color=c,label=mode)
        axs[0,0].plot(t,d.v_max_pu,color=c,linestyle='--')
        axs[0,1].plot(t,d.max_loading_percent,color=c,label=mode)
        axs[1,0].step(t,d.tap_pos,where='post',color=c,label=mode)
        axs[1,1].plot(t,d.curtailed_mw*1000,color=c,label=mode)
    axs[0,0].axhspan(cfg.v_min,cfg.v_max,color='#067f89',alpha=.08)
    for value in [cfg.v_min,cfg.v_max]: axs[0,0].axhline(value,color='#bf4054',linestyle=':',linewidth=1)
    axs[0,1].axhline(cfg.thermal_limit,color='#bf4054',linestyle=':',label='Thermal limit')
    titles=['LV voltage envelope · solid min / dashed max','Maximum line or transformer loading','Automatic tap movements','Commanded PV curtailment']
    units=['Voltage (pu)','Loading (%)','Tap position','Curtailed PV (kW)']
    for ax,title,unit in zip(axs.ravel(),titles,units):
        ax.set(title=title,xlabel='Hours from start',ylabel=unit); ax.grid(alpha=.18)
    axs[0,0].legend(fontsize=8); axs[0,1].legend(fontsize=8)
    return fig

def heatmap_figure(volts,cfg,title):
    fig,ax=plt.subplots(figsize=(12,4),layout='constrained')
    im=ax.imshow(volts.T.to_numpy(),aspect='auto',cmap='RdYlBu_r',vmin=cfg.v_min,vmax=cfg.v_max,interpolation='nearest')
    ax.set(title=title,xlabel='15-minute step',ylabel='LV bus index')
    ax.set_yticks(range(len(volts.columns)),volts.columns)
    fig.colorbar(im,ax=ax,label='Voltage (pu)')
    return fig

