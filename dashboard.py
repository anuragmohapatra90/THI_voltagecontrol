"""Standalone Gradio UI. Simulation state belongs to each browser session."""
from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
import tempfile, json, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from grid_lab import GridLab, Config, PRESETS, MODES
from plots import summary_table, comparison_figure, heatmap_figure

@dataclass
class Session:
    lab: object = None
    runs: dict = field(default_factory=dict)
    run_cfg: object = None
    metadata: dict = field(default_factory=dict)

    def ensure_lab(self,cfg):
        if self.lab is None or any(getattr(self.lab.cfg,k)!=getattr(cfg,k) for k in ['grid_code','groups','retrofit_oltc']):
            self.lab=GridLab(cfg)
        else:
            self.lab.cfg=cfg
        return self.lab


def model_table(lab):
    return pd.DataFrame((lab.training_report or {'Model':'Not trained yet'}).items(),columns=['Validation metric','Value']).astype(str)


def network_note(lab):
    retrofit='**Simulated OLTC retrofit:** HV-side ratio changer, ±8 taps, 1.25% step.' if lab.retrofit else 'Original declared OLTC is used.'
    return f'**{lab.cfg.grid_code}** · {lab.nv} LV buses · {len(lab.pv_ids)} PV inverters · {lab.ng} groups\n\n{retrofit}'


def config_from_values(grid,retrofit,groups,vmin,vmax,thermal,horizon,forecast,maxcurt,delay):
    if vmin>=vmax: raise gr.Error('Minimum voltage must be below maximum voltage.')
    return Config(grid_code=grid,retrofit_oltc=bool(retrofit),groups=int(groups),v_min=float(vmin),v_max=float(vmax),
                  thermal_limit=float(thermal),horizon=int(horizon),forecast=forecast,
                  max_curtailment=float(maxcurt),oltc_delay_steps=int(delay))


def train_model(grid,retrofit,groups,samples,session,progress=gr.Progress()):
    try:
        session=session or Session()
        cfg=Config(grid_code=grid,retrofit_oltc=bool(retrofit),groups=int(groups))
        lab=session.ensure_lab(cfg)
        progress(0,desc='Generating AC cases')
        report=lab.train(int(samples),lambda p:progress(p*.8,desc='Sampling actual SimBench profiles; MLP fitting follows'))
        progress(1,desc='Ready')
        warning=' The MLP reached its iteration limit; inspect its holdout errors.' if report['training_convergence_warning'] else ''
        return session, f'**MLP ready.** Run a controller comparison.{warning}',model_table(lab),network_note(lab)
    except Exception as exc:
        raise gr.Error(f'Training failed: {exc}') from exc


def use_peak_day(grid,retrofit,groups,session):
    try:
        session=session or Session()
        lab=session.ensure_lab(Config(grid_code=grid,retrofit_oltc=bool(retrofit),groups=int(groups)))
        pv=lab.profiles[('sgen','p_mw')].reindex(columns=lab.pv_ids).sum(axis=1).to_numpy()
        return session,int(np.argmax(pv)//96),network_note(lab),model_table(lab)
    except Exception as exc: raise gr.Error(str(exc)) from exc


def detail_result(mode,session):
    if not session or mode not in session.runs:
        return None,pd.DataFrame(),pd.DataFrame()
    steps,volts,actions=session.runs[mode]
    fig=heatmap_figure(volts,session.run_cfg,mode+' · every LV bus')
    plt.close(fig)
    return fig,steps.round(5),actions.round(5)


def export_run(session):
    root=Path(tempfile.mkdtemp(prefix='gridlab_results_'))
    path=root/'grid_experiment.zip'
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('metadata.json',json.dumps(session.metadata,indent=2))
        z.writestr('summary.csv',summary_table(session.runs).to_csv(index=False))
        for mode,(d,v,a) in session.runs.items():
            folder=mode.replace(' ','_').replace('+','and')
            for name,frame in [('steps',d),('bus_voltages',v),('actions',a)]:
                z.writestr(f'{folder}/{name}.csv',frame.to_csv(index=False))
    return str(path)


def run_comparison(grid,retrofit,groups,vmin,vmax,thermal,horizon,forecast,maxcurt,delay,
                   preset,lf,pf,day,quarter,steps,modes,session,progress=gr.Progress()):
    try:
        if not modes: raise gr.Error('Select at least one controller.')
        session=session or Session()
        cfg=config_from_values(grid,retrofit,groups,vmin,vmax,thermal,horizon,forecast,maxcurt,delay)
        lab=session.ensure_lab(cfg)
        if 'MLP + MPC' in modes and lab.model is None:
            raise gr.Error('Train the MLP for this grid/grouping first, or deselect MLP + MPC.')
        if not 0<=int(day)<=(lab.steps-1)//96: raise gr.Error('Day index is outside this profile year.')
        runs={}
        for j,mode in enumerate(modes):
            progress(j/len(modes),desc=mode)
            runs[mode]=lab.run(mode,preset,int(day)*96+int(quarter),int(steps),custom=(float(lf),float(pf)),
                              progress=lambda p,j=j,mode=mode:progress((j+p)/len(modes),desc=mode))
        session.runs=runs;session.run_cfg=cfg
        session.metadata={**lab.metadata(),'preset':preset,'load_factor':float(lf),'pv_factor':float(pf),
                          'start':int(day)*96+int(quarter),'steps':int(steps)}
        failed=sum(int((~r[0].safe).sum()) for r in runs.values())
        status=f'**Comparison complete.** {failed} controller-steps violate a configured limit or fail power flow.' if failed else '**Comparison complete.** All tested controller-steps satisfy the configured AC limits.'
        status+=' Results remain the last completed run until you run again.'
        fig=comparison_figure(runs,cfg);plt.close(fig)
        first=modes[0]
        detail,logs,actions=detail_result(first,session)
        archive=export_run(session)
        return (session,status,summary_table(runs).round(4),fig,model_table(lab),network_note(lab),
                gr.Dropdown(choices=list(runs),value=first),detail,logs,actions,archive)
    except gr.Error: raise
    except Exception as exc: raise gr.Error(f'Experiment failed: {exc}') from exc


CSS='''
.gradio-container {max-width:1500px !important;margin:auto}
#hero {background:linear-gradient(115deg,#0b2139,#124d58);color:white;border-radius:16px;padding:28px;margin-bottom:20px}
#hero h1 {color:white;margin:6px 0;font-size:32px}
#hero p {color:#d2ecee;margin:4px 0}
#hero small {color:#7cded4;letter-spacing:2px}
'''

ASSUMPTIONS='''
### Interpreting the comparison
**Uncontrolled:** neutral tap, available PV P (subject to inverter clipping), PV Q = 0.

**Automatic OLTC:** transformer LV-terminal voltage feedback, 0.99–1.01 pu deadband, at most one tap per 15-minute step. It controls voltage locally; every bus and thermal limit is assessed afterward.

**MLP + MPC:** neutral tap, grouped PV curtailment and reactive-capability commands. A receding forecast horizon is optimized; the first action is AC-screened before application. Solver failures and backup actions are visible in the logs.

PV factors above 1 represent added installed PV/inverter capacity; factors at or below 1 change available output only. Each scenario uses identical equipment across controllers. Clipping and commanded curtailment are recorded separately.

Training uses an MLP with a linear skip connection. Calibration guard margins and simulation holdouts are empirical; they are not a universal safety guarantee. The grid is balanced and quasi-static, with no storage or dynamic frequency model. A perfect forecast uses future SimBench profiles; persistence holds current injections.

The simulation remains on your Colab runtime. Keep it connected while using this browser tab. Changing the network, PV grouping, or retrofit assumption requires a new model.
'''


def build_dashboard(lab=None):
    seed=lab or GridLab()
    cfg=seed.cfg
    pv=seed.profiles[('sgen','p_mw')].reindex(columns=seed.pv_ids).sum(axis=1).to_numpy()
    peak_day=int(np.argmax(pv)//96)
    with gr.Blocks(title='Grid Control Lab',analytics_enabled=False) as app:
        state=gr.State(value=lambda:Session(lab=deepcopy(seed)))
        gr.HTML('<div id="hero"><small>SIMBENCH · PANDAPOWER · MLP + MPC</small><h1>Grid Control Lab</h1><p>Test when transformer taps are sufficient—and when active inverter control is needed.</p></div>')
        with gr.Row():
            with gr.Column(scale=1,min_width=290):
                gr.Markdown('### Network & scenario')
                grid=gr.Dropdown(['1-LV-rural1--0-sw','1-LV-rural2--0-sw','1-LV-rural3--0-sw','1-LV-urban6--0-sw'],value=cfg.grid_code,label='SimBench LV grid')
                retrofit=gr.Checkbox(cfg.retrofit_oltc,label='Allow simulated OLTC retrofit')
                groups=gr.Slider(1,4,value=cfg.groups,step=1,label='PV control groups')
                preset=gr.Dropdown(list(PRESETS),value='Normal',label='Loading preset')
                lf=gr.Slider(.3,3,value=1,step=.05,label='Load P/Q multiplier')
                pf=gr.Slider(0,3.5,value=1,step=.05,label='PV factor')
                with gr.Accordion('Operating limits & controller settings',open=False):
                    vmin=gr.Number(cfg.v_min,label='Minimum voltage (pu)',minimum=.8,maximum=1.)
                    vmax=gr.Number(cfg.v_max,label='Maximum voltage (pu)',minimum=1.,maximum=1.2)
                    thermal=gr.Slider(50,120,value=cfg.thermal_limit,step=1,label='Thermal limit (%)')
                    horizon=gr.Slider(1,8,value=cfg.horizon,step=1,label='MPC horizon (15-minute steps)')
                    forecast=gr.Dropdown(['Perfect','Persistence'],value=cfg.forecast,label='Forecast')
                    maxcurt=gr.Slider(0,1,value=cfg.max_curtailment,step=.05,label='Maximum PV curtailment')
                    delay=gr.Slider(1,4,value=cfg.oltc_delay_steps,step=1,label='OLTC delay (simulation steps)')
                note=gr.Markdown(network_note(seed))
            with gr.Column(scale=3,min_width=600):
                with gr.Tab('Experiment'):
                    with gr.Row():
                        samples=gr.Dropdown([300,600,1000,2000],value=1000,label='MLP training samples')
                        train=gr.Button('1 · Train MLP',variant='secondary')
                    with gr.Row():
                        day=gr.Number(peak_day,label='Day index (0-based)',precision=0,minimum=0,maximum=(seed.steps-1)//96)
                        quarter=gr.Slider(0,95,value=36,step=1,label='Start quarter-hour (36 = 09:00)')
                        steps=gr.Dropdown([8,24,48,96],value=24,label='Duration (15-minute steps)')
                    peak=gr.Button('Use peak-PV day',size='sm')
                    modes=gr.CheckboxGroup(MODES,value=MODES,label='Controllers to compare')
                    run=gr.Button('2 · Run comparison',variant='primary')
                    status=gr.Markdown('**Ready.** Train the MLP, or select OLTC / uncontrolled only.')
                    summary=gr.Dataframe(label='Controller comparison',interactive=False)
                    comparison=gr.Plot(label='Voltage, loading, taps & curtailment')
                    archive=gr.File(label='Download experiment ZIP',interactive=False)
                with gr.Tab('MLP validation'):
                    gr.Markdown('Separate training, calibration, and test sets. Guard margins come from calibration; every applied active-control action is checked by AC power flow.')
                    metrics=gr.Dataframe(model_table(seed),label='Model validation',interactive=False)
                with gr.Tab('Detailed results'):
                    inspect=gr.Dropdown(MODES,value=MODES[0],label='Inspect controller')
                    heat=gr.Plot(label='Every LV bus')
                    logs=gr.Dataframe(label='Step logs & solver status',interactive=False)
                    actions=gr.Dataframe(label='PV group action fractions',interactive=False)
                with gr.Tab('Assumptions'):
                    gr.Markdown(ASSUMPTIONS)
        preset.change(lambda name:PRESETS[name],inputs=preset,outputs=[lf,pf],api_name='set_preset',queue=False)
        common=dict(concurrency_limit=1,concurrency_id='grid_simulation')
        train.click(train_model,[grid,retrofit,groups,samples,state],[state,status,metrics,note],api_name='train_model',**common)
        peak.click(use_peak_day,[grid,retrofit,groups,state],[state,day,note,metrics],api_name='peak_day',**common)
        run.click(run_comparison,[grid,retrofit,groups,vmin,vmax,thermal,horizon,forecast,maxcurt,delay,
                  preset,lf,pf,day,quarter,steps,modes,state],
                  [state,status,summary,comparison,metrics,note,inspect,heat,logs,actions,archive],api_name='compare',**common)
        inspect.change(detail_result,[inspect,state],[heat,logs,actions],api_name='details',**common)
    return app.queue(default_concurrency_limit=1,max_size=8)


def launch_dashboard(app, *, share=True, auth=None):
    """Start a real server, verify it locally, and return URL + login credentials."""
    import secrets
    import requests
    credentials=auth or ('grid',secrets.token_urlsafe(16))
    try:
        _,local_url,share_url=app.launch(
            share=share,inline=False,prevent_thread_lock=True,
            auth=credentials,auth_message='Sign in with the credentials printed by your Colab launch cell.',
            show_error=True,quiet=False,css=CSS,
            theme=gr.themes.Soft(primary_hue='teal',neutral_hue='slate'),
        )
        client=requests.Session()
        client.trust_env=False  # The health check is exclusively to this local Python server.
        response=client.get(local_url,timeout=15)
        response.raise_for_status()
        if share and not share_url:
            raise RuntimeError('Gradio started locally but its share service did not return a browser URL. '
                               'Keep the Colab runtime connected and rerun the launch cell. '
                               'If it fails again, inspect the share/tunnel error printed above.')
        return share_url if share else local_url, credentials
    except Exception:
        app.close()
        raise
