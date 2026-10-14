"""The ADG_datadriven dashboard layout, with live SimBench snapshots in Gradio."""
from dataclasses import dataclass,field
from copy import deepcopy
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import gradio as gr
from grid_lab import GridLab,PRESETS
from live_control import snapshot_mpc


@dataclass
class LiveExperiment:
    lab: object
    t: int = 0
    running: bool = False
    net: object = None
    previous: object = None
    counter: int = 0
    last_mode: str = ''
    last: dict = field(default_factory=dict)
    settings: tuple = (1.,1.,True,'MLP + MPC',1.,.1,1000.)

    def __post_init__(self):
        self.reset()
        pv=self.lab.profiles[('sgen','p_mw')].reindex(columns=self.lab.pv_ids).sum(axis=1).to_numpy()
        self.t=int(np.argmax(pv))-4

    def reset(self):
        self.net=self.lab.new_net();self.previous=np.zeros(2*self.lab.ng)
        self.counter=0;self.last_mode='';self.last={}

    def new_scenario(self):
        self.t=int(np.random.default_rng().integers(self.lab.steps));self.reset()

    def sample(self,lf,pf,enabled,mode,weights=(1.,.1,1000.)):
        lab=self.lab
        lab.rating=lab.base_rating*max(1.,float(pf))
        baseline=lab.new_net()
        available=lab.set_profile(baseline,self.t,float(lf),float(pf))
        lab.control(baseline,available,np.zeros(2*lab.ng))
        y_base=lab.solve(baseline)
        selected=mode if enabled else 'Off'
        if selected!=self.last_mode:
            self.reset();self.last_mode=selected
        av=lab.set_profile(self.net,self.t,float(lf),float(pf))
        info=lab.control(self.net,av,np.zeros(2*lab.ng))
        if selected=='MLP + MPC':
            y,u,info,flags=snapshot_mpc(lab,self.net,av,self.previous,weights)
            self.previous=u
        elif selected=='OLTC':
            y,self.counter,moves,note=lab.oltc(self.net,self.counter)
            self.previous=np.zeros(2*lab.ng)
            flags={'safe':lab.violation(y)<=1e-7,'backup':False,'tap_moves':moves}
        else:
            self.net.trafo.loc[lab.tid,'tap_pos']=lab.tap0
            y=lab.solve(self.net);flags={'safe':lab.violation(y)<=1e-7,'backup':False}
        feasible=np.minimum(av,lab.rating)
        pv=lab.net.sgen.loc[lab.pv_ids,'bus'].to_numpy()
        actions=pd.DataFrame({'Bus':pv,'P (kW)':(self.net.sgen.loc[lab.pv_ids,'p_mw'].to_numpy()-feasible)*1000,
                              'Q (kVar)':self.net.sgen.loc[lab.pv_ids,'q_mvar'].to_numpy()*1000})
        actions=actions.groupby('Bus',as_index=False).sum().round(2)
        self.last={'profile_step':self.t,'baseline':baseline,'controlled':deepcopy(self.net),
                   'y_base':y_base.copy(),'y_controlled':y.copy(),'actions':actions,
                   'mode':selected,'flags':flags,'info':info,
                   'load_p_kw':float(baseline.load.p_mw.sum()*1000),
                   'load_q_kvar':float(baseline.load.q_mvar.sum()*1000),
                   'pv_kw':float(np.sum(available)*1000),
                   'tap_pos':int(self.net.trafo.loc[lab.tid,'tap_pos'])}
        return self.last


def grid_positions(net):
    """Stable one-line layout from actual SimBench connections, not invented feeders."""
    graph=nx.Graph();graph.add_nodes_from(int(i) for i in net.bus.index)
    graph.add_edges_from((int(r.from_bus),int(r.to_bus)) for _,r in net.line[net.line.in_service].iterrows())
    graph.add_edges_from((int(r.hv_bus),int(r.lv_bus)) for _,r in net.trafo[net.trafo.in_service].iterrows())
    root=int(net.ext_grid.iloc[0].bus)
    tree=nx.bfs_tree(graph,root);children={i:list(tree.successors(i)) for i in tree}
    positions={};cursor=[0]
    def place(node,depth):
        if not children[node]:x=cursor[0];cursor[0]+=1
        else:x=float(np.mean([place(c,depth+1) for c in children[node]]))
        positions[node]=(x,-depth);return x
    place(root,0)
    return positions


def voltage_profile(frame,lab):
    net=frame['baseline'];bus=list(net.bus.index)
    base=net.res_bus.loc[bus,'vm_pu'].to_numpy()
    controlled=frame['controlled'].res_bus.loc[bus,'vm_pu'].to_numpy()
    labels=[str(b) for b in bus]
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=labels,y=base,mode='lines+markers',name='Uncontrolled',line={'color':'#ed9b2d','dash':'dash'},marker={'symbol':'circle'}))
    if frame['mode']!='Off':
        fig.add_trace(go.Scatter(x=labels,y=controlled,mode='lines+markers',name=frame['mode']+' Controlled',line={'color':'#16876b','width':3},marker={'symbol':'square'}))
    for value in [lab.cfg.v_min,lab.cfg.v_max]:fig.add_hline(y=value,line_dash='dash',line_color='#d44b52')
    fig.add_hline(y=1.,line_color='#aab3bc',line_width=.7)
    values=np.r_[base,controlled]
    fig.update_layout(template='plotly_white',height=330,margin={'l':55,'r':15,'t':15,'b':45},
                      xaxis_title='Bus',xaxis_type='category',yaxis_title='Voltage (p.u.)',
                      yaxis_range=[min(.88,float(values.min())-.01),max(1.12,float(values.max())+.01)],
                      legend={'orientation':'h','y':1.14},uirevision='profile')
    return fig


def network_heatmap(net,title,positions):
    pv_buses=set(net.sgen.loc[net.sgen.in_service,'bus'].astype(int))
    fig=go.Figure()
    xx=[];yy=[]
    for _,row in net.line[net.line.in_service].iterrows():
        a,b=positions[int(row.from_bus)],positions[int(row.to_bus)]
        xx.extend([a[0],b[0],None]);yy.extend([a[1],b[1],None])
    fig.add_trace(go.Scatter(x=xx,y=yy,mode='lines',line={'color':'#9aa4ae','width':2},hoverinfo='skip',showlegend=False))
    for _,row in net.trafo[net.trafo.in_service].iterrows():
        a,b=positions[int(row.hv_bus)],positions[int(row.lv_bus)]
        fig.add_trace(go.Scatter(x=[a[0],b[0]],y=[a[1],b[1]],mode='lines',line={'color':'#263c4b','width':4},hoverinfo='skip',showlegend=False))
    buses=list(net.bus.index)
    load_p=net.load.groupby('bus').p_mw.sum();load_q=net.load.groupby('bus').q_mvar.sum()
    gen_p=net.sgen.groupby('bus').p_mw.sum()
    data=[[int(b),float(net.res_bus.loc[b,'vm_pu']),float(load_p.get(b,0)*1000),
           float(load_q.get(b,0)*1000),float(gen_p.get(b,0)*1000)] for b in buses]
    fig.add_trace(go.Scatter(x=[positions[int(b)][0] for b in buses],y=[positions[int(b)][1] for b in buses],
        mode='markers+text',text=[str(b) for b in buses],textposition='top center',customdata=data,
        hovertemplate='Bus %{customdata[0]}<br>V %{customdata[1]:.4f} pu<br>Load P %{customdata[2]:.1f} kW<br>Load Q %{customdata[3]:.1f} kVar<br>PV %{customdata[4]:.1f} kW<extra></extra>',
        marker={'symbol':['square' if int(b) in pv_buses else 'circle' for b in buses],
                'size':[20 if int(b) in pv_buses else 15 for b in buses],
                'color':[d[1] for d in data],'colorscale':'RdBu','cmin':.90,'cmax':1.10,
                'line':{'color':'#41505a','width':1},'colorbar':{'title':'V (pu)','thickness':12}},
        showlegend=False))
    fig.update_layout(template='plotly_white',height=385,margin={'l':20,'r':35,'t':35,'b':20},title={'text':title,'font':{'size':15}},
                      xaxis={'visible':False},yaxis={'visible':False},uirevision='grid-layout')
    return fig


def render(session,lf,pf,enabled,mode,wp,wq,wv):
    frame=session.sample(lf,pf,enabled,mode,(wp,wq,wv))
    positions=grid_positions(frame['baseline'])
    profile=voltage_profile(frame,session.lab)
    before=network_heatmap(frame['baseline'],'Uncontrolled',positions)
    after=network_heatmap(frame['controlled'],frame['mode']+' Controlled' if enabled else 'Uncontrolled',positions)
    q=frame['profile_step']%96
    caption=f'{q//4:02d}:{(q%4)*15:02d} · Load {frame["load_p_kw"]:.1f} kW / {frame["load_q_kvar"]:.1f} kVar · PV {frame["pv_kw"]:.1f} kW'
    safe='Grid Safe.' if frame['flags']['safe'] else 'Outside limits.'
    if frame['mode']=='Off':action_note='Controller OFF.'
    elif frame['mode']=='OLTC':action_note=f'Tap: {frame["tap_pos"]:+d} · {safe}'
    else:action_note=safe+(' AC backup used.' if frame['flags']['backup'] else '')
    loading=float(frame['y_controlled'][session.lab.nv:].max())
    action_note+=f' Loading: {loading:.1f}%.'
    return (session,profile,gr.Dataframe(value=frame['actions'],visible=frame['mode']=='MLP + MPC'),
            before,after,caption,action_note)


def update_configuration(lf,pf,enabled,mode,wp,wq,wv,session):
    try:
        session.settings=(lf,pf,enabled,mode,wp,wq,wv)
        return render(session,*session.settings)
    except Exception as exc:raise gr.Error(str(exc)) from exc


def new_scenario(lf,pf,enabled,mode,wp,wq,wv,session):
    session.new_scenario();return update_configuration(lf,pf,enabled,mode,wp,wq,wv,session)


def tick(session):
    if not session.running:return tuple(gr.skip() for _ in range(7))
    session.t=(session.t+1)%session.lab.steps
    return render(session,*session.settings)


def toggle_live(session):
    session.running=not session.running
    return session,gr.Timer(value=2.,active=session.running),gr.Button('⏸ Pause live PQ' if session.running else '▶ Play live PQ')


CSS='''
.gradio-container {max-width:1480px!important;margin:auto}
#title {margin-bottom:18px;border-bottom:3px solid #16876b;padding-bottom:12px}
#controls {background:#f6f8fa;border:1px solid #e4e9ed;border-radius:12px;padding:18px}
'''


def build_dashboard(lab=None):
    seed=lab or GridLab()
    if seed.model is None:
        print('Preparing MLP controller…',flush=True)
        seed.train(1000)
        print('MLP ready.',flush=True)
    with gr.Blocks(title='Active Grid Control',analytics_enabled=False) as app:
        state=gr.State(value=lambda:LiveExperiment(deepcopy(seed)))
        timer=gr.Timer(2.,active=False)
        gr.Markdown('# ⚡ AI-Driven Active Distribution Grid Control',elem_id='title')
        with gr.Row():
            with gr.Column(scale=1,min_width=290,elem_id='controls'):
                gr.Markdown('### 1. Scenario Initialization')
                random=gr.Button('🎲 Generate New Random Scenario')
                preset=gr.Dropdown(list(PRESETS),value='Normal',label='Preset')
                gr.Markdown('### 2. Operational Adjustments')
                lf=gr.Slider(.3,3.,value=1.,step=.05,label='Load Scaling Factor')
                pf=gr.Slider(0,3.5,value=1.,step=.05,label='PV Generation Factor')
                enabled=gr.Checkbox(True,label='Enable Controller')
                mode=gr.Radio(['OLTC','MLP + MPC'],value='MLP + MPC',label='Controller')
                with gr.Group(visible=True) as gains:
                    gr.Markdown('#### MPC Gains')
                    wp=gr.Number(1.,minimum=0,label='Cost P (Active)',step=.1)
                    wq=gr.Number(.1,minimum=0,label='Cost Q (Reactive)',step=.1)
                    wv=gr.Number(1000.,minimum=0,label='Cost Voltage',step=100.)
                simulate=gr.Button('Simulate This Configuration',variant='primary')
                play=gr.Button('▶ Play live PQ')
            with gr.Column(scale=3,min_width=680):
                caption=gr.Markdown('')
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown('### Voltage Profile')
                        profile=gr.Plot(show_label=False)
                    with gr.Column(scale=1,min_width=220):
                        gr.Markdown('### Actions')
                        actions=gr.Dataframe(headers=['Bus','P (kW)','Q (kVar)'],interactive=False,show_label=False)
                        action_note=gr.Markdown('')
                gr.Markdown('### Voltage Heatmap')
                with gr.Row():
                    baseline=gr.Plot(show_label=False)
                    controlled=gr.Plot(show_label=False)
        inputs=[lf,pf,enabled,mode,wp,wq,wv,state]
        outputs=[state,profile,actions,baseline,controlled,caption,action_note]
        common={'concurrency_limit':1,'concurrency_id':'live_grid','show_progress':'hidden'}
        gr.on([lf.input,pf.input,enabled.change,mode.change,wp.change,wq.change,wv.change,simulate.click],
              update_configuration,inputs,outputs,api_name='react',trigger_mode='always_last',**common)
        preset.change(lambda value:PRESETS[value],preset,[lf,pf],queue=False,api_name='preset').then(
              update_configuration,inputs,outputs,trigger_mode='always_last',**common)
        random.click(new_scenario,inputs,outputs,api_name='new_scenario',**common)
        mode.change(lambda e,m:gr.Group(visible=e and m=='MLP + MPC'),[enabled,mode],gains,queue=False,api_visibility='private')
        enabled.change(lambda e,m:gr.Group(visible=e and m=='MLP + MPC'),[enabled,mode],gains,queue=False,api_visibility='private')
        play.click(toggle_live,state,[state,timer,play],queue=False,api_name='play_pause')
        timer.tick(tick,state,outputs,api_name='tick',trigger_mode='once',**common)
        app.load(update_configuration,inputs,outputs,api_name='initial_snapshot',**common)
    return app.queue(default_concurrency_limit=1,max_size=8)


def launch_dashboard(app, *, share=True, auth=None):
    import secrets
    from urllib.request import build_opener,ProxyHandler
    credentials=auth or ('grid',secrets.token_urlsafe(16))
    try:
        _,local_url,share_url=app.launch(share=share,inline=False,prevent_thread_lock=True,
            auth=credentials,show_error=True,quiet=False,css=CSS,
            theme=gr.themes.Soft(primary_hue='emerald',neutral_hue='slate'))
        with build_opener(ProxyHandler({})).open(local_url,timeout=15) as response:
            if response.status!=200:raise RuntimeError('Dashboard health check failed.')
        if share and not share_url:raise RuntimeError('No Gradio share URL was returned. Reconnect Colab and rerun launch.')
        return share_url if share else local_url,credentials
    except Exception:app.close();raise
