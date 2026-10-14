"""ADG's sidebar/profile/actions/heatmap layout, comparing all three controls."""
from dataclasses import dataclass,field
from copy import deepcopy
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr
import pandapower as pp
from grid_lab import GridLab,PRESETS
from live_control import snapshot_mpc
from scenarios import prepare_buckets


def oltc_score(lab,y,tap):
    """Best achievable voltages; thermal compliance remains independently reported."""
    v=y[:lab.nv]
    voltage_violation=max(float(np.max(lab.cfg.v_min-v)),float(np.max(v-lab.cfg.v_max)),0.)
    return (voltage_violation,float(np.mean((v-1.)**2)),lab.violation(y),abs(tap-lab.tap0))


def best_oltc(lab,baseline):
    """Exhaustive AC benchmark of every allowed tap at this fixed PQ snapshot."""
    net=deepcopy(baseline);candidates=[]
    for tap in range(lab.tap_min,lab.tap_max+1):
        net.trafo.loc[lab.tid,'tap_pos']=tap
        try:
            y=lab.solve(net)
            candidates.append((oltc_score(lab,y,tap),tap))
        except (pp.LoadflowNotConverged,ValueError):pass
    if not candidates:raise ValueError('No OLTC tap produced a converged AC solution.')
    _,tap=min(candidates)
    net.trafo.loc[lab.tid,'tap_pos']=tap
    return net,lab.solve(net),tap


@dataclass
class ScenarioExperiment:
    lab: object
    rng: object = field(default_factory=np.random.default_rng)
    t: int = 0
    preset: str = 'Custom'
    last: dict = field(default_factory=dict)

    def __post_init__(self):
        prepare_buckets(self.lab)
        self.t=int(self.rng.integers(self.lab.steps))

    def new_scenario(self,preset=None):
        self.preset=preset or self.preset
        bucket=self.lab.scenario_buckets[self.preset]
        choices=bucket[bucket!=self.t]
        self.t=int(self.rng.choice(choices if len(choices) else bucket))

    def sample(self,lf,pf,weights=(1.,.1,1000.),preset='Custom'):
        lab=self.lab
        if self.preset!=preset:self.new_scenario(preset)
        if preset!='Custom':lf,pf=PRESETS[preset]
        lab.rating=lab.base_rating*max(1.,float(pf))
        baseline=lab.new_net()
        available=lab.set_profile(baseline,self.t,float(lf),float(pf))
        lab.control(baseline,available,np.zeros(2*lab.ng))
        y_base=lab.solve(baseline)
        oltc,y_oltc,tap=best_oltc(lab,baseline)
        mpc=deepcopy(baseline)
        y_mpc,u,info,flags=snapshot_mpc(lab,mpc,available,np.zeros(2*lab.ng),weights)
        feasible=np.minimum(available,lab.rating)
        actions=pd.DataFrame({'Bus':lab.net.sgen.loc[lab.pv_ids,'bus'].to_numpy(),
            'P (kW)':(mpc.sgen.loc[lab.pv_ids,'p_mw'].to_numpy()-feasible)*1000,
            'Q (kVar)':mpc.sgen.loc[lab.pv_ids,'q_mvar'].to_numpy()*1000})
        actions=actions.groupby('Bus',as_index=False).sum().round(2)
        self.last={'profile_step':self.t,'timestamp':lab.profile_times[self.t],
            'baseline':baseline,'oltc':oltc,'mpc':mpc,
            'y_base':y_base,'y_oltc':y_oltc,'y_mpc':y_mpc,'tap_pos':tap,
            'actions':actions,'flags':flags,'preset':preset,'load_factor':lf,'pv_factor':pf,
            'load_p_kw':float(baseline.load.p_mw.sum()*1000),
            'load_q_kvar':float(baseline.load.q_mvar.sum()*1000),
            'pv_kw':float(np.sum(available)*1000)}
        return self.last


def grid_positions(net):
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


def add_heatmap(fig,net,positions,column):
    xx=[];yy=[]
    for table,a_name,b_name in [(net.line,'from_bus','to_bus'),(net.trafo,'hv_bus','lv_bus')]:
        for _,r in table[table.in_service].iterrows():
            a,b=positions[int(r[a_name])],positions[int(r[b_name])]
            xx.extend([a[0],b[0],None]);yy.extend([a[1],b[1],None])
    fig.add_trace(go.Scatter(x=xx,y=yy,mode='lines',line={'color':'#9aa4ae','width':1.5},
        hoverinfo='skip',showlegend=False),row=2,col=column)
    buses=list(net.bus.index);pv_buses=set(net.sgen.loc[net.sgen.in_service,'bus'].astype(int))
    load_p=net.load.groupby('bus').p_mw.sum();load_q=net.load.groupby('bus').q_mvar.sum()
    gen_p=net.sgen.groupby('bus').p_mw.sum()
    data=[[int(b),float(net.res_bus.loc[b,'vm_pu']),float(load_p.get(b,0)*1000),
           float(load_q.get(b,0)*1000),float(gen_p.get(b,0)*1000)] for b in buses]
    fig.add_trace(go.Scatter(x=[positions[int(b)][0] for b in buses],y=[positions[int(b)][1] for b in buses],
        mode='markers+text',text=[str(b) for b in buses],textposition='top center',textfont={'size':9},customdata=data,
        hovertemplate='Bus %{customdata[0]}<br>V %{customdata[1]:.4f} pu<br>Load P %{customdata[2]:.1f} kW<br>Load Q %{customdata[3]:.1f} kVar<br>PV %{customdata[4]:.1f} kW<extra></extra>',
        marker={'symbol':['square' if int(b) in pv_buses else 'circle' for b in buses],
            'size':[12 if int(b) in pv_buses else 9 for b in buses],
            'color':[d[1] for d in data],'coloraxis':'coloraxis',
            'line':{'color':'#41505a','width':.7}},showlegend=False),row=2,col=column)
    fig.update_xaxes(visible=False,row=2,col=column)
    fig.update_yaxes(visible=False,row=2,col=column)


def comparison_plot(frame,lab):
    """Three voltage traces overlaid, then three maps on a shared color scale."""
    fig=make_subplots(rows=2,cols=3,specs=[[{'colspan':2},None,{'type':'table'}],[{},{},{}]],
        column_widths=[.34,.34,.32],row_heights=[.52,.48],vertical_spacing=.17,horizontal_spacing=.05,
        subplot_titles=['Voltage Profile','MLP + MPC Actions','Uncontrolled',
                        f'Best OLTC · Tap {frame["tap_pos"]:+d}','MLP + MPC'])
    bus=list(frame['baseline'].bus.index);labels=[str(b) for b in bus];values=[]
    for key,name,color,symbol,dash in [
        ('baseline','Uncontrolled','#ed9b2d','circle','dash'),
        ('oltc','Best OLTC','#4574c6','diamond','solid'),
        ('mpc','MLP + MPC','#16876b','square','solid')]:
        v=frame[key].res_bus.loc[bus,'vm_pu'].to_numpy();values.extend(v)
        fig.add_trace(go.Scatter(x=labels,y=v,mode='lines+markers',name=name,
            line={'color':color,'dash':dash,'width':2},marker={'symbol':symbol,'size':5}),row=1,col=1)
    for limit in [lab.cfg.v_min,lab.cfg.v_max]:
        fig.add_hline(y=limit,line_dash='dash',line_color='#d44b52',row=1,col=1)
    fig.add_hline(y=1.,line_color='#aab3bc',line_width=.7,row=1,col=1)
    fig.update_xaxes(title_text='Bus',type='category',tickfont={'size':9},row=1,col=1)
    fig.update_yaxes(title_text='Voltage (p.u.)',range=[min(.88,min(values)-.01),max(1.12,max(values)+.01)],row=1,col=1)
    actions=frame['actions']
    fig.add_trace(go.Table(header={'values':list(actions.columns),'fill_color':'#eaf3ef','align':'center','height':23},
        cells={'values':[actions[c].tolist() for c in actions.columns],'format':['d','.2f','.2f'],
               'fill_color':'#fafcfc','align':'center','height':21,'font':{'size':11}}),row=1,col=3)
    positions=grid_positions(frame['baseline'])
    for col,key in enumerate(['baseline','oltc','mpc'],1):add_heatmap(fig,frame[key],positions,col)
    fig.update_layout(template='plotly_white',height=600,autosize=True,
        margin={'l':55,'r':30,'t':70,'b':55},
        legend={'orientation':'h','x':0,'y':1.10,'font':{'size':12}},
        coloraxis={'colorscale':'RdBu','cmin':.90,'cmax':1.10,
            'colorbar':{'title':'V (pu)','orientation':'h','x':.5,'y':-.04,'len':.5,'thickness':10}},
        uirevision='grid-comparison')
    fig.update_annotations(font_size=13)
    return fig


def update_configuration(preset,lf,pf,wp,wq,wv,session):
    try:
        frame=session.sample(lf,pf,(wp,wq,wv),preset)
        timestamp=frame['timestamp'].strftime('%d %b %Y · %H:%M')
        caption=f'**{timestamp}**'
        safe=lambda y:'within limits' if session.lab.violation(y)<=1e-7 else 'outside limits'
        status=f'OLTC: tap {frame["tap_pos"]:+d}, {safe(frame["y_oltc"])}  \nMLP + MPC: {safe(frame["y_mpc"])}'
        return session,comparison_plot(frame,session.lab),caption,status
    except Exception as exc:raise gr.Error(str(exc)) from exc


def new_scenario(preset,lf,pf,wp,wq,wv,session):
    session.new_scenario(preset)
    return update_configuration(preset,lf,pf,wp,wq,wv,session)


def preset_controls(preset):
    lf,pf=PRESETS[preset];custom=preset=='Custom'
    return gr.Slider(value=lf,interactive=custom),gr.Slider(value=pf,interactive=custom)


CSS='''
html,body {margin:0;height:100%;overflow:hidden}
.gradio-container {height:100dvh!important;max-width:none!important;padding:12px 18px!important;box-sizing:border-box}
.gradio-container > .main {height:100%;min-height:0;max-width:none!important}
.gradio-container .app {display:flex;flex-direction:column;flex:1;min-height:0;height:100%;padding:0!important}
.gradio-container .app > .column {flex:1;min-height:0;height:100%}
.gradio-container .fill-height {min-height:0}
#title {flex:0 0 auto!important;margin:0!important;padding-bottom:8px;border-bottom:2px solid #16876b}
#title h1 {font-size:22px;margin:0;line-height:1.3}
#workspace {flex:1!important;min-height:0!important;flex-wrap:nowrap!important;gap:16px}
#controls {flex:0 0 290px!important;min-width:290px!important;max-width:290px!important;min-height:0;gap:4px;
 background:#f6f8fa;border:1px solid #e4e9ed;border-radius:10px;padding:8px;box-sizing:border-box}
#controls h3 {font-size:15px;margin:0}
#controls .prose p {margin:0;font-size:13px}
#comparison {flex:1!important;min-width:0!important;min-height:0!important;height:100%;padding:0!important;border:0!important}
#comparison > div {min-height:0}
#comparison [data-testid="plotly"] {height:100%!important;width:100%!important}
#controls > * {flex-grow:0!important}
#controls .block {padding:4px 8px!important}
#controls .block:has(.prose) {padding:0!important}
#gain-row {gap:6px;flex-wrap:nowrap!important}
#gain-row > div {min-width:0!important}
footer {display:none!important}
'''

LAYOUT_JS='''() => {
  const host=document.getElementById('comparison');
  if (!host || host.dataset.fitted) return;
  host.dataset.fitted='true';
  let pending=false;
  const fit=()=>{
    pending=false;
    const gd=host.querySelector('.js-plotly-plot');
    if (!gd || !window.Plotly || !gd.layout) return;
    const box=host.getBoundingClientRect();
    const height=Math.max(300,Math.floor(window.innerHeight-box.top-12));
    const width=Math.floor(box.width);
    if (Math.abs(box.height-height)>1) host.style.height=height+'px';
    if (Math.abs((gd.layout.height||0)-height)>1 || Math.abs((gd.layout.width||0)-width)>1)
      window.Plotly.relayout(gd,{height,width});
  };
  const schedule=()=>{if (!pending) {pending=true;requestAnimationFrame(fit);}};
  new ResizeObserver(schedule).observe(host);
  new MutationObserver(schedule).observe(host,{childList:true,subtree:true});
  window.addEventListener('resize',schedule);
  schedule();
}'''


def build_dashboard(lab=None):
    seed=lab or GridLab()
    if seed.model is None:
        print('Preparing MLP controller…',flush=True);seed.train(1000);print('MLP ready.',flush=True)
    if not hasattr(seed,'scenario_buckets'):
        print('Preparing voltage-stress scenario buckets…',flush=True);prepare_buckets(seed)
    with gr.Blocks(title='Active Grid Control',analytics_enabled=False,fill_height=True) as app:
        state=gr.State(value=lambda:ScenarioExperiment(deepcopy(seed)))
        gr.Markdown('# ⚡ AI-Driven Active Distribution Grid Control',elem_id='title')
        with gr.Row(elem_id='workspace'):
            with gr.Column(elem_id='controls'):
                gr.Markdown('### 1. Scenario Initialization')
                load=gr.Button('Load new scenario',variant='primary')
                timestamp=gr.Markdown('')
                gr.Markdown('### 2. Operational Adjustments')
                preset=gr.Dropdown(list(PRESETS),value='High PV',label='Preset')
                lf=gr.Slider(.3,seed.cfg.load_scale_max,value=PRESETS['High PV'][0],step=.05,label='Load Scaling Factor',interactive=False)
                pf=gr.Slider(0,seed.cfg.pv_scale_max,value=PRESETS['High PV'][1],step=.05,label='PV Generation Factor',interactive=False)
                gr.Markdown('### MPC Gains')
                with gr.Row(elem_id='gain-row'):
                    wp=gr.Number(1.,minimum=0,label='Cost P',step=.1,min_width=65)
                    wq=gr.Number(.1,minimum=0,label='Cost Q',step=.1,min_width=65)
                    wv=gr.Number(1000.,minimum=0,label='Cost V',step=100.,min_width=65)
                status=gr.Markdown('')
            graph=gr.Plot(show_label=False,elem_id='comparison')
        inputs=[preset,lf,pf,wp,wq,wv,state];outputs=[state,graph,timestamp,status]
        common={'concurrency_limit':1,'concurrency_id':'scenario_grid','show_progress':'hidden'}
        gr.on([lf.input,pf.input,wp.change,wq.change,wv.change],update_configuration,inputs,outputs,
            api_name='react',trigger_mode='always_last',**common)
        preset.change(preset_controls,preset,[lf,pf],queue=False,api_name='preset').then(
            update_configuration,inputs,outputs,trigger_mode='always_last',**common)
        load.click(new_scenario,inputs,outputs,api_name='new_scenario',**common)
        app.load(update_configuration,inputs,outputs,api_name='initial_snapshot',**common)
        app.load(None,js=LAYOUT_JS)
    return app.queue(default_concurrency_limit=1,max_size=8)


def launch_dashboard(app, *, share=True, auth=None):
    """No dashboard login. The auth keyword only accepts None for old launchers."""
    from urllib.request import build_opener,ProxyHandler
    if auth is not None:raise ValueError('This dashboard runs without a login.')
    try:
        _,local_url,share_url=app.launch(share=share,inline=False,prevent_thread_lock=True,
            auth=None,show_error=True,quiet=False,css=CSS,
            theme=gr.themes.Soft(primary_hue='emerald',neutral_hue='slate'))
        with build_opener(ProxyHandler({})).open(local_url,timeout=15) as response:
            if response.status!=200:raise RuntimeError('Dashboard health check failed.')
        if share and not share_url:raise RuntimeError('No Gradio share URL was returned. Reconnect Colab and rerun launch.')
        # Preserve the tuple API for old open notebooks, without authentication.
        return share_url if share else local_url,None
    except Exception:app.close();raise
