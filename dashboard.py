"""ADG's sidebar/profile/actions/heatmap layout, comparing all three controls."""
from dataclasses import dataclass,field
from copy import deepcopy,copy
from html import escape
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr
from pandapower import LoadflowNotConverged
from grid_lab import GridLab
from active_grid import config
from active_grid.local_oltc import settle_oltc
from active_grid.informed_oltc import informed_oltc,prepare_tap_estimator
from live_control import prepare_controller,voltage_violation
from dashboard_diagnostics import control_deltas,voltage_diagnostics,diagnostics_html
from dashboard_assets import CSS,LAYOUT_JS,BEGIN_SCENARIO_JS,END_SCENARIO_JS


MAX_SCENARIO_SCALE = 5.


@dataclass
class ScenarioExperiment:
    lab: object
    rng: object = field(default_factory=np.random.default_rng)
    scenario_id: int = 0
    tap_pos: int = 0
    informed_tap_pos: int = 0
    base: dict = field(default_factory=dict)
    last: dict = field(default_factory=dict)

    def __post_init__(self):
        self.new_scenario()

    def new_scenario(self):
        self.base=self.lab.generate_scenario(int(self.rng.integers(0,2**32-1)))
        self.scenario_id+=1

    def sample(self,lf=1.,pf=1.,weights=(1.,.1,1000.)):
        if not .5<=float(lf)<=MAX_SCENARIO_SCALE or not 0<=float(pf)<=MAX_SCENARIO_SCALE:
            raise ValueError('Use load 0.5–5 and PV 0–5.')
        lab=self.lab
        controller=prepare_controller(lab)
        baseline=lab.new_net()
        lab.set_snapshot(baseline,self.base,lf,pf)
        y_base=lab.solve(baseline)
        oltc,y_oltc,tap,oltc_flags=settle_oltc(lab,baseline,self.tap_pos)
        informed,y_informed,informed_tap,informed_flags=informed_oltc(lab,baseline,self.informed_tap_pos)
        informed_estimates=voltage_diagnostics(informed_flags["predicted_voltages"],informed,range(config.BUS_COUNT))
        mpc=deepcopy(baseline)
        y_mpc,p_set,q_set,flags=controller.run_snapshot(lab,mpc,weights)
        actions,at_limit=control_deltas(baseline,mpc,controller.der_buses)
        estimates=voltage_diagnostics(flags['predicted_voltages'],mpc,controller.bus_ids)
        self.last={'scenario_id':self.scenario_id,'scenario_seed':self.base['seed'],
            'hour':self.base['hour'],'baseline':baseline,'oltc':oltc,'mpc':mpc,
            'y_base':y_base,'y_oltc':y_oltc,'y_mpc':y_mpc,'tap_pos':tap,'oltc_flags':oltc_flags,
            'informed_oltc':informed,'y_informed_oltc':y_informed,'informed_tap_pos':informed_tap,
            'informed_flags':informed_flags,'informed_voltage_estimates':informed_estimates,
            'actions':actions,'at_limit':at_limit,'voltage_estimates':estimates,'flags':flags,'load_factor':float(lf),'pv_factor':float(pf),
            'gross_load_p_kw':float(self.base['load_p'].sum()*lf*1000),
            'load_q_kvar':float(self.base['load_q'].sum()*lf*1000),
            'pv_kw':float(self.base['pv'].sum()*pf*1000),
            'pv_buses':sorted(set(map(int,lab.net.sgen.bus))),
            'load_buses':sorted(set(map(int,lab.net.load.bus))),
            'load_bus_kw':{int(b):float(self.base['load_p'][b]*lf*1000) for b in lab.net.bus.index},
            'pv_bus_kw':{int(b):float(self.base['pv'][b]*pf*1000) for b in lab.net.bus.index}}
        self.tap_pos=tap
        self.informed_tap_pos=informed_tap
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


def add_heatmap(fig,net,positions,column,frame,is_mpc=False,informed=False):
    xx=[];yy=[]
    for table,a_name,b_name in [(net.line,'from_bus','to_bus'),(net.trafo,'hv_bus','lv_bus')]:
        for _,r in table[table.in_service].iterrows():
            a,b=positions[int(r[a_name])],positions[int(r[b_name])]
            xx.extend([a[0],b[0],None]);yy.extend([a[1],b[1],None])
    fig.add_trace(go.Scatter(x=xx,y=yy,mode='lines',line={'color':'#9aa4ae','width':1.5},
        hoverinfo='skip',showlegend=False),row=2,col=column)
    lab_bus=1
    buses=list(net.bus.index);pv_buses=set(frame['pv_buses']);der_buses=set(config.DER_BUSES);load_buses=set(frame['load_buses'])
    limited=set(frame['at_limit'].loc[frame['at_limit'][['ΔP','ΔQ']].any(axis=1),'Bus']) if is_mpc else set()
    load=net.load.groupby('bus')[['p_mw','q_mvar']].sum()
    generation=net.sgen.groupby('bus')[['p_mw','q_mvar']].sum()
    data=[]
    for b in buses:
        if b==lab_bus:
            site='LV busbar' if informed else 'LV busbar · OLTC sensor'
        else:
            installed=[]
            if b in load_buses:installed.append('Load')
            if b in pv_buses:installed.append('PV')
            if b in der_buses:installed.append('control DER')
            site=' + '.join(installed) or 'Grid supply'
        data.append([int(b),float(net.res_bus.loc[b,'vm_pu']),float(load.p_mw.get(b,0)*1000),
            float(load.q_mvar.get(b,0)*1000),frame['pv_bus_kw'].get(int(b),0.),
            float(generation.p_mw.get(b,0)*1000),float(generation.q_mvar.get(b,0)*1000),
            site,
            'LIMIT REACHED' if b in limited else (('MLP band: 0.95–1.05 pu (LV buses)' if informed else 'OLTC band: 0.99–1.01 pu') if b==lab_bus else ''),frame['load_bus_kw'].get(int(b),0.)])
    fig.add_trace(go.Scatter(x=[positions[int(b)][0] for b in buses],y=[positions[int(b)][1] for b in buses],
        mode='markers+text',text=[str(b) for b in buses],textposition='top center',textfont={'size':8},customdata=data,
        hovertemplate='Bus %{customdata[0]} · %{customdata[7]}<br>V %{customdata[1]:.5f} pu<br>Gross load P %{customdata[9]:.2f} kW<br>Load Q %{customdata[3]:.2f} kVar<br>PV P %{customdata[4]:.2f} kW<br>Base net P %{customdata[2]:.2f} kW<br>ΔP %{customdata[5]:.2f} kW · ΔQ %{customdata[6]:.2f} kVar<br>%{customdata[8]}<extra></extra>',
        marker={'symbol':['diamond' if b in der_buses else ('triangle-up' if b in pv_buses else 'circle') for b in buses],
            'size':[11 if b in der_buses else 9 for b in buses],
            'color':[d[1] for d in data],'coloraxis':'coloraxis',
            'line':{'color':['#a45100' if b in limited else '#41505a' for b in buses],
                    'width':[3 if b in limited else .7 for b in buses]}},showlegend=False),row=2,col=column)
    fig.update_xaxes(visible=False,row=2,col=column)
    levels=[p[1] for p in positions.values()]
    fig.update_yaxes(visible=False,range=[min(levels)-3,max(levels)+.8],row=2,col=column)


def comparison_plot(frame,lab,mode='local'):
    """Three voltage traces overlaid, then three maps on a shared color scale."""
    informed=mode=='mlp'
    if informed:
        frame={**frame,'oltc':frame['informed_oltc'],'tap_pos':frame['informed_tap_pos']}
    oltc_name='MLP-informed OLTC' if informed else 'Local OLTC'
    fig=make_subplots(rows=2,cols=3,specs=[[{'colspan':3},None,None],[{},{},{}]],
        column_widths=[1/3]*3,row_heights=[.52,.48],vertical_spacing=.17,horizontal_spacing=.05,
        subplot_titles=['Voltage Profiles · Feeder Order','Uncontrolled',
                        f'{oltc_name} · Tap {frame["tap_pos"]:+d}','MLP + MPC'])
    paths=[[0,1,*range(2,9)],[1,*range(9,15)]]
    bus=[]
    for path in paths:bus.extend(path+[None])
    labels=['' if b is None else str(b) for b in bus];x=list(range(len(bus)));values=[]
    # Each point carries all three AC results, so the last-drawn line cannot hide data.
    shared_voltages=[['',None,None,None] if b is None else
        [str(b),*[float(frame[key].res_bus.loc[b,'vm_pu']) for key in ['baseline','oltc','mpc']]]
        for b in bus]
    relay=frame['oltc_flags']
    lo,hi=relay['deadband_pu']
    if relay['reason']=='deadband':
        decision='Holding in local band' if relay['steps']==0 else f"Settled after {relay['steps']} tap step(s)"
    else:
        decision={'tap limit':'Tap limit reached','deadband crossing':'Holding to avoid tap reversal',
            'next tap power-flow failure':'Holding last valid tap: next tap failed',
            'step limit':'Step limit reached'}.get(relay['reason'],relay['reason'])
    relay_hover=(f'<br>OLTC: {escape(decision)}'
        f'<br>V₁ {relay["secondary_voltage_pu"]:.4f} pu · local band {lo:.2f}–{hi:.2f} pu')
    if informed:
        relay_hover=(f'<br>MLP-informed OLTC: {escape(frame["informed_flags"]["reason"])}'
            '<br>Trigger: estimated feeder voltage outside 0.95–1.05 pu')
    voltage_hover=('<b>Bus %{customdata[0]}</b> · actual AC voltage<br>'
        'Uncontrolled: %{customdata[1]:.5f} pu<br>'
        'OLTC: %{customdata[2]:.5f} pu<br>'
        'MLP + MPC: %{customdata[3]:.5f} pu'+relay_hover+'<extra></extra>')
    for key,name,color,symbol,dash in [
        ('baseline','Uncontrolled','#ed9b2d','circle','dash'),
        ('oltc','OLTC','#4574c6','diamond','solid'),
        ('mpc','MLP + MPC','#16876b','square','solid')]:
        v=[None if b is None else float(frame[key].res_bus.loc[b,'vm_pu']) for b in bus]
        values.extend(a for a in v if a is not None)
        fig.add_trace(go.Scatter(x=x,y=v,customdata=shared_voltages,mode='lines+markers',name=name,
            hovertemplate=voltage_hover,
            line={'color':color,'dash':dash,'width':2},marker={'symbol':symbol,'size':5}),row=1,col=1)
    for limit in [lab.cfg.v_min,lab.cfg.v_max]:
        fig.add_hline(y=limit,line_dash='dash',line_color='#d44b52',row=1,col=1)
    fig.add_hline(y=1.,line_color='#aab3bc',line_width=.7,row=1,col=1)
    fig.update_xaxes(title_text='Bus along each feeder (gaps separate branches)',tickmode='array',
        tickvals=x,ticktext=labels,tickfont={'size':9},row=1,col=1)
    fig.update_yaxes(title_text='Voltage (p.u.)',range=[min(.88,min(values)-.01),max(1.12,max(values)+.01)],row=1,col=1)
    positions=grid_positions(frame['baseline'])
    for col,key in enumerate(['baseline','oltc','mpc'],1):
        add_heatmap(fig,frame[key],positions,col,frame,is_mpc=key=='mpc',informed=informed)
    for name,symbol in [('Load + PV','triangle-up'),('Load + PV + DER','diamond'),('At limit','diamond')]:
        fig.add_trace(go.Scatter(x=[None],y=[None],mode='markers',name=name,
            marker={'symbol':symbol,'size':9,'color':'#e8f1f6','line':{'color':'#a45100' if name=='At limit' else '#41505a',
            'width':3 if name=='At limit' else 1}},legendgroup='sites',showlegend=True),row=1,col=1)
    fig.update_layout(template='plotly_white',height=600,autosize=True,
        margin={'l':48,'r':16,'t':82,'b':55},
        legend={'orientation':'h','x':0,'y':1,'yref':'container','yanchor':'top','itemclick':False,'itemdoubleclick':False,'font':{'size':10}},
        coloraxis={'colorscale':'RdBu','cmin':.90,'cmax':1.10,
            'colorbar':{'title':'V (pu)','orientation':'h','x':.5,'y':-.04,'len':.5,'thickness':10}},
        hovermode='closest',uirevision='grid-comparison',meta={'scenario_id':frame['scenario_id'],'scenario_seed':frame['scenario_seed'],
            'load_factor':frame['load_factor'],'pv_factor':frame['pv_factor'],'oltc_mode':mode})
    fig.update_annotations(font_size=11)
    return fig


def feedback_html(message='Ready', *, error=False):
    flag='true' if error else 'false'
    role='alert' if error else 'status'
    return f'<div id="scenario-feedback" data-error="{flag}" role="{role}" aria-live="polite">{escape(message)}</div>'


def _failed_snapshot(session,message):
    # A recoverable solve error is a completed UI response, with no partial results.
    return session,gr.skip(),gr.skip(),gr.skip(),gr.skip(),gr.skip(),feedback_html(message,error=True)


def status_markdown(frame,lab,mode='local'):
    safe='within limits' if voltage_violation(lab,frame['y_mpc'])<=1e-7 else 'outside limits'
    if mode=='mlp':
        flags=frame['informed_flags']
        limit=' · tap limit' if flags['at_limit'] else ''
        return (f'**OLTC tap position: {frame["informed_tap_pos"]:+d}** · ±5%  \n'
            f'MLP: {flags["reason"]}{limit} · MPC {safe}')
    return (f'**OLTC tap position: {frame["tap_pos"]:+d}** · ±5%  \n'
        f'V₁ {frame["oltc_flags"]["secondary_voltage_pu"]:.4f} pu · MPC {safe}')


def select_mode(session,mode):
    # Tab changes are a view change: no new scenario, training, solve or tap reset.
    if not session.last:return mode,gr.skip(),gr.skip()
    return mode,diagnostics_html(session.last,session.lab,mode),status_markdown(session.last,session.lab,mode)


def show_local(session):return select_mode(session,'local')


def show_informed(session):return select_mode(session,'mlp')


def _evaluate_configuration(lf,pf,wp,wq,wv,session,mode='local',*,resample=False):
    # RNG advances on a failed draw; displayed scenario and results commit only on success.
    candidate=copy(session)
    try:
        if resample:candidate.new_scenario()
        frame=candidate.sample(lf,pf,(wp,wq,wv))
        minutes=int(frame['hour']*60)
        caption=f"**Scenario {frame['scenario_id']} · {minutes//60:02d}:{minutes%60:02d}**  \nSynthetic daily profile"
        status=status_markdown(frame,candidate.lab,mode)
        plot=comparison_plot(frame,candidate.lab)
        informed_plot=comparison_plot(frame,candidate.lab,'mlp')
        diagnostics=diagnostics_html(frame,candidate.lab,mode)
        return candidate,plot,informed_plot,diagnostics,caption,status,feedback_html()
    except LoadflowNotConverged:
        return _failed_snapshot(session,f'Power flow failed (load ×{float(lf):g}, PV ×{float(pf):g}). '
            'Previous result kept. Reduce scales or load another scenario.')
    except Exception as exc:
        return _failed_snapshot(session,f'Simulation failed: {exc}. Previous result kept.')


def update_configuration(lf,pf,wp,wq,wv,session,mode='local'):
    return _evaluate_configuration(lf,pf,wp,wq,wv,session,mode)


def new_scenario(lf,pf,wp,wq,wv,session,mode='local'):
    return _evaluate_configuration(lf,pf,wp,wq,wv,session,mode,resample=True)


def build_dashboard(lab=None, *, scenario_seed=None):
    seed=lab or GridLab()
    if getattr(seed,'voltage_controller',None) is None:
        print('Preparing ADG MLP + MPC controller…',flush=True);prepare_controller(seed);print('MLP ready.',flush=True)
    prepare_tap_estimator(seed)
    with gr.Blocks(title='Voltage control for ADG',analytics_enabled=False,fill_height=True) as app:
        state=gr.State(value=lambda:ScenarioExperiment(deepcopy(seed),rng=np.random.default_rng(scenario_seed)))
        active_mode=gr.State('local')
        gr.Markdown('# Voltage control for ADG',elem_id='title')
        with gr.Row(elem_id='workspace'):
            with gr.Column(elem_id='controls'):
                gr.Markdown('### 1. Scenario Initialization')
                load=gr.Button('Load new scenario',variant='primary',elem_id='new-scenario')
                timestamp=gr.Markdown('')
                gr.Markdown('### 2. Operational Adjustments')
                lf=gr.Slider(.5,MAX_SCENARIO_SCALE,value=1.,step=.05,label='Load Scaling Factor',elem_id='load-scale')
                pf=gr.Slider(0.,MAX_SCENARIO_SCALE,value=1.,step=.05,label='PV Generation Factor',elem_id='pv-scale')
                gr.Markdown('### MPC Gains')
                with gr.Row(elem_id='gain-row'):
                    wp=gr.Number(1.,minimum=0,label='kP · ΔP cost',info='Higher: use less ΔP',step=.1,min_width=65,elem_id='gain-p')
                    wq=gr.Number(.1,minimum=0,label='kQ · ΔQ cost',info='Higher: use less ΔQ',step=.1,min_width=65,elem_id='gain-q')
                    wv=gr.Number(1000.,minimum=0,label='kV · voltage',info='Higher: stronger voltage correction',step=100.,min_width=65,elem_id='gain-v')
                status=gr.Markdown('')
                feedback=gr.HTML(feedback_html(),elem_id='feedback-host')
            with gr.Column(elem_id='comparison-column'):
                with gr.Tabs(elem_id='controller-tabs'):
                    with gr.Tab('Local OLTC',id='local',render_children=True) as local_tab:
                        graph=gr.Plot(show_label=False,elem_id='comparison',elem_classes='comparison-plot')
                    with gr.Tab('MLP-informed OLTC',id='mlp',render_children=True) as informed_tab:
                        informed_graph=gr.Plot(show_label=False,elem_id='comparison-mlp',elem_classes='comparison-plot')
            with gr.Column(elem_id='diagnostics-panel'):
                diagnostics=gr.HTML('',elem_id='diagnostics')
        inputs=[lf,pf,wp,wq,wv,state,active_mode];outputs=[state,graph,informed_graph,diagnostics,timestamp,status,feedback]
        common={'concurrency_limit':1,'concurrency_id':'scenario_grid','show_progress':'hidden'}
        local_tab.select(show_local,[state],[active_mode,diagnostics,status],api_name=False,**common)
        informed_tab.select(show_informed,[state],[active_mode,diagnostics,status],api_name=False,**common)
        gr.on([lf.input,pf.input,wp.change,wq.change,wv.change],update_configuration,inputs,outputs,
            api_name='react',trigger_mode='always_last',**common)
        load.click(None,[],[],js=BEGIN_SCENARIO_JS,queue=False,api_name=False).then(
            new_scenario,inputs,outputs,api_name='new_scenario',**common).then(
            None,[],[],js=END_SCENARIO_JS,queue=False,api_name=False)
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

