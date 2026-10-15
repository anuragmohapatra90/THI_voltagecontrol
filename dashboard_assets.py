"""Responsive, readable dashboard styling and immediate scenario feedback."""

CSS = '''
html,body {margin:0;height:100%;overflow:hidden}
.gradio-container {height:100dvh!important;max-width:none!important;padding:12px 18px!important;box-sizing:border-box;
 color:#172b3a!important;background:#fff!important;color-scheme:light!important}
.gradio-container > .main {height:100%;min-height:0;max-width:none!important}
.gradio-container .app {display:flex;flex-direction:column;flex:1;min-height:0;height:100%;padding:0!important}
.gradio-container .app > .column {flex:1;min-height:0;height:100%}
.gradio-container .fill-height {min-height:0}
#title {flex:0 0 auto!important;margin:0!important;padding-bottom:8px;border-bottom:2px solid #16876b}
#title h1 {font-size:22px;margin:0;line-height:1.3;color:#172b3a!important}
#workspace {flex:1!important;min-height:0!important;flex-wrap:nowrap!important;gap:12px}
#controls {flex:0 0 260px!important;min-width:260px!important;max-width:260px!important;min-height:0;gap:5px;
 background:#f6f8fa!important;border:1px solid #d8e1e8;border-radius:10px;padding:8px;box-sizing:border-box}
#controls h3 {font-size:14px;margin:0;color:#172b3a!important}
#controls,#controls label,#controls label span,#controls .prose,#controls .prose p,#controls .info,
#controls .block,#controls input,#controls span,#controls p,#controls strong,#controls b,#controls .info-text,#controls [data-testid="block-info"] {color:#172b3a!important}
#controls .prose p {margin:0;font-size:12px}
#controls label,#controls label span {background:transparent!important}
#controls .form {background:transparent!important}
#controls input {background:#fff!important;-webkit-text-fill-color:#172b3a!important;border-color:#aebfcb!important}
#controls .block {padding:4px 7px!important;background:#f6f8fa!important;border-color:#d8e1e8!important}
#controls .block:has(.prose),#controls .block:has(#scenario-feedback) {padding:0!important}
#controls > * {flex-grow:0!important}
#controls #new-scenario {position:relative;color:#fff!important;background:#087b64!important;border-color:#087b64!important}
#controls #new-scenario span {color:#fff!important}
#controls #new-scenario[data-busy="true"] span {visibility:hidden}
#controls #new-scenario:disabled {opacity:1!important;background:#4b6570!important;border-color:#4b6570!important}
#controls #new-scenario[data-busy="true"],#controls #new-scenario[data-busy="true"] * {color:transparent!important;-webkit-text-fill-color:transparent!important;pointer-events:none}
#controls #new-scenario[data-busy="true"]::after {content:'Loading scenario…';position:absolute;inset:0;display:grid;place-items:center;color:#fff!important;-webkit-text-fill-color:#fff!important}
#scenario-feedback {color:#405a6b;font-size:10px;line-height:12px;min-height:16px;height:44px;overflow:auto;box-sizing:border-box}
#scenario-feedback[data-error="true"] {color:#8c3026!important;background:#fff1ee;border-radius:4px;padding:3px 5px}
#scenario-feedback[data-busy="true"] {font-size:0!important;color:transparent!important;background:transparent;padding:0}
#scenario-feedback[data-busy="true"]::after {content:'Solving power flow and MPC…';font-size:10px;color:#405a6b;-webkit-text-fill-color:#405a6b}
#scenario-feedback[data-busy="true"]::before {content:'';display:inline-block;width:9px;height:9px;margin-right:6px;
 border:2px solid #b7c8d2;border-top-color:#087b64;border-radius:50%;animation:scenario-spin .7s linear infinite}
@keyframes scenario-spin {to {transform:rotate(360deg)}}
.comparison-plot {flex:1!important;min-width:0!important;min-height:0!important;height:100%;padding:0!important;border:0!important}
.comparison-plot > div {min-height:0}
.comparison-plot [data-testid="plotly"] {height:100%!important;width:100%!important}
.comparison-plot[aria-busy="true"] .js-plotly-plot {opacity:.48;transition:opacity .12s}
#comparison-column {flex:1!important;min-width:0!important;min-height:0!important;gap:0!important}
#controller-tabs {flex:1;min-height:0;padding:0!important;border:0!important;background:#fff!important}
#controller-tabs > .tab-nav {flex:0 0 auto;min-height:32px}
#controller-tabs button {font-size:12px;color:#172b3a!important;background:#f6f8fa!important}
#controller-tabs button[aria-selected='true'] {color:#087b64!important;border-bottom:2px solid #087b64!important;background:#fff!important}
#controller-tabs [role='tabpanel'] {padding:0!important;border:0!important;background:#fff!important}
#controller-tabs [role='tabpanel'] > .block {padding:0!important}
#gain-row {gap:5px;flex-wrap:nowrap!important}
#gain-row > div {min-width:0!important}
#gain-row label,#gain-row label span {font-size:11px!important;background:transparent!important}
#gain-row .info,#gain-row .info-text,#gain-row [data-testid="block-info"] {font-size:10px!important;line-height:13px!important}
#gain-row .info-text {margin-bottom:2px!important}
#diagnostics-panel {flex:0 0 340px!important;min-width:340px!important;max-width:340px!important;min-height:0!important;
 gap:0!important;padding:0!important;border:1px solid #d8e1e8;border-radius:9px;background:#fff!important;box-sizing:border-box}
#diagnostics {flex:1!important;min-height:0!important;padding:8px!important;border:0!important;background:#fff!important}
.diagnostic-layout {display:flex;flex-direction:column;gap:5px;min-height:0;box-sizing:border-box;color:#172b3a!important;
 background:#fff;font-size:11px;font-family:Arial,sans-serif;font-variant-numeric:tabular-nums}
.diagnostic-layout h3 {font-size:12px;line-height:17px;margin:0!important;color:#172b3a!important}
.diagnostic-note {color:#405a6b!important;font-size:10px;line-height:13px}
.diagnostic-layout table {width:100%;border-collapse:collapse;table-layout:fixed;color:#172b3a;background:#fff;margin:0!important}
.diagnostic-layout th,.diagnostic-layout td {text-align:right;white-space:nowrap;padding:0 4px;border-bottom:1px solid #e3eaf0;
 box-sizing:border-box;color:#172b3a!important}
.diagnostic-layout th:first-child {width:33px;text-align:center}
.diagnostic-layout thead th {background:#eaf3f0;height:20px;font-weight:600;font-size:10px}
.action-table td,.action-table tbody th {height:20px;font-size:11px}
.command-at-limit {background:#fff1cf!important;color:#713500!important;font-weight:700}
.max-label {font-size:8px;color:#713500!important;padding-left:2px}
.capacity {font-size:10px;line-height:17px;margin-top:2px;border-radius:4px;padding:0 5px}
.capacity.at-limit {background:#fff1cf;color:#713500!important;font-weight:700}
.capacity.available {background:#eef6f2;color:#28583e!important}
.voltage-section {flex:1;min-height:0;display:flex;flex-direction:column}
.voltage-scroll {min-height:0;overflow:auto;flex:1}
.voltage-table td,.voltage-table tbody th {height:16px;line-height:14px;font-size:10.5px}
.cost-function {flex:0 0 auto;border-top:1px solid #d8e1e8;padding-top:4px;font-size:10.5px;line-height:15px;color:#172b3a!important}
#diagnostics .cost-function,#diagnostics .cost-function div,#diagnostics .cost-function sub {color:#172b3a!important}
.cost-function sub {font-size:8px;line-height:0}
.equation-indent {padding-left:13px}
@media(min-height:760px) {.voltage-table td,.voltage-table tbody th {height:19px;line-height:17px;font-size:11.5px}}
@media(max-height:740px) {
 #controls {gap:3px;padding:7px}
 #controls label,#controls label span {font-size:12px!important;line-height:16px!important}
 #controls .prose p {font-size:11px;line-height:15px}
 .voltage-table td,.voltage-table tbody th {height:15px;line-height:13px;font-size:10px}
 .diagnostic-layout thead th {height:18px}
 .cost-function {font-size:10px;line-height:13px}
 .cost-function h3 {line-height:15px}
}
footer {display:none!important}
'''

LAYOUT_JS = '''() => {
  const column=document.getElementById('comparison-column');
  const panel=document.getElementById('diagnostics-panel');
  const controls=document.getElementById('controls');
  if (!column || column.dataset.fitted) return;
  column.dataset.fitted='true';
  let pending=false;
  const fit=()=>{
    pending=false;
    if (controls) {
      const h=Math.floor(innerHeight-controls.getBoundingClientRect().top-12);
      controls.style.height=h+'px';
      controls.style.maxHeight=h+'px';
    }
    if (panel) {
      const h=Math.floor(innerHeight-panel.getBoundingClientRect().top-12);
      panel.style.height=h+'px';
      const content=panel.querySelector('.diagnostic-layout');
      if (content) content.style.height=(h-18)+'px';
    }
    for (const host of column.querySelectorAll('.comparison-plot')) {
      const box=host.getBoundingClientRect();
      if (!box.width || !box.height) continue;
      const gd=host.querySelector('.js-plotly-plot');
      if (!gd || !window.Plotly || !gd.layout) continue;
      if (!gd._gridSizingHook) {gd._gridSizingHook=true;gd.on('plotly_afterplot',schedule);}
      const height=Math.max(300,Math.floor(innerHeight-box.top-12));
      const width=Math.floor(box.width);
      if (Math.abs(box.height-height)>1) host.style.height=height+'px';
      if (Math.abs((gd.layout.height||0)-height)>1 || Math.abs((gd.layout.width||0)-width)>1)
        window.Plotly.relayout(gd,{height,width});
    }
  };
  const schedule=()=>{if (!pending) {pending=true;requestAnimationFrame(fit);}};
  new ResizeObserver(schedule).observe(column);
  new MutationObserver(schedule).observe(column,{childList:true,subtree:true,attributes:true,attributeFilter:['hidden','aria-selected']});
  if (panel) new MutationObserver(schedule).observe(panel,{childList:true,subtree:true});
  addEventListener('resize',schedule);schedule();
}'''

BEGIN_SCENARIO_JS = '''() => {
  const button=document.getElementById('new-scenario');
  const feedback=document.getElementById('scenario-feedback');
  const graphs=document.querySelectorAll('.comparison-plot');
  document.getElementById('controls')?.setAttribute('inert','');
  document.getElementById('controller-tabs')?.setAttribute('inert','');
  if (button) {button.disabled=true;button.dataset.busy='true';button.setAttribute('aria-label','Loading scenario');}
  if (feedback) {feedback.dataset.busy='true';feedback.setAttribute('aria-busy','true');}
  graphs.forEach(graph=>graph.setAttribute('aria-busy','true'));
  return [];
}'''

END_SCENARIO_JS = '''() => {
  const button=document.getElementById('new-scenario');
  const feedback=document.getElementById('scenario-feedback');
  const graphs=document.querySelectorAll('.comparison-plot');
  document.getElementById('controls')?.removeAttribute('inert');
  document.getElementById('controller-tabs')?.removeAttribute('inert');
  if (button) {button.disabled=false;delete button.dataset.busy;button.removeAttribute('aria-label');}
  if (feedback) {delete feedback.dataset.busy;feedback.setAttribute('aria-busy','false');}
  graphs.forEach(graph=>graph.setAttribute('aria-busy','false'));
  return [];
}'''
