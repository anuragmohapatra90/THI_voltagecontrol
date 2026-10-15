"""Applied control increments and matched MLP/AC diagnostics for one snapshot."""
import numpy as np
import pandas as pd
from active_grid import config

# Half the 0.01 kW/kVar display resolution; includes numerical solutions at a bound.
LIMIT_TOL_KW = .005


def control_deltas(baseline, controlled, der_buses):
    before = baseline.sgen.groupby('bus')[['p_mw', 'q_mvar']].sum()
    after = controlled.sgen.groupby('bus')[['p_mw', 'q_mvar']].sum()
    rows, limits = [], []
    for b in der_buses:
        dp = float((after.loc[b, 'p_mw']-before.loc[b, 'p_mw'])*1000)
        dq = float((after.loc[b, 'q_mvar']-before.loc[b, 'q_mvar'])*1000)
        rows.append({'Bus': int(b), 'ΔP (kW)': dp, 'ΔQ (kVar)': dq})
        limits.append({'Bus': int(b), 'ΔP': abs(dp) >= config.MAX_P_MW*1000-LIMIT_TOL_KW,
                       'ΔQ': abs(dq) >= config.MAX_Q_MVAR*1000-LIMIT_TOL_KW})
    return pd.DataFrame(rows), pd.DataFrame(limits)


def voltage_diagnostics(predicted, net, bus_ids):
    actual = net.res_bus.loc[bus_ids, 'vm_pu'].to_numpy(float)
    estimate = np.asarray(predicted, dtype=float)
    if estimate.shape != actual.shape:
        raise ValueError('MLP and AC bus voltage ordering does not match.')
    return pd.DataFrame({'Bus': bus_ids, 'Estimate (pu)': estimate,
                         'Actual (pu)': actual, 'Error (pu)': estimate-actual})


def cost_function_html(v_min, v_max):
    return f'''<section class="cost-function" aria-label="Full MPC cost function">
<h3>MPC cost function</h3>
<div>J = k<sub>P</sub> Σ<sub>i∈D</sub> ΔP<sub>i</sub>² + k<sub>Q</sub> Σ<sub>i∈D</sub> ΔQ<sub>i</sub>²</div>
<div class="equation-indent">+ k<sub>V</sub> Σ<sub>b∈B</sub> [max(0, {v_min:.2f} − V̂<sub>b</sub>)²</div>
<div class="equation-indent">+ max(0, V̂<sub>b</sub> − {v_max:.2f})²]</div>
<div>V̂ = MLP(P<sub>net</sub> − ΔP, Q<sub>net</sub> − ΔQ)</div>
<div>−{config.MAX_P_MW:.2f} ≤ ΔP<sub>i</sub> ≤ {config.MAX_P_MW:.2f} MW</div>
<div>−{config.MAX_Q_MVAR:.2f} ≤ ΔQ<sub>i</sub> ≤ {config.MAX_Q_MVAR:.2f} MVAr</div>
<div class="diagnostic-note">D: buses 5, 8, 14 · B: all buses · V in pu</div>
</section>'''


def diagnostics_html(frame, lab, mode='local'):
    flags = frame['at_limit'].set_index('Bus')
    action_rows = []
    for _, row in frame['actions'].iterrows():
        b = int(row['Bus'])
        cells = []
        for key, name in [('ΔP', 'ΔP (kW)'), ('ΔQ', 'ΔQ (kVar)')]:
            hit = bool(flags.loc[b, key])
            suffix = ' <strong class="max-label">MAX</strong>' if hit else ''
            attr = f' class="command-at-limit" aria-label="Bus {b} {key} at limit"' if hit else ''
            cells.append(f'<td{attr}>{float(row[name]):+.2f}{suffix}</td>')
        action_rows.append(f'<tr data-bus="{b}"><th scope="row">{b}</th>{"".join(cells)}</tr>')
    count = int(frame['at_limit'][['ΔP', 'ΔQ']].to_numpy().sum())
    capacity = (f'<div class="capacity at-limit" role="status">LIMIT REACHED · {count}/6 commands at ±50</div>'
                if count else '<div class="capacity available" role="status">Capacity available · no command at limit</div>')
    voltage_rows = []
    informed=mode=='mlp'
    estimates=frame['informed_voltage_estimates'] if informed else frame['voltage_estimates']
    heading='Tap-aware MLP estimate vs actual' if informed else 'MLP estimate vs actual'
    note='At applied OLTC tap · error = estimate − actual · pu' if informed else 'After MPC · error = estimate − actual · pu'
    for _, row in estimates.iterrows():
        b = int(row['Bus'])
        values = [float(row[k]) for k in ['Estimate (pu)', 'Actual (pu)', 'Error (pu)']]
        voltage_rows.append(f'<tr data-bus="{b}"><th scope="row">{b}</th><td>{values[0]:.5f}</td>'
                            f'<td>{values[1]:.5f}</td><td>{values[2]:+.5f}</td></tr>')
    return f'''<div class="diagnostic-layout" data-scenario-id="{int(frame['scenario_id'])}" data-oltc-mode="{mode}">
<section class="action-section"><h3>MLP+MPC changes · ΔP / ΔQ</h3>
<div class="diagnostic-note">+ inject / − absorb · relative to uncontrolled</div>
<table class="action-table" aria-label="Applied controller power increments"><thead><tr>
<th scope="col">Bus</th><th scope="col">ΔP (kW)</th><th scope="col">ΔQ (kVar)</th></tr></thead>
<tbody>{''.join(action_rows)}</tbody></table>{capacity}</section>
<section class="voltage-section"><h3>{heading}</h3>
<div class="diagnostic-note">{note}</div>
<div class="voltage-scroll"><table class="voltage-table" aria-label="Per-bus MLP and actual AC voltages">
<thead><tr><th scope="col">Bus</th><th scope="col">Estimate</th><th scope="col">Actual</th><th scope="col">Error</th></tr></thead>
<tbody>{''.join(voltage_rows)}</tbody></table></div></section>
{cost_function_html(lab.cfg.v_min, lab.cfg.v_max)}</div>'''
