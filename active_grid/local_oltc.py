"""Settled snapshot of a local secondary-voltage OLTC relay, without feeder feedback."""
from copy import deepcopy
import pandapower as pp


def settle_oltc(lab, baseline, initial_tap=0):
    """Step using only bus-1 voltage; real switching delay is outside this snapshot model."""
    if int(initial_tap)!=initial_tap or not lab.tap_min<=initial_tap<=lab.tap_max:
        raise ValueError('Invalid initial OLTC tap.')
    tap=int(initial_tap)
    net=deepcopy(baseline)
    net.trafo.loc[lab.tid,'tap_pos']=tap
    y=lab.solve(net)
    history=[]
    reason='deadband'
    for _ in range(lab.tap_max-lab.tap_min+1):
        voltage=float(net.res_bus.loc[lab.lv_bus,'vm_pu'])
        history.append({'tap':tap,'secondary_voltage_pu':voltage})
        if lab.oltc_v_min<=voltage<=lab.oltc_v_max:break
        direction=1 if voltage<lab.oltc_v_min else -1
        next_tap=tap+direction
        if not lab.tap_min<=next_tap<=lab.tap_max:
            reason='tap limit';break
        if any(row['tap']==next_tap for row in history):
            reason='deadband crossing';break
        trial=deepcopy(net)
        trial.trafo.loc[lab.tid,'tap_pos']=next_tap
        try:trial_y=lab.solve(trial)
        except (pp.LoadflowNotConverged,ValueError):
            reason='next tap power-flow failure';break
        net,y,tap=trial,trial_y,next_tap
    else:reason='step limit'
    voltage=float(net.res_bus.loc[lab.lv_bus,'vm_pu'])
    flags={'measured_bus':lab.lv_bus,'secondary_voltage_pu':voltage,
        'deadband_pu':(lab.oltc_v_min,lab.oltc_v_max),'reason':reason,'history':history,
        'at_limit':tap in (lab.tap_min,lab.tap_max),'steps':abs(tap-int(initial_tap))}
    return net,y,tap,flags
