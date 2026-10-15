# Voltage control for ADG: experiment assumptions

## MLP and MPC from the reference project

The dashboard controller ports the structure of [ADG_datadriven](https://github.com/anuragmohapatra90/ADG_datadriven):

- `active_grid/mlp_model.py`: `VoltageEstimator`, with **2×bus-count inputs → 64 ReLU → 32 ReLU → bus-count outputs**.
- Input order is **all bus net P, then all bus net Q**. Positive means consumption: aggregate load minus aggregate generation. Bus IDs are explicitly mapped because SimBench includes non-contiguous IDs such as 42.
- `active_grid/training.py`: StandardScaler for inputs/outputs; Adam learning rate 0.001; MSE loss; batch size 32; 100 epochs. A 3,000-sample AC campaign is split 80/20 for training/validation, with a separate 500-sample test portion. Model weights and scalers are retrained for this actual SimBench network; the reference project's weights belong to a different topology and are not reused.
- `active_grid/controller.py`: `MLP_MPC_Controller.predict_voltages()` subtracts signed DER P/Q actions from current net consumption. `get_optimal_setpoints()` starts at zero and solves the reference one-step SLSQP objective with independent bounded P/Q variables at each DER bus.

`J = wP * sum(P_action_MW²) + wQ * sum(Q_action_MVAr²) + wV * sum(voltage_violation_pu²)`

Voltage limits are 0.95–1.05 pu. The reference action bounds are ±0.05 MW and ±0.05 MVAr per DER. PyTorch autograd supplies the objective derivative and inference uses double precision to avoid finite-difference cancellation in float32. This changes numerical evaluation, not the objective or action bounds. SLSQP uses the reference `ftol=1e-4`.

The previous tanh/Ridge surrogate, grouped PV-curtailment controls, hard calibrated voltage/thermal constraints, and coarse replacement search are no longer used by the dashboard. The separate older offline `GridLab.train()/run()` routines remain in Python and are not the frontend controller.

## Physical actuation and validation

As in the original ADG experiment, the controller acts through **flexible signed DER P/Q injections**. The default keeps four fixed control sites: the interior and end of the longest feeder, plus two other feeder ends (buses 13, 4, 0 and 2 for this grid). This relocates the previous four actuators; it does not increase their number or per-site limits. A separate 100 kVA controllable sgen is added there, with ±50 kW/±50 kVar limits. It is zero in uncontrolled and OLTC cases; MPC applies its optimised signed setpoints. It is distinct from the irradiance-driven PV injection. Positive P/Q injects power, negative P/Q absorbs it. The action table reports these signed actions, matching the reference.

This models flexible DER capacity, including the ability to supply active power when irradiance is low. It is an explicitly added simulation actuator; it is not a claim that an ordinary PV array supplies extra energy at night. No storage energy-state model is included in this snapshot experiment. Existing SimBench load and PV profiles remain identical across the three cases.

All displayed voltages are actual pandapower AC results **after applying the optimiser's actions**, not MLP predictions. If an applied action fails AC convergence, zero commands are restored and the failure is recorded. There is no alternate grid search that silently replaces the MLP-MPC proposal. Solver status, predicted voltages, applied actions, and held-out accuracy remain available in backend metadata, without extra dashboard panels.

The optimisation is voltage-focused, as in the reference: thermal loading is not a penalty or hard constraint. The sidebar therefore explicitly reports **voltage** limit compliance. Line/transformer loading is still calculated and available in results. Good voltage regulation alone does not establish that severe loading has been resolved. Soft voltage penalties and finite actuator limits can leave residual violations.

## Scenarios and comparison

The network is `1-LV-rural1--0-sw`, using the original SimBench calendar (2016, including leap day).

| Mode | Load P/Q scale | PV scale | Sampling |
|---|---:|---:|---|
| High PV | 20.00 | 10.00 | High-generation profiles; max LV voltage ≥1.07 pu and a physical feeder voltage valley |
| High Load | 7.00 | 0.10 | High-demand/low-PV profiles; min LV voltage ≤0.93 pu and substantial branch-to-branch spread |
| Custom | User-defined | User-defined | All 35,136 timestamps |

High PV starts with the upper 10% of annual total PV P. High Load starts with the upper 10% of total load P and requires PV ≤10% of its annual peak. Up to 256 temporally distributed rows plus 16 injection extremes are AC-screened per pool, with up to four random spatial placements per timestamp. Only converged, voltage-stressed layouts enter the bucket. Both presets require an LV voltage spread of at least 0.115 pu and a minimum voltage of at least 0.78 pu, avoiding near-collapse examples. High PV also requires an interior minimum on a real root-to-leaf path: at least 0.003 pu below its start, then at least 0.02 pu recovery toward its end. Screening uses neutral-tap AC voltages only; it never tests MPC performance or chooses scenarios because MPC wins.

Each layout moves the existing SimBench load/PV elements to random feeder positions without changing their element-level P/Q values or PV ratings. Demand is concentrated at an interior site and another branch, with the largest PV element near the far end. High Load instead concentrates demand downstream, retaining its real low-PV profiles; a PV-driven valley is not promised at night. These are deliberately modified stress scenarios, not the original geographical allocation of SimBench customers. The dated profile curves and network line/transformer topology are retained.

Load new scenario chooses another screened timestamp/layout within the preset bucket. Custom draws from the full calendar and generates a fresh random layout. Scalers and MPC gains keep both timestamp and placement fixed. All three cases use exactly the same exogenous injections, connection points and installed controls. The overlaid plot follows actual root-to-leaf paths with gaps between branches and repeated feeder-root labels, rather than joining unrelated buses in table order. Network heatmaps still show every bus.

Training includes these stress buckets, broader annual profiles with load factors 0.3–30 / PV factors 0–10, using the same spatial scenario generator and fixed actuator locations, and random signed DER actions. No artificial sensor noise is added to targets or displayed voltages.

PV factors above one represent added installed PV and inverter capacity. Original PV apparent-power ratings scale with `max(1, PV factor)` and are at least 1.10×nominal PV P. Original PV runs at available/clipped P and Q=0 in all three cases; the separate flexible DER resource supplies the MPC actions.

**Uncontrolled:** neutral tap and zero control-DER actions. **Best OLTC:** exhaustive AC evaluation of every allowed tap at identical PQ, minimizing maximum voltage-limit violation, then nominal-voltage deviation, overall grid-limit violation, and distance from neutral. **MLP + MPC:** neutral tap plus the reference signed-PQ optimiser. The chosen OLTC tap position is displayed explicitly. An OLTC changes the transformer ratio for every branch; distributed P/Q controls can reshape a local voltage profile. Neither is guaranteed to win. Curves may coincide when both choose zero action or settle to similar physical voltages. In particular, the reference MPC only penalises voltage outside the band and charges for actuation, whereas the OLTC benchmark also uses nominal-voltage deviation to break ties. It is valid for OLTC to sit closer to 1 pu in easy cases.

The LV transformer requires an explicitly simulated OLTC retrofit: HV-side ratio changer, ±8 taps at 1.25% per tap. The original SimBench network does not declare that capability. The OLTC result is the best settled tap for the snapshot, not a seconds-scale actuator simulation.

Balanced constant-PQ AC power flow is used. No phase imbalance, frequency dynamics, communications, storage dynamics, or real hardware timing is modeled. The first project supports one two-winding transformer and PV-labelled sgens without original gen/storage elements.

## Checks

CI verifies the reference controller's net-PQ sign mapping, objective derivative, voltage-gain response, independent actuator limits, and **actual AC voltage-violation reduction** across six High PV and six High Load snapshots. It also verifies every spatial stress-bucket row and High PV valley, preservation of dated element-level P/Q under relocation, fixed layout during scaler changes, a mean High PV MPC violation below half the exhaustive OLTC benchmark with improvement in at least half of the paired cases, annual calendar/custom sampling, identical comparison layouts, exhaustive tap selection, Git refresh/launcher behavior, and login-free Gradio event transport.

Chromium checks the renamed heading, explicit tap readout, three overlaid traces, reactive dated scenarios, and the fully visible no-scroll desktop layout down to 1280×650. Model/grid settings and metadata remain in Python, not extra frontend tabs.
