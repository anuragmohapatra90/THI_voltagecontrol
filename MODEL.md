# ADG experiment assumptions and source

The source is [ADG_datadriven at commit 55a6fa8](https://github.com/anuragmohapatra90/ADG_datadriven/tree/55a6fa8e80aa810912ebbfb182c6db11e165fac0). These files are byte-identical to that version, verified against their Git blob hashes in `checks.py`:

| Source file | Role |
|---|---|
| `active_grid/envs.py` | Grid construction and original noisy AC step |
| `active_grid/utils.py` | Synthetic daily load/PV generation |
| `active_grid/config.py` | Bus count, three DER sites, limits, sample counts and sensor noise |
| `active_grid/mlp_model.py` | Original 30 → 64 ReLU → 32 ReLU → 15 estimator |

## Grid and injections

Bus 0 is the 20 kV source; bus 1 is the 0.4 kV substation. The transformer is the original `0.4 MVA 20/0.4 kV` standard type. Feeder A follows 1–2–3–4–5–6–7–8, using cable lengths uniformly drawn from 0.2–0.4 km. Feeder B follows 1–9–10–11–12–13–14, with lengths from 0.3–0.5 km. All lines use `NAYY 4x120 SE`. The 13 loads and 13 original 100 kVA sgens remain at buses 2–14. No elements are relocated and no extra control sgens are added.

The original generator draws a synthetic time uniformly over 24 hours. PV has a Gaussian noon profile centered at 13:00 with 2.5-hour width, clipped to zero below 0.01. Household load combines morning and evening peaks plus a standby component. Per-bus load scale is 2–8 kW, PV scale 0–10 kW, and both receive the reference's shared multiplicative noise with mean 1 and standard deviation 0.1. Load Q is 0.33×load P. These are synthetic daily shapes, without calendar dates or an annual sampling bucket.

Dashboard scalers extend the reference UI ranges to load ×0.5–5 and PV ×0–5; defaults stay at ×1. This changes the experiment inputs, without changing grid equipment or the ±50 kW/kVar command bounds. The original training distribution is retained, so higher-scale inference can be extrapolation. Extreme snapshots may fail AC convergence; they report an error and preserve the last valid display rather than substituting a different scenario or fabricated voltages. A new scenario redraws the original profiles. Slider/gain changes retain the scenario. Like the reference dashboard, exogenous P is represented as net demand `load_P × load_scale − PV × PV_scale` in the load elements; Q is `load_Q × load_scale`. The original sgens carry the signed controllable injections at buses 5, 8 and 14. Unused random generator entries at buses 0/1 are ignored by the original plant and are zeroed in the physical feature vectors.

The reference generates new random line lengths for each environment. This project fixes NumPy seed 42 for one realization and shares that plant across training, testing and the dashboard. The reference's unrelated saved model weights are not assumed to describe this realization. Global NumPy state is restored after reference generation, so concurrent sessions do not alter each other's random sequence.

## MLP and MPC

Training follows the original campaign: 3,000 requested training snapshots and 500 separate requested test snapshots, with random signed actions at the three original DER buses. Non-convergent samples are skipped and counted. The training set is split 80/20; StandardScaler fits training inputs/targets only. Adam uses learning rate 0.001, batch size 32 and 100 epochs. The original sensor-noise standard deviation of 0.005 pu is retained in training targets. Reports include held-out errors against both noisy measurements and underlying physical AC voltages, plus a grid fingerprint.

Inputs are all bus net P followed by all bus net Q. Positive means consumption, so positive DER actions subtract from these features. The unchanged one-step reference objective is:

`J = wP × sum(P_action_MW²) + wQ × sum(Q_action_MVAr²) + wV × sum(voltage_violation_pu²)`

Defaults are `wP=1`, `wQ=0.1`, `wV=1000`, voltage band 0.95–1.05 pu, and independent ±0.05 MW / ±0.05 MVAr bounds per DER. SLSQP starts at zero with `ftol=1e-4`. Double-precision inference and autograd supply stable objective derivatives; float32 finite differences can otherwise cancel the voltage gradient. This numerical correction does not alter the objective or actuator limits.

Commands are applied to the original sgens, followed by actual AC power flow. If that evaluation fails, zero commands are restored and the failure is recorded. There is no substitute control algorithm. Soft voltage penalties, surrogate error and finite capability can leave residual violations. The reference's signed P/Q actuation represents flexible resources; there is no storage state-of-charge model. Voltage control does not enforce thermal loading limits.

## OLTC and comparison

The sole plant addition is an explicitly simulated HV-side ratio tap changer: neutral 0, positions −8…+8, 1.25% per step. Its neutral AC solution is checked against the unmodified reference transformer. The benchmark exhaustively AC-evaluates all 17 taps and minimizes maximum voltage-band violation, then mean squared deviation from 1 pu, then distance from neutral. Its tap position is displayed. This is the best settled snapshot result, not a tap-motor timing simulation.

Uncontrolled and OLTC cases keep DER commands zero. MLP+MPC keeps the transformer at neutral. All three use identical exogenous demand, PV and equipment. Displayed curves/maps are physical AC voltages, with no separate measurement-noise draw for each case. The chart separates the two actual feeder paths rather than connecting unrelated branch endpoints. Coincident curves remain possible in easy cases, and neither controller is guaranteed to perform best. Each voltage point carries the same bus-aligned uncontrolled, OLTC and MPC AC values in its tooltip, so a curve drawn underneath another remains readable by hovering the visible point. Network-site tooltips retain their own PV and command details.


## Dashboard diagnostics

The action table reports applied control increments ΔP/ΔQ: controlled sgen injections minus uncontrolled sgen injections at the same DER bus. Positive supplies power, negative absorbs it. The original uncontrolled commands are zero. Signed limits use the original ±50 kW/kVar capability; a cell is marked MAX within half its 0.01 kW/kVar display resolution (0.005 kW/kVar) of either bound. A limited command is identified individually; the banner counts limited commands rather than claiming every resource is exhausted. Only limited MPC DER sites receive an orange outline.

PV-site symbols reflect the original installations at buses 2–14, even when the daily profile or PV multiplier yields zero power. The PV power hover field comes from the scenario's scaled PV profile. It does not mistake the zero-command control sgens for zero installed PV. The colocated controllable resources remain at buses 5, 8 and 14.

The per-bus diagnostic table compares the MLP prediction **after the applied MPC increments** with the actual AC voltages from that same controlled snapshot, in explicit bus-ID order. Error is prediction minus actual, in pu. It is not a prediction at zero action or a noisy sensor reading. The full objective shows both squared power penalties, both one-sided squared voltage violations, the MLP input mapping and the independent action bounds; P/Q in the objective use MW/MVAr, while the command table uses kW/kVar. Increasing a power penalty discourages that control action; increasing the voltage penalty prioritizes removing predicted violations.

Scenario loading feedback runs in the browser before the solver request. Duplicate clicks are disabled during evaluation. A candidate scenario is solved separately and committed with its plot and diagnostics only on success; an error retains the previous scenario, and the button is re-enabled after either outcome.
