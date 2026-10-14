# Experiment and model assumptions

## Reference experiment

The dashboard follows `anuragmohapatra90/ADG_datadriven`: choose a random base scenario, scale load and PV, enable the controller, adjust P/Q/voltage costs, and compare uncontrolled versus controlled bus voltages and network heatmaps. In this project the grid and absolute PQ injections come from SimBench, as requested; the old synthetic two-feeder grid is not substituted for the benchmark.

The default network is `1-LV-rural1--0-sw`. A random scenario selects a full-year SimBench timestamp. Changing sliders retains that timestamp and recalculates both power flows from the same exogenous injections. Play advances one actual 15-minute profile sample per delivered timer tick; Pause holds the timestamp. The two-second display interval is accelerated visualization rather than physical actuator timing. Live updates use the latest server-side controls, so queued timer calls do not restore older slider settings.

## Presets and injections

| Preset | Load P/Q factor | PV factor |
|---|---:|---:|
| Normal | 1.00 | 1.00 |
| High loading | 2.50 | 0.35 |
| High PV | 0.55 | 3.00 |
| Low PV | 1.50 | 0.10 |

`simbench.get_absolute_values(..., profiles_instead_of_study_cases=True)` supplies the load/generator shapes. Load P and Q scale together. PV Q is zero in the baseline and is actively controlled in the MLP-MPC case. Other sgens retain their PQ profiles. No artificial measurement noise is added to the AC validation voltages.

PV factors above one represent added installed PV/inverter capacity. The original apparent-power rating is at least 1.10 × nominal PV P, multiplied by `max(1, PV factor)`. Factors at or below one reduce availability without reducing installed capacity. Baseline and controlled cases use identical equipment within the same scenario. Inverter clipping above the rating is separate from commanded curtailment.

## Controls

- **Off:** neutral transformer tap, available/clipped PV P, PV Q = 0.
- **OLTC:** transformer LV-terminal voltage feedback with a 0.99–1.01 pu deadband. Adjacent taps are AC-evaluated to handle tap orientation correctly; at most one movement is applied per evaluated sample. Taps persist as PQ evolves.
- **MLP + MPC:** neutral tap, grouped curtailment and reactive-capability fractions. Each inverter satisfies `P² + Q² ≤ S²`. Positive pandapower sgen Q injects reactive power.

The original LV transformer is not declared as an OLTC. The simulation explicitly models an OLTC retrofit: HV-side ratio changer with ±8 taps at 1.25% per tap. This is additional simulated equipment, not original SimBench capability. The assumption is kept in code/model documentation rather than repeated as a dashboard panel.

The live MPC is a one-step receding controller for the current PQ snapshot, matching the old experiment's interaction. Its gain-weighted objective is

`J = wP * sum(delta_P_MW²) + wQ * sum(Q_MVAr²) + wV * sum(voltage_violation_pu²)`

plus the corresponding normalized thermal-violation penalty. The voltage and thermal constraints use empirical calibration margins. Active actions shown in the table are signed changes from feasible available PV: negative P means curtailment. Q is inverter reactive injection.

A direct derivative of the fitted tanh/linear-skip MLP is used by SLSQP to keep snapshot updates responsive. Applied actions are checked against actual pandapower AC results. Out-of-domain/failed/unsafe proposals trigger a finite AC-screened backup search. If no screened action meets all limits, the least-violating converged action is applied and the dashboard says Outside limits. This search is not a proof of infeasibility or a universal safety guarantee.

New random scenarios or controller type changes reset actuator history. Scaling and gain changes keep the current timestamp and controller state. The offline `GridLab.run()` engine remains available for longer-horizon experiments; its advanced settings are not exposed as dashboard tabs.

## Learning and assessment

The default model is prepared automatically once before opening the app, using 1,000 AC samples. It has two 64-unit tanh layers with a Ridge linear skip connection. Inputs are detailed load/sgen P/Q and tap position. Outputs are all monitored LV voltages, energized line loading, and transformer loading.

Training uses full-year shapes, extra high-PV/high-load samples, load factors 0.3–3, PV factors 0–3.5, feasible inverter actions, and tap positions. Seeded splits are 70% train, 15% calibration, and 15% test. Error reports remain available in `lab.training_report`; they are not extra frontend tables. These simulated holdouts do not establish generalization to another topology or real equipment.

Safe means all monitored LV voltages are within 0.95–1.05 pu and all monitored lines/transformers are at or below 100% loading. Balanced, constant-PQ AC power flow is used; no storage, phase imbalance, frequency dynamics, communications, or seconds-scale equipment response is modeled. SimBench's source voltage is retained. `tap_changer_type='Ratio'` is explicitly set, and missing ZIP-load coefficients are treated as constant PQ.

The first implementation supports LV grids with one two-winding transformer and PV-labelled sgens, without gen or storage elements. Model/grid settings are edited in `Config` or Python code. The backend metadata retains the source repository and fetched commit.

## Validation

`python scripts/validate_project.py` checks safe Git refresh and local-edit protection, notebook/source validity, actual tap response, inverter capability, OLTC bounds, the fitted MLP derivatives, gain response, reactive scaling, live profile advancement, pause behavior, controller-off equivalence, and AC-screened live MPC.

The authenticated Gradio server is also exercised for initial plots, slider reactions, presets, play/pause, timer updates, random scenarios, and controller switching. The frontend comprises three plots and one action table, with no diagnostic tabs. A real Colab share link still requires the user's connected runtime and Gradio's share service.
