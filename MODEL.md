# Experiment and model assumptions

## Reference experiment

The dashboard follows `anuragmohapatra90/ADG_datadriven`: scenario/scaling/gain setup on the left, a voltage profile and action table above network voltage heatmaps on the right. The voltage profile overlays all three cases: uncontrolled, best OLTC, and MLP + MPC. The grid and absolute PQ injections come from SimBench.

The default network is `1-LV-rural1--0-sw`. Date/time comes from the original SimBench profile calendar: 1 January–31 December 2016, including leap day. Load new scenario samples within the selected mode, excluding the currently selected row when the bucket has other entries. Custom uses all 35,136 fifteen-minute rows. Custom scalers and MPC gains retain that timestamp and immediately recalculate all three cases. There is no playback timer or controller-selection switch.

## Presets and injections

| Preset | Load P/Q factor | PV factor |
|---|---:|---:|
| High PV | 0.55 | 8.00 |
| High Load | 7.00 | 0.10 |
| Custom | User-defined (default 1.00) | User-defined (default 1.00) |

High PV starts with the upper 10% of annual total PV injections. High Load starts with the upper 10% of total load P, additionally requiring PV at or below 10% of its annual peak. Each pool contributes up to 256 temporally distributed candidate rows plus its 16 strongest injection rows. Neutral-tap AC screening retains only converged High PV rows with max LV voltage ≥1.07 pu and High Load rows with min LV voltage ≤0.93 pu. The finite pre-filtered buckets are cached at startup. Sampling is uniform within the retained bucket; it does not represent annual event frequency. Fixed preset factors cannot be changed by UI/API scaler inputs. Custom applies user scalers to original profiles without stress filtering.

`simbench.get_absolute_values(..., profiles_instead_of_study_cases=True)` supplies the load/generator shapes. Load P and Q scale together. PV Q is zero in the baseline and is actively controlled in the MLP-MPC case. Other sgens retain their PQ profiles. No artificial measurement noise is added to the AC validation voltages.

PV factors above one represent added installed PV/inverter capacity. The original apparent-power rating is at least 1.10 × nominal PV P, multiplied by `max(1, PV factor)`. Factors at or below one reduce availability without reducing installed capacity. Baseline and controlled cases use identical equipment within the same scenario. Inverter clipping above the rating is separate from commanded curtailment.

## Controls

- **Uncontrolled:** neutral transformer tap, available/clipped PV P, PV Q = 0.
- **Best OLTC:** every allowed tap is evaluated with actual AC power flow at identical PQ. The chosen tap minimizes maximum LV voltage-limit violation, then mean squared LV deviation from 1 pu, then overall grid-limit violation and distance from neutral. This exhaustive discrete benchmark shows the best achievable settled tap, not a single delayed move. Thermal compliance is checked independently; correcting voltage cannot remove a severe active-power overload. The separate one-step automatic OLTC routine is retained for offline time-series experiments.
- **MLP + MPC:** neutral tap, grouped curtailment and reactive-capability fractions. Each inverter satisfies `P² + Q² ≤ S²`. Positive pandapower sgen Q injects reactive power.

The original LV transformer is not declared as an OLTC. The simulation explicitly models an OLTC retrofit: HV-side ratio changer with ±8 taps at 1.25% per tap. This is additional simulated equipment, not original SimBench capability. The assumption is kept in code/model documentation rather than repeated as a dashboard panel.

The live MPC is a one-step receding controller for the current PQ snapshot, matching the old experiment's interaction. Its gain-weighted objective is

`J = wP * sum(delta_P_MW²) + wQ * sum(Q_MVAr²) + wV * sum(voltage_violation_pu²)`

plus the corresponding normalized thermal-violation penalty. The voltage and thermal constraints use empirical calibration margins. Active actions shown in the table are signed changes from feasible available PV: negative P means curtailment. Q is inverter reactive injection.

A direct derivative of the fitted tanh/linear-skip MLP is used by SLSQP to keep snapshot updates responsive. Applied actions are checked against actual pandapower AC results. Out-of-domain/failed/unsafe proposals trigger a finite AC-screened backup search. If no screened action meets all limits, the least-violating converged action is applied and the dashboard says Outside limits. This search is not a proof of infeasibility or a universal safety guarantee.

Each snapshot starts both controllers from the same neutral-tap baseline. The MPC starts from zero commands, avoiding dependence on the previous randomly selected scenario. Scaling and gain changes retain the timestamp. The offline `GridLab.run()` engine remains available for longer-horizon experiments; its settings are not exposed as dashboard tabs.

## Learning and assessment

The default model is prepared automatically once before opening the app, using 1,000 AC samples. It has two 64-unit tanh layers with a Ridge linear skip connection. Inputs are detailed load/sgen P/Q and tap position. Outputs are all monitored LV voltages, energized line loading, and transformer loading.

Training uses full-year shapes, extra high-PV/high-load samples, load factors 0.3–8, PV factors 0–10, feasible inverter actions, and tap positions. Seeded splits are 70% train, 15% calibration, and 15% test. Error reports remain available in `lab.training_report`; they are not extra frontend tables. These simulated holdouts do not establish generalization to another topology or real equipment.

Safe means all monitored LV voltages are within 0.95–1.05 pu and all monitored lines/transformers are at or below 100% loading. Balanced, constant-PQ AC power flow is used; no storage, phase imbalance, frequency dynamics, communications, or seconds-scale equipment response is modeled. SimBench's source voltage is retained. `tap_changer_type='Ratio'` is explicitly set, and missing ZIP-load coefficients are treated as constant PQ.

The first implementation supports LV grids with one two-winding transformer and PV-labelled sgens, without gen or storage elements. Model/grid settings are edited in `Config` or Python code. The backend metadata retains the source repository and fetched commit.

## Validation

`python scripts/validate_project.py` checks safe Git refresh and local-edit protection, notebook/source validity, actual tap response, inverter capability, all three cases at identical PQ, exhaustive best-tap selection, the fitted MLP derivatives, gain response, reactive scaling, the calendar and full-year Custom sampling, every stress-bucket row, preset-restricted resampling, and AC-screened MPC.

The Gradio server is exercised without authentication for initial results, three traces on the same voltage axes, slider reactions that retain the date/time, and a new dated scenario. CI additionally uses Chromium at 1440×900, 1366×768, and 1280×720 to check setup left/results right, fully visible controls and graphs, and no page scrolling. A real Colab share link still requires the user's connected runtime and Gradio's share service.
