# Model assumptions and scenarios

## Scenarios

| Preset | Load P/Q factor | PV factor |
|---|---:|---:|
| Normal | 1.00 | 1.00 |
| High loading | 2.50 | 0.35 |
| High PV | 0.55 | 3.00 |
| Low PV | 1.50 | 0.10 |

All PQ shapes originate in `simbench.get_absolute_values(..., profiles_instead_of_study_cases=True)`. Load P and Q scale together. PV Q is controlled rather than inherited from the profile, with unity power factor for the uncontrolled and OLTC cases. Non-PV sgens keep their original PQ profiles.

A PV factor **above 1** represents additional installed PV/inverter capacity. The inverter rating is the original rating (at least 1.10 × nominal PV P) multiplied by `max(1, PV factor)`. A factor **at or below 1** changes available PV output only. Ratings are identical across all compared controllers within a scenario. Physical clipping above the rating is recorded separately from controller-commanded curtailment. This makes high-PV stress electrically feasible at the inverter level and exposes grid constraints.

## Controllers

- **Uncontrolled:** neutral transformer tap; PV at available/clipped P; PV Q = 0.
- **Automatic OLTC:** senses the transformer LV terminal. Default deadband 0.99–1.01 pu, at most one tap per 15-minute step, configurable delay. Adjacent taps are evaluated with AC power flow so HV/LV tap orientation cannot be assumed incorrectly. No inverter control.
- **MLP + MPC:** fixed neutral tap; grouped PV curtailment and reactive capability fractions. Individual inverters obey `P² + Q² ≤ S²`. Positive pandapower sgen Q injects reactive power. A four-step horizon solves an SLSQP constrained nonlinear program; only its first action is applied, and the horizon is replanned at every step.

The MLP uses two 64-unit tanh layers and a linear skip connection (Ridge baseline plus learned nonlinear residual). Its inputs are detailed load/sgen P/Q and tap position; its outputs are every monitored LV voltage, line loading, and transformer loading. The network is algebraic and quasi-static; the horizon contains forecast injections and action changes, not storage dynamics. The objective penalizes voltage deviation, predicted violations, curtailment, Q use, loading, and action changes. Limits are tightened using calibration errors. Constraints also enforce maximum curtailment and normalized action slew.

The training data use full-year SimBench profile shapes, load factors 0.3–3, PV factors 0–3.5, all available taps, and random feasible P/Q actions. The seeded random splits are 70% training, 15% calibration, and 15% test. These are simulation holdouts; they do not establish generalization to another grid, topology, or real measurements. The dashboard reports convergence warnings and errors.

Every applied MPC action is AC-screened. If the solver fails, the proposal leaves the training feature range, or the proposed action violates actual limits, bounded backup actions are also evaluated with AC power flow. A feasible backup is selected when available; otherwise the least-violating converged candidate is applied and flagged. The backup search is finite, so it cannot prove infeasibility or provide a universal safety guarantee. Solver messages and backup usage remain in the result logs. This is a research simulation, not a certified grid controller.

## OLTC assumption and scope

The default SimBench rural LV transformer has fixed/off-load taps and no declared OLTC. The dashboard explicitly permits a simulated retrofit: an HV-side ratio changer with ±8 taps and 1.25% per tap. This changes transformer equipment capability and is not original SimBench equipment. Disable the retrofit to require a declared original OLTC. OLTC safety assessment checks all monitored voltages and thermal loading even though its control sensor is local. A local voltage regulator cannot inherently resolve feeder thermal congestion.

The first project supports LV grids with one two-winding transformer and PV-labelled sgens, without storage or gen elements. Balanced AC power flow is used. The slack source retains its SimBench voltage. SimBench load-model NaNs are filled and power flow is explicitly constant-PQ. `tap_changer_type='Ratio'` is set to make taps effective in pandapower 3.x. Phase imbalance, stochastic forecast uncertainty, seconds-scale tap timing, communications, and inverter hardware dynamics are not modeled. Perfect forecasts are labelled as a best-case benchmark; persistence is selectable.

## Outputs and editing

The dashboard provides voltage envelopes, all-bus voltage heatmaps, loading, tap positions, commanded curtailment, step logs, and group actions. A safe step requires every monitored LV voltage and thermal loading to be within the configured limits; power-flow failures count as unsafe. Export ZIP includes CSVs and JSON settings / model validation. Energy uses a 0.25-hour step.

Edit `PRESETS` in `grid_lab.py` to change defaults. Edit `Config`, the `mpc()` objective, or the `oltc()` deadband logic for research variants. Restart Colab and run the GitHub-backed launcher after merged edits; retrain whenever grid topology, PV grouping, or transformer assumptions change. A module-generated script called `checks.py` verifies tap effects, inverter capability, and both non-MLP controllers; it can also test an already trained lab.

The old private reference notebook could not be read, so this implementation follows the supplied description rather than copying its layout or controller code.

## Primary technical references

- SimBench profiles: https://simbench.readthedocs.io/en/stable/profiles.html
- SimBench grids: https://simbench.readthedocs.io/en/stable/networks.html
- pandapower transformer model: https://pandapower.readthedocs.io/en/v3.2.1/elements/trafo.html
- SimBench tap compatibility issue: https://github.com/e2nIEE/simbench/issues/69
- scikit-learn MLPRegressor: https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
- SciPy SLSQP: https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html
- Gradio sharing: https://www.gradio.app/guides/sharing-your-app
- Gradio launch API: https://www.gradio.app/docs/gradio/blocks
