# Voltage control for ADG

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anuragmohapatra90/THI_voltagecontrol/blob/main/Grid_Control_Lab.ipynb)

The [ADG_datadriven](https://github.com/anuragmohapatra90/ADG_datadriven) dashboard layout in Gradio, with SimBench PQ data and all three control results shown together.

## Run

1. Open the Colab badge and **Run all**. The MLP prepares automatically.
2. Open the printed dashboard link in another tab. There is **no Gradio login**.
3. Choose **High PV**, **High Load**, or **Custom**. **Load new scenario** selects another timestamp and random injection layout within that mode's bucket; its actual date and time appear on the left.
4. In **Custom**, change load/PV scalers. The comparison reacts automatically while keeping the current timestamp and injection positions. MPC gains work in all modes.

Setup is on the left. On the right, one voltage chart overlays **Uncontrolled**, **Best OLTC**, and **MLP + MPC**, with the MPC action table and the corresponding network voltage heatmaps. The layout fits a desktop viewport (tested at 1280×720 and larger) without page scrolling.

Keep Colab connected. After merging a PR, reopen the same GitHub-linked Colab URL, restart the session, and Run all. No replacement notebook uploads are needed.

The GitHub repository is private, so fetching its code still requires the Colab secret `GITHUB_TOKEN` or a hidden read-only token prompt. That is separate from the dashboard, which opens directly without credentials. The token is not saved in Git URLs or project files.

## Comparison

High PV samples high-generation annual profiles with load ×20 / PV ×10. High Load samples high-demand, low-PV profiles with load ×7 / PV ×0.1. Existing load/PV elements are relocated to random feeder positions, preserving their dated P/Q and PV ratings. High PV screens for a real interior voltage dip followed by downstream recovery; High Load concentrates downstream demand. Both require a substantial voltage spread between buses. Only converged AC-screened timestamp/layout pairs enter the preset buckets. Custom uses all 35,136 real 2016 timestamps with user scalers (load up to ×30 / PV up to ×10) and a fresh layout on each new scenario. Slider and gain changes keep that layout fixed. These are deliberately stressed spatial scenarios, with the original SimBench line/transformer network unchanged.

All three cases use the same dated PQ snapshot and inverter ratings. Uncontrolled uses the neutral tap and available/clipped PV with zero inverter Q. **Best OLTC** AC-evaluates every allowed tap position and chooses the best voltage-limit compliance, then the closest voltages to nominal. Its tap position is shown explicitly. This is the best achievable settled tap for the snapshot, rather than one step of a moving tap controller. MLP + MPC uses the original ADG controller structure: net bus P/Q → a 64/32 ReLU MLP → bus voltages, with SLSQP optimising independent signed DER P/Q actions under ±50 kW/±50 kVar limits. Training uses the original Adam/MSE/standard-scalers setup, retrained on the actual SimBench grid. The transformer stays at its neutral tap. The four fixed control sites are buses 0, 2, 4 and 13; each keeps the same capacity as before. The voltage chart follows physical feeder paths, separated by gaps, so unrelated branches are not joined into an artificial voltage shape.

The action table shows signed flexible-DER injections: positive P/Q supplies power; negative P/Q absorbs it. These control resources are separate from the profiled PV, matching the original ADG experiment. The voltage objective does not enforce thermal loading limits; the sidebar explicitly reports voltage compliance. The heatmaps show the actual SimBench network and share one voltage color scale. Curves can overlap in easy cases: MPC penalises violations of the voltage band and action cost, not distance from 1 pu. Spatial stress can expose an OLTC limitation because one tap shifts every branch, while local P/Q can reshape voltages. Neither controller is forced to win. See [MODEL.md](MODEL.md) for model and benchmark assumptions.

## Development

Edit Python on a branch and open a PR. For launcher changes, regenerate `Grid_Control_Lab.ipynb` with `python scripts/build_launcher.py`.

```bash
pip install -r requirements.txt nbformat==5.10.4
python scripts/validate_project.py
```

CI also uses Chromium to check the actual desktop layout, overlaid comparison, dated scenario button, and absence of a login. Browser-test dependencies are installed only in CI.
