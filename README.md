# Voltage control for ADG

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anuragmohapatra90/THI_voltagecontrol/blob/main/Grid_Control_Lab.ipynb)

The [ADG_datadriven](https://github.com/anuragmohapatra90/ADG_datadriven) dashboard layout in Gradio, with SimBench PQ data and all three control results shown together.

## Run

1. Open the Colab badge and **Run all**. The MLP prepares automatically.
2. Open the printed dashboard link in another tab. There is **no Gradio login**.
3. Choose **High PV**, **High Load**, or **Custom**. **Load new scenario** selects another timestamp within that mode's bucket; its actual date and time appear on the left.
4. In **Custom**, change load/PV scalers. The comparison reacts automatically while keeping the current timestamp. MPC gains work in all modes.

Setup is on the left. On the right, one voltage chart overlays **Uncontrolled**, **Best OLTC**, and **MLP + MPC**, with the MPC action table and the corresponding network voltage heatmaps. The layout fits a desktop viewport (tested at 1280×720 and larger) without page scrolling.

Keep Colab connected. After merging a PR, reopen the same GitHub-linked Colab URL, restart the session, and Run all. No replacement notebook uploads are needed.

The GitHub repository is private, so fetching its code still requires the Colab secret `GITHUB_TOKEN` or a hidden read-only token prompt. That is separate from the dashboard, which opens directly without credentials. The token is not saved in Git URLs or project files.

## Comparison

High PV selects from high-generation annual profiles and applies load ×0.55 / PV ×8. High Load selects high-demand, low-PV annual profiles and applies load ×7 / PV ×0.1. Candidate rows are screened with actual AC power flow: every retained High PV snapshot reaches at least 1.07 pu, and every High Load snapshot reaches at most 0.93 pu. New scenarios remain inside the selected pre-filtered bucket. Custom draws from the full 2016 profile calendar and applies the user's load/PV scalers.

All three cases use the same dated PQ snapshot and inverter ratings. Uncontrolled uses the neutral tap and available/clipped PV with zero inverter Q. **Best OLTC** AC-evaluates every allowed tap position and chooses the best voltage-limit compliance, then the closest voltages to nominal. Its tap position is shown explicitly. This is the best achievable settled tap for the snapshot, rather than one step of a moving tap controller. MLP + MPC uses the original ADG controller structure: net bus P/Q → a 64/32 ReLU MLP → bus voltages, with SLSQP optimising independent signed DER P/Q actions under ±50 kW/±50 kVar limits. Training uses the original Adam/MSE/standard-scalers setup, retrained on the actual SimBench grid. The transformer stays at its neutral tap.

The action table shows signed flexible-DER injections: positive P/Q supplies power; negative P/Q absorbs it. These control resources are separate from the profiled PV, matching the original ADG experiment. The voltage objective does not enforce thermal loading limits; the sidebar explicitly reports voltage compliance. The heatmaps show the actual SimBench network and share one voltage color scale. See [MODEL.md](MODEL.md) for model and benchmark assumptions.

## Development

Edit Python on a branch and open a PR. For launcher changes, regenerate `Grid_Control_Lab.ipynb` with `python scripts/build_launcher.py`.

```bash
pip install -r requirements.txt nbformat==5.10.4
python scripts/validate_project.py
```

CI also uses Chromium to check the actual desktop layout, overlaid comparison, dated scenario button, and absence of a login. Browser-test dependencies are installed only in CI.
