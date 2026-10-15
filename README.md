# Voltage control for ADG

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anuragmohapatra90/THI_voltagecontrol/blob/main/Grid_Control_Lab.ipynb)

The grid, household loading, PV generation and three DER control sites come from [ADG_datadriven](https://github.com/anuragmohapatra90/ADG_datadriven). A Gradio dashboard compares **Uncontrolled**, **Best OLTC** and **MLP + MPC** together.

## Run

1. Open the Colab badge, use a CPU runtime and **Run all**. The MLP trains automatically on this grid.
2. Open the printed dashboard link in another tab. There is no dashboard login. Keep Colab connected.
3. **Load new scenario** generates a fresh synthetic daily profile. Adjust load **×0.5–2** and PV **×0–2**, exactly as in the reference dashboard. The graph reacts automatically; changing sliders or MPC gains keeps the underlying scenario fixed.

Setup stays on the left. The right shows three overlaid voltage curves, the three-DER action table and three network voltage maps. The selected OLTC tap is shown explicitly. The scenario time is a synthetic time of day, not a dated annual dataset.

After merging a PR, reopen the same GitHub-linked Colab URL, restart the runtime and Run all. No notebook uploads are needed. This repository is private, so the checkout still needs its read-only GitHub token through the Colab secret `GITHUB_TOKEN` or hidden prompt.

## Experiment

The original 15-bus, two-feeder grid uses a 0.4 MVA 20/0.4 kV transformer and NAYY 4x120 SE cables. Loads and PV remain at the original buses. The controllable DERs are at **5, 8 and 14**, with the original **±50 kW / ±50 kVar** bounds. Positive actions supply power; negative actions absorb power.

The original reference files for grid construction, profile generation, configuration and MLP architecture are copied byte-for-byte. One seeded realization of the reference's random line lengths is shared by model training and the dashboard. The MLP is retrained for that exact plant. Its 64/32 ReLU architecture, Adam/MSE training and SLSQP voltage/action objective are preserved.

The transformer gains an explicit OLTC retrofit. The OLTC comparison checks every permitted tap at the same P/Q snapshot. All curves and maps show the resulting AC power-flow voltages. See [MODEL.md](MODEL.md) for source provenance and assumptions.

## Development

Edit Python on a branch and open a PR. Regenerate the thin notebook after launcher changes:

```bash
python scripts/build_launcher.py
python -m pip install -r requirements.txt nbformat==5.10.4
python scripts/validate_project.py
```

CI checks upstream file identity, reference-grid AC parity, the original scenario/scaler behavior, controller signs and limits, exhaustive tap selection and the login-free browser dashboard on Python 3.12 and 3.13.
