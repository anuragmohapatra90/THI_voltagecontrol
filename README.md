# Active Grid Control

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anuragmohapatra90/THI_voltagecontrol/blob/main/Grid_Control_Lab.ipynb)

The simple dashboard layout from [ADG_datadriven](https://github.com/anuragmohapatra90/ADG_datadriven), rebuilt in Gradio: scenario/scaling controls on the left, voltage profile and actions above two network voltage heatmaps. Simulation uses SimBench PQ profiles with automatic OLTC or MLP + MPC control.

## Run

1. Open the Colab badge and **Run all**. The MLP is prepared automatically once at startup.
2. Open the printed dashboard link in another tab.
3. Change load/PV sliders: the voltage profile, actions, and heatmaps react automatically. Use the preset selector or generate a new random scenario.
4. Choose **OLTC** or **MLP + MPC**. The enable switch compares controller-on versus controller-off operation. The three MPC gain inputs affect the P/Q/voltage objective, as in the original project.
5. **Play live PQ** advances actual SimBench profiles; **Pause** holds the current scenario. Each displayed sample is a 15-minute profile step, played at an accelerated two-second display interval.

Keep Colab connected. After merging a PR, restart the Colab session and run all to fetch the latest `main` source. The same Colab URL works every time; no replacement notebook uploads.

This repository is private. The checkout reads the Colab secret `GITHUB_TOKEN` or a hidden prompt for a read-only token scoped to this repository. It is not saved in Git URLs or project files. The dashboard's temporary link has a generated login password printed in Colab.

## Experiment

A random scenario is an actual SimBench profile timestamp. Scaling changes load P/Q and PV availability; uncontrolled and controlled runs receive identical injections. The controller and its actions update on every evaluated snapshot. OLTC taps persist during playback and are limited to one movement per sample. The live MLP-MPC uses a one-step horizon and the original gain-weighted P/Q/voltage objective, with inverter capability limits and AC verification.

The maps show the actual SimBench network. Circles are load buses; squares mark PV buses. Both maps use the same voltage color scale. Active actions are signed changes relative to available PV (negative P = curtailment); Q is inverter reactive injection.

Model settings, source version, and longer offline experiments remain in Python; they are not extra dashboard pages. See [MODEL.md](MODEL.md) for scientific assumptions, the explicitly simulated OLTC retrofit, forecast/capability limits, and holdout validation.

## Development

Edit the Python files on a branch and open a PR. Changes load through the existing Colab launcher after merge. For bootstrap changes, run `python scripts/build_launcher.py` and commit the generated notebook too.

```bash
pip install -r requirements.txt nbformat==5.10.4
python scripts/validate_project.py
```

CI checks the launcher, safe Git refresh, physical model, reactive controls, live PQ advancement, pause behavior, gain response, and single-step MLP-MPC. The browser frontend uses a timer rather than waiting for a full time-series batch. Display latency depends on the runtime; rapid slider changes retain the latest input, and timer requests do not accumulate while a frame is being computed.
