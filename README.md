# Grid Control Lab

SimBench PQ scenarios, automatic OLTC, and MLP + MPC active inverter control, with a separate-tab Gradio dashboard.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anuragmohapatra90/THI_voltagecontrol/blob/main/Grid_Control_Lab.ipynb)

This badge opens the repository-backed launcher on `main` after the setup PR is merged. This repository is private, so the launcher defaults to authenticated checkout.

## One source of truth

The source lives in GitHub as ordinary Python files. The notebook is a small launcher: fetch source, install dependencies, verify the model, and start the dashboard. It does not embed copies of the source files.

Future changes use a branch and pull request. After a PR is merged into `main`, the same Colab link opens the updated notebook. Every run of the checkout cell also fetches the newest `main` source, so an already-open launcher gets updated Python code when rerun. Nothing needs to be uploaded manually after initial repository setup.

An already-running dashboard does not hot-reload from GitHub. Rerun the checkout/install/launch cells, or use **Runtime → Restart session and run all**. This closes the previous app and loads the new source. The launcher prints the exact commit SHA, which is included in experiment metadata. Restarting creates a new temporary Gradio link and resets models unless you saved them separately.

Do not save a separate copy of the notebook into Drive for routine use; bookmark the repository's **Open in Colab** link. A Drive copy is independent of GitHub, although this launcher will still fetch updated Python files if its checkout cell remains intact.

## Run

1. Open the Colab badge. Use a CPU runtime.
2. Run all cells. The checkout defaults to `main`.
3. Open the printed dashboard URL in another tab and sign in using the generated credentials.
4. Train the MLP, select loading presets, and run controller comparisons.

The Colab runtime must remain connected while the dashboard runs. The endpoint requires a password; credentials are generated in memory at launch, not committed to GitHub.

Public repositories work without GitHub credentials. For a private repository, authorize GitHub access when opening it in Colab, set `PRIVATE_REPO=True`, and provide a read-only token for this repository through the Colab secret `GITHUB_TOKEN` or the hidden prompt. The token is passed to Git through environment configuration only; it is not placed in remote URLs, files, notebook source, or Git command arguments.

## Review a PR before merging

In the notebook's checkout settings, set `GIT_REF = "refs/pull/123/head"` to test PR #123 from the same repository. Restore `GIT_REF = "main"` for the stable project. A branch name, tag, or commit SHA also works. Experiment metadata records the actual fetched commit.

## Repository files

| File | Purpose |
|---|---|
| `Grid_Control_Lab.ipynb` | Stable Colab launcher |
| `grid_lab.py` | SimBench simulation, OLTC, MLP, and MPC |
| `dashboard.py` | Standalone Gradio app and authenticated launcher |
| `plots.py` | Scientific plots and summaries |
| `checks.py` | Tap, inverter capability, and controller checks |
| `requirements.txt` | Pinned dependencies |
| `scripts/colab_checkout.py` | Safe Git refresh logic used by the notebook |
| `scripts/validate_project.py` | Notebook, source, and physical smoke checks |
| `MODEL.md` | Scenario definitions and scientific assumptions |

## Development

Use Python 3.12. Install `requirements.txt` and `nbformat==5.10.4`, then run:

```bash
python scripts/validate_project.py
```

Open a branch and PR for changes. Describe the grid/controller behavior changed and the validation run. No notebook regeneration is needed for changes to the Python modules. If bootstrap logic changes, run `python scripts/build_launcher.py` and commit the resulting notebook alongside it. This keeps the launcher cell synchronized with its reviewed checkout implementation.

The local checkout helper refuses to overwrite modified tracked source. It also refuses a nonempty unrelated directory or a checkout belonging to another repository.

See [MODEL.md](MODEL.md) for the explicit OLTC retrofit, inverter ratings, forecast assumptions, limits, and validation interpretation.
