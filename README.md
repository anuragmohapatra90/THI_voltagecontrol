# Voltage control for ADG

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anuragmohapatra90/THI_voltagecontrol/blob/main/Grid_Control_Lab.ipynb)

The grid, household loading, PV generation and three DER control sites come from [ADG_datadriven](https://github.com/anuragmohapatra90/ADG_datadriven). A Gradio dashboard compares **Uncontrolled**, **OLTC** and **MLP + MPC** together.

## Run

1. Open the Colab badge, use a CPU runtime and **Run all**. The MLP trains automatically on this grid.
2. Open the printed dashboard link in another tab. There is no dashboard login. Keep Colab connected.
3. **Load new scenario** generates a fresh synthetic daily profile. Adjust load **×0.5–5** and PV **×0–5** for stronger stress cases. The graph reacts automatically; changing sliders or MPC gains keeps the underlying scenario fixed.

Setup stays on the left. The center overlays the three physical voltage curves and shows all three network maps. Triangles mark mixed load/PV sites, diamonds mark mixed load/PV sites with a control DER, and orange borders identify a limited MPC resource. Both feeder heads (buses 2 and 9) already have load and PV, as do all customer buses 2–14. Bus 1 is the secondary busbar and has neither. Site hover shows gross load P, load Q, PV P and net P separately, including zero PV at night. Hover any voltage-curve point to see **all three actual AC voltages at that bus**, even when the curves overlap. The same tooltip explains why the OLTC is holding or changing tap and shows its local secondary voltage/deadband. Matching uncontrolled/OLTC curves are expected when the relay holds at neutral. The values are not offset to separate curves.

The right-hand diagnostics show **ΔP/ΔQ relative to uncontrolled**, with **MAX** badges when either signed ±50 kW/kVar limit is reached. Underneath are the MLP estimate, actual controlled AC voltage and signed error (estimate − actual) for every bus, followed by the complete MPC cost function and bounds. The scenario time is a synthetic time of day.

**Load new scenario** immediately shows a loading label and spinner, disables duplicate clicks and marks the previous plot as pending. Results and diagnostics update together. A failed power flow leaves the last valid scenario visible, shows a persistent error and releases the controls so another setting or scenario can be tried. Sidebar labels and values stay readable in light or dark browser themes. Gain hints explain the penalty direction: increasing kP/kQ discourages the corresponding power change; increasing kV puts more weight on voltage correction.

After merging a PR, reopen the same GitHub-linked Colab URL, restart the runtime and Run all. No notebook uploads are needed. This repository is private, so the checkout still needs its read-only GitHub token through the Colab secret `GITHUB_TOKEN` or hidden prompt.

## Experiment

The original 15-bus, two-feeder grid uses a 0.4 MVA 20/0.4 kV transformer and NAYY 4x120 SE cables. Loads and PV remain at the original buses. The controllable DERs are at **5, 8 and 14**, with the original **±50 kW / ±50 kVar** bounds. Positive actions supply power; negative actions absorb power.

The original reference files for grid construction, profile generation, configuration and MLP architecture are copied byte-for-byte. One seeded realization of the reference's random line lengths is shared by model training and the dashboard. The MLP is retrained for that exact plant. The larger dashboard scaling range does not change the reference training campaign; high-scale cases can extrapolate beyond its training data. A non-convergent power flow reports an error and retains the previous result. Its 64/32 ReLU architecture, Adam/MSE training and SLSQP voltage/action objective are preserved.

The transformer gains a local OLTC relay with nine taps (−4…+4 at 1.25%), a **±5% secondary-reference range**. Positive taps raise voltage. It measures **only LV bus 1**, holds its previous tap, and steps until the local voltage reaches the experiment's **0.99–1.01 pu deadband** or a tap limit. Bus 0 is the MV slack; bus 1 already varies with load and tap, so no extra bus/line is required. The sidebar shows the measured secondary voltage.

The display is a settled snapshot, without real tap delay/motor timing. The OLTC has no household or feeder-end voltage feedback. A safe secondary voltage can coexist with an unsafe feeder end; the OLTC then holds while MPC can act at its three DER sites. MPC uses MLP estimates and effort penalties, so it also has estimation/capability limits. All displayed voltages are actual AC results. The old all-bus best-tap search is retained only for offline checks. See [MODEL.md](MODEL.md) for source guidance and assumptions.

## Development

Edit Python on a branch and open a PR. Regenerate the thin notebook after launcher changes:

```bash
python scripts/build_launcher.py
python -m pip install -r requirements.txt nbformat==5.10.4
python scripts/validate_project.py
```

CI checks upstream file identity, reference-grid AC parity, the original scenario/scaler behavior, controller signs and limits, local relay feedback, tap range and rollback, offline exhaustive-tap checks and the login-free browser dashboard on Python 3.12 and 3.13.
