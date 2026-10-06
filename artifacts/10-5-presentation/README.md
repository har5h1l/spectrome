# SGM Graph Filter Progress Update

October 5, 2026. Harshil Shah, Mission San Jose High School.

`presentation.pdf` contains a cover and eight main Beamer frames, delivered as
18 click-through pages. Overlays replace content as the explanation progresses.

`presentation.pptx` preserves the same 18 steps as full-slide images with speaker
notes. Edit `src/main.tex` to change visible content. PowerPoint text and equations
are not separate editable objects.

The design follows the supplied BasinBreakers slides: 16:9, white background,
navy text, green accents, and staged explanations.

## Contents

| Frame | Title | PDF pages |
|---|---|---|
| Cover | SGM Graph Filter Progress Update | 1 |
| 1 | network_transfer_local_alpha | 2–5 |
| 2 | Paper Reproduction | 6–7 |
| 3 | What still needs to be confirmed? | 8–9 |
| 4 | Torch Functions | 10–13 |
| 5 | MLP Implemented So Far | 14–15 |
| 6 | Preliminary Implementation Choices | 16 |
| 7 | MLP Training | 17 |
| 8 | Planned Experiment | 18 |

## Evidence and limits

The spectra image now shows retained record 1 from the measured cohort replay.
A single-record forward rerun regenerated the raw regional responses and verified
that the transformed spectra exactly match the saved replay arrays. The selected
record remains the previously chosen lower median correlation, not a newly
selected best-looking case.

The plot follows the original notebook method: all 68 regions, cyan and blue
curves, red and black means, magnitude smoothing and dB conversion. It applies
no additional normalization. `figures/replay-spectrum-provenance.json` records
the source hashes and numeric check. `figures/replay-example-raw.npz` contains
the exact plotted raw inputs and model responses.

The older `notebook-spectrum.png` remains as a reference but is no longer used
on the slides. Source citations and qualifications are in the speaker notes,
with no visible source footer.

The replay matched all 36 saved scores within 1e-9. `figures/evidence.json`
records measurements, package versions, and hashes. `figures/cohort_scores.csv`
stores per-record scores. `figures/historical_spectra.npz` retains replayed
spectra for inspection; the revised slides do not plot these arrays.

Raj et al. (2020), [Figures 3 and 4c](https://pmc.ncbi.nlm.nih.gov/articles/PMC7336150/),
doi:10.1002/hbm.24991, show the corresponding analyses. The equality check is
against the repository's `individualC_optP.mat` scores. This is a replay with
supplied fits, not fresh fitting or full paper reproduction.

Historical positional pairing is retained. Subject and region identity remain
unresolved. No verified training cache, new parameter fit, or cohort MLP training
run was created for the presentation. Held-out evaluation would still use the
saved MEG-fitted SGM parameters.

The prior focused three-file test run returned **14 passed and 1 failed**. The
failed test expects `unverified`, while the input rejection says `not verified`.
This failure occurs before the test's second rejection check. Details remain in
speaker notes and `figures/test-results.txt`. This revision reran one fixed forward example to generate the displayed figure.
It did not fit parameters, train the MLP, or change scientific source code.

## Rebuild

From the repository root on this Mac:

```bash
zsh artifacts/10-5-presentation/src/build.sh
```

This uses existing MacTeX and the installed Codex workspace runtime. It compiles
the Beamer source, renders each overlay, and exports the validated PPTX without
running the model. `build/` holds disposable compilation and export files.

To deliberately regenerate the displayed replay figure:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/envs/spectrome/bin/python artifacts/10-5-presentation/src/plot_replay_example.py
zsh artifacts/10-5-presentation/src/build.sh
```

This checks source hashes and agreement with the existing replay before plotting.

The source remains editable in Codex. The built-in compiler currently cannot
resolve the external PNG, so use the local build and open `presentation.pdf`
for the complete preview. Keep `src/main.tex` and `figures/replay-spectrum.png`
together in this directory structure.

To deliberately refresh the measured cohort replay, use
`src/collect_evidence.py` in the Spectrome environment. It is not part of the
presentation build.
