# SGM graph-filter update — October 19, 2026

Open `presentation.pdf` to present. Each PDF page advances one Beamer reveal.
The editable source is `src/main.tex`; it uses the October 5 deck's 16:9 white,
navy, and green style. No PowerPoint conversion is included.

## Sequence

- Title page.
- MLP Implementation: four reveals, explanation left and source excerpts right.
- Training and Evaluation: five reveals in the same layout.
- Experiment settings: initial comparison and larger-model/follow-up tables.
- Performance: one table of completed held-out results.
- Runs 0–12: one slide each with training curves and mean normalized spectra.
- Runs 13–15: completed results, each with training curves and mean normalized spectra.

There are 22 logical frames, including the title, and 29 click-through PDF pages.
Runs follow queue order: paired Tanh/Sigmoid settings for 8 units, 32 units,
and two 32-unit layers, then the larger model and the three matched-budget
follow-ups. The October 18 refresh includes all 16 completed runs (0–15).
Run 16 was outside this batch, is absent from the current queue, and has no output folder.
The follow-ups use 5,000 epochs, learning rate 0.0003, Tanh, and seed 7.
Run 13: r 0.862011, 31/36 improved. Run 14: r 0.857671, 26/36 improved.
Run 15: r 0.859865, 29/36 improved.

## Sources and interpretation

`figures/evidence.json` records source hashes, saved configurations, and scores
recomputed from the 36-subject CSV tables. Figures are copied unchanged from
saved completed experiments. Code snippets come from current source, with line
wrapping for slides; the optimizer excerpt explicitly marks omitted checks.

Scores measure mean regional spectral correlation. Spectral figures show
transformed, unit-normalized shapes, not absolute power; bands are sample SD
across subjects, not confidence intervals. Subject-fitted SGM parameters remain
fixed. The comparisons are exploratory and use seed 7. No training was rerun. The collection checks replay saved weights and reconstruct
held-out predictions/scores for runs 13–15; they do not establish independent
scientific replication. See the [collection receipt](../server-launch/basic_mlp_13_15/local_verification.json).

## Rebuild

From the repository root:

```sh
zsh artifacts/10-19-presentation/src/build.sh
```

This compiles the existing source and renders every page. To refresh tables and
figures after completed results have been brought back locally, run
`src/assemble.py` with the Spectrome Python environment, then rebuild. Assembly
regenerates `src/main.tex`, snippets, and figure copies, so preserve manual source
edits before reassembly. Assembly requires completed scores for every queued run
and checks saved settings against the queue. Render checks are in `build/verification.json`.

## Walkthrough style

Use numbered action steps, starting with “1) Compute inputs offline”. Keep the
explanation together on the left in natural presentation language. Put the file
and function on two lines above the code on the right. Fold context into the
main explanation instead of adding a separate bottom callout.

Use Input / Output / Hyperparameters for the MLP description, defining the real
and imaginary outputs before the complex-filter equation. Keep downstream SGM
operations together. Explain each LOOCV fold with a concrete held-out subject;
distinguish that split from spectrum preprocessing.
