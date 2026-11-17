# LOOCV results packet

Open `presentation.pdf` for the current Beamer deck. It shows snapshot
`2026-11-15-runs-0-19`. The figures come from the saved experiment folders.

From the repository root, rebuild with Python and LaTeX:

```bash
python artifacts/results_packet/src/build_packet.py 2026-11-15-runs-0-19
```

The builder reads a finished Spectrome comparison snapshot. Pass a new
snapshot ID after adding runs. The build writes to `presentation.pdf` by
default. Use `--output PATH.pdf` to keep a separate copy.
