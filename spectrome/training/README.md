# Shared graph-filter training

This folder exposes Python functions and a small JSON experiment queue.
The queue runner has no command-line options. No notebook is created or changed.

| Module | Responsibility |
| --- | --- |
| `data.py` | Load chronological cohort records, generate the graph cache, and prepare fixed training tensors. |
| `model.py` | Define the frequency-only complex `FeGraphMLP`. |
| `spectra.py` | Predict observed regional spectra, transform spectra, and calculate correlation. |
| `train.py` | Run LOOCV or coordinate a complete saved experiment. |
| `run.py` | Read `experiments.json` and run missing experiments sequentially. |
| `analysis.py` | Export saved predictions to CSV tables and 300-dpi PNG figures. |
| `helpers.py` | Validate inputs, load checkpoints, capture logs, and write automatic run records. |

The SGM equations remain in `forward/network_transfer_torch.py`. Preparation
uses forward steps 1–2; training repeats step 3 and omits step 4 (FC).

## Experiment queue

Edit `experiments.json` in this folder. Each entry has a `results_dir` relative
to `results/experiments/` and explicit model/training `parameters`. The initial
queue contains the existing 8-unit Tanh/100-epoch run and one next experiment:
8-unit Sigmoid/100 epochs. Both use learning
rate `0.001`, seed `7`, and omega scale `2*pi*45`.

Folders are named `basic_mlp_0` and `basic_mlp_1` in queue order.
`basic_mlp_0` is the completed baseline. Settings stay in the JSON and each
saved run record; folder names do not need to repeat them.
Only the hidden activation changes in this comparison. Review these results
before adding another experiment with more epochs or hidden units.

From the repository root:

```bash
python -m spectrome.training.run
```

The runner skips completed runs only when the recorded settings match and the
results file exists. Older records without activation default to Tanh. An
existing incomplete or mismatched directory stops the queue; inspect it or
choose a new folder name rather than overwrite it. Training errors also stop
the queue. No automatic resume or settings selection is performed. The first
entry points to the actual completed run, so it is skipped on this checkout.
On a checkout without that result folder, the baseline will run too.

`run_experiments()` can also be imported from `spectrome.training.run`.
The queue was added for review; its new experiments have not been launched.

## One complete run from Python

```python
from spectrome.training.train import run_experiment

run_dir = run_experiment(
    hidden_sizes=(8,),
    activation="tanh",
    epochs=100,
    learning_rate=1e-3,
    seed=7,
)
```

These are the accepted first-run defaults: one 8-unit Tanh hidden layer,
omega divided by `2*pi*45` rad/s, Adam, 100 fixed epochs, and seed 7. Hidden
activations can be `tanh` or `sigmoid`; both final outputs remain linear.
Settings are function arguments, not manually maintained configuration files.
Each run automatically records the settings that produced its outputs.

`run_experiment` prepares data, calls `train_filter`, and generates analysis.
Use `run_dir="results/experiments/my_run"` for a named output folder or
`data_dir="/path/to/data"` to select cohort files. Existing run folders are
refused. Runs default to `results/experiments/`. Automated tests use temporary
directories; there is no separate run category or verification output location.

```text
results/experiments/basic_mlp_0/
  summary.md
  config.json
  training.log
  results.pt
  cache/cohort_training_v2.pt
  checkpoints/results.fold-001.pt
  ...
  tables/
    subject_scores.csv
    region_scores.csv
    training_history.csv
    filter_values.csv
  figures/
    summary/training_curves.png
    summary/held_out_scores.png
    subjects/<subject>/spectra.png
    subjects/<subject>/filter.png
```

`config.json` is an automatic run record. It contains actual settings,
source-code hashes, Git state, cache path/hash, subject IDs, frequencies,
timings, and run status. Progress and failures are retained in `training.log`.
Completed fold checkpoints survive a later failure. Automatic resume is not
implemented; use a new run folder to retain earlier evidence.

Use `cache_source="/path/to/cohort_training_v2.pt"` to reuse a compatible frozen
cache. Its purpose, version, and forward-source hash are validated, and it is
copied into the new run folder. Cache reuse cannot be combined with preparation
inputs (`data_dir`, `subjects`, `frequencies_hz`, or `preprocessing_id`). The
cache retains its recorded frequencies, preprocessing, and provenance; reuse
does not refresh source data. Generate a new cache after a forward-source change.

## Data preparation

```python
from spectrome.training.data import prepare_cohort, generate_training_data

cache = generate_training_data(output_path="training_cache.pt")
```

Without supplied records, `generate_training_data` first calls `prepare_cohort`.
The default data directory is `spectrome/data`. Connectivity, MEG, and fitted
parameters share chronological subject order, as confirmed in the Oct. 5
discussion with Anil. One original index is used across all files. A record is
excluded only when both MEG and parameters are empty; a partially missing or
malformed pair fails. The current files yield 36 usable subjects from 39
records, but functions derive the cohort size from data.

Preparation preserves the existing notebook's region permutation, empty nodes,
graph preprocessing, template distances (including its `get_mean_C` operation),
and 68 observed cortical outputs of the 86-node graph. It does not construct a
68-node graph, switch datasets, or refit SGM parameters. Raw nonnegative targets
use the notebook's 40-bin grid from 2 to 45 Hz; the MEG file has no frequency
vector, so this convention is recorded in provenance.

Custom inputs remain supported:

```python
cache = generate_training_data(
    subjects, frequencies_hz, "training_cache.pt",
    preprocessing_id="my_preparation",
)
```

Each record supplies a unique `subject_id`, a `brain` with model-order
`reducedConnectome` and `distance_matrix` in millimeters, seven named fixed
`parameters`, raw `empirical_meg` with shape `[R, F]`, ordered `observed_regions`,
`mapping_verified`, and JSON-compatible provenance. Validation checks metadata
and shapes; it does not infer subject correspondence or reorder subjects.

Version 2 CPU caches hold raw targets, fixed parameters, and a prepared graph
per subject/frequency. At the start of each training call,
`prepare_spectral_subject` computes fixed local responses and transformed
targets in memory. Frequency batching preserves the denominator floor over all
graph modes at each frequency before readout mode selection. Old version 1
fixed-split caches must be regenerated. Historical replay caches remain
available for diagnostics and are rejected by training.

## Model, prediction, and LOOCV

```python
from spectrome.training.model import FeGraphMLP
from spectrome.training.spectra import predict_spectra
from spectrome.training.train import train_filter

model = FeGraphMLP(hidden_sizes=(8,), omega_scale=2 * 3.141592653589793 * 45)
values = model(cache["omega"])
prediction = predict_spectra(cache["subjects"][0], values)

results = train_filter(
    cache, "fe_graph_mlp.pt", hidden_sizes=(8,),
    omega_scale=2 * 3.141592653589793 * 45,
    epochs=100, learning_rate=1e-3, seed=7, activation="tanh",
)
```

The MLP maps angular frequency to one shared complex graph-propagation filter.
Local analytical `Fe`, connectivity, distances, and all seven fitted SGM
parameters remain fixed. Default initialization need not reproduce the
analytical filter.

`train_filter` creates one LOOCV fold per subject, each with a fresh MLP and
Adam optimizer and the same declared seed. Each fold trains on all other
subjects for a fixed epoch budget, then evaluates its held-out subject. With
this cohort, that is 36 folds with 35 training subjects. Training uses CPU
float64 and one full training-subject update per epoch. The loss is one minus
mean subject spectral correlation, with equal region weight within each subject.

Both predictions and targets use magnitude, zero-padded `[1, 2, 5, 2, 1]/11`
smoothing, square root with a `1e-12` floor, and frequency demeaning. Fixed target
transforms are reused. Near-constant/nonfinite spectra and nonfinite
losses/gradients fail. The epoch history records loss before each optimizer
step; final training scores are computed after the final update.

Checkpoints contain weights, model settings (including activation), subject
roles, history, provenance, held-out predictions/scores, final mean training
scores, and learned filter values. Their paths are relative to `results.pt`,
so complete run folders can be moved. Older absolute paths are supported;
checkpoints with no activation setting default to Tanh.

```python
from spectrome.training.helpers import load_fold_checkpoint

fold = load_fold_checkpoint("fe_graph_mlp.pt", results["folds"][0])
model = FeGraphMLP(**fold["model_config"])
model.load_state_dict(fold["state_dict"])
model.eval()
```

No fold selects settings or checkpoints using held-out scores. Tuning requires
inner validation among training subjects for an unbiased outer evaluation.
There is no single final all-subject model in LOOCV. This experiment measures
filter transfer conditional on supplied subject-fitted parameters.

## Analyze saved results

```python
from spectrome.training.analysis import analyze_run

outputs = analyze_run("results/experiments/my_run")
```

Analysis regenerates tables, PNG figures, and `summary.md` without retraining.
It checks cache hashes, frequencies, preprocessing, and fold membership and
recomputes regional correlations to verify saved held-out means.

- Subject tables contain analytical/learned held-out correlations, paired
  differences, and final mean training correlations. Older results leave the
  training-score columns blank.
- Region tables identify zero-based observed rows and model-node indices.
- History tables contain training loss before each step. There are no per-epoch
  held-out curves.
- Filter tables contain real/imaginary components, magnitude, and principal
  phase in radians. Plots unwrap phase. The analytical control uses each
  subject's fixed `tau_e`.

Figures use plain Matplotlib axes, standard colors, compact A/B/C panels, and
300-dpi PNGs. Training plots show all folds and their mean. Score plots show
paired scores, equal-scale scatter with an identity line, and signed
differences. Spectrum and filter figures cover every held-out subject. Example
regions are the first, middle, and last observed rows, fixed independently of
scores. Spectrum overlays show the loss transformation followed by unit-norm
normalization: spectral shape, not calibrated absolute power.

`export_results(results, cache, output_dir)` and
`plot_results(results, cache, output_dir, region_rows=...)` accept saved paths
or loaded dictionaries. Analysis changes only the selected run's derived
outputs; existing scientific results and checkpoints are retained.
