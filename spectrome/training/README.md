# Shared graph-filter training

`cohort.py` loads the current cohort. `filters.py` prepares the frozen graph
cache, defines `FeGraphMLP`, and predicts spectra. `helpers.py` validates inputs.
`train.py::train_filter` performs leave-one-subject-out cross-validation (LOOCV).
`run.py` organizes a complete run; `analysis.py` exports tables and figures.
The Torch forward file owns the SGM equations. Notebook changes are not needed
for these functions.

## One complete run

From the repository root in the `spectrome` environment:

```bash
python -m spectrome.training.run
```

This starts the full cohort experiment with the accepted first-run settings:
one Tanh hidden layer of 8 units, omega scaled by `2*pi*45` rad/s, Adam at
`1e-3`, 100 fixed epochs, and seed 7. It prepares the cache, runs LOOCV, and
generates the analysis. There is no held-out checkpoint or settings selection.
`python -m spectrome.training.train` delegates to the same entry point.

Every run gets a new directory under Git-ignored `results/`. The cache belongs
to that run, alongside its configuration and outputs:

```text
results/fe_graph_loocv_<date-and-time>_seed7/
  config.json
  training.log
  results.pt
  cache/
    cohort_training_v2.pt
  checkpoints/
    results.fold-001.pt
    ...
  tables/
    subject_scores.csv
    region_scores.csv
    training_history.csv
    filter_values.csv
  figures/
    training_curves.png / .pdf
    held_out_scores.png / .pdf
    spectra/fold-001.png / .pdf
    filters/fold-001.png / .pdf
    ...
```

Use `--run-dir results/my_run` to name a run or `--data-dir /path/to/data` to
select the directory holding the current cohort files. Existing run directories
are refused rather than overwritten. `config.json` records settings, source-code
hashes, Git state, cache path/hash, subject IDs, frequencies, stage timings, and
run status. Progress and failures are retained in `training.log`; completed
fold checkpoints remain if a later step fails. Automatic training resume is
not implemented.

## Analyze saved results

To regenerate CSVs and PNG/PDF figures without retraining:

```bash
python -m spectrome.training.analysis results/my_run
```

Analysis checks the cache hash, subject/fold membership, frequencies, and
preprocessing against the saved results. Per-region scores are recomputed with
the training transformation and checked against the saved subject means.

- Subject tables contain analytical/learned held-out correlations and paired
  differences. Region tables identify both the observed row and model node,
  using zero-based indices rather than invented anatomical labels.
- History tables contain training loss before each optimizer step. There are
  no per-epoch held-out curves because held-out subjects are scored only after
  training.
- Filter tables contain frequency, real/imaginary components, magnitude, and
  phase for both learned and analytical graph filters. CSV phase is the
  principal angle in radians; plots unwrap it for readability. The analytical
  control uses each held-out subject's fixed `tau_e`.
- Figures show training curves for all folds, paired held-out score comparisons,
  and spectral/filter diagnostics for every subject. Spectrum figures use the
  first, middle, and last observed rows, chosen independently of scores. They
  show magnitude, smoothing, square root, demeaning, and unit-norm normalization:
  spectral shape used for correlation, not calibrated absolute power.

The public `export_results(results, cache, output_dir)` and
`plot_results(results, cache, output_dir, region_rows=...)` functions also accept
saved paths or loaded dictionaries. A chosen region list must be set without
selecting examples for a favorable outcome. Analysis updates only the selected
run's derived tables and figures.

## Prepare and save

```python
from spectrome.training.filters import generate_training_data

cache = generate_training_data(output_path="training_cache.pt")
```

With no supplied subject records, `generate_training_data` first calls
`prepare_cohort(data_dir)`. The default data directory is `spectrome/data`.
Connectivity, MEG, and fitted parameters share chronological subject order,
as confirmed in the Oct. 5 discussion with Anil. The loader uses the same
original index across all three files and excludes an entry only when both
MEG and parameters are empty. A partially missing or malformed pair fails;
there is no separate subject-matching map. The current files yield 36 usable
subjects from 39 records, but the functions derive the cohort size from data.

Preparation preserves the existing notebook's region permutation, empty
nodes, graph preprocessing, template distances (including its `get_mean_C`
operation), and 68 observed cortical outputs of the 86-node graph. It does
not switch datasets, construct a 68-node graph, or refit SGM parameters.
Targets remain raw nonnegative spectra, with the notebook's default 40-bin
frequency grid from 2 to 45 Hz. The MEG file has no stored frequency vector;
this grid is recorded as a notebook convention in provenance.

For custom paired inputs, the original explicit-record call remains available:

```python
cache = generate_training_data(subjects, frequencies_hz, "training_cache.pt",
                               preprocessing_id="my_preparation")
```

Each record supplies a unique `subject_id`, a `brain` with model-order
`reducedConnectome` and `distance_matrix` in millimeters, seven named fixed
`parameters`, raw `empirical_meg` of shape `[R, F]`, ordered `observed_regions`,
`mapping_verified`, and JSON-compatible `provenance`. The cohort loader records
the meeting confirmation, source paths/hashes, original indices, and exclusions.
There are no permanent split labels. Metadata validation checks IDs, duplicates,
alignment confirmation, and provenance; it does not align or reorder subjects.

Version 2 CPU caches contain raw targets, fixed parameters, and a prepared
graph per subject/frequency. Preparation runs without gradients. Learning
reuses the fixed eigenpairs while carrying gradients to the MLP. Regenerate
old version 1 fixed-split caches before training. `purpose="historical_replay"`
allows diagnostic inputs without confirmed alignment; training rejects those
caches.

## Model and prediction

```python
from spectrome.training.filters import FeGraphMLP, predict_spectra

model = FeGraphMLP(hidden_sizes=(8,), omega_scale=2 * 3.141592653589793 * 45)
Fe_graph = model(cache["omega"])
learned_prediction = predict_spectra(cache["subjects"][0], Fe_graph)
analytical_prediction = predict_spectra(cache["subjects"][0])
```

The MLP takes angular frequency in rad/s divided by a fixed scale. Two real
outputs form one shared complex graph-propagation value per frequency. Local
analytical `Fe`, connectivity, distances, and all seven fitted SGM parameters
remain fixed. Ordinary Linear initialization and Tanh are starter choices;
the initial MLP is not guaranteed to reproduce the analytical control.

## Train and save

```python
from spectrome.training.train import train_filter

results = train_filter(cache, "fe_graph_mlp.pt", hidden_sizes=(8,),
                       omega_scale=2 * 3.141592653589793 * 45,
                       epochs=100, learning_rate=1e-3, seed=7)
```

For normal use, the run entry point above handles all paths and analysis.
The direct function remains available for custom workflows; `checkpoint_dir`
optionally places fold checkpoints in a separate directory. The accepted
first-run settings are fixed choices, not evidence of convergence or optimality.
This implementation uses a fixed epoch budget and no early
stopping or held-out selection; tuning would require a separate inner-validation
protocol among the training subjects.

`train_filter` creates one fold per cached subject. Each fold starts a fresh
MLP and Adam optimizer with the same declared seed, trains on all other
subjects, then evaluates the held-out subject once. With the current cohort,
this gives 36 folds of 35 training subjects and one held-out subject. Training
uses CPU float64 and one full training-subject step per epoch. The loss is one
minus mean subject spectral correlation, with equal region weight per subject.
Both prediction and target undergo magnitude, `[1, 2, 5, 2, 1]/11` zero-padded
smoothing, square root, and frequency demeaning. The existing `1e-12` floor
before square root keeps gradients finite; near-constant/nonfinite spectra fail.

The function returns a results dictionary and saves it at the output path.
Each fold also saves a checkpoint such as `fe_graph_mlp.fold-001.pt`, containing
weights/configuration, train/held-out IDs, epoch history, provenance, analytical
and learned held-out predictions/correlations, and learned `Fe_graph` values.
The combined results contain all fold summaries and mean paired differences.
There is no single final model across all subjects in LOOCV.

```python
import torch

fold = torch.load(results["folds"][0]["checkpoint_path"],
                  map_location="cpu", weights_only=True)
model = FeGraphMLP(**fold["model_config"])
model.load_state_dict(fold["state_dict"])
model.eval()
```

Completed fold checkpoints survive an interrupted run. Automatic resume is not
implemented; use a new output name if retaining an earlier run is required.
This evaluates filter transfer conditional on supplied subject-fitted parameters,
not prediction without subject-specific MEG calibration.
