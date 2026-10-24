# Shared graph-filter MLP

The MLP maps normalized angular frequency to one shared complex graph filter.
The analytical local filter and seven fitted SGM parameters remain fixed.
The SGM equations live in `spectrome/forward/`.

## Code layout

```text
mlp/
  model.py                     PyTorch MLP
  spectra.py                   spectral prediction, correlation, and scoring
  data/
    data.py                    cohort loading and forward-input preparation
    helpers.py                 input and cache checks
  training/
    training.py                train_filter(): fit one model
    loocv.py                   run_loocv(): leave-one-out loop
    nested_cv.py               nested selection and outer refitting
    run.py                     one experiment or a declared batch
    helpers.py                 validation, cache IO, records, checkpoints, splits
    training_runs.json         existing fixed-configuration training settings
  analysis/
    analysis.py                score tables and reports
    plots.py                   spectra, filters, and training curves
    statistics.py              paired statistical comparisons
    helpers.py                 loading, verification, and shared utilities
```

## Data preparation and fitting

`data/data.py` contains the preparation phase:

1. `prepare_cohort()` loads aligned connectivity, fitted parameters, and measured MEG.
2. `generate_training_data()` prepares fixed graph/eigenmode inputs and saves
   them with raw targets in a cache. It does not generate synthetic MEG.
3. `prepare_spectral_subject()` prepares fixed local responses and transforms
   targets once before fitting. The learned-filter-dependent response still
   runs during training.

`training/training.py` contains `train_filter()`. It receives its training
subjects, creates a fresh MLP and Adam optimizer, and returns the fitted model
and loss history. It does not choose held-out subjects, rank configurations,
write run folders, or generate reports.

The training arithmetic is unchanged: CPU float64, a declared seed reset just
before model construction, one full-batch Adam update per epoch, and loss equal
to one minus the mean subject spectral correlation. Each subject score averages
regional correlations. The spectral transformation remains magnitude,
zero-padded `[1,2,5,2,1]/11` smoothing, square root with a `1e-12` floor, and
frequency demeaning. Histories record loss before each optimizer update.

## One experiment entry point

```python
from spectrome.mlp.training.run import run_experiment

run_dir = run_experiment(
    "results/experiments/my_run",
    hidden_sizes=(32,), activation="tanh",
    epochs=500, learning_rate=0.001, seed=7,
)
```

With no nested protocol, this runs a fixed configuration through the
leave-one-subject-out loop in `training/loocv.py`. Each fold trains on the other
subjects and evaluates the held-out subject after the final fixed epoch.

`training/run.py` contains the short drivers. `run_experiment()` follows the
preparation, cross-validation, and analysis steps. `training/helpers.py` owns
input validation, cache IO, logs, saved status, and completed-run checks. It refuses an existing output folder. `create_run_record()`
in `training/helpers.py` only builds metadata; it does not train a model.

`run_experiments()` is the batch entry point. It reads
`training/training_runs.json`, which retains the 16 existing configurations:

```bash
python -m spectrome.mlp.training.run
```

This command skips complete matching runs and starts missing runs. Importing
the package does not start the queue.

## Nested cross-validation uses the same fitter

Pass a nested protocol to the same `run_experiment()` function:

```python
protocol = {
    "outer_strategy": "leave_one_subject_out",
    "outer_folds": None,
    "inner_folds": 5,
    "split_seed": 1701,
    "training_seed": 7,
    "candidate_run_ids": ["basic_mlp_6", "basic_mlp_12", "basic_mlp_13", "basic_mlp_15"],
    "declared_comparisons": ["selected_mlp_vs_analytical"],
}

run_dir = run_experiment(
    "results/nested/example",
    cache_source="/path/to/cohort_training_v2.pt",
    protocol=protocol,
)
```

This is an interface example, not an approved production batch. Nested runs
require a saved compatible cache. Their model settings belong in protocol
candidates, not in the fixed-configuration arguments.

`candidate_run_ids` selects entries by `results_dir` name from
`training/training_runs.json`, in the order supplied. Only those configurations
are compared. Each fit starts with fresh weights; saved historical weights are
not loaded. The candidate seeds must match the protocol's `training_seed`.
Unknown or duplicate IDs and conflicting explicit `candidates` are rejected
before a run folder is created.

For a separate configuration file, pass `training_runs_path="/path/to/runs.json"`
to `run_experiment()`. In a `run_experiments(config_path)` batch, candidate IDs
refer to entries in that same file. Explicit `candidates` remain available for
standalone configurations. The run's `config.json` saves the expanded candidate
settings and IDs; analysis uses that snapshot and does not reread the queue.

Nested cross-validation is a model-selection and evaluation procedure that
requires repeated calls to the unchanged fitter. It is grouped with the run
procedures because it determines which subjects each fit receives.
`training/nested_cv.py` selects settings using inner validation subjects drawn
only from each outer training set. It trains a fresh model on the entire outer
training set and evaluates the outer test subjects. Every fit calls the same
`train_filter()` function. Selection uses pooled mean raw subject correlation,
with equal subject weight; the first declared candidate wins an exact tie.
Split seeds and training seeds are separate.

Nested evaluation requires new inner fits and outer refits. Historical results
cannot become nested results through post-processing. No full nested cohort
experiment has run. Earlier configuration choices used this cohort, so nested
evaluation cannot make it untouched. Fixed SGM parameters were fitted using MEG;
this is not a structure-only prediction claim.

## Analysis API

```python
from spectrome.mlp.analysis import analyze_run

outputs = analyze_run("results/experiments/my_run")
```

The same API handles nested folders. `analyze_run()` loads through helpers and
selects `_write_loocv_report()` or `_write_nested_report()` to format the saved
results. These are internal report writers, not separate training or evaluation
runs. `analysis.py` exports tables and writes reports; `plots.py` draws figures. Normalized spectra and correlation describe
shape, not calibrated absolute power. Sample-SD bands describe variation across
subjects, not confidence intervals.

`analysis/helpers.py` owns data loading for both reports and statistical
comparisons, including configuration, cache, checkpoints, and score tables. It reads checkpoints and
replays predictions without fitting or changing weights. Analysis and queue
skipping share these checks: files, subject membership, selected settings,
seeds, predictions, summaries, and statistics must agree. Invalid runs fail
before reports or tables are rewritten. There is no automatic repair of saved
scientific evidence or automatic resume of incomplete runs.

Statistical comparisons use saved scores and require no training:

```python
from spectrome.mlp.analysis.statistics import (
    compare_saved_run_to_analytical, compare_saved_runs,
)

comparison = compare_saved_run_to_analytical(
    "results/experiments/basic_mlp_12", contrast="run12_vs_analytical",
)
comparison = compare_saved_runs(
    "results/experiments/basic_mlp_12",
    "results/experiments/basic_mlp_6", contrast="run12_vs_run6",
)
```

Each subject contributes its mean regional r, transformed with Fisher's
`atanh`, to a two-sided paired t-test. Pairs are matched by subject ID. These
retrospective tests are exploratory: configuration choices used this cohort,
and cross-validation training sets overlap.

## Checks

Use the project's `spectrome` environment:

```bash
python -m pytest spectrome/tests -q
```

Tests cover seeded training behavior, graph and spectrum parity, subject
isolation, nested selection, checkpoint loading, queue handling, statistical
comparisons, and saved-result consistency.
