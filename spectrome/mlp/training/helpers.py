"""Run bookkeeping, checkpoints, split utilities, and input checks."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch

from ..spectra import SCORING_ID
from ..data.data import generate_training_data
from ..data.helpers import validate_training_cache


DEFAULT_PARAMETERS = {
    "hidden_sizes": (8,), "omega_scale": 2 * math.pi * 45,
    "epochs": 100, "learning_rate": 1e-3, "seed": 7, "activation": "tanh",
}


# Settings and candidate catalogs


def validate_fit_settings(model_config, training_config):
    """Check all settings before creating a model or starting any fit."""
    if set(model_config) != {"hidden_sizes", "omega_scale", "activation"}:
        raise ValueError("model_config must specify hidden_sizes, omega_scale, and activation")
    if set(training_config) != {"epochs", "learning_rate", "seed"}:
        raise ValueError("training_config must specify epochs, learning_rate, and seed")
    widths = model_config["hidden_sizes"]
    if not widths or any(isinstance(size, bool) or not isinstance(size, int) or size < 1
                         for size in widths):
        raise ValueError("hidden_sizes must contain positive integer widths")
    if model_config["activation"] not in {"tanh", "sigmoid"}:
        raise ValueError("activation must be tanh or sigmoid")
    if not 0 < float(model_config["omega_scale"]) < math.inf:
        raise ValueError("omega_scale must be finite and positive (rad/s)")
    epochs = training_config["epochs"]
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    rate = training_config["learning_rate"]
    if isinstance(rate, bool) or not 0 < float(rate) < math.inf:
        raise ValueError("learning_rate must be finite and positive")
    seed = training_config["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")


def normalize_candidates(candidates, training_seed):
    if isinstance(training_seed, bool) or not isinstance(training_seed, int):
        raise ValueError("training_seed must be an integer")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a nonempty list")
    normalized = []
    seen = set()
    for candidate in candidates:
        if set(candidate) != {"candidate_id", "model_config", "training_config"}:
            raise ValueError("Each candidate must define candidate_id, model_config, and training_config")
        candidate_id = candidate["candidate_id"]
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in seen:
            raise ValueError("Candidate IDs must be unique nonempty strings")
        seen.add(candidate_id)
        model_config = candidate["model_config"]
        training_config = candidate["training_config"]
        if set(model_config) != {"hidden_sizes", "omega_scale", "activation"}:
            raise ValueError(f"{candidate_id}: complete model_config is required")
        if set(training_config) != {"epochs", "learning_rate"}:
            raise ValueError(f"{candidate_id}: training_config must specify epochs and learning_rate only")
        validate_fit_settings(model_config, {**training_config, "seed": training_seed})
        normalized.append({
            "candidate_id": candidate_id,
            "model_config": {
                "hidden_sizes": list(model_config["hidden_sizes"]),
                "omega_scale": float(model_config["omega_scale"]),
                "activation": model_config["activation"],
            },
            "training_config": {
                "epochs": training_config["epochs"],
                "learning_rate": float(training_config["learning_rate"]),
            },
        })
    return normalized


def validate_nested_settings(protocol):
    """Reject missing or unsupported protocol settings before any fits."""
    required = {"outer_strategy", "inner_folds", "split_seed", "training_seed", "candidates"}
    if not required <= protocol.keys():
        raise ValueError(f"Nested protocol is missing settings: {sorted(required - protocol.keys())}")
    if protocol["outer_strategy"] not in {"leave_one_out", "leave_one_subject_out", "kfold", "grouped_kfold"}:
        raise ValueError("outer_strategy must be leave_one_subject_out or grouped_kfold")
    for name in ("split_seed", "training_seed"):
        if isinstance(protocol[name], bool) or not isinstance(protocol[name], int) or protocol[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    folds = protocol["inner_folds"]
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("inner_folds must be an integer of at least two")
    comparisons = protocol.get("declared_comparisons", [])
    if not isinstance(comparisons, list) or any(item != "selected_mlp_vs_analytical" for item in comparisons):
        raise ValueError("Supported nested contrast: selected_mlp_vs_analytical")


def validate_run_inputs(parameters, preparation, cache_source, protocol, *, training_runs_path=None):
    """Normalize settings and reject conflicting data or protocol inputs."""
    if cache_source is not None and any(value is not None for value in preparation.values()):
        raise ValueError("cache_source cannot be combined with data preparation inputs")
    parameters = {**parameters, "hidden_sizes": tuple(parameters["hidden_sizes"])}
    if protocol is not None:
        if not isinstance(protocol, dict) or cache_source is None:
            raise ValueError("Nested experiments require a protocol dictionary and cache_source")
        if parameters != DEFAULT_PARAMETERS:
            raise ValueError("Nested model and training settings belong in protocol candidates")
        protocol = resolve_nested_protocol(protocol, training_runs_path)
        protocol["candidates"] = normalize_candidates(protocol["candidates"], protocol["training_seed"])
        validate_nested_settings(protocol)
    elif training_runs_path is not None:
        raise ValueError("training_runs_path requires a nested protocol")
    else:
        validate_fit_settings(
            {key: parameters[key] for key in ("hidden_sizes", "omega_scale", "activation")},
            {key: parameters[key] for key in ("epochs", "learning_rate", "seed")},
        )
    return parameters, protocol


def load_training_runs(config_path=None):
    """Read declared batch settings without starting any run."""
    path = Path(config_path) if config_path is not None else Path(__file__).with_name("training_runs.json")
    return json.loads(path.read_text())


def resolve_nested_protocol(protocol, config_path=None):
    """Snapshot selected training-run settings in declared candidate order.

    Run IDs are the JSON entries' results_dir names, not list positions or
    saved model weights. Saved analysis uses the expanded settings only.
    """
    protocol = json.loads(json.dumps(protocol, allow_nan=False))
    if "candidate_run_ids" not in protocol:
        return protocol
    if "candidates" in protocol:
        raise ValueError("Specify candidate_run_ids or explicit candidates, not both")
    run_ids = protocol.pop("candidate_run_ids")
    if (not isinstance(run_ids, list) or not run_ids
            or any(not isinstance(value, str) or not value for value in run_ids)
            or len(run_ids) != len(set(run_ids))):
        raise ValueError("candidate_run_ids must be a nonempty list of unique run names")
    entries = load_training_runs(config_path)
    by_id = {}
    for entry in entries:
        run_id = entry["results_dir"]
        if run_id in by_id:
            raise ValueError(f"Duplicate training run ID: {run_id}")
        by_id[run_id] = entry
    candidates = []
    for run_id in run_ids:
        if run_id not in by_id:
            raise ValueError(f"Unknown candidate training run: {run_id}")
        entry = by_id[run_id]
        if entry.get("evaluation") == "nested_cross_validation":
            raise ValueError(f"Candidate must be a fixed training configuration: {run_id}")
        parameters = entry.get("parameters", {})
        if set(parameters) != set(DEFAULT_PARAMETERS):
            raise ValueError(f"Candidate {run_id} must specify exactly {tuple(DEFAULT_PARAMETERS)}")
        if parameters["seed"] != protocol["training_seed"]:
            raise ValueError(f"Candidate {run_id} seed differs from protocol training_seed")
        candidates.append({
            "candidate_id": run_id,
            "model_config": {key: parameters[key] for key in
                             ("hidden_sizes", "omega_scale", "activation")},
            "training_config": {key: parameters[key] for key in ("epochs", "learning_rate")},
        })
    protocol["candidates"] = candidates
    return protocol


def queue_run_directory(experiment, root, seen):
    """Require a unique output directory contained by the batch root."""
    directory = (root / experiment["results_dir"]).resolve()
    if directory == root or not directory.is_relative_to(root) or directory in seen:
        raise ValueError(f"Results directory must be unique and inside {root}: {directory}")
    seen.add(directory)
    return directory


def queue_run_arguments(experiment, directory, config_path=None):
    """Validate one queue entry and build arguments for run_experiment()."""
    if experiment.get("evaluation") == "nested_cross_validation":
        cache_source = Path(experiment["cache_source"]).resolve()
        if not cache_source.is_file():
            raise FileNotFoundError(f"Nested queue cache is missing: {cache_source}")
        protocol = resolve_nested_protocol(experiment["protocol"], config_path)
        return {"cache_source": cache_source, "protocol": protocol}
    parameters = experiment["parameters"]
    if set(parameters) != set(DEFAULT_PARAMETERS):
        raise ValueError(f"Specify exactly {tuple(DEFAULT_PARAMETERS)} for {directory.name}")
    return parameters


# Subject splits


def _subject_ids(subject_ids):
    values = list(subject_ids)
    if len(values) < 2 or any(not isinstance(value, str) or not value for value in values):
        raise ValueError("At least two nonempty subject IDs are required")
    if len(values) != len(set(values)):
        raise ValueError("Subject IDs must be unique")
    return values


def inner_split_seed(split_seed, outer_fold):
    """Derive an inner-split seed without consuming training random state."""
    return int(np.random.SeedSequence([split_seed, outer_fold]).generate_state(1)[0])


def make_outer_splits(subject_ids, *, strategy="leave_one_subject_out",
                      n_splits=None, seed=7):
    """Return realized leave-one-out or shuffled grouped outer memberships."""
    ids = _subject_ids(subject_ids)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("split seed must be an integer")
    if strategy in {"leave_one_out", "leave_one_subject_out"}:
        if n_splits not in (None, len(ids)):
            raise ValueError("Leave-one-subject-out has one outer fold per subject")
        tests = [[subject_id] for subject_id in ids]
        strategy = "leave_one_subject_out"
    elif strategy in {"kfold", "grouped_kfold"}:
        if isinstance(n_splits, bool) or not isinstance(n_splits, int):
            raise ValueError("Grouped outer evaluation requires an integer n_splits")
        if n_splits < 2 or n_splits >= len(ids):
            raise ValueError("n_splits must be at least 2 and less than the subject count")
        permutation = np.random.default_rng(seed).permutation(len(ids))
        groups = np.array_split(permutation, n_splits)
        tests = [{ids[index] for index in group.tolist()} for group in groups]
        # Keep the source cohort order in each realized membership record.
        tests = [[subject_id for subject_id in ids if subject_id in group]
                 for group in tests]
        strategy = "grouped_kfold"
    else:
        raise ValueError("strategy must be leave_one_subject_out or grouped_kfold")

    result = []
    seen = set()
    all_ids = set(ids)
    for index, test_ids in enumerate(tests, start=1):
        train_ids = [subject_id for subject_id in ids if subject_id not in test_ids]
        if set(train_ids) & set(test_ids) or set(train_ids) | set(test_ids) != all_ids:
            raise RuntimeError("Generated outer split does not partition the cohort")
        if seen.intersection(test_ids):
            raise RuntimeError("Generated outer test memberships overlap")
        seen.update(test_ids)
        result.append({
            "outer_fold": index,
            "strategy": strategy,
            "training_subject_ids": train_ids,
            "test_subject_ids": list(test_ids),
        })
    if seen != all_ids:
        raise RuntimeError("Generated outer test memberships do not cover the cohort")
    return result


def make_inner_splits(subject_ids, *, n_splits=5, seed=7):
    """Partition outer-training IDs into shuffled, balanced inner folds.

    This function does not lower ``n_splits`` when the requested count is not
    possible. Every returned validation subject occurs exactly once.
    """
    ids = _subject_ids(subject_ids)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("split seed must be an integer")
    if isinstance(n_splits, bool) or not isinstance(n_splits, int):
        raise ValueError("n_splits must be an integer")
    if n_splits < 2 or n_splits > len(ids):
        raise ValueError("n_splits must be at least 2 and no greater than the subject count")
    permutation = np.random.default_rng(seed).permutation(len(ids))
    groups = np.array_split(permutation, n_splits)
    validation_groups = [
        {ids[index] for index in group.tolist()} for group in groups
    ]
    result = []
    seen = set()
    all_ids = set(ids)
    for index, validation_set in enumerate(validation_groups, start=1):
        validation_ids = [subject_id for subject_id in ids if subject_id in validation_set]
        training_ids = [subject_id for subject_id in ids if subject_id not in validation_set]
        if set(training_ids) & set(validation_ids) or set(training_ids) | set(validation_ids) != all_ids:
            raise RuntimeError("Generated inner split does not partition outer training subjects")
        if not validation_ids or seen.intersection(validation_ids):
            raise RuntimeError("Generated inner validation memberships are empty or overlap")
        seen.update(validation_ids)
        result.append({
            "inner_fold": index,
            "training_subject_ids": training_ids,
            "validation_subject_ids": validation_ids,
        })
    if seen != all_ids:
        raise RuntimeError("Generated inner validation memberships do not cover the outer training set")
    return result


# Run preparation, status, and logging


def initialize_run(run_dir, parameters, protocol, cache_source):
    """Create a unique run directory and its initial settings record."""
    repo = Path(__file__).resolve().parents[3]
    if run_dir is None:
        timestamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S_%f")
        name = (f"nested_cv_{timestamp}" if protocol is not None
                else f"fe_graph_loocv_{timestamp}_seed{parameters['seed']}")
        run_dir = repo / "results" / "experiments" / name
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    if protocol is None:
        config = create_run_record(repo, **parameters)
    else:
        config = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "preparing", "evaluation": "nested_cross_validation",
            "format_version": 1, "protocol": protocol,
            "cache_source": str(Path(cache_source).resolve()),
            "cache_path": "cache/cohort_training_v2.pt",
            "results_path": "nested_results.json", "source_sha256": source_hashes(repo),
        }
        config.update(runtime_metadata(repo))
    save_run_record(run_dir, config)
    return run_dir, config


def prepare_run_cache(run_dir, config, preparation, cache_source):
    """Prepare or copy the cache, then record its provenance and run status."""
    started = time.perf_counter()
    cache_path = Path(run_dir) / config["cache_path"]
    if cache_source is None:
        cache = generate_training_data(
            preparation["subjects"], preparation["frequencies_hz"], cache_path,
            data_dir=preparation["data_dir"], preprocessing_id=preparation["preprocessing_id"],
        )
    else:
        cache = torch.load(cache_source, map_location="cpu", weights_only=True)
        validate_training_cache(cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cache_source, cache_path)
        config["cache_source"] = str(Path(cache_source).resolve())
    nested = config.get("evaluation") == "nested_cross_validation"
    config.update({
        "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
        "forward_sha256": cache["forward_sha256"],
        "subject_ids": [subject["subject_id"] for subject in cache["subjects"]],
        "n_subjects": len(cache["subjects"]),
        "frequencies_hz": cache["frequencies_hz"].tolist(),
        "preprocessing_id": cache["preprocessing_id"],
        "status": "running" if nested else "training",
        "preparation_seconds": time.perf_counter() - started,
    })
    save_run_record(run_dir, config)
    return cache


def runtime_metadata(repo):
    """Record the same source revision and numerical runtime for either procedure."""
    record = {
        "torch_version": str(torch.__version__),
        "torch_threads": torch.get_num_threads(),
        "device": "cpu", "dtype": "float64",
    }
    try:
        record["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
        record["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo,
            text=True, stderr=subprocess.DEVNULL,
        ).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        record["git_commit"], record["git_dirty"] = None, None
    return record


def source_hashes(repo):
    """Hash the implementation and all Python forward-model sources."""
    paths = sorted({
        *Path(repo, "spectrome/mlp").rglob("*.py"),
        *Path(repo, "spectrome/forward").rglob("*.py"),
    })
    return {
        str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def create_run_record(repo, *, hidden_sizes, omega_scale,
                      activation, epochs, learning_rate, seed):
    """Capture settings and provenance for a new fixed-configuration LOOCV run."""
    repo = Path(repo).resolve()
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "preparing",
        "evaluation": "leave_one_subject_out",
        "cache_path": "cache/cohort_training_v2.pt",
        "results_path": "results.pt",
        "checkpoint_dir": "checkpoints",
        "model_config": {"hidden_sizes": list(hidden_sizes),
                         "omega_scale": float(omega_scale),
                         "activation": activation},
        "training_config": {"epochs": epochs, "learning_rate": learning_rate,
                            "seed": seed, "optimizer": "Adam",
                            "device": "cpu", "dtype": "float64"},
        "selection_rule": "final epoch of fixed budget; no held-out selection",
        "scoring_id": SCORING_ID,
        "source_sha256": source_hashes(repo),
        "plot_region_policy": "First, middle and last observed rows for every held-out subject",
    }
    record.update(runtime_metadata(repo))
    return record


def record_training_complete(run_dir, config, started):
    """Record fit duration and advance to analysis."""
    key = "evaluation_seconds" if config["evaluation"] == "nested_cross_validation" else "training_seconds"
    config.update({"status": "analyzing", key: time.perf_counter() - started})
    save_run_record(run_dir, config)


def record_run_complete(run_dir, config, started, results):
    """Record successful analysis and the existing fixed-run score summary."""
    config.update({
        "status": "complete", "analysis_seconds": time.perf_counter() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    if config["evaluation"] == "leave_one_subject_out":
        for key in ("mean_analytical_correlation", "mean_learned_correlation",
                    "mean_correlation_difference"):
            config[key] = results[key]
    save_run_record(run_dir, config)


def completed_run_matches(directory, arguments):
    """Skip only complete matching runs; reject partial or conflicting runs."""
    if not directory.exists():
        return False
    config_path = directory / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    if "protocol" not in arguments:
        if (config.get("evaluation", "leave_one_subject_out") != "leave_one_subject_out"
                or config.get("status") != "complete"
                or not (directory / config.get("results_path", "results.pt")).is_file()):
            raise RuntimeError(f"Existing run is incomplete; inspect it or choose a new directory: {directory}")
        model = {"activation": "tanh", **config.get("model_config", {})}
        saved = {**model, **config.get("training_config", {})}
        if any(saved.get(key) != value for key, value in arguments.items()):
            raise ValueError(f"Completed run settings differ from the queue: {directory}")
        from ..analysis.helpers import load_loocv_run
        load_loocv_run(directory, config)
        return True
    result_path = directory / "nested_results.json"
    result = json.loads(result_path.read_text()) if result_path.is_file() else {}
    if config.get("status") != "complete" or result.get("status") != "complete":
        raise RuntimeError(f"Existing nested run is incomplete; inspect it: {directory}")
    expected_cache_hash = hashlib.sha256(arguments["cache_source"].read_bytes()).hexdigest()
    expected_source = source_hashes(Path(__file__).resolve().parents[3])
    copied_cache = directory / config.get("cache_path", "cache/cohort_training_v2.pt")
    if (config.get("protocol") != arguments["protocol"]
            or config.get("source_sha256") != expected_source
            or config.get("cache_sha256") != expected_cache_hash
            or not copied_cache.is_file()
            or hashlib.sha256(copied_cache.read_bytes()).hexdigest() != expected_cache_hash):
        raise ValueError(f"Completed nested run protocol, source, or cache differs: {directory}")
    # Saved-result verification imports checkpoint utilities from this module.
    # Load it here after the helper module has finished initializing.
    from ..analysis.helpers import load_nested_run
    load_nested_run(directory, config, require_complete=True)
    return True


@contextmanager
def run_log(run_dir):
    """Retain stdout and stderr for either training procedure."""
    with (Path(run_dir) / "training.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            yield


class Tee:
    """Keep progress visible while retaining it in the run log."""

    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, value):
        self.console.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.console.flush()
        self.log.flush()


def report_nested_epoch(label, epochs):
    """Return a bounded progress callback for one nested fit."""
    def report(entry):
        epoch = entry["epoch"]
        if epoch in (1, epochs) or epoch % 100 == 0:
            print(f"{label} epoch={epoch}/{epochs} "
                  f"train_loss={entry['train_loss_before_step']:.6f}", flush=True)
    return report


# Nested result records


def create_nested_result(cache, protocol, splits, candidates, execution_order, *,
                         source_sha256, cache_sha256):
    """Build the run record without fitting or writing files."""
    subject_ids = [subject["subject_id"] for subject in cache["subjects"]]
    outer_strategy = protocol["outer_strategy"]
    inner_folds = protocol["inner_folds"]
    split_seed = protocol["split_seed"]
    training_seed = protocol["training_seed"]
    declared_comparisons = protocol.get("declared_comparisons", [])
    protocol_record = {
        "outer_strategy": "leave_one_subject_out" if outer_strategy in {
            "leave_one_out", "leave_one_subject_out"
        } else "grouped_kfold",
        "outer_folds": len(splits),
        "outer_membership": splits,
        "inner_strategy": "shuffled_balanced_kfold",
        "inner_folds": inner_folds,
        "split_seed": split_seed,
        "training_seed": training_seed,
        "selection_metric": "mean_raw_subject_spectral_correlation",
        "subject_weighting": "equal_subject_weight; pooled subject scores across inner folds",
        "tie_rule": "first candidate in declared order",
        "candidate_execution_order": list(execution_order),
        "declared_comparisons": list(declared_comparisons),
    }
    return {
        "format_version": 1,
        "evaluation": "nested_cross_validation",
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scoring_id": SCORING_ID,
        "loss_preprocessing": "magnitude_conv5_sqrt_floor1e-12_demean",
        "subject_ids": subject_ids,
        "subject_metadata": [{
            "subject_id": subject["subject_id"],
            "provenance": subject["provenance"],
            "observed_regions": subject["observed_regions"].tolist(),
            "parameters": {name: value.item() for name, value in subject["parameters"].items()},
        } for subject in cache["subjects"]],
        "preprocessing_id": cache["preprocessing_id"],
        "frequencies_hz": cache["frequencies_hz"].tolist(),
        "forward_sha256": cache["forward_sha256"],
        "cache_sha256": cache_sha256,
        "source_sha256": source_sha256,
        "protocol": protocol_record,
        "candidates": candidates,
        "outer_folds": [],
    }


def save_nested_fold(output_dir, result, outer, inner_splits, selected,
                     candidate_scores, candidate_means, model, history, evaluation):
    """Save weights and predictions, then return one outer-fold record."""
    outer_fold = outer["outer_fold"]
    train_ids = outer["training_subject_ids"]
    test_ids = outer["test_subject_ids"]
    selected_id = selected["candidate_id"]
    selected_train_config = {**selected["training_config"], "seed": result["protocol"]["training_seed"]}
    checkpoint_path = Path("checkpoints") / f"outer-fold-{outer_fold:03d}.pt"
    checkpoint = {
        "format_version": 1,
        "evaluation": "nested_cross_validation",
        "outer_fold": outer_fold,
        "model_config": selected["model_config"],
        "training_config": selected_train_config,
        "selected_candidate_id": selected_id,
        "training_subject_ids": train_ids,
        "test_subject_ids": test_ids,
        "inner_splits": inner_splits,
        "cache_sha256": result["cache_sha256"],
        "forward_sha256": result["forward_sha256"],
        "source_sha256": result["source_sha256"],
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
    }
    save_checkpoint(output_dir / checkpoint_path, checkpoint)
    prediction_path = Path("predictions") / f"outer-fold-{outer_fold:03d}.pt"
    save_checkpoint(output_dir / prediction_path, {
        "outer_fold": outer_fold,
        "selected_candidate_id": selected_id,
        "subject_predictions": {
            row["subject_id"]: {
                key: row[key] for key in (
                    "analytical_prediction", "learned_prediction", "Fe_graph"
                )
            }
            for row in evaluation
        },
    })
    outer_records = [{
        "outer_fold": outer_fold,
        "subject_id": row["subject_id"],
        "selected_candidate_id": selected_id,
        "analytical_r": row["analytical_correlation"],
        "learned_r": row["learned_correlation"],
        "raw_r_difference": row["correlation_difference"],
        "prediction_path": prediction_path.as_posix(),
    } for row in evaluation]
    candidate_records = {}
    for candidate in result["candidates"]:
        candidate_id = candidate["candidate_id"]
        candidate_records[candidate_id] = [{
            "subject_id": subject_id,
            "r": float(candidate_scores[candidate_id][subject_id]),
        } for subject_id in train_ids]
    outer_record = {
        "outer_fold": outer_fold,
        "training_subject_ids": train_ids,
        "test_subject_ids": test_ids,
        "inner_split_seed": inner_split_seed(result["protocol"]["split_seed"], outer_fold),
        "inner_splits": inner_splits,
        "candidate_scores": candidate_records,
        "candidate_mean_raw_r": candidate_means,
        "selected_candidate_id": selected_id,
        "outer_refit": {
            "model_config": selected["model_config"],
            "training_config": selected_train_config,
            "trained_epochs": selected_train_config["epochs"],
            "training_history": history,
        },
        "checkpoint_path": checkpoint_path.as_posix(),
        "prediction_path": prediction_path.as_posix(),
        "outer_subject_records": outer_records,
    }
    return outer_record


def complete_nested_result(result, result_path):
    """Calculate final summaries, validate the records, and save completion."""
    from ..analysis.helpers import summarize_outer_scores, validate_nested_results

    subject_records = [
        row for outer in result["outer_folds"]
        for row in outer["outer_subject_records"]
    ]
    candidate_ids = [candidate["candidate_id"] for candidate in result["candidates"]]
    result["outer_summary"] = summarize_outer_scores(subject_records, candidate_ids)
    result["statistics"] = []
    if "selected_mlp_vs_analytical" in result["protocol"]["declared_comparisons"]:
        from ..analysis.statistics import paired_fisher_z_test
        result["statistics"].append(paired_fisher_z_test(
            {row["subject_id"]: row["learned_r"] for row in subject_records},
            {row["subject_id"]: row["analytical_r"] for row in subject_records},
            left_label="nested_selected_mlp",
            right_label="analytical",
            contrast="selected_mlp_vs_analytical",
            expected_subject_ids=result["subject_ids"],
            provenance={
                "cache_sha256": result["cache_sha256"],
                "preprocessing_id": result["preprocessing_id"],
                "forward_sha256": result["forward_sha256"],
                "scoring_id": SCORING_ID,
            },
        ))
    result["status"] = "complete"
    result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    validate_nested_results(result)
    save_json_atomic(result_path, result)


# Checkpoint and JSON files


def save_checkpoint(path, payload):
    """Save a checkpoint payload; callers supply detached CPU tensors."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return path


def save_fold_checkpoint(path, metadata, fold, state_dict):
    """Save one fold's settings, predictions, and fitted weights."""
    return save_checkpoint(path, {
        **metadata,
        **fold,
        "state_dict": {
            name: value.detach().cpu().clone() for name, value in state_dict.items()
        },
    })


def load_checkpoint(path):
    """Load a CPU checkpoint without allowing arbitrary pickle globals."""
    return torch.load(path, map_location="cpu", weights_only=True)


def load_fold_checkpoint(results_path, fold):
    """Load a checkpoint using its saved absolute or run-relative path."""
    path = Path(fold["checkpoint_path"])
    if not path.is_absolute():
        path = Path(results_path).resolve().parent / path
    return load_checkpoint(path)


def save_run_record(run_dir, record):
    """Atomically replace the run's config.json record."""
    run_dir = Path(run_dir)
    save_json_atomic(run_dir / "config.json", record)


def save_json_atomic(path, value):
    """Write a JSON record atomically without allowing non-finite values."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
