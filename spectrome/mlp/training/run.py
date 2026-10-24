"""Run one training experiment or a declared batch."""

from pathlib import Path
import time
import traceback

from ..analysis import analyze_run
from .loocv import run_loocv
from .nested_cv import nested_cross_validate
from .helpers import (
    DEFAULT_PARAMETERS, validate_run_inputs, initialize_run, run_log,
    prepare_run_cache, record_training_complete, record_run_complete, save_run_record,
    load_training_runs, queue_run_directory, queue_run_arguments, completed_run_matches,
)

HIDDEN_SIZES = DEFAULT_PARAMETERS["hidden_sizes"]
OMEGA_SCALE = DEFAULT_PARAMETERS["omega_scale"]
EPOCHS = DEFAULT_PARAMETERS["epochs"]
LEARNING_RATE = DEFAULT_PARAMETERS["learning_rate"]
SEED = DEFAULT_PARAMETERS["seed"]


def run_experiment(run_dir=None, *, data_dir=None, hidden_sizes=HIDDEN_SIZES,
                   omega_scale=OMEGA_SCALE, epochs=EPOCHS,
                   learning_rate=LEARNING_RATE, seed=SEED,
                   subjects=None, frequencies_hz=None, preprocessing_id=None,
                   activation="tanh", cache_source=None, protocol=None,
                   training_runs_path=None):
    """Prepare data, run the requested CV procedure, and analyze its outputs."""
    parameters = {
        "hidden_sizes": hidden_sizes, "omega_scale": omega_scale,
        "epochs": epochs, "learning_rate": learning_rate, "seed": seed, "activation": activation,
    }
    preparation = {
        "data_dir": data_dir, "subjects": subjects,
        "frequencies_hz": frequencies_hz, "preprocessing_id": preprocessing_id,
    }
    parameters, protocol = validate_run_inputs(
        parameters, preparation, cache_source, protocol,
        training_runs_path=training_runs_path,
    )
    run_dir, config = initialize_run(run_dir, parameters, protocol, cache_source)
    with run_log(run_dir):
        try:
            print(f"Run folder: {run_dir}", flush=True)
            cache = prepare_run_cache(run_dir, config, preparation, cache_source)
            started = time.perf_counter()
            if protocol is None:
                print(f"Prepared {config['n_subjects']} subjects; starting LOOCV", flush=True)
                results = run_loocv(
                    cache, run_dir / "results.pt", checkpoint_dir=run_dir / "checkpoints", **parameters,
                )
            else:
                print(f"Prepared {config['n_subjects']} subjects; starting nested cross-validation", flush=True)
                results = nested_cross_validate(
                    cache, run_dir, protocol, source_sha256=config["source_sha256"],
                    cache_sha256=config["cache_sha256"],
                    candidate_execution_order=protocol.get("candidate_execution_order"),
                )
            record_training_complete(run_dir, config, started)
            started = time.perf_counter()
            analyze_run(run_dir)
            record_run_complete(run_dir, config, started, results)
            print(f"Complete: {run_dir}", flush=True)
        except Exception as error:
            config.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
            save_run_record(run_dir, config)
            traceback.print_exc()
            raise
    return run_dir


def run_experiments(config_path=None, *, results_root=None):
    """Read a batch, check each entry, and run only missing experiments."""
    root = (Path(results_root) if results_root is not None else
            Path(__file__).resolve().parents[3] / "results/experiments").resolve()
    completed, seen = [], set()
    for experiment in load_training_runs(config_path):
        directory = queue_run_directory(experiment, root, seen)
        arguments = queue_run_arguments(experiment, directory, config_path)
        if completed_run_matches(directory, arguments):
            print(f"Skipping completed run: {directory.name}", flush=True)
        else:
            run_experiment(directory, **arguments)
        completed.append(directory)
    return completed


if __name__ == "__main__":
    run_experiments()
