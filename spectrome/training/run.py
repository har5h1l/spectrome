"""Prepare, train, and analyze one self-contained LOOCV experiment folder."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import torch

from .analysis import analyze_run
from .filters import generate_training_data
from .train import EPOCHS, HIDDEN_SIZES, LEARNING_RATE, OMEGA_SCALE, SEED, train_filter


class _Tee:
    """Keep progress visible while retaining it in the run's log."""

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


def run_experiment(run_dir=None, *, data_dir=None, hidden_sizes=HIDDEN_SIZES,
                   omega_scale=OMEGA_SCALE, epochs=EPOCHS,
                   learning_rate=LEARNING_RATE, seed=SEED,
                   subjects=None, frequencies_hz=None, preprocessing_id=None,
                   run_kind="experiment"):
    """Create a new run folder containing the cache, results, CSVs, and figures.

    Defaults are the accepted 100-epoch, seed-7 settings. Explicit subject
    records/frequencies support bounded verification runs. Refuse an existing
    directory to preserve earlier runs. No notebook is modified or executed.
    """
    repo = Path(__file__).resolve().parents[2]
    hidden_sizes = tuple(hidden_sizes)
    if run_dir is None:
        timestamp = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d_%H%M%S_%f")
        run_dir = repo / "results" / f"fe_graph_loocv_{timestamp}_seed{seed}"
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    cache_path = run_dir / "cache" / "cohort_training_v2.pt"
    results_path = run_dir / "results.pt"
    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_kind": run_kind, "status": "preparing",
        "cache_path": "cache/cohort_training_v2.pt", "results_path": "results.pt",
        "checkpoint_dir": "checkpoints",
        "model_config": {"hidden_sizes": list(hidden_sizes), "omega_scale": float(omega_scale)},
        "training_config": {"epochs": epochs, "learning_rate": learning_rate, "seed": seed,
                            "optimizer": "Adam", "device": "cpu", "dtype": "float64"},
        "selection_rule": "final epoch of fixed budget; no held-out selection",
        "torch_version": str(torch.__version__), "torch_threads": torch.get_num_threads(),
        "source_sha256": {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in [*Path(__file__).parent.glob("*.py"),
                                       repo / "spectrome/forward/network_transfer_torch.py"]},
        "plot_region_policy": "First, middle and last observed rows for every held-out subject",
    }
    # Git metadata is useful when available, but does not prevent use of a
    # downloaded source tree. Exact participating source hashes are saved above.
    try:
        config["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip()
        config["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True, stderr=subprocess.DEVNULL).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        config["git_commit"], config["git_dirty"] = None, None

    def save_config():
        temporary = run_dir / "config.json.tmp"
        temporary.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
        temporary.replace(run_dir / "config.json")

    save_config()
    with (run_dir / "training.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            try:
                print(f"Run folder: {run_dir}")
                started = time.perf_counter()
                cache = generate_training_data(
                    subjects, frequencies_hz, cache_path, data_dir=data_dir,
                    preprocessing_id=preprocessing_id,
                )
                config.update({
                    "cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                    "subject_ids": [subject["subject_id"] for subject in cache["subjects"]],
                    "n_subjects": len(cache["subjects"]),
                    "frequencies_hz": cache["frequencies_hz"].tolist(),
                    "preprocessing_id": cache["preprocessing_id"],
                    "status": "training",
                    "preparation_seconds": time.perf_counter() - started,
                })
                save_config()
                print(f"Prepared {config['n_subjects']} subjects; starting LOOCV")
                started = time.perf_counter()
                results = train_filter(
                    cache, results_path, checkpoint_dir=run_dir / "checkpoints",
                    hidden_sizes=hidden_sizes, omega_scale=omega_scale,
                    epochs=epochs, learning_rate=learning_rate, seed=seed,
                )
                config["status"] = "analyzing"
                config["training_seconds"] = time.perf_counter() - started
                save_config()
                print("Exporting tables and figures")
                started = time.perf_counter()
                analyze_run(run_dir)
                config.update({"status": "complete",
                               "analysis_seconds": time.perf_counter() - started,
                               "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                               "mean_analytical_correlation": results["mean_analytical_correlation"],
                               "mean_learned_correlation": results["mean_learned_correlation"],
                               "mean_correlation_difference": results["mean_correlation_difference"]})
                save_config()
                print(f"Complete: {run_dir}")
            except Exception as error:
                config.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
                save_config()
                traceback.print_exc()
                raise
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="New output folder; defaults to a dated folder under results/")
    parser.add_argument("--data-dir", type=Path, help="Current chronological cohort source directory")
    args = parser.parse_args()
    run_experiment(args.run_dir, data_dir=args.data_dir)


if __name__ == "__main__":
    main()
