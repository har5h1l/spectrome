"""Run the experiments.json queue sequentially; retain completed runs."""

import json
from pathlib import Path

from .train import run_experiment


def run_experiments(config_path=None, *, results_root=None):
    """Skip completed, matching runs; stop on existing incomplete/mismatched runs.

    Folder names are relative to results/experiments, independent of the
    working directory. No existing outputs are overwritten or resumed.
    """
    config_path = Path(config_path) if config_path is not None else Path(__file__).with_name("experiments.json")
    root = (Path(results_root) if results_root is not None else
            Path(__file__).resolve().parents[2] / "results/experiments").resolve()
    experiments = json.loads(config_path.read_text())
    completed, seen = [], set()
    for experiment in experiments:
        directory = (root / experiment["results_dir"]).resolve()
        if directory == root or not directory.is_relative_to(root) or directory in seen:
            raise ValueError(f"Results directory must be unique and inside {root}: {directory}")
        seen.add(directory)
        parameters = experiment["parameters"]
        model_keys = ("hidden_sizes", "activation", "omega_scale")
        training_keys = ("epochs", "learning_rate", "seed")
        if set(parameters) != set(model_keys + training_keys):
            raise ValueError(f"Specify exactly {model_keys + training_keys} for {directory.name}")
        if directory.exists():
            record_path = directory / "config.json"
            record = json.loads(record_path.read_text()) if record_path.is_file() else {}
            if record.get("status") != "complete" or not (directory / record.get("results_path", "results.pt")).is_file():
                raise RuntimeError(f"Existing run is incomplete; inspect it or choose a new directory: {directory}")
            model = {"activation": "tanh", **record.get("model_config", {})}
            saved = {**model, **record.get("training_config", {})}
            if any(saved.get(key) != value for key, value in parameters.items()):
                raise ValueError(f"Completed run settings differ from the queue: {directory}")
            print(f"Skipping completed run: {directory.name}", flush=True)
        else:
            print(f"Starting run: {directory.name}", flush=True)
            run_experiment(directory, **parameters)
        completed.append(directory)
    return completed


if __name__ == "__main__":
    run_experiments()
