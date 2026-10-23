"""Run only the approved three experiments, using the existing frozen cache."""
import hashlib
import json
from pathlib import Path
import sys

packet = Path(__file__).resolve().parent
manifest = json.loads((packet / "manifest.json").read_text())
root = Path(manifest["server_dir"])
entries = json.loads((packet / "experiments.json").read_text())
expected_names = [f"basic_mlp_{i}" for i in (13, 14, 15)]
if [entry["results_dir"] for entry in entries] != expected_names:
    raise RuntimeError("Launch packet must contain only runs 13, 14, and 15")
queue = json.loads((root / "spectrome/training/experiments.json").read_text())
if queue[13:16] != entries:
    raise RuntimeError("Launch packet differs from the saved experiment queue")
for relative, expected_hash in manifest["files"].items():
    if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected_hash:
        raise RuntimeError(f"Source hash mismatch: {relative}")
cache = root / "inputs/cohort_training_v2.pt"
if hashlib.sha256(cache.read_bytes()).hexdigest() != manifest["cache_sha256"]:
    raise RuntimeError("Frozen cache hash mismatch")
for entry, widths in zip(entries, ([128] * 5, [32], [32, 32])):
    expected = dict(hidden_sizes=widths, activation="tanh", epochs=5000,
                    learning_rate=0.0003, seed=7, omega_scale=282.7433388230814)
    if entry["parameters"] != expected:
        raise RuntimeError("Settings differ from the approved experiment")
    if (root / "results/experiments" / entry["results_dir"]).exists():
        raise RuntimeError(f"Refusing to overwrite or restart {entry['results_dir']}")
print("Validated launch: only " + ", ".join(expected_names), flush=True)
if "--check" in sys.argv:
    sys.exit(0)

sys.path.insert(0, str(root))
import torch
from spectrome.training.train import run_experiment

torch.set_num_threads(4)
for entry in entries:
    print(f"Starting {entry['results_dir']}", flush=True)
    run_experiment(root / "results/experiments" / entry["results_dir"],
                   cache_source=cache, **entry["parameters"])
print("All three approved experiments completed", flush=True)
