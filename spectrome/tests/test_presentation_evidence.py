"""Historical evidence provenance survives training-module reorganization."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def collector():
    path = ROOT / "artifacts/10-5-presentation/src/collect_evidence.py"
    spec = importlib.util.spec_from_file_location("historical_evidence_collector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evidence_hashes_resolve_current_training_modules(collector):
    hashes = collector.collect_source_hashes()
    training_paths = {name for name in hashes if name.startswith("spectrome/training/")}
    assert training_paths == {
        f"spectrome/training/{name}.py"
        for name in ("data", "model", "spectra", "train", "helpers", "analysis")
    }
    for name, digest in hashes.items():
        assert digest == hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def test_missing_source_fails_before_replay_or_output_changes(collector, monkeypatch, tmp_path):
    monkeypatch.setattr(collector, "ROOT", tmp_path / "missing_source")
    monkeypatch.setattr(collector, "OUT", tmp_path / "presentation")

    def unexpected_replay(*args, **kwargs):
        pytest.fail("Missing source must fail before replay begins")

    monkeypatch.setattr(collector, "loadmat", unexpected_replay)
    with pytest.raises(FileNotFoundError):
        collector.main()
    assert not collector.OUT.exists()
