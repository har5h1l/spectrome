"""Historical evidence provenance survives training-module reorganization."""

import hashlib
from pathlib import Path

import pytest

pytest.importorskip("torch")

from spectrome.tests import historical_evidence as collector

ROOT = Path(__file__).resolve().parents[2]


def test_evidence_hashes_resolve_current_mlp_modules():
    hashes = collector.collect_source_hashes()
    assert not any(name.startswith("spectrome/training/") for name in hashes)
    mlp_paths = {name for name in hashes if name.startswith("spectrome/mlp/")}
    assert "spectrome/mlp/training/training.py" in mlp_paths
    assert "spectrome/mlp/training/loocv.py" in mlp_paths
    assert "spectrome/mlp/analysis/statistics.py" in mlp_paths
    assert "spectrome/forward/network_transfer_torch.py" in hashes
    for name, digest in hashes.items():
        assert digest == hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def test_missing_source_fails_before_replay_or_output_changes(monkeypatch, tmp_path):
    output_dir = tmp_path / "presentation"

    def unexpected_replay(*args, **kwargs):
        pytest.fail("Missing source must fail before replay begins")

    monkeypatch.setattr(collector, "loadmat", unexpected_replay)
    with pytest.raises(FileNotFoundError):
        collector.collect_evidence(output_dir, root=tmp_path / "missing_source")
    assert not output_dir.exists()
