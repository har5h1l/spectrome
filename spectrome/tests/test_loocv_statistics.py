import csv
import json

import pytest

from spectrome.mlp.analysis import loocv_statistics


def test_snapshot_family_stays_fixed_when_ranking_grows(tmp_path, monkeypatch):
    subject_ids = ["a", "b", "c", "d"]
    analytical = dict(zip(subject_ids, [0.6, 0.65, 0.68, 0.72]))
    run_ids = ["control", "wider", "deeper"]
    experiments = tmp_path / "experiments"
    output = tmp_path / "analysis"

    for index, run_id in enumerate(run_ids):
        directory = experiments / run_id
        directory.mkdir(parents=True)
        config = {
            "status": "complete",
            "cache_sha256": "shared-cache",
            "model_config": {"hidden_sizes": [8 * (index + 1)], "omega_scale": 1},
            "training_config": {"epochs": 100, "learning_rate": 0.001, "seed": 7},
        }
        (directory / "config.json").write_text(json.dumps(config))

    def load_scores(run_dir):
        index = run_ids.index(run_dir.name)
        learned = {
            key: analytical[key] + 0.02 * index + 0.005 * position
            for position, key in enumerate(subject_ids)
        }
        return {
            "learned": learned, "analytical": analytical,
            "provenance": {"subject_ids": subject_ids, "forward_sha256": "forward-source"},
        }

    monkeypatch.setattr(loocv_statistics, "load_loocv_scores", load_scores)
    paths = ["--experiments", str(experiments), "--output", str(output),
             "--reference-run", "control"]
    snapshot = ["snapshot", "--snapshot-id", "original", *paths, "--run-ids", *run_ids[:2]]
    loocv_statistics.main(snapshot)
    directory = output / "snapshots" / "original"
    saved = (directory / "comparisons.csv").read_bytes()
    with (directory / "comparisons.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["run_id"] for row in rows] == ["wider", "control"]
    for row in rows:
        assert int(row["family_size"]) == 2
        assert float(row["p_bonferroni"]) == pytest.approx(min(1.0, 2 * float(row["p_raw"])))

    loocv_statistics.main(["ranking", *paths, "--run-ids", *run_ids])
    loocv_statistics.main(snapshot)
    with (output / "ranking.csv").open(newline="") as stream:
        assert [row["run_id"] for row in csv.DictReader(stream)] == ["deeper", "wider", "control"]
    assert (directory / "comparisons.csv").read_bytes() == saved

    with pytest.raises(ValueError, match="snapshot differs"):
        loocv_statistics.main([*snapshot, "deeper"])
