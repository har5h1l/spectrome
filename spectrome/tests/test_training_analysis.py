"""Real-record tests for export accuracy and the self-contained run workflow."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spectrome.tests.test_cohort_training import records, small_cache
from spectrome.training.analysis import analyze_run, export_results, plot_results
from spectrome.training.train import run_experiment, train_filter
from spectrome.training import train


@pytest.fixture
def saved_result(small_cache, tmp_path):
    result = train_filter(small_cache, tmp_path / "results.pt", hidden_sizes=(4,),
                          omega_scale=100, epochs=2, learning_rate=1e-3, seed=7,
                          checkpoint_dir=tmp_path / "checkpoints")
    return result


def _rows(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_csv_metrics_match_saved_held_out_results(small_cache, saved_result, tmp_path):
    tables = export_results(saved_result, small_cache, tmp_path / "tables")
    subjects = _rows(tables["subject_scores"])
    regions = _rows(tables["region_scores"])
    history = _rows(tables["training_history"])
    filters = _rows(tables["filter_values"])
    assert len(subjects) == 3 and len(regions) == 3 * 68
    assert len(history) == 3 * 2 and len(filters) == 3 * 5
    for row, fold in zip(subjects, saved_result["folds"]):
        assert row["subject_id"] == fold["held_out_subject_id"]
        assert float(row["learned_training_correlation"]) == pytest.approx(fold["learned_training_correlation"])
        assert float(row["analytical_training_correlation"]) == pytest.approx(fold["analytical_training_correlation"])
        assert float(row["learned_correlation"]) == pytest.approx(fold["learned_correlation"])
        assert float(row["analytical_correlation"]) == pytest.approx(fold["analytical_correlation"])
        regional = [float(r["learned_correlation"]) for r in regions if r["subject_id"] == row["subject_id"]]
        assert np.mean(regional) == pytest.approx(float(row["learned_correlation"]))
    value = saved_result["folds"][0]["Fe_graph"][0]
    assert float(filters[0]["learned_real"]) == pytest.approx(value.real.item())
    assert float(filters[0]["learned_imaginary"]) == pytest.approx(value.imag.item())
    assert float(filters[0]["learned_magnitude"]) == pytest.approx(value.abs().item())
    assert Path(saved_result["folds"][0]["checkpoint_path"]).parent == Path("checkpoints")


def test_analysis_rejects_wrong_cache_and_duplicate_folds(small_cache, saved_result, tmp_path):
    with pytest.raises(ValueError, match="frequencies"):
        export_results(saved_result, {**small_cache, "frequencies_hz": small_cache["frequencies_hz"] + 1},
                       tmp_path / "wrong")
    duplicated = {**saved_result, "folds": [saved_result["folds"][0]] * 3}
    with pytest.raises(ValueError, match="exactly one"):
        export_results(duplicated, small_cache, tmp_path / "duplicate")
    changed = {**small_cache, "subjects": [dict(s) for s in small_cache["subjects"]]}
    changed["subjects"][0]["empirical_meg"] = changed["subjects"][0]["empirical_meg"].flip(-1)
    with pytest.raises(ValueError, match="Recomputed scores"):
        export_results(saved_result, changed, tmp_path / "changed")


def test_figures_cover_all_subjects_and_close_handles(small_cache, saved_result, tmp_path):
    import matplotlib.pyplot as plt
    paths = plot_results(saved_result, small_cache, tmp_path / "figures")
    assert len(paths) == 2 + 2 * 3
    assert all(path.suffix == ".png" for path in paths)
    assert not list((tmp_path / "figures").rglob("*.pdf"))
    assert len(list((tmp_path / "figures/subjects").glob("*/spectra.png"))) == 3
    assert len(list((tmp_path / "figures/subjects").glob("*/filter.png"))) == 3
    assert all(path.is_file() and path.stat().st_size > 100 for path in paths)
    assert not plt.get_fignums()


def test_run_folder_owns_cache_and_rejects_overwrite(small_cache, monkeypatch, tmp_path):
    def prepare(*args, **kwargs):
        path = Path(args[2])
        path.parent.mkdir(parents=True)
        torch.save(small_cache, path)
        return small_cache

    monkeypatch.setattr(train, "generate_training_data", prepare)
    directory = run_experiment(tmp_path / "verification", epochs=2, hidden_sizes=(4,),
                               omega_scale=100, run_kind="verification")
    config = json.loads((directory / "config.json").read_text())
    assert config["status"] == "complete" and config["n_subjects"] == 3
    assert (directory / config["cache_path"]).is_file()
    assert config["training_config"]["epochs"] == 2 and config["training_config"]["seed"] == 7
    assert len(list((directory / "checkpoints").glob("*.pt"))) == 3
    assert len(_rows(directory / "tables/subject_scores.csv")) == 3
    assert "fold=3/3 epoch=2" in (directory / "training.log").read_text()
    with pytest.raises(FileExistsError):
        run_experiment(directory)
    # Figures/tables can be regenerated without calling train_filter.
    def no_training(*args, **kwargs):
        raise AssertionError("Analysis must not retrain")
    monkeypatch.setattr(train, "train_filter", no_training)
    regenerated = analyze_run(directory)
    assert len(regenerated["tables"]) == 4
    assert len(regenerated["figures"]) == 8
    assert ".pdf" not in regenerated["summary"].read_text()
    assert "Subjects improved" in regenerated["summary"].read_text()
    with (directory / config["cache_path"]).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        analyze_run(directory)


def test_failed_run_retains_error_and_log(tmp_path):
    directory = tmp_path / "failed"
    with pytest.raises(FileNotFoundError):
        run_experiment(directory, data_dir=tmp_path / "missing_data")
    config = json.loads((directory / "config.json").read_text())
    assert config["status"] == "failed"
    assert "FileNotFoundError" in config["error"]
    assert "Traceback" in (directory / "training.log").read_text()


def test_cache_reuse_copies_and_rejects_stale_source(small_cache, tmp_path, monkeypatch):
    source = tmp_path / "source.pt"
    torch.save(small_cache, source)
    monkeypatch.setattr(train, "generate_training_data", lambda *a, **k: pytest.fail("cache was rebuilt"))
    monkeypatch.setattr(train, "analyze_run", lambda *a, **k: None)
    directory = run_experiment(tmp_path / "reused", cache_source=source,
                               epochs=1, hidden_sizes=(4,), omega_scale=100,
                               activation="sigmoid", run_kind="verification")
    config = json.loads((directory / "config.json").read_text())
    assert (directory / config["cache_path"]).read_bytes() == source.read_bytes()
    assert config["model_config"]["activation"] == "sigmoid"
    with pytest.raises(ValueError, match="cannot be combined"):
        run_experiment(tmp_path / "conflict", cache_source=source, data_dir=tmp_path)
    torch.save({**small_cache, "forward_sha256": "stale"}, source)
    with pytest.raises(ValueError, match="Forward source changed"):
        run_experiment(tmp_path / "stale", cache_source=source)
