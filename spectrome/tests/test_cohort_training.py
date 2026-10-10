"""Chronological loading and bounded LOOCV checks with real cohort records."""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

torch = pytest.importorskip("torch")

from spectrome.tests.test_historical_torch_parity import DATA, FREQS, prepared_brain
from spectrome.forward.network_transfer import network_transfer_local_alpha
from spectrome.training import cohort, filters
from spectrome.training.filters import FeGraphMLP, generate_training_data, predict_spectra
from spectrome.training.train import train_filter


@pytest.fixture(scope="module")
def records():
    return cohort.prepare_cohort(DATA)


def test_chronological_cohort_uses_one_exclusion_rule(records):
    raw_c = loadmat(DATA / "individual_subjects.mat")["A_all_subjs_final"]
    raw_meg = loadmat(DATA / "freqMEGdata.mat")["freqMEGdata"].ravel()
    raw_p = loadmat(DATA / "SCFC_opparam_individual.mat")["output"].ravel()
    indices = [i for i, (m, p) in enumerate(zip(raw_meg, raw_p))
               if m["psd"].size and p["param"].size]
    assert len(records) == len(indices) == 36
    assert len({record["subject_id"] for record in records}) == len(records)
    assert [record["provenance"]["raw_connectivity_index"] for record in records] == indices
    for record, index in zip(records, indices):
        assert "split" not in record
        assert record["mapping_verified"] is True
        assert record["provenance"]["raw_meg_index"] == index
        assert record["provenance"]["raw_param_index"] == index
        assert record["provenance"]["excluded_original_indices"] == [21, 22, 28]
        np.testing.assert_array_equal(record["empirical_meg"], raw_meg[index]["psd"])
        np.testing.assert_array_equal(list(record["parameters"].values()), raw_p[index]["param"][:, 0])
    # Check a retained record after the first exclusions, using its original C
    # slice rather than its compacted position. Preserve the notebook operator.
    record = records[21]
    assert record["provenance"]["raw_connectivity_index"] == 23
    D = np.genfromtxt(DATA / "mean80_fiberlength.csv", delimiter=",")
    expected = prepared_brain(raw_c[:, :, 23], D)
    np.testing.assert_allclose(record["brain"].reducedConnectome, expected.reducedConnectome)
    np.testing.assert_array_equal(record["brain"].distance_matrix, expected.distance_matrix)


def test_partial_missing_record_fails_instead_of_realigning(monkeypatch):
    original_loadmat = cohort.loadmat

    def broken_parameters(path):
        data = original_loadmat(path)
        if Path(path).name == "SCFC_opparam_individual.mat":
            data["output"]["param"][0, 23] = np.empty((0, 0))
        return data

    monkeypatch.setattr(cohort, "loadmat", broken_parameters)
    with pytest.raises(ValueError, match="Record 23"):
        cohort.prepare_cohort(DATA)


@pytest.fixture
def small_cache(records, tmp_path):
    # Three actual subjects, five actual bins: a short implementation check,
    # not a 36-subject experiment or a selected scientific configuration.
    subjects = [{**record, "empirical_meg": record["empirical_meg"][:, :5]}
                for record in records[:3]]
    return generate_training_data(subjects, FREQS[:5], tmp_path / "cache.pt",
                                  preprocessing_id="test_real_cohort_five_bins")


def test_generate_training_data_calls_cohort_preparation(records, monkeypatch, tmp_path):
    called = []

    def prepare(data_dir):
        called.append(data_dir)
        return [{**records[0], "empirical_meg": records[0]["empirical_meg"][:, :5]}]

    monkeypatch.setattr(filters, "prepare_cohort", prepare)
    path = tmp_path / "automatic.pt"
    cache = generate_training_data(frequencies_hz=FREQS[:5], output_path=path, data_dir=DATA)
    saved = torch.load(path, weights_only=True)
    assert called == [DATA]
    assert cache["format_version"] == saved["format_version"] == 2
    assert "split" not in saved["subjects"][0]
    assert cache["preprocessing_id"] == cohort.COHORT_PREPROCESSING_ID


def test_full_default_cache_and_chronological_analytical_baseline(tmp_path):
    path = tmp_path / "full_cohort.pt"
    generate_training_data(output_path=path)
    cache = torch.load(path, weights_only=True)
    assert len(cache["subjects"]) == 36
    torch.testing.assert_close(cache["frequencies_hz"], torch.from_numpy(FREQS))
    for subject in cache["subjects"]:
        assert "split" not in subject
        assert len(subject["graphs"]) == 40
        assert subject["empirical_meg"].shape == (68, 40)
        assert all(value.device.type == "cpu" and not value.requires_grad
                   for graph in subject["graphs"] for value in graph.values()
                   if torch.is_tensor(value))
        prediction = predict_spectra(subject)
        assert prediction.shape == (68, 40) and torch.isfinite(prediction).all()
    # Independently evaluate a record after exclusions on its original slice.
    subject = cache["subjects"][21]
    index = subject["provenance"]["raw_connectivity_index"]
    assert index == 23
    C = loadmat(DATA / "individual_subjects.mat")["A_all_subjs_final"][:, :, index]
    D = np.genfromtxt(DATA / "mean80_fiberlength.csv", delimiter=",")
    brain = prepared_brain(C, D)
    parameters = {name: value.item() for name, value in subject["parameters"].items()}
    reference = np.stack([network_transfer_local_alpha(brain, parameters, 2 * np.pi * f)[3][:68]
                          for f in FREQS], axis=-1)
    np.testing.assert_allclose(predict_spectra(subject).numpy(), reference, atol=1e-9, rtol=1e-9)


def test_loocv_folds_fresh_models_and_reload(small_cache, monkeypatch, tmp_path):
    initial_parameters, optimizers = [], []
    original_adam = torch.optim.Adam

    class RecordingAdam(original_adam):
        def __init__(self, parameters, **kwargs):
            parameters = list(parameters)
            initial_parameters.append([p.detach().clone() for p in parameters])
            super().__init__(parameters, **kwargs)
            assert not self.state
            optimizers.append(self)

    monkeypatch.setattr(torch.optim, "Adam", RecordingAdam)
    output = tmp_path / "result.pt"
    before = small_cache["subjects"][0]["parameters"]["alpha"].clone()
    result = train_filter(small_cache, output, hidden_sizes=[4], omega_scale=100,
                          epochs=2, learning_rate=0.001, seed=7)
    ids = [subject["subject_id"] for subject in small_cache["subjects"]]
    assert len(result["folds"]) == len(optimizers) == len(ids)
    assert [fold["held_out_subject_id"] for fold in result["folds"]] == ids
    for initial in initial_parameters[1:]:
        for actual, expected in zip(initial, initial_parameters[0]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    parameter_ids = [{id(p) for p in opt.param_groups[0]["params"]} for opt in optimizers]
    assert not parameter_ids[0] & parameter_ids[1]
    for index, fold in enumerate(result["folds"]):
        assert set(fold["training_subject_ids"]) == set(ids) - {ids[index]}
        assert len(fold["history"]) == fold["trained_epochs"] == 2
        checkpoint = torch.load(fold["checkpoint_path"], weights_only=True)
        model = FeGraphMLP(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["state_dict"])
        with torch.no_grad():
            predicted = predict_spectra(small_cache["subjects"][index], model(small_cache["omega"]))
        torch.testing.assert_close(predicted, fold["learned_prediction"])
    saved = torch.load(output, weights_only=True)
    assert saved["mean_correlation_difference"] == pytest.approx(
        np.mean([fold["correlation_difference"] for fold in result["folds"]]))
    torch.testing.assert_close(before, small_cache["subjects"][0]["parameters"]["alpha"])


def test_held_out_targets_cannot_change_its_trained_weights(small_cache, tmp_path):
    settings = dict(hidden_sizes=[4], omega_scale=100, epochs=2, learning_rate=0.001, seed=7)
    before = train_filter(small_cache, tmp_path / "before.pt", **settings)
    held_out = small_cache["subjects"][0]
    held_out["empirical_meg"] = held_out["empirical_meg"].flip(-1)
    after = train_filter(small_cache, tmp_path / "after.pt", **settings)
    first = torch.load(before["folds"][0]["checkpoint_path"], weights_only=True)
    second = torch.load(after["folds"][0]["checkpoint_path"], weights_only=True)
    assert first["history"] == second["history"]
    for name in first["state_dict"]:
        torch.testing.assert_close(first["state_dict"][name], second["state_dict"][name], rtol=0, atol=0)


def test_loocv_rejects_single_subject_and_old_split_cache(small_cache, tmp_path):
    settings = dict(hidden_sizes=[4], omega_scale=100, epochs=1, learning_rate=0.001, seed=7)
    with pytest.raises(ValueError, match="at least two"):
        train_filter({**small_cache, "subjects": small_cache["subjects"][:1]},
                     tmp_path / "single.pt", **settings)
    with pytest.raises(ValueError, match="version 2"):
        train_filter({**small_cache, "format_version": 1}, tmp_path / "old.pt", **settings)
