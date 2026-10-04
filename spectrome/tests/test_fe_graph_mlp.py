"""Cache and gradient checks using an existing real historical record.

No synthetic dataset and no cohort training. Historical pairing is deliberately
marked unverified; these tests establish implementation behavior only.
"""

import numpy as np
import pytest
from scipy.io import loadmat

torch = pytest.importorskip("torch")

from spectrome.tests.test_historical_torch_parity import DATA, FREQS, prepared_brain, transformed_spectra
from spectrome.forward.network_transfer_torch import run_local_coupling_forward_torch
from spectrome.training.filters import (
    FeGraphMLP, generate_training_data, predict_spectra,
    spectral_correlation, transform_spectra,
)
from spectrome.training.train import train_filter


@pytest.fixture(scope="module")
def historical_record():
    C = loadmat(DATA / "individual_subjects.mat")["A_all_subjs_final"][:, :, 0]
    D = np.genfromtxt(DATA / "mean80_fiberlength.csv", delimiter=",")
    parameters = np.squeeze(loadmat(DATA / "SCFC_opparam_individual.mat")["output"]["param"])[0]
    target = np.squeeze(loadmat(DATA / "freqMEGdata.mat")["freqMEGdata"]["psd"])[0]
    names = ("tau_e", "tau_i", "alpha", "speed", "gei", "gii", "tauC")
    return {"subject_id": "historical-record-0", "brain": prepared_brain(C, D),
            "parameters": dict(zip(names, np.squeeze(parameters).astype(float))),
            "empirical_meg": target.astype(float), "observed_regions": np.arange(68),
            "split": "train", "mapping_verified": False,
            "provenance": {"basis": "Historical notebook replay; structural identity unverified",
                           "raw_meg_index": 0, "raw_param_index": 0, "raw_connectivity_index": 0}}


@pytest.fixture(scope="module")
def replay_cache(historical_record, tmp_path_factory):
    path = tmp_path_factory.mktemp("fe_graph") / "cache.pt"
    generate_training_data([historical_record], FREQS, path,
                           preprocessing_id="historical_notebook_preparation",
                           purpose="historical_replay")
    return torch.load(path, weights_only=True)


def test_saved_cache_reproduces_real_analytical_prediction(replay_cache, historical_record):
    subject = replay_cache["subjects"][0]
    assert all(not v.requires_grad for g in subject["graphs"] for v in g.values()
               if torch.is_tensor(v))
    actual = predict_spectra(subject)
    expected = run_local_coupling_forward_torch(
        historical_record["brain"], historical_record["parameters"], FREQS
    )[0][:68]
    torch.testing.assert_close(actual, expected)
    np.testing.assert_allclose(transform_spectra(actual).numpy(),
                               transformed_spectra(actual.numpy()), atol=1e-10, rtol=1e-10)


def test_mlp_loss_gradient_through_real_cached_graphs(replay_cache):
    # Width and scale here are test settings, not selected experiment settings.
    torch.manual_seed(7)
    model = FeGraphMLP(hidden_sizes=[4], omega_scale=2 * np.pi * 45)
    subject = replay_cache["subjects"][0]

    def loss():
        filters = model(replay_cache["omega"])
        assert filters.shape == (40,) and filters.dtype == torch.complex128
        return 1 - spectral_correlation(predict_spectra(subject, filters), subject["empirical_meg"])

    loss().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    parameter = model.network[-1].bias
    gradient = parameter.grad[0].item()
    with torch.no_grad():
        original = parameter[0].item()
        parameter[0] = original + 1e-6
        plus = loss().item()
        parameter[0] = original - 1e-6
        minus = loss().item()
        parameter[0] = original
    assert abs(gradient) > 1e-10
    assert gradient == pytest.approx((plus - minus) / 2e-6, rel=1e-4, abs=1e-6)
    reloaded = FeGraphMLP([4], 2 * np.pi * 45)
    reloaded.load_state_dict(model.state_dict())
    torch.testing.assert_close(reloaded(replay_cache["omega"]), model(replay_cache["omega"]))


def test_unverified_data_cannot_be_prepared_or_trained(historical_record, replay_cache, tmp_path):
    with pytest.raises(ValueError, match="unverified"):
        generate_training_data([historical_record], FREQS, tmp_path / "invalid.pt",
                               preprocessing_id="historical_notebook_preparation")
    assert not (tmp_path / "invalid.pt").exists()
    with pytest.raises(ValueError, match="historical replay"):
        train_filter(replay_cache, tmp_path / "model.pt", hidden_sizes=[4],
                     omega_scale=100, epochs=1, learning_rate=0.001, seed=7)
