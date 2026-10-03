"""Reproduce the notebook's saved scores with the historical pairing."""

from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
from scipy.io import loadmat
from scipy.stats import pearsonr
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]

from spectrome.forward.network_transfer import network_transfer_local_alpha
from spectrome.forward.network_transfer_torch import network_transfer_local_alpha_torch


DATA = ROOT / "spectrome" / "data"
FREQS = np.linspace(2, 45, 40)
LPF = np.array([1, 2, 5, 2, 1]) / 11


def prepared_brain(raw_connectivity, template_distance):
    # Literal spectral_correlation.ipynb cell 2 mapping and cell 7 operations.
    cort_lh = np.array(
        [0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 16,
         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 5, 32, 33, 9]
    )
    sub_lh = np.array([0, 40, 36, 39, 38, 37, 35, 34, 0])
    perm = np.concatenate([cort_lh, cort_lh + 41, sub_lh, sub_lh + 35])
    C = raw_connectivity[np.ix_(perm, perm)].copy()
    empty = np.array([68, 77, 76, 85])
    C[empty, :] = 0
    C[:, empty] = 0

    rotation = np.concatenate([np.arange(18, 86), np.arange(18)])
    C = C[np.ix_(rotation, rotation)]
    D = template_distance[np.ix_(rotation, rotation)].copy()

    left = np.concatenate([np.arange(34), np.arange(68, 77)])
    right = np.concatenate([np.arange(34, 68), np.arange(77, 86)])
    q = np.maximum(C[np.ix_(left, left)], C[np.ix_(right, right)])
    q1 = np.maximum(C[np.ix_(left, right)], C[np.ix_(right, left)])
    C[np.ix_(left, left)] = q
    C[np.ix_(right, right)] = q
    C[np.ix_(left, right)] = q1
    C[np.ix_(right, left)] = q1

    threshold = 7 * np.mean(C[C > 0])
    C = np.minimum(C, threshold)
    C = 0.95 * C + 0.05 * C  # Brain.reduce_extreme_dir, as implemented

    D = (D + D.T) / 2
    sort_indices = np.argsort(-D[:])
    D = np.minimum(D, sort_indices[int(np.round(0.01 * len(sort_indices)))])
    return SimpleNamespace(reducedConnectome=C, distance_matrix=D)


def transformed_spectra(raw):
    out = np.empty((68, len(FREQS)))
    for region in range(68):
        values = np.convolve(np.abs(raw[region]), LPF, "same")
        values = np.sqrt(np.abs(values))
        out[region] = values - np.mean(values)
    return out


def score(model, empirical):
    return float(np.mean([pearsonr(empirical[i], model[i])[0] for i in range(68)]))


def test_torch_reproduces_saved_historical_scores():
    raw_c = loadmat(DATA / "individual_subjects.mat")["A_all_subjs_final"]
    raw_p = np.squeeze(loadmat(DATA / "SCFC_opparam_individual.mat")["output"]["param"])
    raw_meg = np.squeeze(loadmat(DATA / "freqMEGdata.mat")["freqMEGdata"]["psd"])
    params = [np.squeeze(item).astype(float) for item in raw_p if item.shape == (7, 1)]
    meg = [item.astype(float) for item in raw_meg if item.shape == (68, 40)]
    assert len(params) == len(meg) == 36
    D = np.genfromtxt(DATA / "mean80_fiberlength.csv", delimiter=",")
    saved = np.squeeze(loadmat(ROOT / "spectrome/notebooks/individualC_optP.mat")["ind_opt_corr"])
    names = ("tau_e", "tau_i", "alpha", "speed", "gei", "gii", "tauC")

    for s in range(36):
        # Preserve the notebook's compacted C index here; this is historical parity.
        brain = prepared_brain(raw_c[:, :, s], D)
        p = dict(zip(names, params[s]))
        numpy_raw, torch_raw = [], []
        for f in FREQS:
            w = 2 * np.pi * f
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                numpy_raw.append(network_transfer_local_alpha(brain, p, w)[3][:68])
            torch_raw.append(network_transfer_local_alpha_torch(brain, p, w)[3][:68].detach().numpy())
        numpy_raw = np.asarray(numpy_raw).T
        torch_raw = np.asarray(torch_raw).T
        np.testing.assert_allclose(torch_raw, numpy_raw, atol=1e-9, rtol=1e-9)
        numpy_spectra = transformed_spectra(numpy_raw)
        torch_spectra = transformed_spectra(torch_raw)
        np.testing.assert_allclose(torch_spectra, numpy_spectra, atol=1e-9, rtol=1e-9)

        empirical = np.empty((68, 40))
        for region in range(68):
            values = np.convolve(meg[s][region], LPF, "same")
            values = np.sqrt(np.abs(values))
            empirical[region] = values - np.mean(values)
        numpy_score = score(numpy_spectra, empirical)
        torch_score = score(torch_spectra, empirical)
        assert np.isfinite(numpy_score) and np.isfinite(torch_score)
        assert abs(numpy_score - saved[s]) < 1e-9
        assert abs(torch_score - numpy_score) < 1e-9
