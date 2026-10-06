"""Replay the historical notebook path for presentation figures, without training.

Run with the spectrome environment. Reuses the repeatable parity test's literal
data preparation. This preserves historical positional pairing deliberately.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
from scipy.io import loadmat
import torch
from spectrome.tests.test_historical_torch_parity import (
    prepared_brain, transformed_spectra, score, FREQS, LPF,
)
from spectrome.forward.network_transfer import network_transfer_local_alpha
from spectrome.forward.network_transfer_torch import network_transfer_local_alpha_torch


def main():
    data = ROOT / "spectrome/data"
    raw_c = loadmat(data / "individual_subjects.mat")["A_all_subjs_final"]
    raw_p = np.squeeze(loadmat(data / "SCFC_opparam_individual.mat")["output"]["param"])
    raw_meg = np.squeeze(loadmat(data / "freqMEGdata.mat")["freqMEGdata"]["psd"])
    retained = [i for i, item in enumerate(raw_p) if item.shape == (7, 1)]
    assert retained == [i for i, item in enumerate(raw_meg) if item.shape == (68, 40)]
    params = [np.squeeze(raw_p[i]).astype(float) for i in retained]
    meg = [raw_meg[i].astype(float) for i in retained]
    D = np.genfromtxt(data / "mean80_fiberlength.csv", delimiter=",")
    saved = np.squeeze(loadmat(ROOT / "spectrome/notebooks/individualC_optP.mat")["ind_opt_corr"])
    names = ("tau_e", "tau_i", "alpha", "speed", "gei", "gii", "tauC")
    rows, spectra, empirical_spectra = [], [], []
    for s in range(36):
        brain = prepared_brain(raw_c[:, :, s], D)
        p = dict(zip(names, params[s]))
        numpy_raw, torch_raw = [], []
        for f in FREQS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                numpy_raw.append(network_transfer_local_alpha(brain, p, 2*np.pi*f)[3][:68])
            torch_raw.append(network_transfer_local_alpha_torch(brain, p, 2*np.pi*f)[3][:68].detach().numpy())
        numpy_raw, torch_raw = np.asarray(numpy_raw).T, np.asarray(torch_raw).T
        x, xt = transformed_spectra(numpy_raw), transformed_spectra(torch_raw)
        empirical = np.asarray([np.sqrt(np.abs(np.convolve(m, LPF, "same"))) for m in meg[s]])
        empirical -= empirical.mean(axis=1, keepdims=True)
        rows.append(dict(record=s+1, retained_raw_index=retained[s], structural_raw_index=s,
                         saved=float(saved[s]), numpy=score(x, empirical), torch=score(xt, empirical),
                         raw_error=float(np.max(np.abs(torch_raw-numpy_raw))),
                         transformed_error=float(np.max(np.abs(xt-x)))))
        spectra.append(x)
        empirical_spectra.append(empirical)
    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    with (figures / "cohort_scores.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    # Choose the lower median by cohort mean correlation, without inspecting plots.
    selected = int(np.argsort([r["numpy"] for r in rows])[17])
    np.savez(figures / "historical_spectra.npz", frequencies=FREQS,
             prediction=np.asarray(spectra), empirical=np.asarray(empirical_spectra),
             example_index=selected, displayed_region_rows=np.array([0,22,45]))
    files = [
        "spectrome/data/individual_subjects.mat", "spectrome/data/SCFC_opparam_individual.mat",
        "spectrome/data/freqMEGdata.mat", "spectrome/data/mean80_fiberlength.csv",
        "spectrome/notebooks/individualC_optP.mat", "spectrome/tests/test_historical_torch_parity.py",
        "spectrome/forward/network_transfer.py", "spectrome/forward/network_transfer_torch.py",
        "spectrome/training/filters.py", "spectrome/training/train.py",
    ]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "historical_replay_only", "records": len(rows),
        "regions": 68, "frequencies": len(FREQS), "frequency_range_hz": [2,45],
        "mean_numpy_correlation": float(np.mean([r["numpy"] for r in rows])),
        "max_saved_error": max(abs(r["numpy"]-r["saved"]) for r in rows),
        "max_torch_correlation_error": max(abs(r["torch"]-r["numpy"]) for r in rows),
        "max_raw_error": max(r["raw_error"] for r in rows),
        "max_transformed_error": max(r["transformed_error"] for r in rows),
        "example_record": selected+1, "example_selection": "lower median of mean regional correlation",
        "region_rows_one_based": [1,23,46],
        "empty_raw_indices_zero_based": sorted(set(range(39))-set(retained)),
        "positional_mismatches": sum(s != i for s,i in enumerate(retained)),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "torch": torch.__version__},
        "source_sha256": {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},
        "limitations": ["Historical compacted structural indexing is preserved.",
                        "This does not verify subject or region identity.",
                        "Supplied parameter fits are reused. No fitting or MLP training is run."]
    }
    assert summary["max_saved_error"] < 1e-9
    assert summary["max_torch_correlation_error"] < 1e-9
    assert summary["max_raw_error"] < 1e-9
    (figures / "evidence.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k:v for k,v in summary.items() if k not in {"source_sha256","limitations"}}, indent=2))


if __name__ == "__main__":
    main()
