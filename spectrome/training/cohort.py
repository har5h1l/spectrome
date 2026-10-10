"""Load the current cohort in its shared chronological subject order."""

import hashlib
from pathlib import Path

import numpy as np
from scipy.io import loadmat

PARAMETER_NAMES = ("tau_e", "tau_i", "alpha", "speed", "gei", "gii", "tauC")
COHORT_PREPROCESSING_ID = "chronological_cohort_notebook_graph_cortical_outputs_v1"


def prepare_cohort(data_dir=None):
    """Return paired records using one inclusion rule for all three files.

    Anil confirmed the chronological subject order and existing cortical
    alignment. Preserve the notebook's graph preparation and 68 observed
    outputs; this does not introduce a cortex-only graph or refit parameters.
    Targets remain raw, so the training objective transforms them only once.
    """
    from spectrome.brain.Brain import Brain

    data_dir = Path(data_dir) if data_dir is not None else Path(__file__).parents[1] / "data"
    paths = {name: data_dir / name for name in (
        "individual_subjects.mat", "SCFC_opparam_individual.mat",
        "freqMEGdata.mat", "mean80_fibercount.csv", "mean80_fiberlength.csv",
    )}
    sources = {name: {"path": str(path.resolve()),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
               for name, path in paths.items()}
    connectivity = loadmat(paths["individual_subjects.mat"])["A_all_subjs_final"]
    fitted = loadmat(paths["SCFC_opparam_individual.mat"])["output"].ravel()
    meg = loadmat(paths["freqMEGdata.mat"])["freqMEGdata"].ravel()
    if connectivity.shape != (82, 82, len(meg)) or len(fitted) != len(meg):
        raise ValueError("Cohort connectivity, parameter, and MEG record counts/shapes differ")

    included, excluded = [], []
    for index, (fit, observation) in enumerate(zip(fitted, meg)):
        parameters, target = fit["param"], observation["psd"]
        if parameters.size == 0 and target.size == 0:
            excluded.append(index)
            continue
        if parameters.shape != (7, 1) or target.shape != (68, 40):
            raise ValueError(f"Record {index}: expected paired 7x1 parameters and 68x40 MEG")
        subject_id = str(observation["name"].item())
        if subject_id != str(fit["subjname"].item()):
            raise ValueError(f"Record {index}: chronological MEG and parameter names differ")
        included.append((index, subject_id, parameters, target))
    if not included:
        raise ValueError("Cohort has no usable paired subjects")

    # Literal individual-connectome region preparation from the cohort notebook.
    cortex_left = np.array([0, 1, 2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14,
                            15, 17, 16, 18, 19, 20, 21, 22, 23, 24, 25,
                            26, 27, 28, 29, 30, 31, 5, 32, 33, 9])
    subcortex_left = np.array([0, 40, 36, 39, 38, 37, 35, 34, 0])
    permutation = np.concatenate([cortex_left, cortex_left + 41,
                                  subcortex_left, subcortex_left + 35])
    empty_nodes = np.array([68, 77, 76, 85])
    subjects = []
    for index, subject_id, parameters, target in included:
        C = connectivity[:, :, index][np.ix_(permutation, permutation)].copy()
        C[empty_nodes, :] = 0
        C[:, empty_nodes] = 0
        brain = Brain()
        brain.add_connectome(str(data_dir))
        brain.connectome = C
        brain.reorder_connectome(brain.connectome, brain.distance_matrix)
        brain.bi_symmetric_c()
        brain.reduce_extreme_dir()
        # Preserve the notebook-local get_mean_C operation, including its
        # index-based cap. Changing distance preparation is a separate change.
        D = (brain.distance_matrix + brain.distance_matrix.T) / 2
        indices = np.argsort(-D[:])
        brain.distance_matrix = np.minimum(D, indices[int(np.round(0.01 * len(indices)))])
        subjects.append({
            "subject_id": subject_id,
            "brain": brain,
            "parameters": dict(zip(PARAMETER_NAMES, parameters[:, 0])),
            "empirical_meg": target.copy(),
            "observed_regions": np.arange(68),
            "mapping_verified": True,
            "provenance": {
                "alignment_basis": "Anil meeting 2026-10-05; shared chronological subject order and existing cortical ordering confirmed by Harshil",
                "sources": sources,
                "raw_meg_index": index,
                "raw_param_index": index,
                "raw_connectivity_index": index,
                "excluded_original_indices": excluded,
                "exclusion_rule": "Both MEG and fitted parameters empty; one shared inclusion rule",
                "preprocessing_id": COHORT_PREPROCESSING_ID,
                "region_preparation": "Existing notebook permutation, empty nodes, rotation, symmetry and clipping; first 68 outputs",
                "graph_nodes": 86,
                "observed_region_indices": list(range(68)),
                "distance_units": "mm",
                "distance_preparation": "Template distance; notebook get_mean_C unchanged",
                "frequency_basis": "Notebook linspace(2, 45, 40) Hz; source MEG has no stored frequency vector",
            },
        })
    return subjects
