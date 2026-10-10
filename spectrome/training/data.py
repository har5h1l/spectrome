"""Load the chronological cohort and prepare fixed graph-filter training data.

prepare_cohort loads paired records. generate_training_data saves the frozen
graph cache (forward step 1). prepare_spectral_subject prepares local responses
(forward step 2) and transformed targets in memory before training.
"""

import hashlib
from pathlib import Path

import numpy as np
from scipy.io import loadmat
import torch

from spectrome.forward import network_transfer_torch as nt
from .spectra import transform_spectra
from .helpers import (
    validate_brain_matrices,
    validate_frequencies,
    validate_observations,
    validate_parameters,
    validate_prepared_graph,
    validate_subject_metadata,
)

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


@torch.no_grad()
def generate_training_data(subjects=None, frequencies_hz=None,
                           output_path="training_cache.pt", *,
                           preprocessing_id=None, data_dir=None, use_smalleigs=True,
                           purpose="training"):
    """Prepare and save fixed graphs and raw empirical targets on CPU.

    With no ``subjects``, prepare the current chronological cohort first.
    Alternatively supply paired records and an explicit preprocessing ID.
    Cache subjects once without split labels; train_filter assigns LOOCV roles.
    """
    if purpose not in {"training", "historical_replay"}:
        raise ValueError("purpose must be 'training' or 'historical_replay'")
    if subjects is None:
        subjects = prepare_cohort(data_dir)
        preprocessing_id = preprocessing_id or COHORT_PREPROCESSING_ID
    elif data_dir is not None:
        raise ValueError("Supply subjects or data_dir, not both")
    if not preprocessing_id:
        raise ValueError("preprocessing_id must identify the input preparation")
    if frequencies_hz is None:
        frequencies_hz = torch.linspace(2, 45, 40, dtype=torch.float64)

    frequencies = validate_frequencies(frequencies_hz)
    records, seen_ids = [], set()

    # Validate each subject, prepare its graph at each frequency, and cache it.
    for subject in subjects:
        subject_id, provenance = validate_subject_metadata(
            subject, purpose, seen_ids
        )
        parameters = validate_parameters(subject_id, subject.get("parameters"))
        brain = subject.get("brain")
        C, _ = validate_brain_matrices(subject_id, brain)
        regions, target = validate_observations(
            subject, subject_id, C.shape[0], frequencies.numel()
        )

        graphs = []
        for frequency in frequencies:
            graph = nt.prepare_graph_torch(
                brain, parameters, 2 * torch.pi * frequency,
                use_smalleigs=use_smalleigs,
            )
            graph = {
                name: value.detach().cpu().clone() if torch.is_tensor(value) else value
                for name, value in graph.items()
            }
            validate_prepared_graph(graph, subject_id, frequency.item())
            graphs.append(graph)

        records.append({
            "subject_id": subject_id,
            "parameters": parameters,
            "graphs": graphs,
            "empirical_meg": target,
            "observed_regions": regions,
            "provenance": provenance,
        })

    if not records:
        raise ValueError("subjects must contain at least one subject record")

    cache = {
        "format_version": 2,
        "purpose": purpose,
        "frequencies_hz": frequencies,
        "omega": 2 * torch.pi * frequencies,
        "preprocessing_id": preprocessing_id,
        "use_smalleigs": use_smalleigs,
        "forward_sha256": hashlib.sha256(Path(nt.__file__).read_bytes()).hexdigest(),
        "subjects": records,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)
    return cache


@torch.no_grad()
def prepare_spectral_subject(subject):
    """Prepare fixed frequency batches and target transforms once per training call.

    Return a separate dictionary; never mutate the raw cache or its targets.
    """
    terms = [nt.prepare_local_response_torch(graph, subject["parameters"])
             for graph in subject["graphs"]]
    if len({term["K"] for term in terms}) != 1:
        raise ValueError("All frequencies must use the same mode count")
    response = {key: torch.stack([term[key] for term in terms])
                for key in terms[0] if key != "K"}
    response["K"] = terms[0]["K"]
    target = transform_spectra(subject["empirical_meg"])
    norm = target.norm(dim=-1)
    if not torch.isfinite(target).all() or (norm <= 1e-12).any():
        raise ValueError("Spectral correlation requires finite, nonconstant regional spectra")
    return {**subject, "spectral_response": response,
            "transformed_target": target, "target_norm": norm}
