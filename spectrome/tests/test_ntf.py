"""NumPy forward-model smoke checks using the packaged template connectome."""

import numpy as np
import pytest

from ..brain.Brain import Brain
from ..forward import network_transfer as nt
from ..forward import runforward
from ..utils import path


@pytest.fixture
def brain():
    result = Brain()
    result.add_connectome(path.get_data_path())
    result.reorder_connectome(result.connectome, result.distance_matrix)
    result.bi_symmetric_c()
    result.reduce_extreme_dir()
    return result


def test_distance_matrix(brain):
    assert brain.distance_matrix.shape == (86, 86)


def test_connectome_preprocessing(brain):
    assert brain.reducedConnectome[0, 1] == 0.125


def test_ntf(brain):
    response, eigenvalues, eigenvectors, prediction, fc = nt.network_transfer_function(
        brain, parameters=brain.ntf_params, w=2 * np.pi
    )
    assert response.shape == eigenvalues.shape == prediction.shape == (86,)
    assert eigenvectors.shape == (86, 57)
    assert fc.shape == (86, 86)
    assert np.isfinite(prediction).all()


def test_run_forward(brain):
    prediction, response, eigenvalues, eigenvectors = runforward.run_forward(
        brain, brain.ntf_params, np.array([40.0])
    )
    assert prediction.shape == (86, 1)
    assert response.shape == eigenvalues.shape == (1, 86)
    assert eigenvectors.shape == (1, 86, 57)
    assert np.isfinite(prediction).all()
