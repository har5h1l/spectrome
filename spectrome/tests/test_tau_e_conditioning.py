"""Small implementation checks for optional subject tau_e conditioning."""

import pytest

torch = pytest.importorskip("torch")

from spectrome.mlp.model import FeGraphMLP


def test_conditioned_model_uses_scaled_subject_tau_and_keeps_parameter_names():
    torch.manual_seed(7)
    model = FeGraphMLP((3,), 100.0, tau_e_conditioned=True)
    omega = torch.tensor([10.0, 20.0], dtype=torch.float64)
    first = model(omega, 0.01)
    second = model(omega, 0.02)
    assert first.shape == second.shape == (2,)
    assert first.dtype == torch.complex128
    assert not torch.allclose(first, second)
    assert set(model.state_dict()) == {"omega_scale", "network.0.weight", "network.0.bias",
                                       "network.2.weight", "network.2.bias"}


@pytest.mark.parametrize("tau_e", [None, 0.0, -0.01, float("nan"), float("inf"), [0.01, 0.02]])
def test_conditioned_model_rejects_missing_or_invalid_tau_e(tau_e):
    model = FeGraphMLP((3,), 100.0, tau_e_conditioned=True)
    with pytest.raises(ValueError, match="tau_e"):
        model(torch.tensor([10.0], dtype=torch.float64), tau_e)


def test_frequency_only_model_still_accepts_legacy_call_and_state_dict():
    torch.manual_seed(7)
    legacy = FeGraphMLP((3,), 100.0)
    torch.manual_seed(7)
    explicit_false = FeGraphMLP((3,), 100.0, tau_e_conditioned=False)
    omega = torch.tensor([10.0, 20.0], dtype=torch.float64)
    torch.testing.assert_close(legacy(omega), explicit_false(omega))
    assert legacy.state_dict().keys() == explicit_false.state_dict().keys()

