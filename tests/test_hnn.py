import numpy as np
import pytest

torch = pytest.importorskip("torch")

from vectorfield.hnn import HNN, HamiltonianNeuralNetwork, fit_HNN, fit_hamiltonian_network


def test_hnn_uses_tanh_and_supports_grid_helpers() -> None:
    grid_x, grid_y = np.meshgrid(np.linspace(-1.0, 1.0, 3), np.linspace(-2.0, 2.0, 4))
    model = HamiltonianNeuralNetwork(hidden_dims=(8, 4), grid_x=grid_x, grid_y=grid_y)

    assert any(isinstance(module, torch.nn.Tanh) for module in model.network)

    field = model.vfield()
    hamiltonian = model.hamiltonian()

    assert field.shape == grid_x.shape + (2,)
    assert hamiltonian.shape == grid_x.shape


def test_vector_field_aliases_match() -> None:
    model = HNN(hidden_dims=(8,))
    points = torch.tensor([[0.2, -0.3], [1.0, 0.5]], dtype=torch.float32, requires_grad=True)

    via_vfield = model.vfield(points)
    via_vector_field = model.vector_field(points)

    assert torch.allclose(via_vfield, via_vector_field)


def test_fit_hnn_logs_expanded_losses(capsys: pytest.CaptureFixture[str]) -> None:
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    vectors = np.zeros_like(points)

    model = fit_hamiltonian_network(points, vectors, epochs=2, verbose=True, hidden_dims=(4,))

    captured = capsys.readouterr().out

    assert isinstance(model, HamiltonianNeuralNetwork)
    assert "total loss" in captured
    assert "MSE" in captured
    assert "boundary loss" in captured
    assert "topology loss" in captured


def test_fit_hnn_alias_returns_model() -> None:
    points = np.array([[0.0, 0.0], [0.5, -0.5]], dtype=np.float32)
    vectors = np.zeros_like(points)

    model = fit_HNN(points, vectors, epochs=1, verbose=False, hidden_dims=(4,))

    assert isinstance(model, HamiltonianNeuralNetwork)
