import numpy as np
import pytest

from vectorfield import (
    VectorField2D,
    divergence_curl,
    gradient_field,
    hamiltonian_field,
    integrate_field,
    integrate_interpolated,
)


def test_vector_field_from_potential_gradient_and_hamiltonian() -> None:
    x_grid, y_grid = np.meshgrid(np.arange(4.0), np.arange(3.0))
    potential = x_grid + 2.0 * y_grid

    gradient = VectorField2D.from_potential(potential, kind="gradient")
    hamiltonian = VectorField2D.from_potential(potential, kind="hamiltonian")

    assert gradient.shape == potential.shape
    assert np.allclose(gradient.x, 1.0)
    assert np.allclose(gradient.y, 2.0)
    assert np.allclose(hamiltonian.x, -2.0)
    assert np.allclose(hamiltonian.y, 1.0)


def test_potential_helper_functions_match_class_constructor() -> None:
    x_grid, y_grid = np.meshgrid(np.arange(4.0), np.arange(3.0))
    potential = x_grid + 2.0 * y_grid

    gradient = gradient_field(potential)
    hamiltonian = hamiltonian_field(potential)
    divergence, curl = divergence_curl(gradient)

    assert np.allclose(gradient.as_array(), VectorField2D.from_potential(potential).as_array())
    assert np.allclose(hamiltonian.x, -2.0)
    assert np.allclose(divergence, 0.0)
    assert np.allclose(curl, 0.0)


def test_vector_field_divergence_and_curl_for_linear_gradient_field() -> None:
    x_grid, y_grid = np.meshgrid(np.arange(5.0), np.arange(4.0))
    potential = x_grid + y_grid
    gradient = VectorField2D.from_potential(potential, kind="gradient")

    divergence, curl = gradient.divergence_curl()

    assert divergence.shape == potential.shape
    assert curl.shape == potential.shape
    assert np.allclose(divergence, 0.0)
    assert np.allclose(curl, 0.0)


def test_integrate_field_recovers_expected_cumulative_pattern() -> None:
    vector = VectorField2D(np.ones((3, 4)), np.zeros((3, 4)))

    integrated = integrate_field(vector)

    expected = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (3, 1))
    assert np.allclose(integrated, expected)


def test_integrate_interpolated_returns_finite_array_with_same_shape() -> None:
    x_grid, y_grid = np.meshgrid(np.linspace(-1.0, 1.0, 6), np.linspace(-1.0, 1.0, 5))
    potential = x_grid**2 + y_grid**2
    vector = VectorField2D.from_potential(potential, kind="gradient")

    integrated = integrate_interpolated(vector, oversampling=2)

    assert integrated.shape == potential.shape
    assert np.isfinite(integrated).all()


def test_vector_field_from_array_validates_shape() -> None:
    with pytest.raises(ValueError):
        VectorField2D.from_array(np.zeros((3, 4)))


def test_vector_field_from_potential_rejects_invalid_kind() -> None:
    with pytest.raises(ValueError):
        VectorField2D.from_potential(np.zeros((3, 4)), kind="invalid")
