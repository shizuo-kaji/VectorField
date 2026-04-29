import numpy as np

from vectorfield import VectorField2D, fft_helmholtz_hodge_decomposition


def test_fft_helmholtz_hodge_decomposition_preserves_shape_and_reconstruction() -> None:
    x_grid, y_grid = np.meshgrid(np.linspace(-1.0, 1.0, 10), np.linspace(-1.0, 1.0, 8))
    scalar_a = np.sin(x_grid) + 0.5 * y_grid**2
    scalar_b = 0.25 * x_grid**2 + x_grid * y_grid

    hamiltonian = VectorField2D.from_potential(scalar_a, kind="hamiltonian")
    gradient = VectorField2D.from_potential(scalar_b, kind="gradient")
    mixed = VectorField2D(hamiltonian.x + gradient.x, hamiltonian.y + gradient.y)

    decomposition = fft_helmholtz_hodge_decomposition(mixed)
    reconstructed = decomposition.hamiltonian.as_array() + decomposition.gradient.as_array()

    assert decomposition.hamiltonian.shape == mixed.shape
    assert decomposition.gradient.shape == mixed.shape
    assert decomposition.harmonic.shape == mixed.shape
    assert decomposition.hamiltonian_potential.shape == mixed.shape
    assert decomposition.gradient_potential.shape == mixed.shape
    assert np.allclose(reconstructed, mixed.as_array())
