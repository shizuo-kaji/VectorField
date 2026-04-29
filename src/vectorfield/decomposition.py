from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import Array2D, VectorField2D, coerce_vector_field, integrate_interpolated


@dataclass(slots=True)
class HelmholtzHodgeDecomposition:
    hamiltonian: VectorField2D
    gradient: VectorField2D
    harmonic: VectorField2D
    hamiltonian_potential: Array2D
    gradient_potential: Array2D


def fft_helmholtz_hodge_decomposition(
    field: VectorField2D | np.ndarray,
) -> HelmholtzHodgeDecomposition:
    vector_field = coerce_vector_field(field)
    values = vector_field.as_array()

    kx = np.fft.fftfreq(values.shape[0]).reshape(-1, 1)
    ky = np.fft.fftfreq(values.shape[1])
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.0

    dk = (np.fft.fftn(values[:, :, 0]) * kx + np.fft.fftn(values[:, :, 1]) * ky) / k2
    hamiltonian_values = np.stack(
        (
            np.fft.ifftn(dk * kx).real,
            np.fft.ifftn(dk * ky).real,
        ),
        axis=-1,
    )
    gradient_values = np.stack(
        (
            values[:, :, 0] - hamiltonian_values[:, :, 0],
            values[:, :, 1] - hamiltonian_values[:, :, 1],
        ),
        axis=-1,
    )

    hamiltonian = VectorField2D.from_array(hamiltonian_values, spacing=vector_field.spacing)
    gradient = VectorField2D.from_array(gradient_values, spacing=vector_field.spacing)
    harmonic = VectorField2D(
        np.zeros_like(hamiltonian.x),
        np.zeros_like(hamiltonian.y),
        spacing=vector_field.spacing,
    )

    hamiltonian_potential = integrate_interpolated(
        VectorField2D(hamiltonian.y, -hamiltonian.x, spacing=vector_field.spacing)
    )
    gradient_potential = integrate_interpolated(gradient)

    return HelmholtzHodgeDecomposition(
        hamiltonian=hamiltonian,
        gradient=gradient,
        harmonic=harmonic,
        hamiltonian_potential=hamiltonian_potential,
        gradient_potential=gradient_potential,
    )
