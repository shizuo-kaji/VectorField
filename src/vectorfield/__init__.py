from __future__ import annotations

from .core import (
    VectorField2D,
    divergence_curl,
    gradient_field,
    hamiltonian_field,
    integrate_field,
    integrate_interpolated,
)
from .decomposition import HelmholtzHodgeDecomposition, fft_helmholtz_hodge_decomposition
from .interpolation import RBFVectorFieldInterpolator
from .koopman import build_koopman_generator, build_koopman_generator_scharr
from .plotting import plot_scalar_field, plot_vector_field
from .transport import PointCloudTransport, optimal_assignment_transport

__all__ = [
    "HelmholtzHodgeDecomposition",
    "PointCloudTransport",
    "RBFVectorFieldInterpolator",
    "VectorField2D",
    "build_koopman_generator",
    "build_koopman_generator_scharr",
    "divergence_curl",
    "fft_helmholtz_hodge_decomposition",
    "gradient_field",
    "hamiltonian_field",
    "integrate_field",
    "integrate_interpolated",
    "optimal_assignment_transport",
    "plot_scalar_field",
    "plot_vector_field",
]

__version__ = "0.1.0"
