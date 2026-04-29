from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import interpolate

Array2D = npt.NDArray[np.float64]
Array3D = npt.NDArray[np.float64]
Spacing = tuple[float, float]


def _as_float_2d(array: npt.ArrayLike, *, name: str) -> Array2D:
    values = np.asarray(array, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {values.shape!r}.")
    return values


@dataclass(slots=True)
class VectorField2D:
    x: Array2D
    y: Array2D
    spacing: Spacing = (1.0, 1.0)

    def __post_init__(self) -> None:
        self.x = _as_float_2d(self.x, name="x")
        self.y = _as_float_2d(self.y, name="y")
        if self.x.shape != self.y.shape:
            raise ValueError("x and y components must have the same shape.")

    @classmethod
    def from_array(cls, field: npt.ArrayLike, *, spacing: Spacing = (1.0, 1.0)) -> "VectorField2D":
        values = np.asarray(field, dtype=float)
        if values.ndim != 3 or values.shape[-1] != 2:
            raise ValueError("field must have shape (ny, nx, 2).")
        return cls(values[..., 0], values[..., 1], spacing=spacing)

    @classmethod
    def from_potential(
        cls,
        potential: npt.ArrayLike,
        *,
        spacing: Spacing = (1.0, 1.0),
        kind: str = "gradient",
    ) -> "VectorField2D":
        scalar = _as_float_2d(potential, name="potential")
        grad_y, grad_x = np.gradient(scalar, spacing[0], spacing[1])
        if kind == "gradient":
            return cls(grad_x, grad_y, spacing=spacing)
        if kind == "hamiltonian":
            return cls(-grad_y, grad_x, spacing=spacing)
        raise ValueError("kind must be either 'gradient' or 'hamiltonian'.")

    @property
    def shape(self) -> tuple[int, int]:
        return self.x.shape

    @property
    def magnitude(self) -> Array2D:
        return np.hypot(self.x, self.y)

    def as_array(self) -> Array3D:
        return np.stack((self.x, self.y), axis=-1)

    def divergence_curl(self) -> tuple[Array2D, Array2D]:
        dudy, dudx = np.gradient(self.x, self.spacing[0], self.spacing[1])
        dvdy, dvdx = np.gradient(self.y, self.spacing[0], self.spacing[1])
        divergence = dudx + dvdy
        curl = dvdx - dudy
        return divergence, curl


def gradient_field(
    potential: npt.ArrayLike,
    *,
    spacing: Spacing = (1.0, 1.0),
) -> VectorField2D:
    return VectorField2D.from_potential(potential, spacing=spacing, kind="gradient")


def hamiltonian_field(
    potential: npt.ArrayLike,
    *,
    spacing: Spacing = (1.0, 1.0),
) -> VectorField2D:
    return VectorField2D.from_potential(potential, spacing=spacing, kind="hamiltonian")


def divergence_curl(
    field: VectorField2D | npt.ArrayLike,
    *,
    spacing: Spacing = (1.0, 1.0),
) -> tuple[Array2D, Array2D]:
    return coerce_vector_field(field, spacing=spacing).divergence_curl()


def coerce_vector_field(
    field: VectorField2D | npt.ArrayLike,
    *,
    spacing: Spacing = (1.0, 1.0),
) -> VectorField2D:
    if isinstance(field, VectorField2D):
        return field
    return VectorField2D.from_array(field, spacing=spacing)


def integrate_field(
    field: VectorField2D | npt.ArrayLike,
    *,
    spacing: Spacing = (1.0, 1.0),
) -> Array2D:
    vector_field = coerce_vector_field(field, spacing=spacing)
    cumulative_x = np.cumsum(vector_field.x, axis=1) * vector_field.spacing[1]
    cumulative_y = np.cumsum(vector_field.y, axis=0) * vector_field.spacing[0]
    midpoint = cumulative_x.shape[1] // 2
    potential = cumulative_y[:, [midpoint]] + cumulative_x - cumulative_x[:, [midpoint]]
    potential += cumulative_x[0, midpoint]
    return potential


def integrate_interpolated(
    field: VectorField2D | npt.ArrayLike,
    *,
    oversampling: int = 2,
    spacing: Spacing = (1.0, 1.0),
) -> Array2D:
    vector_field = coerce_vector_field(field, spacing=spacing)
    rows, cols = vector_field.shape

    x_coords = np.linspace(0.0, 1.0, cols)
    y_coords = np.linspace(0.0, 1.0, rows)
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    sample_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    interp_x = interpolate.CloughTocher2DInterpolator(sample_points, vector_field.x.ravel())
    interp_y = interpolate.CloughTocher2DInterpolator(sample_points, vector_field.y.ravel())

    interp_x_coords = np.linspace(0.0, 1.0, oversampling * cols)
    interp_y_coords = np.linspace(0.0, 1.0, oversampling * rows)
    fine_x, fine_y = np.meshgrid(interp_x_coords, interp_y_coords)
    fine_points = np.column_stack((fine_x.ravel(), fine_y.ravel()))

    dphi_dx = interp_x(fine_points).reshape(fine_x.shape)
    dphi_dy = interp_y(fine_points).reshape(fine_x.shape)

    cumulative_x = np.nancumsum(dphi_dx, axis=1) * vector_field.spacing[1] / oversampling
    cumulative_y = np.nancumsum(dphi_dy, axis=0) * vector_field.spacing[0] / oversampling
    midpoint = cumulative_x.shape[1] // 2
    potential = cumulative_y[:, [midpoint]] + cumulative_x - cumulative_x[:, [midpoint]]
    potential += cumulative_x[0, midpoint]
    return potential[::oversampling, ::oversampling]
