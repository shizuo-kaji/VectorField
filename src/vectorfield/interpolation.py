from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree

from .core import VectorField2D

Array2D = npt.NDArray[np.float64]


def _as_points(values: npt.ArrayLike, *, name: str) -> Array2D:
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2).")
    return points


@dataclass(slots=True)
class RBFVectorFieldInterpolator:
    centers: Array2D
    weights_x: npt.NDArray[np.float64]
    weights_y: npt.NDArray[np.float64]
    neighbors: int = 5
    sigma: float = 1.0
    _tree: cKDTree = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.centers = _as_points(self.centers, name="centers")
        self.weights_x = np.asarray(self.weights_x, dtype=float)
        self.weights_y = np.asarray(self.weights_y, dtype=float)
        if self.centers.shape[0] != self.weights_x.shape[0] or self.centers.shape[0] != self.weights_y.shape[0]:
            raise ValueError("weights must match the number of centers.")
        self._tree = cKDTree(self.centers)

    @staticmethod
    def _basis(points: Array2D, centers: Array2D, sigma: float) -> npt.NDArray[np.float64]:
        return np.exp(-np.sum((points - centers) ** 2, axis=1) / sigma)

    @classmethod
    def fit(
        cls,
        sample_points: npt.ArrayLike,
        sample_vectors: npt.ArrayLike,
        *,
        domain: tuple[tuple[float, float], tuple[float, float]] | None = None,
        grid_shape: tuple[int, int] = (10, 10),
        neighbors: int = 5,
        sigma: float = 1.0,
    ) -> "RBFVectorFieldInterpolator":
        points = _as_points(sample_points, name="sample_points")
        vectors = _as_points(sample_vectors, name="sample_vectors")
        if points.shape[0] != vectors.shape[0]:
            raise ValueError("sample_points and sample_vectors must have the same length.")

        if domain is None:
            x_bounds = (float(points[:, 0].min()), float(points[:, 0].max()))
            y_bounds = (float(points[:, 1].min()), float(points[:, 1].max()))
        else:
            x_bounds, y_bounds = domain

        center_x, center_y = np.meshgrid(
            np.linspace(x_bounds[0], x_bounds[1], grid_shape[0]),
            np.linspace(y_bounds[0], y_bounds[1], grid_shape[1]),
        )
        centers = np.column_stack((center_x.ravel(), center_y.ravel()))
        tree = cKDTree(centers)

        neighbor_count = min(neighbors, len(centers))
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []

        for row_index, point in enumerate(points):
            _, center_indices = tree.query(point, k=neighbor_count)
            center_indices = np.atleast_1d(center_indices)
            basis_values = cls._basis(np.repeat(point[None, :], len(center_indices), axis=0), centers[center_indices], sigma)
            rows.extend([row_index] * len(center_indices))
            cols.extend(center_indices.tolist())
            data.extend(basis_values.tolist())

        design = csc_matrix((data, (rows, cols)), shape=(len(points), len(centers)))
        weights_x = lsqr(design, vectors[:, 0])[0]
        weights_y = lsqr(design, vectors[:, 1])[0]
        return cls(centers, weights_x, weights_y, neighbors=neighbor_count, sigma=sigma)

    def evaluate(self, points: npt.ArrayLike) -> Array2D:
        query_points = _as_points(points, name="points")
        _, center_indices = self._tree.query(query_points, k=self.neighbors)
        center_indices = np.asarray(center_indices)
        if center_indices.ndim == 1:
            center_indices = center_indices[:, None]

        vectors = np.zeros((len(query_points), 2), dtype=float)
        for row_index, point in enumerate(query_points):
            indices = np.atleast_1d(center_indices[row_index])
            basis_values = self._basis(
                np.repeat(point[None, :], len(indices), axis=0),
                self.centers[indices],
                self.sigma,
            )
            vectors[row_index, 0] = np.sum(self.weights_x[indices] * basis_values)
            vectors[row_index, 1] = np.sum(self.weights_y[indices] * basis_values)
        return vectors

    def evaluate_grid(self, x_grid: npt.ArrayLike, y_grid: npt.ArrayLike) -> VectorField2D:
        grid_x = np.asarray(x_grid, dtype=float)
        grid_y = np.asarray(y_grid, dtype=float)
        if grid_x.shape != grid_y.shape:
            raise ValueError("x_grid and y_grid must have the same shape.")
        vectors = self.evaluate(np.column_stack((grid_x.ravel(), grid_y.ravel())))
        return VectorField2D(
            vectors[:, 0].reshape(grid_x.shape),
            vectors[:, 1].reshape(grid_x.shape),
        )
