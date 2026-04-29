from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

Array2D = npt.NDArray[np.float64]


def _as_points(values: npt.ArrayLike, *, name: str) -> Array2D:
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2).")
    if points.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one point.")
    return points


@dataclass(slots=True)
class PointCloudTransport:
    source_points: Array2D
    target_points: Array2D
    matched_target_points: Array2D
    vectors: Array2D
    source_indices: npt.NDArray[np.int64]
    target_indices: npt.NDArray[np.int64]
    total_cost: float


def optimal_assignment_transport(
    source_points: npt.ArrayLike,
    target_points: npt.ArrayLike,
    *,
    squared: bool = True,
) -> PointCloudTransport:
    """Match two point clouds with a minimum-cost one-to-one assignment.

    This is a discrete equal-weight transport helper for examples and small
    point clouds. If the clouds have different sizes, the returned transport is
    defined on the largest matchable subset.
    """

    source = _as_points(source_points, name="source_points")
    target = _as_points(target_points, name="target_points")

    distances = cdist(source, target, metric="sqeuclidean" if squared else "euclidean")
    source_indices, target_indices = linear_sum_assignment(distances)
    matched_source = source[source_indices]
    matched_target = target[target_indices]
    vectors = matched_target - matched_source

    return PointCloudTransport(
        source_points=matched_source,
        target_points=target,
        matched_target_points=matched_target,
        vectors=vectors,
        source_indices=source_indices,
        target_indices=target_indices,
        total_cost=float(distances[source_indices, target_indices].sum()),
    )
