import numpy as np
import pytest

from vectorfield import optimal_assignment_transport


def test_optimal_assignment_transport_returns_vectors_aligned_to_sources() -> None:
    source_points = np.array([[0.0, 0.0], [2.0, 0.0]])
    target_points = np.array([[3.0, 0.0], [1.0, 0.0]])

    transport = optimal_assignment_transport(source_points, target_points)

    assert transport.source_points.shape == (2, 2)
    assert np.allclose(transport.matched_target_points, [[1.0, 0.0], [3.0, 0.0]])
    assert np.allclose(transport.vectors, [[1.0, 0.0], [1.0, 0.0]])
    assert transport.total_cost == pytest.approx(2.0)


def test_optimal_assignment_transport_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError):
        optimal_assignment_transport(np.zeros((0, 2)), np.zeros((1, 2)))
