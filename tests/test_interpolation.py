import numpy as np
import pytest

from vectorfield import RBFVectorFieldInterpolator


def test_rbf_interpolator_fit_and_evaluate_grid() -> None:
    sample_points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    sample_vectors = np.tile(np.array([[1.0, -0.5]]), (4, 1))

    interpolator = RBFVectorFieldInterpolator.fit(
        sample_points,
        sample_vectors,
        grid_shape=(2, 2),
        neighbors=4,
        sigma=1.5,
    )

    evaluated = interpolator.evaluate(np.array([[0.5, 0.5]]))
    grid_x, grid_y = np.meshgrid(np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3))
    grid_field = interpolator.evaluate_grid(grid_x, grid_y)

    assert evaluated.shape == (1, 2)
    assert np.allclose(evaluated[0], [1.0, -0.5], atol=0.3)
    assert grid_field.shape == (3, 3)


def test_rbf_interpolator_rejects_bad_input_shapes() -> None:
    with pytest.raises(ValueError):
        RBFVectorFieldInterpolator.fit(np.zeros((3, 2)), np.zeros((3, 3)))


def test_rbf_interpolator_rejects_mismatched_grid_shapes() -> None:
    interpolator = RBFVectorFieldInterpolator.fit(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([[1.0, 1.0], [1.0, 1.0]]),
        grid_shape=(2, 2),
    )

    with pytest.raises(ValueError):
        interpolator.evaluate_grid(np.zeros((2, 2)), np.zeros((3, 3)))
