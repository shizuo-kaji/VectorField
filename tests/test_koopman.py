import numpy as np
from scipy.sparse import csr_matrix

from vectorfield import build_koopman_generator, build_koopman_generator_scharr


def test_build_koopman_generator_returns_expected_shape() -> None:
    vx = np.ones((4, 5))
    vy = np.ones((4, 5))

    generator = build_koopman_generator(vx, vy)

    assert isinstance(generator, csr_matrix)
    assert generator.shape == (20, 20)
    assert generator.nnz > 0


def test_build_koopman_generator_with_constraints_adds_rows() -> None:
    vx = np.ones((4, 5))
    vy = np.ones((4, 5))

    generator = build_koopman_generator(vx, vy, constraint_indices=[0, 3], constraint_weight=10.0)

    assert generator.shape == (22, 20)


def test_build_koopman_generator_scharr_returns_expected_shape() -> None:
    vx = np.ones((4, 4))
    vy = np.ones((4, 4))

    generator = build_koopman_generator_scharr(vx, vy, constraint_indices=[0], constraint_weight=5.0)

    assert isinstance(generator, csr_matrix)
    assert generator.shape == (17, 16)
    assert generator.nnz > 0
