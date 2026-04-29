from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import sparse


def build_koopman_generator(
    vx: np.ndarray,
    vy: np.ndarray,
    *,
    spacing: tuple[float, float] = (1.0, 1.0),
    constraint_indices: Sequence[int] = (),
    constraint_weight: float = 1e3,
) -> sparse.csr_matrix:
    ny, nx = vx.shape
    gradient_x = sparse.dok_matrix((nx * ny + len(constraint_indices), nx * ny))
    for row_index, index in enumerate(constraint_indices):
        gradient_x[nx * ny + row_index, index] = constraint_weight

    for y in range(ny):
        gradient_x[y * nx, y * nx + 1] = vx[y, 0] / spacing[1]
        gradient_x[y * nx, y * nx] = -vx[y, 0] / spacing[1]
        gradient_x[y * nx + nx - 1, y * nx + nx - 1] = vx[y, nx - 1] / spacing[1]
        gradient_x[y * nx + nx - 1, y * nx + nx - 2] = -vx[y, nx - 1] / spacing[1]
        for x in range(1, nx - 1):
            gradient_x[y * nx + x, y * nx + x + 1] = 0.5 * vx[y, x] / spacing[1]
            gradient_x[y * nx + x, y * nx + x - 1] = -0.5 * vx[y, x] / spacing[1]

    gradient_y = sparse.dok_matrix((nx * ny + len(constraint_indices), nx * ny))
    for x in range(nx):
        gradient_y[x, nx + x] = vy[0, x] / spacing[0]
        gradient_y[x, x] = -vy[0, x] / spacing[0]
        gradient_y[(ny - 1) * nx + x, (ny - 1) * nx + x] = vy[ny - 1, x] / spacing[0]
        gradient_y[(ny - 1) * nx + x, (ny - 2) * nx + x] = -vy[ny - 1, x] / spacing[0]
        for y in range(1, ny - 1):
            gradient_y[y * nx + x, (y + 1) * nx + x] = 0.5 * vy[y, x] / spacing[0]
            gradient_y[y * nx + x, (y - 1) * nx + x] = -0.5 * vy[y, x] / spacing[0]

    return gradient_x.tocsr() + gradient_y.tocsr()


def build_koopman_generator_scharr(
    vx: np.ndarray,
    vy: np.ndarray,
    *,
    spacing: tuple[float, float] = (1.0, 1.0),
    a: float = 10,
    b: float = 3,
    constraint_indices: Sequence[int] = (),
    constraint_weight: float = 1e3,
) -> sparse.csr_matrix:
    ny, nx = vx.shape
    total = a + 2 * b
    a_norm = a / total
    b_norm = b / total
    a_edge = a_norm / (a_norm + b_norm)
    b_edge = b_norm / (a_norm + b_norm)

    gradient_x = sparse.dok_matrix((nx * ny + len(constraint_indices), nx * ny))
    for row_index, index in enumerate(constraint_indices):
        gradient_x[nx * ny + row_index, index] = constraint_weight

    gradient_x[0, 1] = 2 * a_edge * vx[0, 0] / spacing[1]
    gradient_x[0, 0] = -2 * a_edge * vx[0, 0] / spacing[1]
    gradient_x[0, nx + 1] = 2 * b_edge * vx[0, 0] / spacing[1]
    gradient_x[0, nx] = -2 * b_edge * vx[0, 0] / spacing[1]

    gradient_x[nx - 1, nx - 1] = 2 * a_edge * vx[0, nx - 1] / spacing[1]
    gradient_x[nx - 1, nx - 2] = -2 * a_edge * vx[0, nx - 1] / spacing[1]
    gradient_x[nx - 1, 2 * nx - 1] = 2 * b_edge * vx[0, nx - 1] / spacing[1]
    gradient_x[nx - 1, 2 * nx - 2] = -2 * b_edge * vx[0, nx - 1] / spacing[1]

    gradient_x[(ny - 1) * nx, (ny - 1) * nx + 1] = 2 * a_edge * vx[ny - 1, 0] / spacing[1]
    gradient_x[(ny - 1) * nx, (ny - 1) * nx] = -2 * a_edge * vx[ny - 1, 0] / spacing[1]
    gradient_x[(ny - 1) * nx, (ny - 2) * nx + 1] = 2 * b_edge * vx[ny - 1, 0] / spacing[1]
    gradient_x[(ny - 1) * nx, (ny - 2) * nx] = -2 * b_edge * vx[ny - 1, 0] / spacing[1]

    gradient_x[ny * nx - 1, ny * nx - 1] = 2 * a_edge * vx[ny - 1, nx - 1] / spacing[1]
    gradient_x[ny * nx - 1, ny * nx - 2] = -2 * a_edge * vx[ny - 1, nx - 1] / spacing[1]
    gradient_x[ny * nx - 1, (ny - 1) * nx - 1] = 2 * b_edge * vx[ny - 1, nx - 1] / spacing[1]
    gradient_x[ny * nx - 1, (ny - 1) * nx - 2] = -2 * b_edge * vx[ny - 1, nx - 1] / spacing[1]

    for x in range(1, nx - 1):
        gradient_x[x, x + 1] = a_edge * vx[0, x] / spacing[1]
        gradient_x[x, x - 1] = -a_edge * vx[0, x] / spacing[1]
        gradient_x[x, nx + x + 1] = b_edge * vx[0, x] / spacing[1]
        gradient_x[x, nx + x - 1] = -b_edge * vx[0, x] / spacing[1]

    for x in range(1, nx - 1):
        gradient_x[(ny - 1) * nx + x, (ny - 1) * nx + x + 1] = a_edge * vx[ny - 1, x] / spacing[1]
        gradient_x[(ny - 1) * nx + x, (ny - 1) * nx + x - 1] = -a_edge * vx[ny - 1, x] / spacing[1]
        gradient_x[(ny - 1) * nx + x, (ny - 2) * nx + x + 1] = b_edge * vx[ny - 1, x] / spacing[1]
        gradient_x[(ny - 1) * nx + x, (ny - 2) * nx + x - 1] = -b_edge * vx[ny - 1, x] / spacing[1]

    for y in range(1, ny - 1):
        gradient_x[y * nx, y * nx + 1] = 2 * a_norm * vx[y, 0] / spacing[1]
        gradient_x[y * nx, y * nx] = -2 * a_norm * vx[y, 0] / spacing[1]
        gradient_x[y * nx, (y - 1) * nx + 1] = 2 * b_norm * vx[y, 0] / spacing[1]
        gradient_x[y * nx, (y - 1) * nx] = -2 * b_norm * vx[y, 0] / spacing[1]
        gradient_x[y * nx, (y + 1) * nx + 1] = 2 * b_norm * vx[y, 0] / spacing[1]
        gradient_x[y * nx, (y + 1) * nx] = -2 * b_norm * vx[y, 0] / spacing[1]

    for y in range(1, ny - 1):
        gradient_x[y * nx + nx - 1, y * nx + nx - 1] = 2 * a_norm * vx[y, nx - 1] / spacing[1]
        gradient_x[y * nx + nx - 1, y * nx + nx - 2] = -2 * a_norm * vx[y, nx - 1] / spacing[1]
        gradient_x[y * nx + nx - 1, (y - 1) * nx + nx - 1] = 2 * b_norm * vx[y, nx - 1] / spacing[1]
        gradient_x[y * nx + nx - 1, (y - 1) * nx + nx - 2] = -2 * b_norm * vx[y, nx - 1] / spacing[1]
        gradient_x[y * nx + nx - 1, (y + 1) * nx + nx - 1] = 2 * b_norm * vx[y, nx - 1] / spacing[1]
        gradient_x[y * nx + nx - 1, (y + 1) * nx + nx - 2] = -2 * b_norm * vx[y, nx - 1] / spacing[1]

    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            gradient_x[y * nx + x, y * nx + x + 1] = a_norm * vx[y, x] / spacing[1]
            gradient_x[y * nx + x, y * nx + x - 1] = -a_norm * vx[y, x] / spacing[1]
            gradient_x[y * nx + x, (y - 1) * nx + x + 1] = b_norm * vx[y, x] / spacing[1]
            gradient_x[y * nx + x, (y + 1) * nx + x + 1] = b_norm * vx[y, x] / spacing[1]
            gradient_x[y * nx + x, (y - 1) * nx + x - 1] = -b_norm * vx[y, x] / spacing[1]
            gradient_x[y * nx + x, (y + 1) * nx + x - 1] = -b_norm * vx[y, x] / spacing[1]

    gradient_y = sparse.dok_matrix((nx * ny + len(constraint_indices), nx * ny))
    gradient_y[0, nx] = 2 * a_edge * vy[0, 0] / spacing[0]
    gradient_y[0, 0] = -2 * a_edge * vy[0, 0] / spacing[0]
    gradient_y[0, nx + 1] = 2 * b_edge * vy[0, 0] / spacing[0]
    gradient_y[0, 1] = -2 * b_edge * vy[0, 0] / spacing[0]

    gradient_y[(ny - 1) * nx, (ny - 1) * nx] = 2 * a_edge * vy[ny - 1, 0] / spacing[0]
    gradient_y[(ny - 1) * nx, (ny - 2) * nx] = -2 * a_edge * vy[ny - 1, 0] / spacing[0]
    gradient_y[(ny - 1) * nx, (ny - 1) * nx + 1] = 2 * b_edge * vy[ny - 1, 0] / spacing[0]
    gradient_y[(ny - 1) * nx, (ny - 2) * nx + 1] = -2 * b_edge * vy[ny - 1, 0] / spacing[0]

    gradient_y[nx - 1, nx - 1] = -2 * a_edge * vy[0, nx - 1] / spacing[0]
    gradient_y[nx - 1, 2 * nx - 1] = 2 * a_edge * vy[0, nx - 1] / spacing[0]
    gradient_y[nx - 1, nx - 2] = -2 * b_edge * vy[0, nx - 1] / spacing[0]
    gradient_y[nx - 1, 2 * nx - 2] = 2 * b_edge * vy[0, nx - 1] / spacing[0]

    gradient_y[ny * nx - 1, ny * nx - 1] = 2 * a_edge * vy[ny - 1, nx - 1] / spacing[0]
    gradient_y[ny * nx - 1, (ny - 1) * nx - 1] = -2 * a_edge * vy[ny - 1, nx - 1] / spacing[0]
    gradient_y[ny * nx - 1, ny * nx - 2] = 2 * b_edge * vy[ny - 1, nx - 1] / spacing[0]
    gradient_y[ny * nx - 1, (ny - 1) * nx - 2] = -2 * b_edge * vy[ny - 1, nx - 1] / spacing[0]

    for y in range(1, ny - 1):
        gradient_y[y * nx, (y + 1) * nx] = a_edge * vy[y, 0] / spacing[0]
        gradient_y[y * nx, (y - 1) * nx] = -a_edge * vy[y, 0] / spacing[0]
        gradient_y[y * nx, (y + 1) * nx + 1] = b_edge * vy[y, 0] / spacing[0]
        gradient_y[y * nx, (y - 1) * nx + 1] = -b_edge * vy[y, 0] / spacing[0]

    for y in range(1, ny - 1):
        gradient_y[y * nx + nx - 1, (y + 1) * nx + nx - 1] = a_edge * vy[y, nx - 1] / spacing[0]
        gradient_y[y * nx + nx - 1, (y - 1) * nx + nx - 1] = -a_edge * vy[y, nx - 1] / spacing[0]
        gradient_y[y * nx + nx - 1, (y + 1) * nx + nx - 2] = b_edge * vy[y, nx - 1] / spacing[0]
        gradient_y[y * nx + nx - 1, (y - 1) * nx + nx - 2] = -b_edge * vy[y, nx - 1] / spacing[0]

    for x in range(1, nx - 1):
        gradient_y[x, nx + x] = 2 * a_norm * vy[0, x] / spacing[0]
        gradient_y[x, x] = -2 * a_norm * vy[0, x] / spacing[0]
        gradient_y[x, nx + x + 1] = 2 * b_norm * vy[0, x] / spacing[0]
        gradient_y[x, x + 1] = -2 * b_norm * vy[0, x] / spacing[0]
        gradient_y[x, nx + x - 1] = 2 * b_norm * vy[0, x] / spacing[0]
        gradient_y[x, x - 1] = -2 * b_norm * vy[0, x] / spacing[0]

    for x in range(1, nx - 1):
        gradient_y[(ny - 1) * nx + x, (ny - 1) * nx + x] = 2 * a_norm * vy[ny - 1, x] / spacing[0]
        gradient_y[(ny - 1) * nx + x, (ny - 2) * nx + x] = -2 * a_norm * vy[ny - 1, x] / spacing[0]
        gradient_y[(ny - 1) * nx + x, (ny - 1) * nx + x + 1] = 2 * b_norm * vy[ny - 1, x] / spacing[0]
        gradient_y[(ny - 1) * nx + x, (ny - 2) * nx + x + 1] = -2 * b_norm * vy[ny - 1, x] / spacing[0]
        gradient_y[(ny - 1) * nx + x, (ny - 1) * nx + x - 1] = 2 * b_norm * vy[ny - 1, x] / spacing[0]
        gradient_y[(ny - 1) * nx + x, (ny - 2) * nx + x - 1] = -2 * b_norm * vy[ny - 1, x] / spacing[0]

    for y in range(1, ny - 1):
        gradient_y[y * nx + nx - 1, (y + 1) * nx + nx - 1] = a_edge * vy[y, nx - 1] / spacing[0]
        gradient_y[y * nx + nx - 1, (y - 1) * nx + nx - 1] = -a_edge * vy[y, nx - 1] / spacing[0]
        gradient_y[y * nx + nx - 1, (y + 1) * nx + nx - 2] = b_edge * vy[y, nx - 1] / spacing[0]
        gradient_y[y * nx + nx - 1, (y - 1) * nx + nx - 2] = -b_edge * vy[y, nx - 1] / spacing[0]

    for y in range(1, ny - 1):
        for x in range(1, nx - 1):
            gradient_y[y * nx + x, (y + 1) * nx + x] = a_norm * vy[y, x] / spacing[0]
            gradient_y[y * nx + x, (y - 1) * nx + x] = -a_norm * vy[y, x] / spacing[0]
            gradient_y[y * nx + x, (y + 1) * nx + x + 1] = b_norm * vy[y, x] / spacing[0]
            gradient_y[y * nx + x, (y - 1) * nx + x + 1] = -b_norm * vy[y, x] / spacing[0]
            gradient_y[y * nx + x, (y + 1) * nx + x - 1] = b_norm * vy[y, x] / spacing[0]
            gradient_y[y * nx + x, (y - 1) * nx + x - 1] = -b_norm * vy[y, x] / spacing[0]

    return gradient_x.tocsr() + gradient_y.tocsr()
