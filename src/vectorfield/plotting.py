from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .core import VectorField2D, coerce_vector_field


def _axis_coordinates(values: np.ndarray | None, size: int, *, axis: str) -> np.ndarray:
    if values is None:
        return np.arange(size)
    array = np.asarray(values)
    if array.ndim == 1:
        return array
    if array.ndim == 2:
        if axis == "x" and array.shape[1] == size:
            return array[0]
        if axis == "y" and array.shape[0] == size:
            return array[:, 0]
        if array.shape[1] == size:
            return array[0]
        if array.shape[0] == size:
            return array[:, 0]
        raise ValueError("2D coordinate grids must match the field shape.")
    raise ValueError("Coordinates must be 1D or 2D arrays.")


def plot_scalar_field(
    field: np.ndarray,
    *,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    levels: int | list[float] = 3,
    title: str | None = None,
    norm=None,
    cmap: str = "coolwarm",
) -> plt.Axes:
    scalar = np.asarray(field, dtype=float)
    if scalar.ndim != 2:
        raise ValueError("field must be a 2D array.")

    plot_axes = ax or plt.subplots(1, 1)[1]
    x_coords = _axis_coordinates(x, scalar.shape[1], axis="x")
    y_coords = _axis_coordinates(y, scalar.shape[0], axis="y")

    image = plot_axes.imshow(scalar[::-1, :], cmap=cmap, norm=norm)
    plot_axes.contour(x_coords, y_coords, scalar[::-1, :], levels=levels, colors="k")
    if title is not None:
        plot_axes.set_title(title)

    divider = make_axes_locatable(plot_axes)
    color_axes = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(image, cax=color_axes)
    return plot_axes


def plot_vector_field(
    field: VectorField2D | np.ndarray,
    *,
    x: np.ndarray | None = None,
    y: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    title: str | None = None,
    quiver: bool = False,
    norm=None,
    density: tuple[float, float] = (0.5, 1.0),
    cmap: str = "coolwarm",
) -> plt.Axes:
    vector_field = coerce_vector_field(field)
    plot_axes = ax or plt.subplots(1, 1)[1]
    magnitude = vector_field.magnitude

    x_coords = _axis_coordinates(x, vector_field.shape[1], axis="x")
    y_coords = _axis_coordinates(y, vector_field.shape[0], axis="y")

    if quiver:
        stride = 20
        artist = plot_axes.quiver(
            x_coords[::stride],
            y_coords[::stride],
            vector_field.x[::stride, ::stride],
            vector_field.y[::stride, ::stride],
            magnitude[::stride, ::stride],
            pivot="tail",
            cmap=cmap,
            scale=350,
            scale_units="width",
            width=0.005,
        )
        color_source = artist
    else:
        stream = plot_axes.streamplot(
            x_coords,
            y_coords,
            vector_field.x,
            vector_field.y,
            color=magnitude,
            linewidth=2,
            cmap=cmap,
            norm=norm,
            density=density,
        )
        color_source = stream.lines

    plot_axes.set(
        xlim=(np.min(x_coords), np.max(x_coords)),
        ylim=(np.min(y_coords), np.max(y_coords)),
    )
    plot_axes.set_aspect(1)
    plot_axes.axis("off")
    plot_axes.set_title(title or f"min {magnitude.min():.4e}, max {magnitude.max():.4e}")

    divider = make_axes_locatable(plot_axes)
    color_axes = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(color_source, cax=color_axes)
    return plot_axes
