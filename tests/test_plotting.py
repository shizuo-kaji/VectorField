import numpy as np
from matplotlib import pyplot as plt

from vectorfield import VectorField2D, plot_scalar_field, plot_vector_field


def test_plot_scalar_field_returns_axis_and_sets_title() -> None:
    fig, ax = plt.subplots()
    field = np.arange(12.0).reshape(3, 4)

    returned_ax = plot_scalar_field(field, ax=ax, title="Scalar")

    assert returned_ax is ax
    assert ax.get_title() == "Scalar"
    plt.close(fig)


def test_plot_vector_field_returns_axis_for_streamplot() -> None:
    fig, ax = plt.subplots()
    x_grid, y_grid = np.meshgrid(np.linspace(-1.0, 1.0, 6), np.linspace(-1.0, 1.0, 5))
    vector = VectorField2D.from_potential(x_grid**2 + y_grid**2, kind="gradient")

    returned_ax = plot_vector_field(vector, x=x_grid, y=y_grid, ax=ax, title="Vector")

    assert returned_ax is ax
    assert ax.get_title() == "Vector"
    plt.close(fig)


def test_plot_vector_field_accepts_square_coordinate_grids() -> None:
    fig, ax = plt.subplots()
    x_grid, y_grid = np.meshgrid(np.linspace(-1.0, 1.0, 6), np.linspace(-1.0, 1.0, 6))
    vector = VectorField2D.from_potential(x_grid**2 + y_grid**2, kind="gradient")

    returned_ax = plot_vector_field(vector, x=x_grid, y=y_grid, ax=ax, title="Square grid")

    assert returned_ax is ax
    assert ax.get_title() == "Square grid"
    plt.close(fig)


def test_plot_vector_field_returns_axis_for_quiver() -> None:
    fig, ax = plt.subplots()
    vector = VectorField2D(np.ones((30, 30)), np.zeros((30, 30)))

    returned_ax = plot_vector_field(vector, ax=ax, quiver=True)

    assert returned_ax is ax
    plt.close(fig)
