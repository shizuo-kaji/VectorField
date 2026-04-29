# VectorField

`vectorfield` is a small Python package for analyzing, visualizing, decomposing,
and interpolating two-dimensional vector fields.

## Features

- Build gradient and Hamiltonian vector fields from scalar potentials.
- Compute divergence and curl.
- Visualize scalar fields and vector fields with Matplotlib.
- Run FFT-based Helmholtz-Hodge decomposition.
- Interpolate sparse vector observations with radial basis functions.
- Build infinitesimal Koopman generator matrices.
- Fit Hamiltonian neural networks when the optional HNN dependencies are installed.
- Convert matched point clouds into a vector field through discrete optimal assignment.

## Project Layout

- `src/vectorfield/core.py`: `VectorField2D` and core differential operators.
- `src/vectorfield/decomposition.py`: FFT-based Helmholtz-Hodge decomposition.
- `src/vectorfield/interpolation.py`: `RBFVectorFieldInterpolator`.
- `src/vectorfield/plotting.py`: Matplotlib plotting helpers.
- `src/vectorfield/koopman.py`: Koopman generator builders.
- `src/vectorfield/transport.py`: point-cloud assignment helpers.
- `src/vectorfield/hnn.py`: optional Hamiltonian neural network utilities.
- `vfield.ipynb`: demo notebook using the current API.

## Installation

Install the core package in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install optional HNN dependencies when you need the neural-network examples:

```bash
pip install -e ".[hnn]"
```

## Quick Start

```python
import numpy as np

from vectorfield import VectorField2D, plot_scalar_field, plot_vector_field

nx, ny = 120, 100
extent = 2.0
x, y = np.meshgrid(np.linspace(-extent, extent, nx), np.linspace(-extent, extent, ny))

potential = x**2 + y**2
gradient = VectorField2D.from_potential(potential, kind="gradient")
hamiltonian = VectorField2D.from_potential(potential, kind="hamiltonian")

plot_scalar_field(potential, x=x, y=y, title="Potential")
plot_vector_field(gradient, x=x, y=y, title="Gradient field")
plot_vector_field(hamiltonian, x=x, y=y, title="Hamiltonian field")
```

## Notebook

The demo notebook is `vfield.ipynb`. It uses the current `vectorfield` API and
does not depend on the removed `vfield_util` package.

## License

MIT
