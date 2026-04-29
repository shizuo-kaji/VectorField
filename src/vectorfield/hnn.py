from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import cv2
import numpy as np
import numpy.typing as npt
import torch
from scipy.ndimage import gaussian_filter, sobel
from torch import nn

Array2D = npt.NDArray[np.float64]


@dataclass(slots=True)
class BoundaryNormals:
    indices: npt.NDArray[np.int32]
    normals: Array2D

    def coordinates(self, x_grid: npt.ArrayLike, y_grid: npt.ArrayLike) -> Array2D:
        x_values = np.asarray(x_grid, dtype=float)
        y_values = np.asarray(y_grid, dtype=float)
        return np.column_stack(
            (
                x_values[self.indices[:, 1], self.indices[:, 0]],
                y_values[self.indices[:, 1], self.indices[:, 0]],
            )
        )


def estimate_boundary_normals(mask: npt.ArrayLike, *, sigma: float = 5.0) -> BoundaryNormals:
    binary_mask = np.asarray(mask, dtype=np.uint8)
    contour_result = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = contour_result[0] if len(contour_result) == 2 else contour_result[1]

    blurred = gaussian_filter(binary_mask.astype(float), sigma=sigma)
    grad_x = sobel(blurred, axis=1, mode="constant")
    grad_y = sobel(blurred, axis=0, mode="constant")
    magnitude = np.hypot(grad_x, grad_y)
    grad_x /= np.maximum(magnitude, 1e-10)
    grad_y /= np.maximum(magnitude, 1e-10)

    if not contours:
        return BoundaryNormals(
            indices=np.zeros((0, 2), dtype=np.int32),
            normals=np.zeros((0, 2), dtype=float),
        )

    points = contours[0].reshape(-1, 2)
    valid = (
        (points[:, 0] > 0)
        & (points[:, 0] < binary_mask.shape[1] - 1)
        & (points[:, 1] > 0)
        & (points[:, 1] < binary_mask.shape[0] - 1)
    )
    indices = points[valid].astype(np.int32)
    normals = np.column_stack((grad_x[indices[:, 1], indices[:, 0]], grad_y[indices[:, 1], indices[:, 0]]))
    return BoundaryNormals(indices=indices, normals=normals)


class HamiltonianNeuralNetwork(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 2,
        hidden_dims: Sequence[int] = (128, 128),
        output_dim: int = 1,
        grid_x: npt.ArrayLike | None = None,
        grid_y: npt.ArrayLike | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layer = nn.Linear(last_dim, hidden_dim)
            nn.init.orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)
            layers.extend((layer, nn.Tanh()))
            last_dim = hidden_dim

        output_layer = nn.Linear(last_dim, output_dim, bias=False)
        nn.init.orthogonal_(output_layer.weight)
        layers.append(output_layer)

        self.network = nn.Sequential(*layers)
        self.register_buffer("symplectic_matrix", torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32))
        self.register_buffer(
            "grid_x",
            self._as_optional_grid_tensor(grid_x),
            persistent=False,
        )
        self.register_buffer(
            "grid_y",
            self._as_optional_grid_tensor(grid_y),
            persistent=False,
        )

    @staticmethod
    def _as_optional_grid_tensor(values: npt.ArrayLike | None) -> torch.Tensor | None:
        if values is None:
            return None
        return torch.as_tensor(np.asarray(values, dtype=np.float32), dtype=torch.float32)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.network(points)

    def vfield(self, points: torch.Tensor | None = None) -> torch.Tensor:
        if points is None:
            if self.grid_x is None or self.grid_y is None:
                raise ValueError("grid_x and grid_y must be set when calling vfield() without points.")
            points = torch.stack((self.grid_x, self.grid_y), dim=-1)
        if not points.requires_grad:
            points = points.clone().detach().requires_grad_(True)
        hamiltonian = self.forward(points)
        gradient = torch.autograd.grad(hamiltonian.sum(), points, create_graph=True)[0]
        return gradient @ self.symplectic_matrix

    def vector_field(self, points: torch.Tensor) -> torch.Tensor:
        return self.vfield(points)

    def hamiltonian(self) -> torch.Tensor:
        if self.grid_x is None or self.grid_y is None:
            raise ValueError("grid_x and grid_y must be set before calling hamiltonian().")
        points = torch.stack((self.grid_x, self.grid_y), dim=-1)
        return self.forward(points).reshape(self.grid_x.shape)

    @torch.no_grad()
    def predict_hamiltonian(self, points: npt.ArrayLike, *, device: torch.device | str | None = None) -> Array2D:
        tensor_points = torch.as_tensor(np.asarray(points, dtype=np.float32), device=device)
        return self.forward(tensor_points).cpu().numpy()

    def predict_vector_field(self, points: npt.ArrayLike, *, device: torch.device | str | None = None) -> Array2D:
        tensor_points = torch.as_tensor(np.asarray(points, dtype=np.float32), device=device).requires_grad_(True)
        return self.vfield(tensor_points).detach().cpu().numpy()


def fit_hamiltonian_network(
    points: npt.ArrayLike,
    vectors: npt.ArrayLike,
    *,
    boundary_points: npt.ArrayLike | None = None,
    boundary_normals: npt.ArrayLike | None = None,
    model: HamiltonianNeuralNetwork | None = None,
    epochs: int = 10_000,
    batch_size: int | None = None,
    learning_rate: float = 1e-3,
    boundary_weight: float = 1.0,
    topology_weight: float = 0.0,
    topology_loss_fn: Callable[[HamiltonianNeuralNetwork, torch.Tensor], torch.Tensor] | None = None,
    hidden_dims: Sequence[int] = (128, 128),
    grid_x: npt.ArrayLike | None = None,
    grid_y: npt.ArrayLike | None = None,
    device: torch.device | str | None = None,
    verbose: bool = True,
) -> HamiltonianNeuralNetwork:
    sample_points = np.asarray(points, dtype=np.float32)
    sample_vectors = np.asarray(vectors, dtype=np.float32)
    if sample_points.ndim != 2 or sample_points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2).")
    if sample_vectors.shape != sample_points.shape:
        raise ValueError("vectors must have shape (n, 2).")

    runtime_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    point_tensor = torch.as_tensor(sample_points, device=runtime_device, dtype=torch.float32).requires_grad_(True)
    vector_tensor = torch.as_tensor(sample_vectors, device=runtime_device)

    if boundary_points is not None and boundary_normals is not None:
        boundary_point_tensor = torch.as_tensor(
            np.asarray(boundary_points, dtype=np.float32),
            device=runtime_device,
            dtype=torch.float32,
        ).requires_grad_(True)
        boundary_normal_tensor = torch.as_tensor(np.asarray(boundary_normals, dtype=np.float32), device=runtime_device)
    else:
        boundary_point_tensor = None
        boundary_normal_tensor = None

    if model is None:
        model = HamiltonianNeuralNetwork(hidden_dims=hidden_dims, grid_x=grid_x, grid_y=grid_y)
    model = model.to(runtime_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    loss_function = nn.MSELoss(reduction="mean").to(runtime_device)

    if batch_size is None:
        batch_size = len(sample_points)

    for step in range(epochs):
        indices = torch.randperm(point_tensor.shape[0], device=runtime_device)[:batch_size]
        predicted_vectors = model.vfield(point_tensor[indices])
        fitting_loss = loss_function(vector_tensor[indices], predicted_vectors)
        loss = fitting_loss

        boundary_loss = torch.tensor(0.0, device=runtime_device)
        if boundary_point_tensor is not None and boundary_normal_tensor is not None and boundary_weight > 0:
            boundary_vectors = model.vfield(boundary_point_tensor)
            boundary_loss = (torch.sum(boundary_vectors * boundary_normal_tensor, dim=1) ** 2).mean()
            loss = loss + boundary_weight * boundary_loss

        topology_loss = torch.tensor(0.0, device=runtime_device)
        if topology_weight > 0:
            if topology_loss_fn is None:
                raise ValueError("topology_loss_fn must be provided when topology_weight > 0.")
            topology_loss = topology_loss_fn(model, point_tensor)
            loss = loss + topology_weight * topology_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.detach())

        if verbose and step % max(1, epochs // 10) == 0:
            print(
                "iter "
                f"{step}, total loss {loss.item():.6f}, MSE {fitting_loss.item():.6f}, "
                f"boundary loss {boundary_loss.item():.6f}, topology loss {topology_loss.item():.6f}"
            )

    return model.eval()


HNN = HamiltonianNeuralNetwork
fit_HNN = fit_hamiltonian_network
