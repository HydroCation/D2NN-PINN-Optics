"""
Diffractive Deep Neural Network (D2NN) optical neural network models.

Implements multi-layer physical phase-only diffractive networks interconnected
by free-space Angular Spectrum Method (ASM) propagation kernels.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from .propagation import AngularSpectrumPropagator


class DiffractiveLayer(nn.Module):
    """Single diffractive phase mask layer."""

    def __init__(
        self,
        grid_n: int,
        phase_init: str = "uniform",
        init_range: Tuple[float, float] = (-np.pi / 2.0, np.pi / 2.0),
    ) -> None:
        """Initialize a diffractive phase modulation plate.

        Args:
            grid_n: Dimension of square phase grid (grid_n x grid_n).
            phase_init: Initialization strategy ('uniform', 'zeros', or 'normal').
            init_range: (min, max) phase in radians for uniform initialization.
        """
        super().__init__()
        self.grid_n = grid_n

        if phase_init == "uniform":
            low, high = init_range
            init_tensor = (torch.rand(grid_n, grid_n) * (high - low)) + low
        elif phase_init == "zeros":
            init_tensor = torch.zeros(grid_n, grid_n)
        elif phase_init == "normal":
            init_tensor = torch.randn(grid_n, grid_n) * 0.1
        else:
            raise ValueError(f"Unknown phase_init: {phase_init}")

        self.phase = nn.Parameter(init_tensor)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Apply complex phase transmission: T = exp(j * phi)."""
        transmission = torch.exp(1j * self.phase)
        return u * transmission


class OpticalD2NN(nn.Module):
    """Phase-only Diffractive Deep Neural Network (D2NN).

    Consists of K cascaded phase-modulation layers spaced by drift distance d,
    with an initial drift from input plane to Layer 1, and a final drift from
    Layer K to the detector output plane.
    """

    def __init__(
        self,
        grid_n: int = 128,
        fov_size: int = 8,
        num_layers: int = 8,
        wavelength: float = 0.75e-3,
        drift_distance: float = 14 * 0.75e-3,
        dx: Optional[float] = None,
        use_aperture: bool = False,
        aperture_radius_ratio: float = 0.45,
        aperture_order: int = 8,
        phase_init: str = "uniform",
    ) -> None:
        """Initialize the D2NN architecture.

        Args:
            grid_n: Computational window size (grid_n x grid_n).
            fov_size: Centered Field of View dimension (fov_size x fov_size).
            num_layers: Number of cascaded diffractive phase plates K.
            wavelength: Operating optical wavelength in meters.
            drift_distance: Inter-layer drift distance d in meters.
            dx: Spatial sampling pitch in meters (default wavelength / 2).
            use_aperture: Whether to enforce radial aperture beam confinement.
            aperture_radius_ratio: Normalized aperture cutoff radius [0, 1].
            aperture_order: Exponent in super-Gaussian spatial mask.
            phase_init: Phase initialization scheme ('uniform', 'zeros', etc.).
        """
        super().__init__()
        self.grid_n = grid_n
        self.fov_size = fov_size
        self.num_layers = num_layers
        self.wavelength = wavelength
        self.drift_distance = drift_distance
        self.dx = dx if dx is not None else (wavelength / 2.0)
        self.use_aperture = use_aperture

        # Centered Field of View bounds
        self.fov_start = (self.grid_n - self.fov_size) // 2
        self.fov_end = self.fov_start + self.fov_size

        # Angular Spectrum Wave Propagator
        self.propagator = AngularSpectrumPropagator(
            grid_n=self.grid_n,
            wavelength=self.wavelength,
            drift_distance=self.drift_distance,
            dx=self.dx,
        )

        # Cascaded Phase Plates
        self.phases = nn.ParameterList([
            nn.Parameter((torch.rand(self.grid_n, self.grid_n) - 0.5) * np.pi)
            if phase_init == "uniform"
            else nn.Parameter(torch.zeros(self.grid_n, self.grid_n))
            for _ in range(num_layers)
        ])

        # Optional radial super-Gaussian aperture confinement mask
        if use_aperture:
            fx = torch.fft.fftfreq(self.grid_n, d=self.dx)
            fy = torch.fft.fftfreq(self.grid_n, d=self.dx)
            FY, FX = torch.meshgrid(fy, fx, indexing="ij")
            r_grid = torch.sqrt(FX**2 + FY**2)
            r_norm = r_grid / (r_grid.max() + 1e-12)
            mask = torch.exp(-((r_norm / aperture_radius_ratio) ** aperture_order))
            self.register_buffer("spatial_mask", mask.to(torch.float32))
        else:
            self.spatial_mask = None

    def embed_fov(self, small_patch: torch.Tensor) -> torch.Tensor:
        """Embed an (..., fov_size, fov_size) patch into centered (..., grid_n, grid_n) field.

        Args:
            small_patch: Complex or real tensor with trailing dims (fov_size, fov_size).

        Returns:
            Zero-padded full field tensor with trailing dims (grid_n, grid_n).
        """
        shape = list(small_patch.shape[:-2]) + [self.grid_n, self.grid_n]
        full_field = torch.zeros(
            shape, dtype=torch.complex64, device=small_patch.device
        )
        full_field[..., self.fov_start : self.fov_end, self.fov_start : self.fov_end] = (
            small_patch.to(torch.complex64)
        )
        return full_field

    def extract_fov(self, full_field: torch.Tensor) -> torch.Tensor:
        """Extract the centered (..., fov_size, fov_size) patch from (..., grid_n, grid_n).

        Args:
            full_field: Complex field tensor with trailing dims (grid_n, grid_n).

        Returns:
            Extracted sub-field tensor with trailing dims (fov_size, fov_size).
        """
        return full_field[
            ..., self.fov_start : self.fov_end, self.fov_start : self.fov_end
        ]

    def forward(self, u_in: torch.Tensor) -> torch.Tensor:
        """Forward optical propagation through the cascaded D2NN.

        Args:
            u_in: Input full complex field tensor (..., grid_n, grid_n).

        Returns:
            Output full complex field tensor at the detector plane.
        """
        u = u_in
        for phi in self.phases:
            u = self.propagator(u)
            u = u * torch.exp(1j * phi)
            if self.spatial_mask is not None:
                u = u * self.spatial_mask
        # Final drift to detector plane
        u = self.propagator(u)
        return u

    def forward_fov(self, small_patch: torch.Tensor) -> torch.Tensor:
        """End-to-end forward pass: embed FOV -> propagate -> extract FOV."""
        u_in = self.embed_fov(small_patch)
        u_out = self.forward(u_in)
        return self.extract_fov(u_out)

    def get_phase_tensors(self) -> Dict[str, torch.Tensor]:
        """Return dictionary of cloned phase tensors."""
        return {f"layer_{i}": p.detach().clone() for i, p in enumerate(self.phases)}
