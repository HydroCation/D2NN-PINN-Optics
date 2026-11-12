"""
Angular Spectrum Method (ASM) wave propagation module.

Provides physical scalar diffraction modeling using the Rayleigh-Sommerfeld
formulation via Fourier-domain angular spectrum propagation.
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


def compute_asm_transfer_function(
    grid_n: int,
    dx: float,
    wavelength: float,
    drift_distance: float,
    device: Optional[Union[torch.device, str]] = None,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """Compute the complex transfer function H(fx, fy) for the Angular Spectrum Method.

    Args:
        grid_n: Number of sampling points along each dimension (NxN).
        dx: Spatial sampling pitch in meters (typically wavelength / 2).
        wavelength: Free-space wavelength in meters.
        drift_distance: Propagation distance d in meters along the optical axis.
        device: Device to place the transfer kernel on.
        dtype: Complex tensor data type (default torch.complex64).

    Returns:
        torch.Tensor of shape (grid_n, grid_n) representing H(fx, fy).
    """
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    k0 = 2.0 * np.pi / wavelength

    # Spatial frequency coordinates (cycles per meter)
    fx = torch.fft.fftfreq(grid_n, d=dx, device=device)
    fy = torch.fft.fftfreq(grid_n, d=dx, device=device)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")

    # Transverse wavevectors (rad/m)
    kx = 2.0 * np.pi * FX
    ky = 2.0 * np.pi * FY

    # Longitudinal wavevector component squared: kz^2 = k0^2 - kx^2 - ky^2
    kz_sq = k0**2 - kx**2 - ky**2

    # Propagating modes (kz_sq > 0); evanescent modes (kz_sq <= 0) are filtered
    prop_mask = kz_sq > 0
    kz = torch.zeros_like(kz_sq)
    kz[prop_mask] = torch.sqrt(kz_sq[prop_mask])

    # Transfer function kernel
    h_phase = torch.exp(1j * kz * drift_distance)
    h_kernel = torch.where(
        prop_mask,
        h_phase.to(dtype),
        torch.zeros_like(kz_sq, dtype=dtype),
    )
    return h_kernel


class AngularSpectrumPropagator(nn.Module):
    """Angular Spectrum Method (ASM) wave propagation layer.

    Propagates a 2D complex optical field forward by distance d in homogeneous free space.
    The transfer function is precalculated and registered as a persistent buffer.
    """

    def __init__(
        self,
        grid_n: int = 128,
        wavelength: float = 0.75e-3,
        drift_distance: float = 14 * 0.75e-3,
        dx: Optional[float] = None,
    ) -> None:
        """Initialize the ASM propagator.

        Args:
            grid_n: Number of grid points along each spatial axis (grid_n x grid_n).
            wavelength: Optical wavelength in meters.
            drift_distance: Propagation distance d in meters.
            dx: Spatial sampling pitch. Defaults to wavelength / 2.0 (Nyquist rate).
        """
        super().__init__()
        self.grid_n = grid_n
        self.wavelength = wavelength
        self.drift_distance = drift_distance
        self.dx = dx if dx is not None else (wavelength / 2.0)

        # Precompute initial transfer function on CPU and register as buffer
        kernel = compute_asm_transfer_function(
            grid_n=self.grid_n,
            dx=self.dx,
            wavelength=self.wavelength,
            drift_distance=self.drift_distance,
            device=torch.device("cpu"),
        )
        self.register_buffer("H_kernel", kernel)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        """Forward wave propagation through distance d.

        Args:
            field: Complex optical wave field tensor with shape (... , grid_n, grid_n).

        Returns:
            Diffracted complex optical field with identical shape.
        """
        if not field.is_complex():
            raise TypeError(f"Expected complex tensor input, got {field.dtype}")

        # Fourier transform to angular spectrum
        u_fft = torch.fft.fft2(field)
        # Spectrum multiplication by transfer function
        u_prop_fft = u_fft * self.H_kernel
        # Inverse Fourier transform back to spatial domain
        return torch.fft.ifft2(u_prop_fft)

    def extra_repr(self) -> str:
        return (
            f"grid_n={self.grid_n}, wavelength={self.wavelength:.3e}m, "
            f"drift_distance={self.drift_distance:.3e}m, dx={self.dx:.3e}m"
        )
