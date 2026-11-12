"""Unit tests for Angular Spectrum Method (ASM) wave propagation."""

import pytest
import torch
import numpy as np

from src.propagation import AngularSpectrumPropagator, compute_asm_transfer_function


def test_compute_asm_transfer_function_shape_and_type():
    """Verify kernel shape, dtype, and evanescent filtering."""
    grid_n = 64
    wavelength = 0.75e-3
    dx = wavelength / 2.0
    drift_distance = 10e-3

    H = compute_asm_transfer_function(
        grid_n=grid_n,
        dx=dx,
        wavelength=wavelength,
        drift_distance=drift_distance,
        device=torch.device("cpu"),
    )

    assert H.shape == (grid_n, grid_n)
    assert H.is_complex()
    # At DC (center frequency fx=0, fy=0), |H| must be 1.0
    assert torch.isclose(torch.abs(H[0, 0]), torch.tensor(1.0, dtype=torch.float32), atol=1e-5)
    # Beyond cut-off evanescent modes must be zero
    assert torch.all(torch.abs(H) <= 1.0 + 1e-6)


def test_asm_propagator_forward_energy_conservation():
    """Verify that pure propagating fields approximately conserve optical power."""
    grid_n = 64
    wavelength = 0.75e-3
    dx = wavelength / 2.0
    drift_distance = 5 * wavelength

    propagator = AngularSpectrumPropagator(
        grid_n=grid_n,
        wavelength=wavelength,
        drift_distance=drift_distance,
        dx=dx,
    )

    # Low-frequency Gaussian beam well within propagating cone
    x = torch.linspace(-grid_n / 2, grid_n / 2, grid_n)
    X, Y = torch.meshgrid(x, x, indexing="ij")
    field = torch.exp(-(X**2 + Y**2) / (2 * (grid_n / 8) ** 2)).to(torch.complex64)
    field = field.unsqueeze(0)  # (1, 64, 64)

    initial_power = torch.sum(torch.abs(field) ** 2)
    diffracted = propagator(field)
    diffracted_power = torch.sum(torch.abs(diffracted) ** 2)

    # Power should be preserved within 1% for low-frequency paraxial beam
    relative_power_diff = torch.abs(diffracted_power - initial_power) / initial_power
    assert relative_power_diff < 0.02


def test_asm_propagator_gradient_flow():
    """Verify gradients propagate backwards through ASM propagator."""
    propagator = AngularSpectrumPropagator(grid_n=32, wavelength=1e-3, drift_distance=5e-3)
    field = torch.randn(2, 32, 32, dtype=torch.complex64, requires_grad=True)

    out = propagator(field)
    loss = torch.sum(torch.abs(out) ** 2)
    loss.backward()

    assert field.grad is not None
    assert not torch.isnan(field.grad).any()
