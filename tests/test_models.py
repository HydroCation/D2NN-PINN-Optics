"""Unit tests for Optical D2NN models."""

import pytest
import torch
import numpy as np

from src.models import OpticalD2NN, DiffractiveLayer


def test_diffractive_layer_phase():
    """Verify single diffractive layer phase modulation."""
    layer = DiffractiveLayer(grid_n=32, phase_init="uniform")
    u_in = torch.ones(1, 32, 32, dtype=torch.complex64)
    u_out = layer(u_in)

    # Pure phase modulation should not change magnitude of unit input
    assert torch.allclose(torch.abs(u_out), torch.ones_like(torch.abs(u_out)), atol=1e-5)


def test_optical_d2nn_forward_and_fov_embedding():
    """Verify FOV embedding, extraction, and end-to-end forward pass."""
    grid_n = 64
    fov_size = 8
    num_layers = 4
    model = OpticalD2NN(grid_n=grid_n, fov_size=fov_size, num_layers=num_layers)

    # Test embed and extract roundtrip on isolated patch
    patch = torch.randn(2, fov_size, fov_size, dtype=torch.complex64)
    full = model.embed_fov(patch)
    assert full.shape == (2, grid_n, grid_n)

    extracted = model.extract_fov(full)
    assert extracted.shape == (2, fov_size, fov_size)
    assert torch.allclose(extracted, patch)

    # Test forward_fov
    out_fov = model.forward_fov(patch)
    assert out_fov.shape == (2, fov_size, fov_size)
    assert out_fov.is_complex()


def test_optical_d2nn_parameter_gradients():
    """Verify backward gradient flow to all layer phase parameters."""
    model = OpticalD2NN(grid_n=32, fov_size=4, num_layers=3)
    inputs = torch.randn(2, 4, 4, dtype=torch.complex64)

    out = model.forward_fov(inputs)
    loss = torch.sum(torch.abs(out) ** 2)
    loss.backward()

    for i, phase in enumerate(model.phases):
        assert phase.grad is not None, f"Phase {i} missing gradient"
        assert not torch.isnan(phase.grad).any(), f"Phase {i} has NaN gradients"


def test_optical_d2nn_complex_modulation():
    """Verify OpticalD2NN with complex modulation (trainable amplitude + phase)."""
    grid_n = 32
    fov_size = 4
    num_layers = 3
    model = OpticalD2NN(
        grid_n=grid_n,
        fov_size=fov_size,
        num_layers=num_layers,
        modulation="complex",
    )

    # 1. Parameter structure and initial amplitude boundedness in (0, 1)
    assert hasattr(model, "raw_amplitudes")
    assert model.raw_amplitudes is not None
    assert len(model.raw_amplitudes) == num_layers
    assert len(model.phases) == num_layers

    # Initial values should be 2.0, sigmoid(2.0) ≈ 0.8808, strictly in (0, 1)
    expected_sigmoid_2 = torch.sigmoid(torch.tensor(2.0))
    for raw_amp in model.raw_amplitudes:
        amp = torch.sigmoid(raw_amp)
        assert torch.all(amp > 0.0)
        assert torch.all(amp < 1.0)
        assert torch.allclose(amp, expected_sigmoid_2, atol=1e-5)

    # 2. Forward pass execution
    inputs = torch.randn(2, fov_size, fov_size, dtype=torch.complex64)
    out = model.forward_fov(inputs)
    assert out.shape == (2, fov_size, fov_size)
    assert out.is_complex()

    # 3. Backward gradient flow to both raw_amplitudes and phases
    loss = torch.sum(torch.abs(out) ** 2)
    loss.backward()

    for idx, (phase, raw_amp) in enumerate(zip(model.phases, model.raw_amplitudes)):
        assert phase.grad is not None, f"Phase layer {idx} missing gradient"
        assert not torch.isnan(phase.grad).any(), f"Phase layer {idx} has NaN gradient"
        assert torch.any(phase.grad != 0.0), f"Phase layer {idx} has zero gradient"

        assert raw_amp.grad is not None, f"Raw amplitude layer {idx} missing gradient"
        assert not torch.isnan(raw_amp.grad).any(), f"Raw amplitude layer {idx} has NaN gradient"
        assert torch.any(raw_amp.grad != 0.0), f"Raw amplitude layer {idx} has zero gradient"

    # 4. Verify DiffractiveLayer complex modulation unit behavior
    diff_layer = DiffractiveLayer(grid_n=32, modulation="complex")
    u_in = torch.ones(1, 32, 32, dtype=torch.complex64)
    u_out = diff_layer(u_in)
    assert torch.allclose(torch.abs(u_out), torch.full_like(torch.abs(u_out), expected_sigmoid_2), atol=1e-5)


def test_optical_d2nn_apertured_modulation():
    """Verify OpticalD2NN with apertured phase-only modulation (super-Gaussian spatial mask)."""
    grid_n = 32
    fov_size = 4
    num_layers = 3
    aperture_radius_ratio = 0.45
    aperture_order = 4

    model = OpticalD2NN(
        grid_n=grid_n,
        fov_size=fov_size,
        num_layers=num_layers,
        modulation="apertured",
        aperture_radius_ratio=aperture_radius_ratio,
        aperture_order=aperture_order,
    )

    # 1. Verification of spatial aperture buffer presence and shape
    assert hasattr(model, "spatial_aperture")
    assert model.spatial_aperture is not None
    assert model.spatial_aperture.shape == (grid_n, grid_n)

    # 2. Verification of aperture flat-top transmission profile:
    # Center pixel (grid_n // 2, grid_n // 2) is unattenuated (> 0.99)
    # Border / corner pixel (0, 0) is heavily suppressed (< 0.01)
    center_idx = grid_n // 2
    assert model.spatial_aperture[center_idx, center_idx].item() > 0.99
    assert model.spatial_aperture[0, 0].item() < 0.01

    # 3. Forward pass execution
    inputs = torch.randn(2, fov_size, fov_size, dtype=torch.complex64)
    out = model.forward_fov(inputs)
    assert out.shape == (2, fov_size, fov_size)
    assert out.is_complex()

    # 4. Backward gradient flow to phase parameters
    loss = torch.sum(torch.abs(out) ** 2)
    loss.backward()

    for idx, phase in enumerate(model.phases):
        assert phase.grad is not None, f"Phase layer {idx} missing gradient"
        assert not torch.isnan(phase.grad).any(), f"Phase layer {idx} has NaN gradient"
        assert torch.any(phase.grad != 0.0), f"Phase layer {idx} has zero gradient"

    # 5. DiffractiveLayer modulation validation
    diff_layer = DiffractiveLayer(grid_n=32, modulation="apertured")
    u_in = torch.ones(1, 32, 32, dtype=torch.complex64)
    u_out = diff_layer(u_in)
    assert torch.allclose(torch.abs(u_out), torch.ones_like(torch.abs(u_out)), atol=1e-5)

