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
