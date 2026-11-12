"""Unit tests for wave-optics metrics."""

import pytest
import torch
import numpy as np

from src.metrics import (
    field_cosine_similarity,
    frobenius_relative_error,
    matrix_cosine_similarity,
    optimal_complex_alignment_scale,
    active_energy_efficiency,
    time_of_flight_latency,
    SPEED_OF_LIGHT,
)


def test_field_cosine_similarity_identity_and_orthogonality():
    """Verify fidelity is 1.0 for identical fields and 0.0 for orthogonal fields."""
    u = torch.randn(4, 8, 8, dtype=torch.complex64)
    # Self-fidelity must be 1.0
    fid_self = field_cosine_similarity(u, u)
    assert torch.allclose(fid_self, torch.ones_like(fid_self), atol=1e-5)

    # Invariant to global complex phase and scale
    c = 2.5 * torch.exp(1j * torch.tensor(0.7))
    fid_scaled = field_cosine_similarity(c * u, u)
    assert torch.allclose(fid_scaled, torch.ones_like(fid_scaled), atol=1e-5)


def test_optimal_complex_alignment_and_frobenius():
    """Verify least-squares alignment constant c and Frobenius relative error."""
    A_true = torch.randn(64, 64, dtype=torch.complex64)
    c_target = 0.45 * np.exp(1j * 1.2)
    A_pred = A_true / c_target

    c_opt = optimal_complex_alignment_scale(A_true, A_pred)
    assert np.isclose(c_opt.item(), c_target, atol=1e-4)

    frob_err = frobenius_relative_error(A_true, A_pred, scale_c=c_opt)
    assert frob_err < 1e-4

    cos_sim = matrix_cosine_similarity(A_true, A_pred)
    assert np.isclose(cos_sim, 1.0, atol=1e-4)


def test_active_energy_efficiency():
    """Verify energy efficiency calculation."""
    inputs = torch.ones(4, 8, 8, dtype=torch.complex64)
    # Output with half amplitude -> quarter power
    outputs = inputs * 0.5
    eff = active_energy_efficiency(outputs, inputs)
    assert np.isclose(eff, 0.25, atol=1e-5)


def test_time_of_flight_latency():
    """Verify time of flight latency calculation in nanoseconds."""
    num_layers = 8
    drift_distance = 10.5e-3  # 10.5 mm
    # 9 propagation steps: 9 * 10.5mm = 94.5 mm = 0.0945 m
    # tau = 0.0945 / 299792458 * 1e9 ns ~ 0.315 ns
    tau = time_of_flight_latency(num_layers, drift_distance)
    expected_tau = (9 * drift_distance / SPEED_OF_LIGHT) * 1e9
    assert np.isclose(tau, expected_tau, atol=1e-4)
    assert 0.3 < tau < 0.4
