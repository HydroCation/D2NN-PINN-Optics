"""
Wave-optics SciML evaluation metrics and physical performance criteria.

Provides mathematically rigorous, deterministic metrics for:
- Field fidelity / complex cosine similarity
- Operator transfer matrix reconstruction (Frobenius error, alignment scale, matrix cosine similarity)
- Physical energy efficiency and time-of-flight latency
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch

# Speed of light in vacuum (m/s)
SPEED_OF_LIGHT: float = 299792458.0


def field_cosine_similarity(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    dim: Tuple[int, ...] = (-2, -1),
) -> torch.Tensor:
    """Compute the complex field cosine similarity (fidelity) between optical fields.

    Fidelity is defined as:
        F(u, v) = |<u, v>| / (||u||_2 * ||v||_2)
                = |sum(u * conj(v))| / (sqrt(sum(|u|^2)) * sqrt(sum(|v|^2)))

    Args:
        pred: Predicted complex optical field tensor.
        target: Target ground-truth complex optical field tensor.
        eps: Small epsilon for numerical division stability.
        dim: Spatial dimensions to compute inner product over.

    Returns:
        Tensor of fidelity values in range [0, 1].
    """
    overlap = torch.sum(pred * torch.conj(target), dim=dim)
    norm_pred = torch.sqrt(torch.sum(torch.abs(pred) ** 2, dim=dim) + eps)
    norm_targ = torch.sqrt(torch.sum(torch.abs(target) ** 2, dim=dim) + eps)
    return torch.abs(overlap) / (norm_pred * norm_targ)


def optimal_complex_alignment_scale(
    A_true: torch.Tensor,
    A_pred: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Calculate the global complex alignment constant c minimizing ||A_true - c * A_pred||_F^2.

    Derived via least-squares complex projection:
        c = <A_pred, A_true> / ||A_pred||_F^2
          = sum(A_true * conj(A_pred)) / sum(|A_pred|^2)

    Args:
        A_true: Ground truth complex operator matrix.
        A_pred: Empirically probed complex operator matrix.
        eps: Small epsilon for numerical stability.

    Returns:
        Scalar complex tensor c.
    """
    numerator = torch.sum(A_true * torch.conj(A_pred))
    denominator = torch.sum(torch.abs(A_pred) ** 2) + eps
    return numerator / denominator


def frobenius_relative_error(
    A_true: torch.Tensor,
    A_pred: torch.Tensor,
    scale_c: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> float:
    """Compute Frobenius relative error: ||A_true - c * A_pred||_F / ||A_true||_F.

    Args:
        A_true: Ground truth complex matrix (M, N).
        A_pred: Predicted / synthesized complex matrix (M, N).
        scale_c: Optional precomputed complex scale factor. If None, computes optimal c.
        eps: Epsilon for denominator stabilization.

    Returns:
        Scalar relative error as float.
    """
    if scale_c is None:
        scale_c = optimal_complex_alignment_scale(A_true, A_pred, eps=eps)

    A_aligned = scale_c * A_pred
    diff_norm = torch.linalg.norm(A_true - A_aligned, ord="fro")
    true_norm = torch.linalg.norm(A_true, ord="fro") + eps
    return float((diff_norm / true_norm).item())


def matrix_cosine_similarity(
    A_true: torch.Tensor,
    A_pred: torch.Tensor,
    eps: float = 1e-8,
) -> float:
    """Compute Complex Matrix Cosine Similarity:

        |Tr(A^H * A')| / (||A||_F * ||A'||_F)

    Invariant to global complex scalar multiplication.

    Args:
        A_true: Ground truth complex matrix.
        A_pred: Empirical complex matrix.
        eps: Stability constant.

    Returns:
        Scalar cosine similarity in [0, 1] as float.
    """
    # Tr(A_true^H * A_pred) = sum(conj(A_true) * A_pred)
    inner_prod = torch.sum(torch.conj(A_true) * A_pred)
    norm_true = torch.linalg.norm(A_true, ord="fro")
    norm_pred = torch.linalg.norm(A_pred, ord="fro") + eps
    sim = torch.abs(inner_prod) / (norm_true * norm_pred)
    return float(sim.item())


def active_energy_efficiency(
    pred_fov: torch.Tensor,
    in_fov: torch.Tensor,
    eps: float = 1e-8,
    dim: Tuple[int, ...] = (-2, -1),
) -> float:
    """Compute the Active Energy Efficiency E[ ||o'||^2 / ||i||^2 ].

    Measures fraction of optical energy coupled into the designated output FOV
    relative to total incident energy at the input FOV.

    Args:
        pred_fov: Optical field in output FOV.
        in_fov: Optical field in input FOV.
        eps: Small constant to avoid zero-division.
        dim: Spatial dimensions.

    Returns:
        Mean efficiency as float in range [0, 1].
    """
    pred_power = torch.sum(torch.abs(pred_fov) ** 2, dim=dim)
    in_power = torch.sum(torch.abs(in_fov) ** 2, dim=dim)
    eff = torch.mean(pred_power / (in_power + eps))
    return float(eff.item())


def time_of_flight_latency(
    num_layers: int,
    drift_distance: float,
    speed_of_light: float = SPEED_OF_LIGHT,
) -> float:
    """Calculate the physical time-of-flight inference latency in nanoseconds.

    For K diffractive layers, light undergoes (K + 1) drift propagation steps:
        tau = (K + 1) * d / c

    Args:
        num_layers: Number of diffractive phase plates K.
        drift_distance: Drift distance d per stage in meters.
        speed_of_light: Speed of light in the propagation medium (m/s).

    Returns:
        Latency tau in nanoseconds (ns).
    """
    total_distance = (num_layers + 1) * drift_distance
    tau_seconds = total_distance / speed_of_light
    return tau_seconds * 1e9  # Convert seconds to ns
