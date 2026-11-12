"""
Diffractive Deep Neural Network (D2NN) Wave-Optics Package.

This package provides a modular, production-grade PyTorch framework for simulating
and optimizing physical diffractive optical networks using the Angular Spectrum Method (ASM).
"""

from .propagation import AngularSpectrumPropagator, compute_asm_transfer_function
from .models import OpticalD2NN, DiffractiveLayer
from .metrics import (
    field_cosine_similarity,
    frobenius_relative_error,
    matrix_cosine_similarity,
    optimal_complex_alignment_scale,
    active_energy_efficiency,
    time_of_flight_latency,
    compute_psnr,
    compute_ssim_2d,
)
from .logger import (
    RunConfig,
    MetricsTracker,
    OperatorBenchmarker,
    ExperimentLogger,
    BenchmarkResults,
)
from .data import load_mnist_benchmark_dataset, generate_synthetic_digits_dataset

__all__ = [
    "AngularSpectrumPropagator",
    "compute_asm_transfer_function",
    "OpticalD2NN",
    "DiffractiveLayer",
    "field_cosine_similarity",
    "frobenius_relative_error",
    "matrix_cosine_similarity",
    "optimal_complex_alignment_scale",
    "active_energy_efficiency",
    "time_of_flight_latency",
    "compute_psnr",
    "compute_ssim_2d",
    "RunConfig",
    "MetricsTracker",
    "OperatorBenchmarker",
    "ExperimentLogger",
    "BenchmarkResults",
    "load_mnist_benchmark_dataset",
    "generate_synthetic_digits_dataset",
]
