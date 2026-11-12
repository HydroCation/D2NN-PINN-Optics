"""Unit tests for multi-regime evaluation and benchmarking suite (src/evaluate.py and src/data.py)."""

from pathlib import Path
import pytest
import torch
import numpy as np

from src.data import generate_synthetic_digits_dataset, load_mnist_benchmark_dataset
from src.evaluate import (
    HarmonicBenchmark,
    ImageDatasetBenchmark,
    ModelEvaluator,
    RobustnessBenchmark,
)
from src.logger import RunConfig
from src.models import OpticalD2NN


def test_synthetic_digit_dataset_generation():
    """Verify procedural synthetic digit dataset generation."""
    n_samples = 50
    fov_size = 8
    imgs, labels = generate_synthetic_digits_dataset(
        n_samples=n_samples, fov_size=fov_size, seed=42, device="cpu"
    )

    assert imgs.shape == (n_samples, fov_size, fov_size)
    assert imgs.is_complex()
    assert imgs.dtype == torch.complex64
    assert labels.shape == (n_samples,)
    # Range check
    assert imgs.real.min().item() >= 0.0
    assert imgs.real.max().item() <= 1.0
    assert torch.all(imgs.imag == 0.0)


def test_harmonic_benchmark_generation_and_evaluation(tmp_path: Path):
    """Verify HarmonicBenchmark plane wave generation, peak localization, and plot creation."""
    fov_size = 4  # Fast test with 4x4
    bm = HarmonicBenchmark(fov_size=fov_size, device="cpu")

    waves, coords = bm.generate_plane_waves()
    assert waves.shape == (16, 4, 4)
    assert len(coords) == 16

    impulses = bm.generate_kronecker_impulses()
    assert impulses.shape == (16, 4, 4)

    gaussians = bm.generate_gaussians()
    assert gaussians.shape == (2, 4, 4)

    model = OpticalD2NN(grid_n=16, fov_size=fov_size, num_layers=2)
    fig_path = tmp_path / "test_harmonic.png"
    result = bm.evaluate(model=model, output_figure_path=fig_path)

    assert 0.0 <= result.spectral_peak_localization_accuracy_pct <= 100.0
    assert 0.0 <= result.mean_field_fidelity_pct <= 100.0
    assert fig_path.exists()
    assert fig_path.stat().st_size > 0


def test_image_dataset_benchmark(tmp_path: Path):
    """Verify ImageDatasetBenchmark runs with procedural fallback, computing PSNR and SSIM."""
    fov_size = 4
    bm = ImageDatasetBenchmark(fov_size=fov_size, device="cpu")
    model = OpticalD2NN(grid_n=16, fov_size=fov_size, num_layers=2)

    fig_path = tmp_path / "test_gallery.png"
    result = bm.evaluate(
        model=model,
        n_samples=16,
        force_synthetic=True,
        output_figure_path=fig_path,
    )

    assert result.n_samples == 16
    assert result.data_source == "procedural_synthetic"
    assert not np.isnan(result.psnr_db)
    assert -1.0 <= result.ssim <= 1.0
    assert 0.0 <= result.mean_field_fidelity_pct <= 100.0
    assert fig_path.exists()
    assert fig_path.stat().st_size > 0


def test_robustness_benchmark(tmp_path: Path):
    """Verify phase quantization, noise injection, and dual-axis plot generation."""
    fov_size = 4
    bm = RobustnessBenchmark(fov_size=fov_size, device="cpu")

    # Test phase quantization formula
    phase = torch.linspace(-np.pi, np.pi, 100)
    q2 = bm.quantize_phase(phase, bits=2)
    unique_levels = torch.unique(q2)
    # 2 bits has (2^2 - 1) + 1 levels = 4 levels
    assert len(unique_levels) <= 4

    model = OpticalD2NN(grid_n=16, fov_size=fov_size, num_layers=2)
    test_in = torch.randn(8, fov_size, fov_size, dtype=torch.complex64)
    test_targ = torch.fft.fft2(test_in)

    fig_path = tmp_path / "test_robustness.png"
    result = bm.evaluate(
        model=model,
        test_inputs=test_in,
        test_targets=test_targ,
        quantization_bits=(2, 4),
        noise_sigmas_rad=(0.01, 0.1),
        n_noise_trials=2,
        output_figure_path=fig_path,
    )

    assert len(result.quantization_fidelity_pct) == 2
    assert len(result.noise_fidelity_mean_pct) == 2
    assert fig_path.exists()
    assert fig_path.stat().st_size > 0


def test_model_evaluator_end_to_end(tmp_path: Path):
    """Verify ModelEvaluator loads checkpoint, runs all benchmarks, and saves summary JSON."""
    model = OpticalD2NN(grid_n=16, fov_size=4, num_layers=2)
    config = RunConfig(grid_n=16, fov_size=4, num_layers=2, device="cpu")

    ckpt_path = tmp_path / "phases_final.pt"
    torch.save(
        {
            "config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "phases": [p.detach().cpu() for p in model.phases],
        },
        ckpt_path,
    )

    evaluator = ModelEvaluator(checkpoint_path=ckpt_path, device="cpu")
    summary = evaluator.run_all(
        output_dir=tmp_path / "figures",
        summary_path=tmp_path / "eval_summary.json",
        force_synthetic=True,
        n_samples=20,
    )

    assert "harmonic_benchmark" in summary
    assert "image_dataset_benchmark" in summary
    assert "robustness_benchmark" in summary
    assert (tmp_path / "eval_summary.json").exists()
    assert (tmp_path / "figures" / "harmonic_spectra_comparison.png").exists()
    assert (tmp_path / "figures" / "mnist_optical_fft_gallery.png").exists()
    assert (tmp_path / "figures" / "robustness_ablation.png").exists()
