"""
Multi-Regime Evaluation & Benchmarking Suite for Diffractive Optical Neural Networks (D2NN).

Evaluates synthesized optical 2D DFT against PyTorch native torch.fft.fftshift(torch.fft.fft2(x))
across three comprehensive experimental regimes:
1. HarmonicBenchmark: Procedural 2D plane waves, Kronecker impulses, and 2D Gaussians.
2. ImageDatasetBenchmark: 1,000 real or synthetic digits (MNIST) downsampled to active FOV.
3. RobustnessBenchmark: Physical hardware-aware stress testing (phase quantization and noise).
"""

import argparse
import copy
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from .data import load_mnist_benchmark_dataset
from .logger import RunConfig, seed_everything
from .metrics import (
    active_energy_efficiency,
    compute_psnr,
    compute_ssim_2d,
    field_cosine_similarity,
    optimal_complex_alignment_scale,
)
from .models import OpticalD2NN

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] (%(name)s) %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("d2nn.evaluate")


# ---------------------------------------------------------------------------
# Benchmark Result Containers
# ---------------------------------------------------------------------------

@dataclass
class HarmonicBenchmarkResult:
    """Evaluation metrics for HarmonicBenchmark."""

    spectral_peak_localization_accuracy_pct: float
    pslr_mean_db: float
    mean_field_fidelity_pct: float
    active_energy_efficiency_pct: float
    plot_filepath: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ImageDatasetBenchmarkResult:
    """Evaluation metrics for ImageDatasetBenchmark."""

    n_samples: int
    data_source: str
    mean_field_fidelity_pct: float
    psnr_db: float
    ssim: float
    active_energy_efficiency_pct: float
    c_real: float
    c_imag: float
    c_magnitude: float
    plot_filepath: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RobustnessBenchmarkResult:
    """Evaluation metrics for RobustnessBenchmark."""

    quantization_bits: List[int]
    quantization_fidelity_pct: List[float]
    quantization_efficiency_pct: List[float]
    noise_sigmas_rad: List[float]
    noise_fidelity_mean_pct: List[float]
    noise_fidelity_std_pct: List[float]
    noise_efficiency_mean_pct: List[float]
    noise_efficiency_std_pct: List[float]
    plot_filepath: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 1. Harmonic Benchmark
# ---------------------------------------------------------------------------

class HarmonicBenchmark:
    """Evaluates optical DFT against analytical Fourier harmonics, impulses, and Gaussians."""

    def __init__(
        self,
        fov_size: int = 8,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Initialize HarmonicBenchmark.

        Args:
            fov_size: Active Field of View grid size (default 8).
            device: Computing device.
        """
        self.fov_size = fov_size
        self.device = torch.device(device)

    def generate_plane_waves(self) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """Generate all discrete 2D Fourier plane wave basis functions u(x, y) = exp(j 2pi (u0 x + v0 y) / N).

        Returns:
            Tuple of (waves_tensor, freq_coords):
            - waves_tensor: (fov_size^2, fov_size, fov_size) complex64
            - freq_coords: List of (u0, v0) frequency tuples.
        """
        N = self.fov_size
        y_grid, x_grid = torch.meshgrid(
            torch.arange(N, device=self.device, dtype=torch.float32),
            torch.arange(N, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        waves = []
        coords = []
        for u0 in range(N):
            for v0 in range(N):
                # Spatial plane wave phase: 2 * pi * (u0 * x + v0 * y) / N
                spatial_phase = 2.0 * np.pi * (u0 * x_grid + v0 * y_grid) / float(N)
                wave = torch.exp(1j * spatial_phase)
                waves.append(wave)
                coords.append((u0, v0))

        return torch.stack(waves, dim=0).to(torch.complex64), coords

    def generate_kronecker_impulses(self) -> torch.Tensor:
        """Generate complete standard impulse basis delta(x - x0, y - y0)."""
        N = self.fov_size
        impulses = torch.zeros((N * N, N, N), dtype=torch.complex64, device=self.device)
        for idx in range(N * N):
            impulses[idx, idx // N, idx % N] = 1.0 + 0j
        return impulses

    def generate_gaussians(self, waists: Tuple[float, ...] = (1.0, 2.0)) -> torch.Tensor:
        """Generate centered 2D Gaussians with spatial beam waists w0 in pixels."""
        N = self.fov_size
        y, x = torch.meshgrid(
            torch.arange(N, device=self.device, dtype=torch.float32),
            torch.arange(N, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        cy, cx = (N - 1.0) / 2.0, (N - 1.0) / 2.0
        gaussians = []
        for w0 in waists:
            r_sq = (x - cx) ** 2 + (y - cy) ** 2
            g = torch.exp(-r_sq / (2.0 * (w0**2)))
            gaussians.append(g.to(torch.complex64))
        return torch.stack(gaussians, dim=0)

    def evaluate(
        self,
        model: OpticalD2NN,
        output_figure_path: Union[str, Path] = "outputs/figures/harmonic_spectra_comparison.png",
    ) -> HarmonicBenchmarkResult:
        """Execute the Harmonic test suite.

        Args:
            model: Trained OpticalD2NN network.
            output_figure_path: Target file path for the 4-panel comparison.

        Returns:
            HarmonicBenchmarkResult dataclass.
        """
        model.eval()
        N = self.fov_size

        # 1. Plane waves evaluation
        plane_waves, coords = self.generate_plane_waves()
        targets_pw = torch.fft.fftshift(
            torch.fft.fft2(plane_waves), dim=(-2, -1)
        ) / float(N)

        with torch.no_grad():
            preds_pw = model.forward_fov(plane_waves)

        # Global optimal complex scaling for plane waves
        scale_c = optimal_complex_alignment_scale(targets_pw, preds_pw)
        preds_pw_aligned = scale_c * preds_pw

        # Peak localization accuracy
        correct_peaks = 0
        pslr_values = []
        total_pw = len(coords)

        for i in range(total_pw):
            target_mag_sq = torch.abs(targets_pw[i]) ** 2
            pred_mag_sq = torch.abs(preds_pw[i]) ** 2

            targ_peak_idx = int(torch.argmax(target_mag_sq).item())
            pred_peak_idx = int(torch.argmax(pred_mag_sq).item())

            if targ_peak_idx == pred_peak_idx:
                correct_peaks += 1

            # Peak-to-Sidelobe Ratio (PSLR in dB)
            peak_power = pred_mag_sq.view(-1)[pred_peak_idx].item()
            pred_flat = pred_mag_sq.view(-1).clone()
            pred_flat[pred_peak_idx] = -1.0  # Mask out main lobe peak
            sidelobe_power = pred_flat.max().item()

            pslr_db = 10.0 * np.log10(peak_power / (max(sidelobe_power, 1e-12)))
            pslr_values.append(pslr_db)

        peak_accuracy_pct = (correct_peaks / total_pw) * 100.0
        mean_pslr_db = float(np.mean(pslr_values))

        # 2. Combined harmonic suite metrics (plane waves + impulses + gaussians)
        impulses = self.generate_kronecker_impulses()
        gaussians = self.generate_gaussians()
        all_inputs = torch.cat([plane_waves, impulses, gaussians], dim=0)

        all_targets = torch.fft.fftshift(
            torch.fft.fft2(all_inputs), dim=(-2, -1)
        ) / float(N)

        with torch.no_grad():
            all_preds = model.forward_fov(all_inputs)

        fidelities = field_cosine_similarity(all_preds, all_targets)
        mean_fidelity_pct = float(torch.mean(fidelities).item()) * 100.0
        active_eff_pct = active_energy_efficiency(all_preds, all_inputs) * 100.0

        # 3. Render 4-panel comparison plot
        # Pick an illustrative diagonal grating frequency (u0=2, v0=3)
        sample_idx = 2 * N + 3  # (u0=2, v0=3)
        self._render_4panel_plot(
            sample_input=plane_waves[sample_idx],
            sample_target=targets_pw[sample_idx],
            sample_pred=preds_pw_aligned[sample_idx],
            save_path=output_figure_path,
            freq_label=f"(u0={coords[sample_idx][0]}, v0={coords[sample_idx][1]})",
        )

        return HarmonicBenchmarkResult(
            spectral_peak_localization_accuracy_pct=peak_accuracy_pct,
            pslr_mean_db=mean_pslr_db,
            mean_field_fidelity_pct=mean_fidelity_pct,
            active_energy_efficiency_pct=active_eff_pct,
            plot_filepath=str(output_figure_path),
        )

    def _render_4panel_plot(
        self,
        sample_input: torch.Tensor,
        sample_target: torch.Tensor,
        sample_pred: torch.Tensor,
        save_path: Union[str, Path],
        freq_label: str,
    ) -> None:
        """Render 4-panel comparison: Input -> Target FFT -> Optical Output -> Error."""
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), dpi=300)

        in_real = sample_input.real.detach().cpu().numpy()
        targ_mag = torch.abs(sample_target).detach().cpu().numpy()
        pred_mag = torch.abs(sample_pred).detach().cpu().numpy()
        residual = np.abs(pred_mag - targ_mag)

        # Scale colormap dynamically to ground-truth scale
        targ_max = max(float(targ_mag.max()), 1e-4)

        # Panel 1: Input grating
        im0 = axes[0].imshow(in_real, cmap="viridis")
        axes[0].set_title(f"Input Grating Re(u)\n{freq_label}", fontsize=10)
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Panel 2: Target FFT magnitude
        im1 = axes[1].imshow(targ_mag, cmap="inferno", vmin=0.0, vmax=targ_max)
        axes[1].set_title("Analytical Target |A u|\n(torch.fft.fft2)", fontsize=10)
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3: Optical Synthesized output
        im2 = axes[2].imshow(pred_mag, cmap="inferno", vmin=0.0, vmax=targ_max)
        axes[2].set_title("Optical Synthesized |c o'|\n(D2NN Physics)", fontsize=10)
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        # Panel 4: Pointwise residual matching ground-truth scale
        im3 = axes[3].imshow(residual, cmap="magma", vmin=0.0, vmax=targ_max)
        axes[3].set_title("Pointwise Residual\n||c o'| - |A u||", fontsize=10)
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

        for ax in axes:
            ax.set_xticks(range(self.fov_size))
            ax.set_yticks(range(self.fov_size))
            ax.tick_params(labelsize=8)

        plt.tight_layout()
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_file, dpi=300, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Image Dataset Benchmark
# ---------------------------------------------------------------------------

class ImageDatasetBenchmark:
    """Evaluates optical DFT over 1,000 real or synthetic downsampled digit images."""

    def __init__(
        self,
        fov_size: int = 8,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        self.fov_size = fov_size
        self.device = torch.device(device)

    def evaluate(
        self,
        model: OpticalD2NN,
        n_samples: int = 1000,
        mnist_root: Union[str, Path] = "data/mnist",
        force_synthetic: bool = False,
        output_figure_path: Union[str, Path] = "outputs/figures/mnist_optical_fft_gallery.png",
    ) -> ImageDatasetBenchmarkResult:
        """Run image dataset benchmarking suite.

        Args:
            model: Trained OpticalD2NN model.
            n_samples: Number of evaluation test samples (default 1000).
            mnist_root: Data directory.
            force_synthetic: If True, forces procedural digit generator.
            output_figure_path: Path for 8-sample comparison gallery plot.

        Returns:
            ImageDatasetBenchmarkResult.
        """
        model.eval()
        N = self.fov_size

        # Load evaluation test images
        inputs, labels, data_source = load_mnist_benchmark_dataset(
            n_samples=n_samples,
            fov_size=self.fov_size,
            root=mnist_root,
            force_synthetic=force_synthetic,
            device=self.device,
        )

        # Ground truth analytical 2D-DFT
        targets = torch.fft.fftshift(
            torch.fft.fft2(inputs), dim=(-2, -1)
        ) / float(N)

        # Optical inference
        with torch.no_grad():
            preds = model.forward_fov(inputs)

        # Global optimal complex alignment scalar c = <o_opt, o_true> / ||o_opt||^2
        scale_c = optimal_complex_alignment_scale(targets, preds)
        c_complex = complex(scale_c.item() if hasattr(scale_c, "item") else scale_c)
        preds_aligned = scale_c * preds

        # 1. Mean Complex Field Fidelity
        fidelities = field_cosine_similarity(preds, targets)
        mean_fid_pct = float(torch.mean(fidelities).item()) * 100.0

        # 2. Active energy efficiency
        eff_pct = active_energy_efficiency(preds, inputs) * 100.0

        # 3. Power spectrum PSNR and SSIM
        pred_power = torch.abs(preds_aligned) ** 2
        targ_power = torch.abs(targets) ** 2

        psnr_val = compute_psnr(pred=pred_power, target=targ_power)
        ssim_val = compute_ssim_2d(pred=pred_power, target=targ_power)

        # 4. Render 8-sample Gallery
        self._render_gallery(
            inputs=inputs[:8],
            targets=targets[:8],
            preds_aligned=preds_aligned[:8],
            labels=labels[:8],
            save_path=output_figure_path,
        )

        return ImageDatasetBenchmarkResult(
            n_samples=len(inputs),
            data_source=data_source,
            mean_field_fidelity_pct=mean_fid_pct,
            psnr_db=psnr_val,
            ssim=ssim_val,
            active_energy_efficiency_pct=eff_pct,
            c_real=float(c_complex.real),
            c_imag=float(c_complex.imag),
            c_magnitude=float(abs(c_complex)),
            plot_filepath=str(output_figure_path),
        )

    def _render_gallery(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        preds_aligned: torch.Tensor,
        labels: torch.Tensor,
        save_path: Union[str, Path],
    ) -> None:
        """Render 8 sample digits gallery: Input -> Target FFT -> Optical Output -> Pointwise Error."""
        n_rows = 8
        fig, axes = plt.subplots(n_rows, 4, figsize=(10, 16), dpi=300)

        for i in range(n_rows):
            img_in = inputs[i].real.detach().cpu().numpy()
            targ_mag = torch.abs(targets[i]).detach().cpu().numpy()
            pred_mag = torch.abs(preds_aligned[i]).detach().cpu().numpy()
            error = np.abs(pred_mag - targ_mag)
            digit_label = int(labels[i].item())

            # Row-specific dynamic ground-truth scale
            vmax_row = max(float(targ_mag.max()), 1e-4)

            # Col 0: Input digit
            axes[i, 0].imshow(img_in, cmap="gray", vmin=0, vmax=1)
            axes[i, 0].set_ylabel(f"Digit '{digit_label}'", fontsize=9, rotation=0, labelpad=25)
            if i == 0:
                axes[i, 0].set_title("Input (8x8)", fontsize=10, fontweight="bold")

            # Col 1: Target FFT magnitude
            axes[i, 1].imshow(targ_mag, cmap="inferno", vmin=0.0, vmax=vmax_row)
            if i == 0:
                axes[i, 1].set_title("Target |A i|", fontsize=10, fontweight="bold")

            # Col 2: Optical Output magnitude
            axes[i, 2].imshow(pred_mag, cmap="inferno", vmin=0.0, vmax=vmax_row)
            if i == 0:
                axes[i, 2].set_title("Optical |c o'|", fontsize=10, fontweight="bold")

            # Col 3: Pointwise Error on ground truth scale
            axes[i, 3].imshow(error, cmap="magma", vmin=0.0, vmax=vmax_row)
            if i == 0:
                axes[i, 3].set_title("Pointwise Error", fontsize=10, fontweight="bold")

            for col in range(4):
                axes[i, col].set_xticks([])
                axes[i, col].set_yticks([])

        plt.tight_layout()
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_file, dpi=300, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Robustness Benchmark
# ---------------------------------------------------------------------------

class RobustnessBenchmark:
    """Hardware stress-testing on frozen model: phase quantization and additive Gaussian noise."""

    def __init__(
        self,
        fov_size: int = 8,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        self.fov_size = fov_size
        self.device = torch.device(device)

    @staticmethod
    def quantize_phase(phase: torch.Tensor, bits: int) -> torch.Tensor:
        """Quantize continuous phase phi in [-pi, pi) into B bits.

        Formula:
            phi_q = round( (phi + pi) / (2pi) * (2^B - 1) ) * (2pi / (2^B - 1)) - pi

        Args:
            phase: Original continuous phase tensor in radians.
            bits: Bit depth integer B >= 1.

        Returns:
            Discretized phase tensor.
        """
        levels = (2**bits) - 1
        phi_norm = (phase + np.pi) / (2.0 * np.pi)
        phi_q_norm = torch.round(phi_norm * levels) / levels
        return phi_q_norm * (2.0 * np.pi) - np.pi

    def evaluate(
        self,
        model: OpticalD2NN,
        test_inputs: torch.Tensor,
        test_targets: torch.Tensor,
        quantization_bits: Tuple[int, ...] = (2, 3, 4, 5, 8),
        noise_sigmas_rad: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.2),
        n_noise_trials: int = 5,
        output_figure_path: Union[str, Path] = "outputs/figures/robustness_ablation.png",
    ) -> RobustnessBenchmarkResult:
        """Run hardware stress testing suite.

        Args:
            model: Trained OpticalD2NN.
            test_inputs: Complex test input batch.
            test_targets: Analytical DFT ground-truth batch.
            quantization_bits: List of bit depths B to evaluate.
            noise_sigmas_rad: List of Gaussian phase noise sigmas (rad).
            n_noise_trials: Number of random noise trials per sigma.
            output_figure_path: Target figure path.

        Returns:
            RobustnessBenchmarkResult dataclass.
        """
        model.eval()
        original_phases = [p.detach().clone() for p in model.phases]

        # 1. Phase Quantization Sweep
        q_fidelities = []
        q_efficiencies = []

        for b in quantization_bits:
            with torch.no_grad():
                for i, p_orig in enumerate(original_phases):
                    model.phases[i].copy_(self.quantize_phase(p_orig, bits=b))

                preds = model.forward_fov(test_inputs)
                fid = float(torch.mean(field_cosine_similarity(preds, test_targets)).item()) * 100.0
                eff = active_energy_efficiency(preds, test_inputs) * 100.0
                q_fidelities.append(fid)
                q_efficiencies.append(eff)

        # Restore original continuous phases
        with torch.no_grad():
            for i, p_orig in enumerate(original_phases):
                model.phases[i].copy_(p_orig)

        # 2. Additive Gaussian Phase Noise Sweep
        noise_fid_means = []
        noise_fid_stds = []
        noise_eff_means = []
        noise_eff_stds = []

        for sigma in noise_sigmas_rad:
            trial_fids = []
            trial_effs = []
            for trial in range(n_noise_trials):
                torch.manual_seed(1000 + trial * 37)
                with torch.no_grad():
                    for i, p_orig in enumerate(original_phases):
                        noise = torch.randn_like(p_orig) * sigma
                        model.phases[i].copy_(p_orig + noise)

                    preds = model.forward_fov(test_inputs)
                    fid = float(torch.mean(field_cosine_similarity(preds, test_targets)).item()) * 100.0
                    eff = active_energy_efficiency(preds, test_inputs) * 100.0
                    trial_fids.append(fid)
                    trial_effs.append(eff)

            noise_fid_means.append(float(np.mean(trial_fids)))
            noise_fid_stds.append(float(np.std(trial_fids)))
            noise_eff_means.append(float(np.mean(trial_effs)))
            noise_eff_stds.append(float(np.std(trial_effs)))

        # Restore original continuous phases
        with torch.no_grad():
            for i, p_orig in enumerate(original_phases):
                model.phases[i].copy_(p_orig)

        # 3. Render dual-axis line plots
        self._render_robustness_plot(
            quantization_bits=list(quantization_bits),
            q_fidelities=q_fidelities,
            q_efficiencies=q_efficiencies,
            noise_sigmas=list(noise_sigmas_rad),
            noise_fid_means=noise_fid_means,
            noise_fid_stds=noise_fid_stds,
            noise_eff_means=noise_eff_means,
            noise_eff_stds=noise_eff_stds,
            save_path=output_figure_path,
        )

        return RobustnessBenchmarkResult(
            quantization_bits=list(quantization_bits),
            quantization_fidelity_pct=q_fidelities,
            quantization_efficiency_pct=q_efficiencies,
            noise_sigmas_rad=list(noise_sigmas_rad),
            noise_fidelity_mean_pct=noise_fid_means,
            noise_fidelity_std_pct=noise_fid_stds,
            noise_efficiency_mean_pct=noise_eff_means,
            noise_efficiency_std_pct=noise_eff_stds,
            plot_filepath=str(output_figure_path),
        )

    def _render_robustness_plot(
        self,
        quantization_bits: List[int],
        q_fidelities: List[float],
        q_efficiencies: List[float],
        noise_sigmas: List[float],
        noise_fid_means: List[float],
        noise_fid_stds: List[float],
        noise_eff_means: List[float],
        noise_eff_stds: List[float],
        save_path: Union[str, Path],
    ) -> None:
        """Render dual-axis line plots for bit-depth and phase noise ablation."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

        # --- Left: Quantization Ablation ---
        color_fid = "#1f77b4"
        color_eff = "#ff7f0e"

        ax1.plot(
            quantization_bits,
            q_fidelities,
            marker="o",
            color=color_fid,
            linewidth=2,
            label="Field Fidelity (%)",
        )
        ax1.set_xlabel("Phase Quantization Bit-Depth (B)", fontsize=11)
        ax1.set_ylabel("Complex Field Fidelity (%)", color=color_fid, fontsize=11)
        ax1.tick_params(axis="y", labelcolor=color_fid)
        ax1.set_xticks(quantization_bits)
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax1_twin = ax1.twinx()
        ax1_twin.plot(
            quantization_bits,
            q_efficiencies,
            marker="s",
            linestyle="--",
            color=color_eff,
            linewidth=2,
            label="Active Efficiency (%)",
        )
        ax1_twin.set_ylabel("Active Energy Efficiency (%)", color=color_eff, fontsize=11)
        ax1_twin.tick_params(axis="y", labelcolor=color_eff)
        ax1.set_title("Robustness vs. Phase Quantization", fontsize=12, fontweight="bold")

        # --- Right: Additive Phase Noise Ablation ---
        sigmas_np = np.array(noise_sigmas)
        fid_means_np = np.array(noise_fid_means)
        fid_stds_np = np.array(noise_fid_stds)
        eff_means_np = np.array(noise_eff_means)
        eff_stds_np = np.array(noise_eff_stds)

        ax2.plot(
            sigmas_np,
            fid_means_np,
            marker="o",
            color=color_fid,
            linewidth=2,
            label="Field Fidelity (%)",
        )
        ax2.fill_between(
            sigmas_np,
            fid_means_np - fid_stds_np,
            fid_means_np + fid_stds_np,
            alpha=0.25,
            color=color_fid,
        )
        ax2.set_xlabel(r"Gaussian Phase Noise $\sigma$ (radians)", fontsize=11)
        ax2.set_ylabel("Complex Field Fidelity (%)", color=color_fid, fontsize=11)
        ax2.tick_params(axis="y", labelcolor=color_fid)
        ax2.grid(True, linestyle="--", alpha=0.5)

        ax2_twin = ax2.twinx()
        ax2_twin.plot(
            sigmas_np,
            eff_means_np,
            marker="s",
            linestyle="--",
            color=color_eff,
            linewidth=2,
            label="Active Efficiency (%)",
        )
        ax2_twin.fill_between(
            sigmas_np,
            eff_means_np - eff_stds_np,
            eff_means_np + eff_stds_np,
            alpha=0.20,
            color=color_eff,
        )
        ax2_twin.set_ylabel("Active Energy Efficiency (%)", color=color_eff, fontsize=11)
        ax2_twin.tick_params(axis="y", labelcolor=color_eff)
        ax2.set_title(r"Robustness vs. Phase Noise $\mathcal{N}(0, \sigma^2)$", fontsize=12, fontweight="bold")

        plt.tight_layout()
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_file, dpi=300, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Orchestration Facade & CLI Entrypoint
# ---------------------------------------------------------------------------

class ModelEvaluator:
    """Facade orchestrating Harmonic, ImageDataset, and Robustness benchmarks."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """Initialize ModelEvaluator with model checkpoint and configuration.

        Args:
            checkpoint_path: Path to trained PyTorch checkpoint (*.pt).
            config_path: Optional path to YAML/JSON configuration.
            device: Computing device ('cuda' or 'cpu').
        """
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load checkpoint payload
        logger.info(f"Loading checkpoint from: {self.checkpoint_path}")
        self.checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Reconstruct RunConfig
        if config_path and Path(config_path).exists():
            cfg_p = Path(config_path)
            self.config = (
                RunConfig.from_yaml(cfg_p)
                if cfg_p.suffix.lower() in [".yaml", ".yml"]
                else RunConfig.from_json(cfg_p)
            )
        elif "config" in self.checkpoint and isinstance(self.checkpoint["config"], dict):
            self.config = RunConfig.from_dict(self.checkpoint["config"])
        else:
            self.config = RunConfig()

        self.config.device = str(self.device)

        # Build and load OpticalD2NN
        self.model = OpticalD2NN(
            grid_n=self.config.grid_n,
            fov_size=self.config.fov_size,
            num_layers=self.config.num_layers,
            wavelength=self.config.wavelength,
            drift_distance=self.config.drift_distance,
            use_aperture=self.config.use_aperture,
            aperture_radius_ratio=self.config.aperture_radius_ratio,
            aperture_order=self.config.aperture_order,
        ).to(self.device)

        if "model_state_dict" in self.checkpoint:
            self.model.load_state_dict(self.checkpoint["model_state_dict"])
        elif "phases" in self.checkpoint:
            for i, p_tensor in enumerate(self.checkpoint["phases"]):
                self.model.phases[i].data.copy_(p_tensor.to(self.device))

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        logger.info(
            f"Successfully initialized OpticalD2NN ({self.config.num_layers} layers) on {self.device}."
        )

    def run_all(
        self,
        output_dir: Union[str, Path] = "outputs/figures",
        summary_path: Optional[Union[str, Path]] = None,
        force_synthetic: bool = False,
        n_samples: int = 1000,
    ) -> Dict[str, Any]:
        """Execute all 3 benchmark regimes and persist summary JSON and figures.

        Args:
            output_dir: Output directory for figures.
            summary_path: Optional JSON summary destination.
            force_synthetic: Force procedural synthetic digit generation.
            n_samples: Number of image dataset samples.

        Returns:
            Dictionary containing full benchmark results.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=== Regime 1: Harmonic Benchmark ===")
        harmonic_bm = HarmonicBenchmark(fov_size=self.config.fov_size, device=self.device)
        harmonic_res = harmonic_bm.evaluate(
            model=self.model,
            output_figure_path=out_dir / "harmonic_spectra_comparison.png",
        )
        logger.info(
            f"Harmonic Peak Localization Accuracy: {harmonic_res.spectral_peak_localization_accuracy_pct:.2f}% | "
            f"PSLR: {harmonic_res.pslr_mean_db:.2f} dB | Fidelity: {harmonic_res.mean_field_fidelity_pct:.2f}%"
        )

        logger.info("=== Regime 2: Image Dataset Benchmark ===")
        image_bm = ImageDatasetBenchmark(fov_size=self.config.fov_size, device=self.device)
        image_res = image_bm.evaluate(
            model=self.model,
            n_samples=n_samples,
            mnist_root="data/mnist",
            force_synthetic=force_synthetic,
            output_figure_path=out_dir / "mnist_optical_fft_gallery.png",
        )
        logger.info(
            f"Image Dataset ({image_res.data_source}) Mean Fidelity: {image_res.mean_field_fidelity_pct:.2f}% | "
            f"PSNR: {image_res.psnr_db:.2f} dB | SSIM: {image_res.ssim:.4f} | Eff: {image_res.active_energy_efficiency_pct:.2f}%"
        )

        logger.info("=== Regime 3: Robustness & Stress Benchmark ===")
        robust_bm = RobustnessBenchmark(fov_size=self.config.fov_size, device=self.device)
        # Use first 256 samples from harmonic plane waves + impulses as test set
        pw, _ = harmonic_bm.generate_plane_waves()
        imp = harmonic_bm.generate_kronecker_impulses()
        rob_inputs = torch.cat([pw, imp], dim=0)
        rob_targets = torch.fft.fftshift(
            torch.fft.fft2(rob_inputs), dim=(-2, -1)
        ) / float(self.config.fov_size)

        robust_res = robust_bm.evaluate(
            model=self.model,
            test_inputs=rob_inputs,
            test_targets=rob_targets,
            output_figure_path=out_dir / "robustness_ablation.png",
        )
        logger.info(
            f"Robustness ablation: 8-bit fid={robust_res.quantization_fidelity_pct[-1]:.2f}%, "
            f"2-bit fid={robust_res.quantization_fidelity_pct[0]:.2f}%"
        )

        summary = {
            "checkpoint_path": str(self.checkpoint_path),
            "device": str(self.device),
            "hyperparameters": self.config.to_dict(),
            "harmonic_benchmark": harmonic_res.to_dict(),
            "image_dataset_benchmark": image_res.to_dict(),
            "robustness_benchmark": robust_res.to_dict(),
        }

        benchmark_results = {
            "checkpoint_path": str(self.checkpoint_path),
            "device": str(self.device),
            "harmonic_benchmark": harmonic_res.to_dict(),
            "image_dataset_benchmark": image_res.to_dict(),
            "robustness_benchmark": robust_res.to_dict(),
        }

        # Route evaluation_summary.json and benchmark_results.json directly into outputs/
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(parents=True, exist_ok=True)

        primary_summary_path = outputs_dir / "evaluation_summary.json"
        with open(primary_summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        primary_bench_path = outputs_dir / "benchmark_results.json"
        with open(primary_bench_path, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, f, indent=2)

        # Handle explicit summary_path or run directory copies
        if summary_path is not None:
            custom_summary_path = Path(summary_path)
            custom_summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(custom_summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            target_reported = custom_summary_path
        else:
            run_dir = self.checkpoint_path.parent
            if run_dir.exists() and run_dir.name.startswith("run_"):
                with open(run_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)
                with open(run_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
                    json.dump(benchmark_results, f, indent=2)
            target_reported = primary_summary_path

        logger.info(f"Evaluation summary written to: {target_reported}")
        logger.info(f"Benchmark results written to: {primary_bench_path}")
        return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Regime Evaluation and Benchmarking Suite for Diffractive Optical Neural Networks."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint (phases_final.pt).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML or JSON config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/figures",
        help="Directory to save output figures.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help="Explicit destination path for evaluation_summary.json.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device ('cuda', 'cpu').",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of images to evaluate in ImageDatasetBenchmark.",
    )
    parser.add_argument(
        "--force-synthetic",
        action="store_true",
        help="Force procedural geometric synthetic digit generation.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI execution function."""
    args = parse_args()
    seed_everything(42)

    evaluator = ModelEvaluator(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
    )

    evaluator.run_all(
        output_dir=args.output_dir,
        summary_path=args.summary_path,
        force_synthetic=args.force_synthetic,
        n_samples=args.n_samples,
    )


if __name__ == "__main__":
    main()
