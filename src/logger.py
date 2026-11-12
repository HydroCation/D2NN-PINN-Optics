"""
Atomic OOP Benchmarking & Logging Engine for Diffractive Optical Neural Networks.

Adheres strictly to pure-Python, zero-heavy-dependency logging principles:
- RunConfig: Dataclass parameter specification
- MetricsTracker: Running epoch trajectories and CSV persistence
- OperatorBenchmarker: Transfer matrix impulse response probing, metric calculation, and visualization
- ExperimentLogger: Facade managing timestamped runs, dual stream/file logging, and JSON summaries
"""

from dataclasses import asdict, dataclass, field
import csv
import datetime
import json
import logging
import os
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from .metrics import (
    SPEED_OF_LIGHT,
    active_energy_efficiency,
    field_cosine_similarity,
    frobenius_relative_error,
    matrix_cosine_similarity,
    optimal_complex_alignment_scale,
    time_of_flight_latency,
)


def seed_everything(seed: int = 42) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch.

    Args:
        seed: Integer random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class RunConfig:
    """Dataclass storing all wave-optics experiment hyperparameters."""

    grid_n: int = 128
    fov_size: int = 8
    wavelength: float = 0.75e-3
    drift_distance: float = 14 * 0.75e-3
    num_layers: int = 8
    lr: float = 0.015
    batch_size: int = 64
    epochs: int = 2500
    target_eff: float = 0.035
    loss_mode: str = "balanced"  # 'balanced' | 'exponential' | 'kulce'
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tag: str = "d2nn_dft"
    model_type: str = "phase_only"  # 'phase_only' | 'complex' | 'apertured'
    c_M: float = 1.0
    c_D: float = 0.15
    alpha: float = 25.0
    eta_min: float = 5e-4
    use_aperture: bool = False
    aperture_radius_ratio: float = 0.85
    aperture_order: int = 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def save_json(self, filepath: Union[str, Path]) -> None:
        """Serialize configuration atomically to a JSON file.

        Args:
            filepath: Destination file path.
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        tmp_path.replace(path)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunConfig":
        """Instantiate RunConfig from dictionary, filtering unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, filepath: Union[str, Path]) -> "RunConfig":
        """Load RunConfig from a JSON file.

        Args:
            filepath: Source JSON path.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> "RunConfig":
        """Load RunConfig from a YAML file.

        Args:
            filepath: Source YAML path.
        """
        import yaml  # Lazy import PyYAML

        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


class MetricsTracker:
    """Maintains running training trajectories per epoch and persists to CSV."""

    FIELDNAMES = [
        "epoch",
        "loss_total",
        "loss_fidelity",
        "loss_efficiency",
        "active_efficiency",
        "field_fidelity",
        "lr",
    ]

    def __init__(self, csv_filepath: Optional[Union[str, Path]] = None) -> None:
        """Initialize the tracker.

        Args:
            csv_filepath: Optional path to metrics.csv for incremental logging.
        """
        self.csv_filepath = Path(csv_filepath) if csv_filepath else None
        self.history: List[Dict[str, float]] = []
        self._csv_initialized: bool = False

        if self.csv_filepath:
            self._init_csv()

    def _init_csv(self) -> None:
        """Create CSV file and write header if not existing."""
        if self.csv_filepath is None:
            return
        self.csv_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
            writer.writeheader()
        self._csv_initialized = True

    def update(
        self,
        epoch: int,
        loss_total: float,
        loss_fidelity: float,
        loss_efficiency: float,
        active_efficiency: float,
        field_fidelity: float,
        lr: Optional[float] = None,
    ) -> Dict[str, float]:
        """Record an epoch's metrics, append to CSV, and update history.

        Args:
            epoch: Current epoch index.
            loss_total: Overall scalar loss.
            loss_fidelity: Fidelity component of the loss.
            loss_efficiency: Optical power efficiency component of the loss.
            active_efficiency: Measured ratio of output FOV power to input FOV power.
            field_fidelity: Complex cosine similarity between prediction and target.
            lr: Current optimizer learning rate.

        Returns:
            Dictionary containing the recorded metrics.
        """
        record: Dict[str, float] = {
            "epoch": float(epoch),
            "loss_total": float(loss_total),
            "loss_fidelity": float(loss_fidelity),
            "loss_efficiency": float(loss_efficiency),
            "active_efficiency": float(active_efficiency),
            "field_fidelity": float(field_fidelity),
            "lr": float(lr) if lr is not None else 0.0,
        }
        self.history.append(record)

        if self.csv_filepath:
            if not self._csv_initialized:
                self._init_csv()
            with open(self.csv_filepath, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.FIELDNAMES)
                writer.writerow(record)

        return record

    def compute_rolling_averages(self, window: int = 10) -> Dict[str, float]:
        """Compute rolling average of recorded metrics over the most recent epochs.

        Args:
            window: Number of recent epochs to average.

        Returns:
            Dictionary of averaged metric values.
        """
        if not self.history:
            return {}

        recent = self.history[-window:]
        keys = [
            "loss_total",
            "loss_fidelity",
            "loss_efficiency",
            "active_efficiency",
            "field_fidelity",
        ]
        return {k: float(np.mean([r[k] for r in recent])) for k in keys}

    def get_latest(self) -> Dict[str, float]:
        """Return the most recently recorded metrics record."""
        return self.history[-1] if self.history else {}

    def get_best_fidelity(self) -> Dict[str, float]:
        """Find the record corresponding to the highest field fidelity."""
        if not self.history:
            return {}
        return max(self.history, key=lambda r: r["field_fidelity"])

    def get_min_loss(self) -> Dict[str, float]:
        """Find the record corresponding to the minimum total loss."""
        if not self.history:
            return {}
        return min(self.history, key=lambda r: r["loss_total"])


@dataclass
class BenchmarkResults:
    """Container for OperatorBenchmarker evaluation outputs."""

    c_real: float
    c_imag: float
    c_abs: float
    frobenius_error: float
    matrix_cosine_similarity: float
    active_energy_efficiency: float
    tof_latency_ns: float
    plot_filepath: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize benchmark results to dictionary."""
        return asdict(self)


class OperatorBenchmarker:
    """Independent evaluation engine probing optical transfer matrices via delta impulses."""

    def __init__(
        self,
        fov_size: int = 8,
        num_layers: int = 8,
        drift_distance: float = 14 * 0.75e-3,
        device: Optional[Union[torch.device, str]] = None,
    ) -> None:
        """Initialize the OperatorBenchmarker.

        Args:
            fov_size: Centered FOV grid dimension (fov_size x fov_size).
            num_layers: Number of cascaded diffractive layers.
            drift_distance: Free-space distance d between adjacent layers.
            device: Computing device.
        """
        self.fov_size = fov_size
        self.matrix_dim = fov_size * fov_size  # e.g., 64 for 8x8 FOV
        self.num_layers = num_layers
        self.drift_distance = drift_distance
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    def extract_empirical_operator(
        self, model: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Probe the model with N_i standard basis delta impulses to extract empirical operator A'.

        Constructs ground-truth 2D-DFT matrix A and empirical synthesized matrix A' in C^(N x N).

        Args:
            model: Trained OpticalD2NN module.

        Returns:
            Tuple of (A_true, A_pred) complex64 tensors of shape (matrix_dim, matrix_dim).
        """
        A_true = torch.zeros(
            (self.matrix_dim, self.matrix_dim),
            dtype=torch.complex64,
            device=self.device,
        )
        A_pred = torch.zeros(
            (self.matrix_dim, self.matrix_dim),
            dtype=torch.complex64,
            device=self.device,
        )

        model.eval()
        with torch.no_grad():
            for idx in range(self.matrix_dim):
                delta_in = torch.zeros(
                    (1, self.fov_size, self.fov_size),
                    dtype=torch.complex64,
                    device=self.device,
                )
                delta_in[0, idx // self.fov_size, idx % self.fov_size] = 1.0 + 0j

                # Ground truth 2D discrete Fourier transform
                dft_out = torch.fft.fftshift(
                    torch.fft.fft2(delta_in), dim=(-2, -1)
                ) / float(self.fov_size)
                A_true[:, idx] = dft_out.reshape(-1)

                # Synthesized optical response through physical D2NN
                if hasattr(model, "forward_fov"):
                    pred_out = model.forward_fov(delta_in)
                elif hasattr(model, "embed_fov") and hasattr(model, "extract_fov"):
                    full_in = model.embed_fov(delta_in)
                    full_out = model(full_in)
                    pred_out = model.extract_fov(full_out)
                else:
                    pred_out = model(delta_in)

                A_pred[:, idx] = pred_out.reshape(-1)

        return A_true, A_pred

    def render_verification_plot(
        self,
        A_true: torch.Tensor,
        A_pred_aligned: torch.Tensor,
        save_path: Union[str, Path],
    ) -> None:
        """Render and save the 3-panel transfer matrix verification plot.

        Panels:
        1. Ground-truth target phase angle(A)
        2. Synthesized optical phase angle(A')
        3. Absolute phase residual |Delta phi| = |angle(exp(j*(phi_pred - phi_true)))|

        Args:
            A_true: Ground-truth operator matrix.
            A_pred_aligned: Scaled synthesized operator matrix (c * A').
            save_path: Destination path for the plot PNG.
        """
        phase_true = torch.angle(A_true).detach().cpu().numpy()
        phase_pred = torch.angle(A_pred_aligned).detach().cpu().numpy()
        phase_diff = np.abs(np.angle(np.exp(1j * (phase_pred - phase_true))))

        fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=300)

        # Panel 1: Ground-truth phase
        im0 = axes[0].imshow(phase_true, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axes[0].set_title(r"Ground-Truth Target Phase $\angle A$", fontsize=11)
        axes[0].set_xlabel("Input Vector Index")
        axes[0].set_ylabel("Output Vector Index")
        cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        cbar0.set_ticks([-np.pi, 0, np.pi])
        cbar0.set_ticklabels([r"$-\pi$", r"$0$", r"$\pi$"])

        # Panel 2: Synthesized optical phase
        im1 = axes[1].imshow(phase_pred, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axes[1].set_title(r"Synthesized Optical Phase $\angle (c A')$", fontsize=11)
        axes[1].set_xlabel("Input Vector Index")
        cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        cbar1.set_ticks([-np.pi, 0, np.pi])
        cbar1.set_ticklabels([r"$-\pi$", r"$0$", r"$\pi$"])

        # Panel 3: Absolute phase residual
        im2 = axes[2].imshow(phase_diff, cmap="magma", vmin=0, vmax=np.pi)
        axes[2].set_title(r"Absolute Phase Residual $|\Delta \phi|$", fontsize=11)
        axes[2].set_xlabel("Input Vector Index")
        cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        cbar2.set_ticks([0, np.pi / 2, np.pi])
        cbar2.set_ticklabels([r"$0$", r"$\pi/2$", r"$\pi$"])

        plt.tight_layout()
        save_file = Path(save_path)
        save_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_file, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def evaluate(
        self,
        model: nn.Module,
        output_dir: Union[str, Path],
        test_inputs: Optional[torch.Tensor] = None,
        plot_filename: str = "transfer_matrix_verification.png",
    ) -> BenchmarkResults:
        """Run complete operator matrix probing, compute metrics, and export verification plot.

        Args:
            model: Trained D2NN optical neural network.
            output_dir: Run directory where the figure is saved.
            test_inputs: Optional evaluation input field batch. If None, synthesizes a test set.
            plot_filename: Name of the verification plot file.

        Returns:
            BenchmarkResults dataclass instance.
        """
        A_true, A_pred = self.extract_empirical_operator(model)

        # 1. Optimal complex alignment scalar c
        scale_c = optimal_complex_alignment_scale(A_true, A_pred)
        c_complex = scale_c.item() if hasattr(scale_c, "item") else complex(scale_c)
        A_pred_aligned = scale_c * A_pred

        # 2. Frobenius relative error ||A - c*A'||_F / ||A||_F
        frob_err = frobenius_relative_error(A_true, A_pred, scale_c=scale_c)

        # 3. Complex matrix cosine similarity |Tr(A^H (c*A'))| / (||A||_F * ||c*A'||_F)
        mat_cos_sim = matrix_cosine_similarity(A_true, A_pred_aligned)

        # 4. Active Energy Efficiency across test set E[ ||o'||^2 / ||i||^2 ]
        model.eval()
        with torch.no_grad():
            if test_inputs is None:
                # Deterministic coherent evaluation inputs (batch=128)
                torch.manual_seed(1337)
                r_re = torch.randn(
                    128, self.fov_size, self.fov_size, device=self.device
                )
                r_im = torch.randn(
                    128, self.fov_size, self.fov_size, device=self.device
                )
                test_inputs = torch.complex(r_re, r_im)

            if hasattr(model, "forward_fov"):
                test_preds = model.forward_fov(test_inputs)
            elif hasattr(model, "embed_fov") and hasattr(model, "extract_fov"):
                full_in = model.embed_fov(test_inputs)
                test_preds = model.extract_fov(model(full_in))
            else:
                test_preds = model(test_inputs)

            eff = active_energy_efficiency(test_preds, test_inputs)

        # 5. Time-of-flight latency tau = (K + 1) * d / c (ns)
        tau_ns = time_of_flight_latency(
            num_layers=self.num_layers,
            drift_distance=self.drift_distance,
            speed_of_light=SPEED_OF_LIGHT,
        )

        # 6. Render and save the 3-panel figure
        plot_path = Path(output_dir) / plot_filename
        self.render_verification_plot(A_true, A_pred_aligned, plot_path)

        return BenchmarkResults(
            c_real=float(c_complex.real),
            c_imag=float(c_complex.imag),
            c_abs=float(abs(c_complex)),
            frobenius_error=frob_err,
            matrix_cosine_similarity=mat_cos_sim,
            active_energy_efficiency=eff,
            tof_latency_ns=tau_ns,
            plot_filepath=str(plot_path),
        )


class ExperimentLogger:
    """Facade orchestrating run directories, dual console/file logging, checkpoints, and summaries."""

    def __init__(
        self,
        config: RunConfig,
        base_dir: Union[str, Path] = "outputs/runs",
    ) -> None:
        """Initialize the ExperimentLogger facade.

        Creates a timestamped run directory:
            <base_dir>/run_<YYYYMMDD_HHMMSS>_<tag>/

        Args:
            config: RunConfig containing experiment parameters.
            base_dir: Directory where runs are organized.
        """
        self.config = config
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"run_{self.timestamp}_{self.config.tag}"
        self.run_dir = Path(base_dir) / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Initialize subcomponents
        self.logger = self._setup_logging()
        self.tracker = MetricsTracker(self.run_dir / "metrics.csv")
        self.benchmarker = OperatorBenchmarker(
            fov_size=config.fov_size,
            num_layers=config.num_layers,
            drift_distance=config.drift_distance,
            device=config.device,
        )

        # Save initial configuration
        self.config.save_json(self.run_dir / "config.json")
        self.logger.info(
            f"Initialized atomic experiment run: {self.run_name} at {self.run_dir}"
        )
        self.logger.info(f"Target device: {self.config.device} | Seed: {self.config.seed}")

    def _setup_logging(self) -> logging.Logger:
        """Configure dual stream (console) and file (run.log) logging with clean formatting."""
        logger_name = f"d2nn.{self.run_name}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.handlers.clear()  # Avoid handler accumulation

        # Console StreamHandler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        # FileHandler to run.log
        log_file = self.run_dir / "run.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] (%(filename)s:%(lineno)d) %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        logger.propagate = False
        return logger

    def log_epoch(
        self,
        epoch: int,
        loss_total: float,
        loss_fidelity: float,
        loss_efficiency: float,
        active_efficiency: float,
        field_fidelity: float,
        lr: Optional[float] = None,
        log_frequency: int = 100,
    ) -> Dict[str, float]:
        """Record epoch metrics in tracker and log progress to console and run.log.

        Args:
            epoch: Current epoch number.
            loss_total: Total scalar loss.
            loss_fidelity: Fidelity loss component.
            loss_efficiency: Efficiency loss component.
            active_efficiency: Fraction of optical power in FOV.
            field_fidelity: Complex cosine similarity.
            lr: Current learning rate.
            log_frequency: Frequency of printing to console.

        Returns:
            Dictionary of recorded metrics.
        """
        record = self.tracker.update(
            epoch=epoch,
            loss_total=loss_total,
            loss_fidelity=loss_fidelity,
            loss_efficiency=loss_efficiency,
            active_efficiency=active_efficiency,
            field_fidelity=field_fidelity,
            lr=lr,
        )

        is_first = epoch == 1
        is_last = epoch == self.config.epochs
        is_interval = epoch % log_frequency == 0

        if is_first or is_last or is_interval:
            self.logger.info(
                f"Epoch {epoch:4d}/{self.config.epochs} | "
                f"Fidelity: {field_fidelity * 100:6.2f}% | "
                f"Efficiency: {active_efficiency * 100:5.2f}% | "
                f"LossTotal: {loss_total:.4f} | "
                f"LossEff: {loss_efficiency:.4f} | "
                f"LR: {lr if lr is not None else 0.0:.5f}"
            )
        return record

    def save_checkpoint(
        self,
        model: nn.Module,
        filename: str = "phases_final.pt",
    ) -> Path:
        """Save model phase checkpoints to run directory.

        Args:
            model: Trained OpticalD2NN module.
            filename: Target checkpoint filename.

        Returns:
            Path to the saved checkpoint.
        """
        checkpoint_path = self.run_dir / filename
        checkpoint_payload = {
            "epoch": len(self.tracker.history),
            "config": self.config.to_dict(),
            "model_state_dict": model.state_dict(),
        }
        if hasattr(model, "phases"):
            checkpoint_payload["phases"] = [p.detach().cpu() for p in model.phases]
        if hasattr(model, "raw_amplitudes") and model.raw_amplitudes is not None:
            checkpoint_payload["raw_amplitudes"] = [
                a.detach().cpu() for a in model.raw_amplitudes
            ]

        torch.save(checkpoint_payload, checkpoint_path)
        self.logger.info(f"Saved phase checkpoint to {checkpoint_path}")
        return checkpoint_path

    def run_benchmark(
        self,
        model: nn.Module,
        test_inputs: Optional[torch.Tensor] = None,
        plot_filename: str = "transfer_matrix_verification.png",
    ) -> BenchmarkResults:
        """Execute OperatorBenchmarker and log all calculated criteria.

        Args:
            model: Trained D2NN optical neural network.
            test_inputs: Optional test inputs for active efficiency measurement.
            plot_filename: Name of output verification plot file.

        Returns:
            BenchmarkResults instance.
        """
        self.logger.info("Executing OperatorBenchmarker transfer matrix evaluation...")
        results = self.benchmarker.evaluate(
            model=model,
            output_dir=self.run_dir,
            test_inputs=test_inputs,
            plot_filename=plot_filename,
        )

        c_str = f"{results.c_real:+.4e} {results.c_imag:+.4e}j (|c|={results.c_abs:.4f})"
        self.logger.info(f"Operator Alignment Scale c: {c_str}")
        self.logger.info(
            f"Transfer Matrix Frobenius Relative Error: {results.frobenius_error * 100:.2f}%"
        )
        self.logger.info(
            f"Complex Matrix Cosine Similarity: {results.matrix_cosine_similarity * 100:.2f}%"
        )
        self.logger.info(
            f"Active Energy Efficiency: {results.active_energy_efficiency * 100:.2f}%"
        )
        self.logger.info(
            f"Physical Time-of-Flight Latency: {results.tof_latency_ns:.3f} ns"
        )
        self.logger.info(f"Saved 3-panel verification plot to: {results.plot_filepath}")
        return results

    def generate_summary(
        self,
        benchmark_results: Optional[BenchmarkResults] = None,
        convergence_epoch: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate and save experiment_summary.json containing hardware info and final metrics.

        Args:
            benchmark_results: Optional BenchmarkResults from OperatorBenchmarker.
            convergence_epoch: Epoch number at which model met convergence criterion.

        Returns:
            Complete summary dictionary.
        """
        best_fid = self.tracker.get_best_fidelity()
        min_loss = self.tracker.get_min_loss()
        latest = self.tracker.get_latest()

        if convergence_epoch is None and best_fid:
            convergence_epoch = int(best_fid.get("epoch", 0))

        # Runtime hardware inspection
        cuda_available = torch.cuda.is_available()
        cuda_device_name = (
            torch.cuda.get_device_name(0) if cuda_available else "N/A"
        )
        cuda_device_count = torch.cuda.device_count() if cuda_available else 0

        hardware_info = {
            "cuda_available": cuda_available,
            "cuda_device_name": cuda_device_name,
            "cuda_device_count": cuda_device_count,
            "pytorch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "os_platform": platform.platform(),
            "cpu_architecture": platform.machine(),
        }

        summary: Dict[str, Any] = {
            "run_id": self.run_name,
            "timestamp": self.timestamp,
            "hardware": hardware_info,
            "hyperparameters": self.config.to_dict(),
            "training_summary": {
                "total_epochs_completed": len(self.tracker.history),
                "convergence_epoch": convergence_epoch,
                "latest_metrics": latest,
                "best_fidelity_record": best_fid,
                "min_loss_record": min_loss,
            },
            "benchmark_results": benchmark_results.to_dict()
            if benchmark_results
            else None,
        }

        summary_path = self.run_dir / "experiment_summary.json"
        tmp_path = summary_path.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        tmp_path.replace(summary_path)

        self.logger.info(f"Saved experiment summary to {summary_path}")
        return summary
