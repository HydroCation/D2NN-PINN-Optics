#!/usr/bin/env python3
"""
Main CLI entrypoint for D2NN Optical Operator Learning Experiments.

Usage:
    python run_experiment.py --config configs/default.yaml
    python run_experiment.py --config configs/kulce_optimized.yaml --epochs 1000
    python run_experiment.py --quick-test  # Fast 10-epoch validation run
"""

import argparse
from pathlib import Path
import sys
import torch
import torch.optim as optim

from src.logger import ExperimentLogger, RunConfig, seed_everything
from src.metrics import active_energy_efficiency, field_cosine_similarity
from src.models import OpticalD2NN


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Diffractive Deep Neural Network (D2NN) for optical 2D-DFT."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML or JSON configuration file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override initial learning rate.",
    )
    parser.add_argument(
        "--loss-mode",
        type=str,
        default=None,
        choices=["balanced", "exponential", "kulce"],
        help="Loss mode ('balanced' or 'exponential').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computing device ('cuda', 'cpu', or device index).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic initialization.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Identifier tag for run folder naming.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="phase_only",
        choices=["phase_only", "complex", "apertured"],
        help="Modulation model type ('phase_only', 'complex', or 'apertured').",
    )
    parser.add_argument(
        "--target-eff",
        type=float,
        default=None,
        help="Target optical efficiency in active FOV (e.g. 0.035 for 3.5%).",
    )
    parser.add_argument(
        "--barrier-weight",
        type=float,
        default=None,
        help="Barrier weight / scaling coefficient for efficiency penalty (c_D).",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run a rapid 10-epoch sanity test.",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> RunConfig:
    """Load configuration from file and apply command-line overrides."""
    config_path = Path(args.config)
    if not config_path.exists():
        # Fallback to default JSON if YAML is missing
        json_alt = config_path.with_suffix(".json")
        if json_alt.exists():
            config_path = json_alt
        else:
            raise FileNotFoundError(f"Configuration file not found: {args.config}")

    if config_path.suffix.lower() in [".yaml", ".yml"]:
        config = RunConfig.from_yaml(config_path)
    else:
        config = RunConfig.from_json(config_path)

    # CLI Overrides
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
    if args.loss_mode is not None:
        config.loss_mode = args.loss_mode
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    if args.tag is not None:
        config.tag = args.tag
    if "--model-type" in sys.argv:
        config.model_type = args.model_type
    if args.target_eff is not None:
        config.target_eff = args.target_eff
    if args.barrier_weight is not None:
        config.c_D = args.barrier_weight

    if args.quick_test:
        config.epochs = 10
        config.batch_size = 16
        config.tag = f"test_{config.tag}"

    # Auto device validation
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        config.device = "cpu"

    return config


def main() -> None:
    """Run optical D2NN training and operator benchmarking."""
    args = parse_args()
    config = load_config(args)

    # Set deterministic random seed
    seed_everything(config.seed)
    device = torch.device(config.device)

    # Initialize atomic logging and run management
    exp_logger = ExperimentLogger(config=config, base_dir="outputs/runs")
    exp_logger.logger.info("Initializing physical OpticalD2NN architecture...")

    # Instantiate model
    model = OpticalD2NN(
        grid_n=config.grid_n,
        fov_size=config.fov_size,
        num_layers=config.num_layers,
        wavelength=config.wavelength,
        drift_distance=config.drift_distance,
        use_aperture=config.use_aperture,
        aperture_radius_ratio=config.aperture_radius_ratio,
        aperture_order=config.aperture_order,
        modulation=config.model_type,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.eta_min
    )

    exp_logger.logger.info(
        f"Starting training: {config.epochs} epochs | Loss mode: '{config.loss_mode}' | "
        f"FOV: {config.fov_size}x{config.fov_size} in {config.grid_n}x{config.grid_n} grid"
    )

    log_freq = max(1, config.epochs // 10) if not args.quick_test else 1

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()

        # Generate coherent complex random wavefront inputs
        rand_re = torch.randn(
            config.batch_size, config.fov_size, config.fov_size, device=device
        )
        rand_im = torch.randn(
            config.batch_size, config.fov_size, config.fov_size, device=device
        )
        raw_inputs = torch.complex(rand_re, rand_im)

        if config.loss_mode in ["exponential", "kulce"]:
            # Unit power normalization as in Kulce et al.
            power_norm = torch.sqrt(
                torch.sum(torch.abs(raw_inputs) ** 2, dim=(-2, -1), keepdim=True) + 1e-8
            )
            inputs_fov = raw_inputs / power_norm
        else:
            inputs_fov = raw_inputs

        # Ground-truth 2D-DFT target
        targets_fov = torch.fft.fftshift(
            torch.fft.fft2(inputs_fov), dim=(-2, -1)
        ) / float(config.fov_size)

        # Forward pass through physical wave-optics pipeline
        preds_fov = model.forward_fov(inputs_fov)

        # Efficiency: fraction of optical power preserved in designated FOV
        eff = active_energy_efficiency(preds_fov, inputs_fov)

        # Fidelity: complex field cosine similarity
        fid_tensor = field_cosine_similarity(preds_fov, targets_fov)
        mean_fidelity = float(torch.mean(fid_tensor).item())
        loss_fidelity = 1.0 - torch.mean(fid_tensor)

        # Loss calculation based on selected mode
        if config.loss_mode in ["exponential", "kulce"]:
            # Kulce exponential penalty: L_D = exp(-alpha * eff)
            eff_tensor = torch.mean(
                torch.sum(torch.abs(preds_fov) ** 2, dim=(-2, -1))
                / (torch.sum(torch.abs(inputs_fov) ** 2, dim=(-2, -1)) + 1e-8)
            )
            loss_eff = torch.exp(-config.alpha * eff_tensor)
            total_loss = config.c_M * loss_fidelity + config.c_D * loss_eff
        else:
            # Balanced logarithmic attraction toward target_eff
            eff_tensor = torch.mean(
                torch.sum(torch.abs(preds_fov) ** 2, dim=(-2, -1))
                / (torch.sum(torch.abs(inputs_fov) ** 2, dim=(-2, -1)) + 1e-8)
            )
            eff_ratio = torch.clamp(eff_tensor / config.target_eff, min=1e-4, max=1.0)
            loss_eff = -0.08 * torch.log(eff_ratio)
            total_loss = loss_fidelity + loss_eff

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        exp_logger.log_epoch(
            epoch=epoch,
            loss_total=float(total_loss.item()),
            loss_fidelity=float(loss_fidelity.item()),
            loss_efficiency=float(loss_eff.item()),
            active_efficiency=eff,
            field_fidelity=mean_fidelity,
            lr=current_lr,
            log_frequency=log_freq,
        )

    # Save model phase checkpoint
    exp_logger.save_checkpoint(model, filename="phases_final.pt")

    # Run operator transfer matrix benchmarking
    benchmark_results = exp_logger.run_benchmark(model)

    # Save experiment summary JSON
    exp_logger.generate_summary(benchmark_results=benchmark_results)
    exp_logger.logger.info("Experiment run completed successfully.")


if __name__ == "__main__":
    main()
