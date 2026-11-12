"""Unit tests for atomic logging and evaluation engine (src/logger.py)."""

import csv
import json
from pathlib import Path
import pytest
import torch
import numpy as np

from src.logger import (
    BenchmarkResults,
    ExperimentLogger,
    MetricsTracker,
    OperatorBenchmarker,
    RunConfig,
    seed_everything,
)
from src.models import OpticalD2NN


def test_run_config_serialization(tmp_path: Path):
    """Verify RunConfig conversion to dict, JSON, and YAML roundtrip."""
    config = RunConfig(
        grid_n=64,
        fov_size=4,
        wavelength=1e-3,
        drift_distance=5e-3,
        num_layers=4,
        lr=0.01,
        batch_size=32,
        epochs=50,
        target_eff=0.05,
        loss_mode="balanced",
        seed=123,
        device="cpu",
    )

    d = config.to_dict()
    assert d["grid_n"] == 64
    assert d["seed"] == 123

    # JSON roundtrip
    json_path = tmp_path / "test_config.json"
    config.save_json(json_path)
    assert json_path.exists()

    loaded_json = RunConfig.from_json(json_path)
    assert loaded_json.grid_n == config.grid_n
    assert loaded_json.lr == config.lr
    assert loaded_json.loss_mode == config.loss_mode

    # YAML roundtrip
    yaml_path = tmp_path / "test_config.yaml"
    import yaml

    with open(yaml_path, "w") as f:
        yaml.safe_dump(config.to_dict(), f)
    loaded_yaml = RunConfig.from_yaml(yaml_path)
    assert loaded_yaml.num_layers == config.num_layers


def test_metrics_tracker_and_csv(tmp_path: Path):
    """Verify MetricsTracker records history, computes rolling averages, and writes CSV."""
    csv_path = tmp_path / "metrics.csv"
    tracker = MetricsTracker(csv_filepath=csv_path)

    for epoch in range(1, 11):
        tracker.update(
            epoch=epoch,
            loss_total=1.0 / epoch,
            loss_fidelity=0.5 / epoch,
            loss_efficiency=0.5 / epoch,
            active_efficiency=0.01 * epoch,
            field_fidelity=0.1 * epoch,
            lr=0.001,
        )

    assert len(tracker.history) == 10
    best_fid = tracker.get_best_fidelity()
    assert best_fid["epoch"] == 10
    assert np.isclose(best_fid["field_fidelity"], 1.0)

    # Rolling averages over window of 5
    rolling = tracker.compute_rolling_averages(window=5)
    assert "field_fidelity" in rolling
    # Average of [0.6, 0.7, 0.8, 0.9, 1.0] = 0.8
    assert np.isclose(rolling["field_fidelity"], 0.8)

    # Verify CSV file contents
    assert csv_path.exists()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 10
        assert float(reader[0]["epoch"]) == 1.0
        assert float(reader[-1]["epoch"]) == 10.0


def test_operator_benchmarker_evaluation(tmp_path: Path):
    """Verify OperatorBenchmarker impulse probing and 3-panel plot creation."""
    seed_everything(42)
    fov_size = 4  # 4x4 -> 16x16 operator matrix for fast test
    model = OpticalD2NN(grid_n=32, fov_size=fov_size, num_layers=2)
    model.eval()

    benchmarker = OperatorBenchmarker(
        fov_size=fov_size,
        num_layers=2,
        drift_distance=5e-3,
        device=torch.device("cpu"),
    )

    results = benchmarker.evaluate(
        model=model,
        output_dir=tmp_path,
        plot_filename="test_verification.png",
    )

    assert isinstance(results, BenchmarkResults)
    assert results.matrix_cosine_similarity >= 0.0
    assert results.frobenius_error >= 0.0
    assert results.active_energy_efficiency >= 0.0
    assert results.tof_latency_ns > 0.0

    plot_file = Path(results.plot_filepath)
    assert plot_file.exists()
    assert plot_file.stat().st_size > 0


def test_experiment_logger_facade(tmp_path: Path):
    """Verify ExperimentLogger run folder creation, dual logging, checkpoints, and summary."""
    config = RunConfig(
        grid_n=32,
        fov_size=4,
        num_layers=2,
        epochs=5,
        device="cpu",
        tag="unit_test",
    )

    logger_facade = ExperimentLogger(config=config, base_dir=tmp_path)
    assert logger_facade.run_dir.exists()
    assert (logger_facade.run_dir / "config.json").exists()
    assert (logger_facade.run_dir / "run.log").exists()

    # Log an epoch
    logger_facade.log_epoch(
        epoch=1,
        loss_total=0.5,
        loss_fidelity=0.3,
        loss_efficiency=0.2,
        active_efficiency=0.03,
        field_fidelity=0.85,
        lr=0.01,
        log_frequency=1,
    )

    # Save model checkpoint
    model = OpticalD2NN(grid_n=32, fov_size=4, num_layers=2)
    ckpt_path = logger_facade.save_checkpoint(model, filename="phases_final.pt")
    assert ckpt_path.exists()

    # Benchmark and summary
    bench_results = logger_facade.run_benchmark(model)
    summary = logger_facade.generate_summary(benchmark_results=bench_results)

    assert (logger_facade.run_dir / "experiment_summary.json").exists()
    assert "hardware" in summary
    assert "pytorch_version" in summary["hardware"]
    assert "benchmark_results" in summary
    assert summary["benchmark_results"]["frobenius_error"] == bench_results.frobenius_error
