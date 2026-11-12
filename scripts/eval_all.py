"""Batch evaluation script across all trained models (A, B, C).

Iteratively resolves the latest checkpoints for Model A, Model B, and Model C,
running src.evaluate with figure export to outputs/figures/.
"""

import glob
import os
from pathlib import Path
import subprocess
import sys

MODELS = [
    {
        "name": "Model A (Phase-Only Baseline)",
        "patterns": ["*model_a*", "*baseline*", "*coopt*"],
        "config": "configs/default.yaml",
    },
    {
        "name": "Model B (Complex Modulation)",
        "patterns": ["*model_b*", "*complex*"],
        "config": "configs/complex_modulation.yaml",
    },
    {
        "name": "Model C (Apertured Cascade)",
        "patterns": ["*model_c*", "*apertured*"],
        "config": "configs/apertured_phase.yaml",
    },
]


def resolve_latest_checkpoint(patterns: list[str]) -> Path | None:
    """Find the most recent checkpoint matching any of the glob patterns."""
    matches: list[str] = []
    for pat in patterns:
        matches.extend(glob.glob(f"outputs/runs/{pat}/phases_final.pt"))
    if not matches:
        return None
    sorted_matches = sorted(list(set(matches)), key=os.path.getmtime)
    return Path(sorted_matches[-1])


def main() -> None:
    print("=" * 70)
    print("D2NN Multi-Model Batch Evaluation Dispatcher")
    print("=" * 70)

    evaluated = 0
    for model_info in MODELS:
        name = model_info["name"]
        patterns = model_info["patterns"]
        config = model_info["config"]

        ckpt = resolve_latest_checkpoint(patterns)
        if ckpt is None or not ckpt.exists():
            print(f"[SKIP] No checkpoint found for {name} (patterns: {patterns})")
            continue

        print(f"\nEvaluating {name}...")
        print(f"  Checkpoint: {ckpt}")
        print(f"  Config:     {config}")

        cmd = [
            sys.executable,
            "-m",
            "src.evaluate",
            "--checkpoint",
            str(ckpt),
            "--config",
            config,
            "--export-figures",
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"[ERROR] Evaluation failed for {name} with code {res.returncode}")
            sys.exit(res.returncode)
        evaluated += 1

    print("\n" + "=" * 70)
    print(f"Batch evaluation complete. Successfully evaluated {evaluated} models.")
    print("=" * 70)


if __name__ == "__main__":
    main()
