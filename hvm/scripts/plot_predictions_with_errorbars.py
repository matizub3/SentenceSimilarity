#!/usr/bin/env python3
"""
Plot baseline test predictions with posterior uncertainty error bars.

Default behavior uses the #1 run from leaderboard_top10.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot test predictions with +/- one posterior stddev error bars."
    )
    parser.add_argument(
        "--hparam-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "hparam_results",
        help="Directory containing run folders and leaderboard CSVs.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Specific run directory name under hparam_results (optional).",
    )
    parser.add_argument(
        "--leaderboard",
        type=str,
        default="leaderboard_top10.csv",
        help="Leaderboard file used to pick default run when --run-dir is omitted.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="1-indexed rank in leaderboard used when --run-dir is omitted.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=120,
        help="Maximum number of test points shown for readability.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <hparam-dir>/plots).",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def choose_run_dir(args: argparse.Namespace, hparam_dir: Path) -> str:
    if args.run_dir:
        return args.run_dir

    rows = read_rows(hparam_dir / args.leaderboard)
    if len(rows) == 0:
        raise RuntimeError(f"Leaderboard is empty: {args.leaderboard}")
    idx = args.rank - 1
    if idx < 0 or idx >= len(rows):
        raise RuntimeError(f"Requested rank {args.rank} outside leaderboard range 1..{len(rows)}")
    return rows[idx]["run_dir"]


def downsample_indices(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points).astype(int)


def main() -> None:
    args = parse_args()
    hparam_dir = args.hparam_dir.resolve()
    out_dir = (args.out_dir or (hparam_dir / "plots")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = choose_run_dir(args, hparam_dir)
    run_path = hparam_dir / run_dir
    pred_path = run_path / "predictions.npz"

    if not pred_path.exists():
        raise RuntimeError(f"Missing predictions file: {pred_path}")

    data = np.load(pred_path)
    required = ["test_pred_mean", "test_pred_std"]
    missing = [k for k in required if k not in data.files]
    if missing:
        raise RuntimeError(
            f"predictions.npz missing required keys {missing}. Found keys: {list(data.files)}"
        )
    if "y_test_true" in data.files:
        y_true_key = "y_test_true"
    elif "y_test_N" in data.files:
        y_true_key = "y_test_N"
    else:
        raise RuntimeError(
            "predictions.npz missing target key. Expected one of "
            "['y_test_true', 'y_test_N']. "
            f"Found keys: {list(data.files)}"
        )

    y_pred = np.asarray(data["test_pred_mean"]).reshape(-1)
    y_std = np.asarray(data["test_pred_std"]).reshape(-1)
    y_true = np.asarray(data[y_true_key]).reshape(-1)

    n = min(len(y_pred), len(y_std), len(y_true))
    y_pred = y_pred[:n]
    y_std = y_std[:n]
    y_true = y_true[:n]

    keep = downsample_indices(n, args.max_points)
    x = np.arange(n)[keep]
    y_pred_k = y_pred[keep]
    y_std_k = y_std[keep]
    y_true_k = y_true[keep]

    fig, ax = plt.subplots(1, 1, figsize=(11, 5.2))
    ax.errorbar(
        x,
        y_pred_k,
        yerr=y_std_k,
        fmt="o",
        markersize=3.0,
        linewidth=1.0,
        elinewidth=1.0,
        capsize=2.0,
        alpha=0.85,
        color="tab:blue",
        label="Prediction ±1 std",
    )
    ax.scatter(
        x,
        y_true_k,
        s=14,
        alpha=0.7,
        color="tab:orange",
        label="Ground truth",
    )
    ax.set_title(
        f"Test Predictions with Posterior Uncertainty\n{run_dir}",
        fontsize=10,
    )
    ax.set_xlabel("Test sample index")
    ax.set_ylabel("Similarity score")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    out_path = out_dir / f"pred_errorbars_{run_dir}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
