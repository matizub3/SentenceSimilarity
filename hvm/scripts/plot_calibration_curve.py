#!/usr/bin/env python3
"""
Plot calibration curve (reliability diagram) for baseline predictions.

Assumptions:
  - test_pred_mean represents a probability-like score for positive class.
  - y_test_true / y_test_N can be converted to binary labels.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate calibration curve (reliability diagram)."
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
        help="Leaderboard used when --run-dir is omitted.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="1-indexed rank in leaderboard used when --run-dir is omitted.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of equal-width probability bins.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold to convert y_true to binary when labels are not already {0,1}.",
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


def load_targets(npz_data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "y_test_true" in npz_data.files:
        return np.asarray(npz_data["y_test_true"]).reshape(-1)
    if "y_test_N" in npz_data.files:
        return np.asarray(npz_data["y_test_N"]).reshape(-1)
    raise RuntimeError(
        "predictions.npz missing target key. Expected one of ['y_test_true', 'y_test_N']."
    )


def to_binary_labels(y_true: np.ndarray, threshold: float) -> np.ndarray:
    uniq = np.unique(y_true)
    if np.all(np.isin(uniq, [0.0, 1.0])):
        return y_true.astype(int)
    return (y_true >= threshold).astype(int)


def compute_calibration(prob: np.ndarray, y_bin: np.ndarray, n_bins: int):
    prob = np.clip(prob, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(prob, edges[1:-1], right=False)

    x_mean = []
    y_freq = []
    counts = []

    for b in range(n_bins):
        mask = bin_ids == b
        c = int(mask.sum())
        if c == 0:
            continue
        x_mean.append(float(prob[mask].mean()))
        y_freq.append(float(y_bin[mask].mean()))
        counts.append(c)

    x_mean = np.asarray(x_mean)
    y_freq = np.asarray(y_freq)
    counts = np.asarray(counts)
    weights = counts / counts.sum()

    ece = float(np.sum(weights * np.abs(y_freq - x_mean)))
    brier = float(np.mean((prob - y_bin) ** 2))
    return x_mean, y_freq, counts, ece, brier


def main() -> None:
    args = parse_args()
    hparam_dir = args.hparam_dir.resolve()
    out_dir = (args.out_dir or (hparam_dir / "plots")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = choose_run_dir(args, hparam_dir)
    pred_path = hparam_dir / run_dir / "predictions.npz"
    if not pred_path.exists():
        raise RuntimeError(f"Missing predictions file: {pred_path}")

    data = np.load(pred_path)
    if "test_pred_mean" not in data.files:
        raise RuntimeError(
            f"predictions.npz missing 'test_pred_mean'. Found keys: {list(data.files)}"
        )

    y_prob = np.asarray(data["test_pred_mean"]).reshape(-1)
    y_true = load_targets(data)
    n = min(len(y_prob), len(y_true))
    y_prob = y_prob[:n]
    y_true = y_true[:n]
    y_bin = to_binary_labels(y_true, args.threshold)

    x_mean, y_freq, counts, ece, brier = compute_calibration(y_prob, y_bin, args.bins)

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 5.8))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5, label="Perfect calibration")
    ax.plot(x_mean, y_freq, marker="o", linewidth=2.0, color="tab:blue", label="Model")

    for x, y, c in zip(x_mean, y_freq, counts):
        ax.annotate(str(int(c)), (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability (bin mean)")
    ax.set_ylabel("Observed positive frequency")
    ax.set_title(
        f"Calibration Curve\n{run_dir}\nECE={ece:.4f}, Brier={brier:.4f}, bins={args.bins}",
        fontsize=10,
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    out_path = out_dir / f"calibration_curve_{run_dir}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"Wrote: {out_path}")
    print(f"ECE={ece:.6f} Brier={brier:.6f} samples={n}")


if __name__ == "__main__":
    main()
