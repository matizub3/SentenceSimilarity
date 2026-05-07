#!/usr/bin/env python3
"""
Plot test residuals (y_true - y_pred_mean) versus predicted posterior mean.

By default uses a pinned best baseline run directory under hparam_results.
Pass --from-leaderboard to select via leaderboard CSV instead.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_RUN_DIR = (
    "arch_100_iters_2000_bs_64_mc_50_lr_0p001_prior_0p5_lik_0p05_qstd_0p05_sigmoid_seed_101"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Residuals vs predicted mean on the test set."
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
        help=(
            "Run directory under --hparam-dir. "
            "If omitted (and not using --from-leaderboard), uses the pinned default "
            "best run (see DEFAULT_RUN_DIR in this script)."
        ),
    )
    parser.add_argument(
        "--from-leaderboard",
        action="store_true",
        help="Select run from leaderboard (--leaderboard, --rank) instead of the default pinned run.",
    )
    parser.add_argument(
        "--leaderboard",
        type=str,
        default="leaderboard_top10.csv",
        help="Leaderboard file used with --from-leaderboard.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="1-indexed rank in leaderboard used with --from-leaderboard.",
    )
    parser.add_argument(
        "--trend-bins",
        type=int,
        default=15,
        help="Equal-width bins on predicted mean for a smoothed mean residual curve "
        "(0 disables).",
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
    if args.from_leaderboard:
        rows = read_rows(hparam_dir / args.leaderboard)
        if len(rows) == 0:
            raise RuntimeError(f"Leaderboard is empty: {args.leaderboard}")
        idx = args.rank - 1
        if idx < 0 or idx >= len(rows):
            raise RuntimeError(
                f"Requested rank {args.rank} outside leaderboard range 1..{len(rows)}"
            )
        return rows[idx]["run_dir"]

    if args.run_dir:
        return args.run_dir

    return DEFAULT_RUN_DIR


def load_targets(npz_data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "y_test_true" in npz_data.files:
        return np.asarray(npz_data["y_test_true"], dtype=np.float64).reshape(-1)
    if "y_test_N" in npz_data.files:
        return np.asarray(npz_data["y_test_N"], dtype=np.float64).reshape(-1)
    raise RuntimeError(
        "predictions.npz missing target key. Expected one of "
        "['y_test_true', 'y_test_N']."
    )


def binned_mean_residual(
    y_pred: np.ndarray, residual: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-width bins on y_pred; return bin centers and mean residual per bin."""
    lo, hi = float(np.min(y_pred)), float(np.max(y_pred))
    if hi <= lo:
        return np.array([]), np.array([])
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (y_pred >= edges[i]) & (y_pred < edges[i + 1])
        if i == n_bins - 1:
            mask = (y_pred >= edges[i]) & (y_pred <= edges[i + 1])
        if np.any(mask):
            means[i] = float(np.mean(residual[mask]))
    valid = ~np.isnan(means)
    return centers[valid], means[valid]


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
            "predictions.npz missing 'test_pred_mean'. "
            f"Found keys: {list(data.files)}"
        )

    y_pred = np.asarray(data["test_pred_mean"], dtype=np.float64).reshape(-1)
    y_true = load_targets(data)

    n = min(len(y_pred), len(y_true))
    y_pred = y_pred[:n]
    y_true = y_true[:n]
    residual = y_true - y_pred

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.2))
    ax.axhline(0.0, color="gray", linewidth=1.0, linestyle="--", zorder=1)
    ax.scatter(
        y_pred,
        residual,
        s=12,
        alpha=0.35,
        color="tab:blue",
        edgecolors="none",
        zorder=2,
        label="Test points",
    )

    if args.trend_bins > 0:
        cx, cy = binned_mean_residual(y_pred, residual, args.trend_bins)
        if len(cx) > 0:
            ax.plot(
                cx,
                cy,
                color="darkred",
                linewidth=2.0,
                zorder=3,
                label=f"Binned mean ({args.trend_bins} bins)",
            )

    ax.set_title(f"Residuals vs predicted mean\n{run_dir}", fontsize=10)
    ax.set_xlabel("Posterior predictive mean (test)")
    ax.set_ylabel("Residual (y_true − pred_mean)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    out_path = out_dir / f"residuals_vs_pred_{run_dir}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
