#!/usr/bin/env python3
"""
Plot baseline test predictions with posterior uncertainty error bars.

Points are ordered by ground-truth similarity (low to high). By default every
test point is plotted (--max-points 0); pass a positive cap to subsample.
Overlays a linear fit to predicted means vs rank and a band equal to that fit
± smoothed local epistemic standard deviation along the rank axis.
Default behavior uses the #1 run from leaderboard_top10.csv.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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
        default=0,
        metavar="N",
        help="Plot at most N test points after sorting; use 0 for all (default).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <hparam-dir>/plots).",
    )
    parser.add_argument(
        "--fit-std-bins",
        type=int,
        default=None,
        metavar="B",
        help="Bins along rank for smoothing mean(test_pred_std) around the linear "
        "fit band (default: ~max(8, n_points//25), capped).",
    )
    parser.add_argument(
        "--no-prediction-fit",
        action="store_true",
        help="Do not draw the linear fit or ±std band on predictions.",
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


def downsample_sorted_positions(n: int, max_points: int) -> np.ndarray:
    """Indices along an array already sorted by ground truth (0 .. n-1)."""
    if n <= max_points:
        return np.arange(n)
    return np.linspace(0, n - 1, max_points).astype(int)


def binned_mean_on_axis(
    axis_vals: np.ndarray, values: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-width bins on axis_vals; mean(values) per bin (drops empty bins)."""
    axis_vals = np.asarray(axis_vals, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    lo, hi = float(np.min(axis_vals)), float(np.max(axis_vals))
    if hi <= lo:
        return np.array([lo]), np.array([float(np.mean(values))])
    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (axis_vals >= edges[i]) & (axis_vals < edges[i + 1])
        if i == n_bins - 1:
            mask = (axis_vals >= edges[i]) & (axis_vals <= edges[i + 1])
        if np.any(mask):
            means[i] = float(np.mean(values[mask]))
    valid = ~np.isnan(means)
    return centers[valid], means[valid]


def default_fit_std_bins(n_points: int) -> int:
    if n_points <= 1:
        return 1
    return int(max(8, min(48, n_points // 25)))


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

    sort_order = np.argsort(y_true, kind="stable")
    y_pred = y_pred[sort_order]
    y_std = y_std[sort_order]
    y_true = y_true[sort_order]

    if args.max_points < 0:
        raise ValueError("--max-points must be >= 0 (use 0 for all test points)")
    if args.fit_std_bins is not None and args.fit_std_bins < 1:
        raise ValueError("--fit-std-bins must be >= 1 when provided")
    max_points = n if args.max_points == 0 else args.max_points
    keep = downsample_sorted_positions(n, max_points)
    x = np.arange(len(keep))
    y_pred_k = y_pred[keep]
    y_std_k = y_std[keep]
    y_true_k = y_true[keep]

    mpts = len(keep)
    fig_w = max(11.0, min(40.0, 0.052 * mpts + 7.0))
    mrk = float(max(1.1, min(3.2, 380.0 / max(mpts, 1))))
    ecap = float(max(0.4, min(2.0, 140.0 / max(mpts, 1))))
    elw = float(max(0.35, min(1.0, 120.0 / max(mpts, 1))))

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, 5.5))

    x_f = x.astype(np.float64)
    if (
        not args.no_prediction_fit
        and len(y_pred_k) >= 2
        and (np.max(x_f) - np.min(x_f)) > 0
    ):
        n_bins = (
            args.fit_std_bins
            if args.fit_std_bins is not None
            else default_fit_std_bins(len(x))
        )
        n_bins = max(1, int(n_bins))

        coef = np.polyfit(x_f, y_pred_k.astype(np.float64), 1)
        y_fit = np.polyval(coef, x_f).astype(np.float64)

        cx, mean_std = binned_mean_on_axis(x_f, y_std_k.astype(np.float64), n_bins)
        if len(cx) == 0:
            sigma_line = np.full_like(x_f, float(np.mean(y_std_k)))
        else:
            sigma_line = np.interp(x_f, cx, mean_std)

        ax.fill_between(
            x_f,
            y_fit - sigma_line,
            y_fit + sigma_line,
            alpha=0.22,
            color="tab:green",
            zorder=1,
            label="Linear fit ± smoothed epistemic σ",
        )
        ax.plot(
            x_f,
            y_fit,
            color="darkgreen",
            linewidth=2.0,
            linestyle="-",
            zorder=2,
            label="Linear fit (predictions vs rank)",
        )

    ax.errorbar(
        x,
        y_pred_k,
        yerr=y_std_k,
        fmt="o",
        markersize=mrk,
        linewidth=0.8,
        elinewidth=elw,
        capsize=ecap,
        alpha=0.78,
        color="tab:blue",
        zorder=3,
        label="Prediction ±1 std",
    )
    ax.scatter(
        x,
        y_true_k,
        s=max(8.0, mrk * 4.2),
        alpha=0.65,
        color="tab:orange",
        zorder=4,
        label="Ground truth",
    )
    ax.set_title(
        f"Test Predictions with Posterior Uncertainty\n{run_dir}",
        fontsize=10,
    )
    ax.set_xlabel("Rank along test set sorted by ground truth (low → high)")
    ax.set_ylabel("Similarity score")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    out_path = out_dir / f"pred_errorbars_{run_dir}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"Plotted {mpts} test point(s) (of {n} in predictions). Wrote: {out_path}")


if __name__ == "__main__":
    main()
