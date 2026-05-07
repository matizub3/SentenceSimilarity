#!/usr/bin/env python3
"""
Plot training curves for one or more runs:
  - iteration vs valid_rmse
  - iteration vs train_elbo
on the same figure using dual y-axes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train ELBO + valid RMSE for specific runs.")
    parser.add_argument(
        "--hparam-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "hparam_results",
        help="Directory containing run folders.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Run directory name under hparam_results. Pass multiple times for multiple runs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: <hparam-dir>/plots).",
    )
    return parser.parse_args()


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def read_history(path: Path) -> dict[str, list[float]]:
    out = {"iter": [], "train_elbo": [], "valid_rmse": []}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out["iter"].append(float(row["iter"]))
            out["train_elbo"].append(float(row["train_elbo"]))
            out["valid_rmse"].append(float(row["valid_rmse"]))
    return out


def plot_one(run_dir: str, history: dict[str, list[float]], out_path: Path) -> None:
    fig, ax1 = plt.subplots(1, 1, figsize=(8.0, 4.8))
    ax2 = ax1.twinx()

    line_rmse = ax1.plot(
        history["iter"],
        history["valid_rmse"],
        color="tab:blue",
        linewidth=2.2,
        label="valid_rmse",
    )[0]
    line_elbo = ax2.plot(
        history["iter"],
        history["train_elbo"],
        color="tab:orange",
        linewidth=2.2,
        label="train_elbo",
    )[0]

    ax1.set_title(f"Training Curves: {run_dir}", fontsize=10)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Valid RMSE", color="tab:blue")
    ax2.set_ylabel("Train ELBO", color="tab:orange")
    ax1.grid(alpha=0.25)

    lines = [line_rmse, line_elbo]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    hparam_dir = args.hparam_dir.resolve()
    out_dir = (args.out_dir or (hparam_dir / "plots")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for run_dir in args.run_dir:
        history_path = hparam_dir / run_dir / "history.csv"
        if not history_path.exists():
            raise RuntimeError(f"Missing history.csv for run: {run_dir}")

        history = read_history(history_path)
        out_path = out_dir / f"train_curves_{safe_name(run_dir)}.png"
        plot_one(run_dir, history, out_path)
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
