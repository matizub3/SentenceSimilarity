#!/usr/bin/env python3
"""
Plot top-k model training curves from leaderboard CSVs.

Generates:
  - topk_rmse_models.png  (top-k by test RMSE)
  - topk_elbo_models.png  (top-k by best ELBO)

Each panel overlays validation RMSE and train ELBO over iterations
using dual y-axes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot top-k runs from RMSE and ELBO leaderboards."
    )
    parser.add_argument(
        "--hparam-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "hparam_results",
        help="Directory containing leaderboard CSVs and run folders.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top models to plot for each leaderboard.",
    )
    parser.add_argument(
        "--rmse-leaderboard",
        type=str,
        default="leaderboard_top10.csv",
        help="RMSE leaderboard filename inside hparam-dir.",
    )
    parser.add_argument(
        "--elbo-leaderboard",
        type=str,
        default="leaderboard_elbo_top10.csv",
        help="ELBO leaderboard filename inside hparam-dir.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for figures (default: <hparam-dir>/plots).",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_history(path: Path) -> dict[str, list[float]]:
    out = {"iter": [], "valid_rmse": [], "train_elbo": []}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out["iter"].append(float(row["iter"]))
            out["valid_rmse"].append(float(row["valid_rmse"]))
            out["train_elbo"].append(float(row["train_elbo"]))
    return out


def resolve_history_from_rmse_row(hparam_dir: Path, row: dict[str, str]) -> Path:
    results_rel = Path(row["results_json"])
    if not results_rel.is_absolute():
        # results_json is typically like: hvm/hparam_results/<run_dir>/results.json
        # trim prefix up to hparam_results to map under local hparam_dir.
        parts = list(results_rel.parts)
        if "hparam_results" in parts:
            idx = parts.index("hparam_results")
            results_rel = Path(*parts[idx + 1 :])
    return hparam_dir / results_rel.parent / "history.csv"


def resolve_history_from_elbo_row(hparam_dir: Path, row: dict[str, str]) -> Path:
    history_rel = Path(row["history_csv"])
    if not history_rel.is_absolute():
        parts = list(history_rel.parts)
        if "hparam_results" in parts:
            idx = parts.index("hparam_results")
            history_rel = Path(*parts[idx + 1 :])
    return hparam_dir / history_rel


def plot_group(
    rows: list[dict[str, str]],
    title: str,
    out_path: Path,
    history_resolver,
    rank_col: str,
) -> None:
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.6), squeeze=False)
    axes = axes[0]

    for i, row in enumerate(rows):
        ax1 = axes[i]
        ax2 = ax1.twinx()

        history_path = history_resolver(row)
        hist = read_history(history_path)

        rmse_line = ax1.plot(
            hist["iter"],
            hist["valid_rmse"],
            color="tab:blue",
            linewidth=2.0,
            label="valid_rmse",
        )[0]
        elbo_line = ax2.plot(
            hist["iter"],
            hist["train_elbo"],
            color="tab:orange",
            linewidth=2.0,
            label="train_elbo",
        )[0]

        rank = row.get(rank_col, str(i + 1))
        run_name = row["run_dir"]
        test_rmse = row.get("test_rmse", "NA")
        valid_rmse = row.get("valid_rmse", "NA")
        ax1.set_title(f"#{rank} {run_name}\nTest {test_rmse} | Valid {valid_rmse}", fontsize=9)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Valid RMSE", color="tab:blue")
        ax2.set_ylabel("Train ELBO", color="tab:orange")
        ax1.grid(alpha=0.25)

        lines = [rmse_line, elbo_line]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="best", fontsize=8)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    hparam_dir = args.hparam_dir.resolve()
    out_dir = (args.out_dir or (hparam_dir / "plots")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rmse_rows = read_csv_rows(hparam_dir / args.rmse_leaderboard)[: args.top_k]
    elbo_rows = read_csv_rows(hparam_dir / args.elbo_leaderboard)[: args.top_k]

    if len(rmse_rows) == 0 or len(elbo_rows) == 0:
        raise RuntimeError("One or both leaderboard files are empty.")

    plot_group(
        rmse_rows,
        title=f"Top {len(rmse_rows)} Models by Test RMSE",
        out_path=out_dir / f"top{len(rmse_rows)}_rmse_models.png",
        history_resolver=lambda row: resolve_history_from_rmse_row(hparam_dir, row),
        rank_col="rank_test_rmse",
    )

    plot_group(
        elbo_rows,
        title=f"Top {len(elbo_rows)} Models by Best ELBO",
        out_path=out_dir / f"top{len(elbo_rows)}_elbo_models.png",
        history_resolver=lambda row: resolve_history_from_elbo_row(hparam_dir, row),
        rank_col="rank_best_elbo",
    )

    print(f"Wrote plots to: {out_dir}")


if __name__ == "__main__":
    main()
