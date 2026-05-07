#!/usr/bin/env python3
"""
Summarize baseline test performance by STS language pair (MAE, RMSE, counts).

Loads the same predictions.npz + feature npz alignment as export_test_predictions_with_sentences.py.
Writes a CSV table and a bar-chart figure (MAE and RMSE per pair).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HVM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HVM_ROOT))

from baseline_setup import stratified_train_valid_test_split

LANG_PAIRS = [
    "ar-ar",
    "en-ar",
    "en-de",
    "en-en",
    "en-tr",
    "es-en",
    "es-es",
    "fr-en",
    "it-en",
    "nl-en",
    "ko-ko",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per language-pair test MAE/RMSE table + bar chart."
    )
    p.add_argument(
        "--feature-file",
        type=Path,
        default=HVM_ROOT / "sts17_octen_features.npz",
        help="Feature .npz used when training the run.",
    )
    p.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Path to predictions.npz (default: <hparam-dir>/<run-dir>/predictions.npz).",
    )
    p.add_argument(
        "--hparam-dir",
        type=Path,
        default=HVM_ROOT / "hparam_results",
        help="Used with --run-dir when --predictions is omitted.",
    )
    p.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run folder under hparam-dir.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Split seed if predictions.npz lacks test_global_row_index "
        "(else from results.json, else 101).",
    )
    p.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Output CSV (default: cwd test_perf_by_language_pair_<run-name>.csv).",
    )
    p.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="Output PNG (default: same stem as CSV with .png).",
    )
    p.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing CSV (still prints table and can plot).",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip writing the figure.",
    )
    return p.parse_args()


def resolve_predictions_path(args: argparse.Namespace) -> Path:
    if args.predictions is not None:
        return args.predictions.resolve()
    if args.run_dir is None:
        raise SystemExit("Provide --predictions or --run-dir (optional --hparam-dir).")
    return (args.hparam_dir.resolve() / args.run_dir / "predictions.npz").resolve()


def read_seed_from_results(predictions_path: Path) -> Optional[int]:
    results_path = predictions_path.parent / "results.json"
    if not results_path.is_file():
        return None
    with results_path.open() as f:
        data = json.load(f)
    seed = data.get("seed")
    return int(seed) if seed is not None else None


def load_test_prediction_frame(
    feature_path: Path,
    pred_path: Path,
    seed_arg: Optional[int],
) -> pd.DataFrame:
    feats = np.load(feature_path, allow_pickle=True)
    n_rows = int(feats["x_ND"].shape[0])
    pair_N = feats["pair_N"] if "pair_N" in feats.files else np.array(["unknown"] * n_rows)
    y_scaled = (feats["y_raw_N"].astype(np.float64) / 5.0).reshape(-1)

    pred = np.load(pred_path, allow_pickle=True)
    y_hat = np.asarray(pred["test_pred_mean"], dtype=np.float64).reshape(-1)
    y_sd = np.asarray(pred["test_pred_std"], dtype=np.float64).reshape(-1)
    if "y_test_N" in pred.files:
        y_obs = np.asarray(pred["y_test_N"], dtype=np.float64).reshape(-1)
    elif "y_test_true" in pred.files:
        y_obs = np.asarray(pred["y_test_true"], dtype=np.float64).reshape(-1)
    else:
        raise RuntimeError(f"No y_test labels in {pred_path}")

    if "test_global_row_index" in pred.files:
        global_idx = np.asarray(pred["test_global_row_index"], dtype=np.int64).reshape(-1)
    else:
        seed = seed_arg if seed_arg is not None else read_seed_from_results(pred_path)
        if seed is None:
            seed = 101
        _, _, test_ids = stratified_train_valid_test_split(pair_N, seed=seed)
        global_idx = np.asarray(test_ids, dtype=np.int64).reshape(-1)

    if not (len(y_hat) == len(y_sd) == len(y_obs) == len(global_idx)):
        raise RuntimeError("Length mismatch between predictions and index vector.")

    y_from_table = y_scaled[global_idx]
    if np.max(np.abs(y_from_table - y_obs)) > 1e-4:
        raise RuntimeError(
            "Label mismatch vs feature npz at row indices — check --feature-file / seed."
        )

    rows = []
    for i in range(len(global_idx)):
        gid = int(global_idx[i])
        rows.append(
            {
                "language_pair": str(pair_N[gid]),
                "y_true": float(y_obs[i]),
                "y_pred_mean": float(y_hat[i]),
                "y_pred_std": float(y_sd[i]),
            }
        )
    df = pd.DataFrame(rows)
    df["abs_err"] = (df["y_true"] - df["y_pred_mean"]).abs()
    df["sq_err"] = (df["y_true"] - df["y_pred_mean"]) ** 2
    return df


def aggregate_by_pair(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair, sub in df.groupby("language_pair", sort=False):
        err = sub["y_true"].to_numpy(dtype=np.float64) - sub["y_pred_mean"].to_numpy(
            dtype=np.float64
        )
        rows.append(
            {
                "language_pair": pair,
                "n_test": int(len(sub)),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "bias": float(np.mean(err)),
                "mean_pred_std": float(np.mean(sub["y_pred_std"].to_numpy())),
            }
        )
    return pd.DataFrame(rows)


def sort_pairs_by_lang_order(df_summary: pd.DataFrame) -> pd.DataFrame:
    present = set(df_summary["language_pair"].astype(str))
    ordered = [p for p in LANG_PAIRS if p in present]
    ordered += sorted(present - set(ordered))
    cat = pd.Categorical(df_summary["language_pair"], categories=ordered, ordered=True)
    return (
        df_summary.assign(_k=cat)
        .sort_values("_k")
        .drop(columns="_k")
        .reset_index(drop=True)
    )


def overall_row(df: pd.DataFrame) -> dict:
    err = df["y_true"].to_numpy(dtype=np.float64) - df["y_pred_mean"].to_numpy(
        dtype=np.float64
    )
    return {
        "language_pair": "ALL (full test)",
        "n_test": int(len(df)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "bias": float(np.mean(err)),
        "mean_pred_std": float(np.mean(df["y_pred_std"].to_numpy())),
    }


def plot_mae_rmse(summary_pairs: pd.DataFrame, title: str, out_path: Path) -> None:
    pairs = summary_pairs["language_pair"].tolist()
    x = np.arange(len(pairs))
    fig_w = max(10.0, 0.72 * len(pairs) + 4.0)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 5.4))

    axes[0].bar(x, summary_pairs["mae"], color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(pairs, rotation=45, ha="right")
    axes[0].set_ylabel("MAE")
    axes[0].set_title("Mean absolute error (test)")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, summary_pairs["rmse"], color="coral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pairs, rotation=45, ha="right")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("RMSE (test)")
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    feature_path = args.feature_file.resolve()
    pred_path = resolve_predictions_path(args)

    if not feature_path.is_file():
        raise RuntimeError(f"Missing feature file: {feature_path}")
    if not pred_path.is_file():
        raise RuntimeError(f"Missing predictions file: {pred_path}")

    df = load_test_prediction_frame(feature_path, pred_path, args.seed)
    by_pair = sort_pairs_by_lang_order(aggregate_by_pair(df))
    overall = pd.DataFrame([overall_row(df)])
    table_out = pd.concat([by_pair, overall], ignore_index=True)

    pd.set_option("display.max_rows", 30)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda v: f"{v:.5f}")
    print("\nTest performance by language_pair:\n")
    print(table_out.to_string(index=False))

    run_label = pred_path.parent.name
    default_stem = f"test_perf_by_language_pair_{run_label}"

    csv_path = args.csv_out
    if csv_path is None:
        csv_path = Path.cwd() / f"{default_stem}.csv"
    else:
        csv_path = csv_path.resolve()

    plot_path = args.plot_out
    if plot_path is None:
        plot_path = csv_path.with_suffix(".png")
    else:
        plot_path = plot_path.resolve()

    if not args.no_csv:
        table_out.to_csv(csv_path, index=False)
        print(f"\nWrote table CSV: {csv_path}")

    if not args.no_plot:
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_mae_rmse(
            by_pair,
            title=f"Test error by language pair\n{run_label}",
            out_path=plot_path,
        )
        print(f"Wrote plot: {plot_path}")


if __name__ == "__main__":
    main()