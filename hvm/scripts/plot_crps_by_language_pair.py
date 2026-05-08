#!/usr/bin/env python3
"""
Mean CRPS per language pair from predictions.npz + feature file pair labels.

Uses test_global_row_index to align each test prediction with sts17_octen_features.npz
rows and their pair_N. Falls back to stratified_train_valid_test_split if index missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HVM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HVM_ROOT))

from baseline_setup import stratified_train_valid_test_split


def load_targets(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    if "y_test_true" in npz.files:
        return np.asarray(npz["y_test_true"], dtype=np.float64).reshape(-1)
    return np.asarray(npz["y_test_N"], dtype=np.float64).reshape(-1)


def orient_samples(samples: np.ndarray, n_targets: int) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 2:
        raise RuntimeError(f"test_pred_samples must be 2D; got {arr.shape}")
    if arr.shape[1] == n_targets:
        return arr.T
    if arr.shape[0] == n_targets:
        return arr
    raise RuntimeError(
        f"Cannot align test_pred_samples shape {arr.shape} with n_targets={n_targets}."
    )


def crps_per_example(samples_NS: np.ndarray, y_N: np.ndarray) -> np.ndarray:
    _, n_samples = samples_NS.shape
    if n_samples < 2:
        raise RuntimeError("Need at least 2 predictive samples per example.")

    term1 = np.mean(np.abs(samples_NS - y_N[:, None]), axis=1)
    diffs = np.abs(samples_NS[:, :, None] - samples_NS[:, None, :])
    term2 = 0.5 * np.mean(diffs, axis=(1, 2))
    return term1 - term2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot mean CRPS by language pair.")
    p.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Path to predictions.npz",
    )
    p.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run folder under hparam_results (alternative to --predictions).",
    )
    p.add_argument(
        "--hparam-dir",
        type=Path,
        default=HVM_ROOT / "hparam_results",
    )
    p.add_argument(
        "--feature-file",
        type=Path,
        default=HVM_ROOT / "sts17_octen_features.npz",
        help="Must match training; provides pair_N.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path (default: plots/crps_by_pair_<stem>.png).",
    )
    p.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="CSV summary path (default: same stem as PNG, .csv).",
    )
    p.add_argument(
        "--no-csv",
        action="store_true",
        help="Do not write CSV summary.",
    )
    return p.parse_args()


def resolve_pred_path(args: argparse.Namespace) -> Path:
    if args.predictions is not None:
        return args.predictions.resolve()
    if args.run_dir is None:
        raise SystemExit("Provide --predictions or --run-dir.")
    return (args.hparam_dir.resolve() / args.run_dir / "predictions.npz").resolve()


def read_seed(predictions_path: Path) -> int:
    rp = predictions_path.parent / "results.json"
    if rp.is_file():
        with rp.open() as f:
            d = json.load(f)
        if d.get("seed") is not None:
            return int(d["seed"])
    return 101


def main() -> None:
    args = parse_args()
    pred_path = resolve_pred_path(args)
    feat_path = args.feature_file.resolve()

    if not pred_path.is_file():
        raise RuntimeError(f"Missing predictions: {pred_path}")
    if not feat_path.is_file():
        raise RuntimeError(f"Missing features: {feat_path}")

    with np.load(feat_path, allow_pickle=True) as feats:
        n_rows = int(feats["x_ND"].shape[0])
        pair_N = feats["pair_N"] if "pair_N" in feats.files else np.array(["unknown"] * n_rows)

    with np.load(pred_path, allow_pickle=True) as pred:
        if "test_pred_samples" not in pred.files:
            raise RuntimeError(f"{pred_path} missing test_pred_samples.")
        y = load_targets(pred)
        samples = orient_samples(pred["test_pred_samples"], len(y))
        if "test_global_row_index" in pred.files:
            gidx = np.asarray(pred["test_global_row_index"], dtype=np.int64).reshape(-1)
        else:
            seed = read_seed(pred_path)
            _, _, test_ids = stratified_train_valid_test_split(pair_N, seed=seed)
            gidx = np.asarray(test_ids, dtype=np.int64).reshape(-1)

    n = min(len(y), samples.shape[0], len(gidx))
    y, samples, gidx = y[:n], samples[:n], gidx[:n]
    crps = crps_per_example(samples, y)
    pairs = np.array([str(pair_N[int(gi)]) for gi in gidx])

    unique_pairs = sorted(np.unique(pairs))
    rows = []
    means = []
    counts = []

    for p in unique_pairs:
        mask = pairs == p
        m = float(np.mean(crps[mask]))
        c = int(mask.sum())
        rows.append({"language_pair": p, "n": c, "mean_crps": m, "std_crps": float(np.std(crps[mask]))})
        means.append(m)
        counts.append(c)

    stem = pred_path.parent.name
    plots_dir = args.hparam_dir.resolve() / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_png = (
        args.out.resolve()
        if args.out
        else plots_dir / f"crps_by_language_pair_{stem}.png"
    )

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.35 * len(unique_pairs))))
    order = np.argsort(means)
    y_pos = np.arange(len(unique_pairs))
    sorted_pairs = [unique_pairs[i] for i in order]
    sorted_means = [means[i] for i in order]
    sorted_counts = [counts[i] for i in order]

    bars = ax.barh(y_pos, sorted_means, color="steelblue", edgecolor="navy", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{p} (n={c})" for p, c in zip(sorted_pairs, sorted_counts)])
    ax.set_xlabel("Mean CRPS (lower is better)")
    ax.set_title("CRPS by language pair")
    ax.grid(axis="x", alpha=0.35)
    for i, (bar, m) in enumerate(zip(bars, sorted_means)):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{m:.4f}",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    print(f"Wrote {out_png}")
    print(f"n_test examples: {n}")
    for r in sorted(rows, key=lambda x: x["mean_crps"], reverse=False):
        print(f"  {r['language_pair']}: mean_crps={r['mean_crps']:.6f}, n={r['n']}")

    if not args.no_csv:
        import csv as csv_mod

        csv_path = (
            args.csv_out.resolve()
            if args.csv_out is not None
            else out_png.with_suffix(".csv")
        )
        with csv_path.open("w", newline="") as fp:
            w = csv_mod.DictWriter(
                fp,
                fieldnames=["language_pair", "n", "mean_crps", "std_crps"],
            )
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
