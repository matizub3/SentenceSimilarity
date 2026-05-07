#!/usr/bin/env python3
"""
Plot per-example predictive CDFs against the true CDF for continuous targets.

For each selected test example:
  - Approximate predictive CDF from posterior samples in predictions.npz
  - True CDF for point mass at y_true (step function)
  - Metadata in title: y_true, pred_mean, pred_std
  - Sentence pair text
"""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HVM_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot predictive CDFs for selected test examples.")
    p.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help="Path to predictions.npz. If omitted, use --run-dir under --hparam-dir.",
    )
    p.add_argument(
        "--hparam-dir",
        type=Path,
        default=HVM_ROOT / "hparam_results",
        help="Used with --run-dir if --predictions is omitted.",
    )
    p.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run folder name under hparam-dir.",
    )
    p.add_argument(
        "--feature-file",
        type=Path,
        default=HVM_ROOT / "sts17_octen_features.npz",
        help="Feature npz containing sentence1_N/sentence2_N and y_raw_N.",
    )
    p.add_argument("--n-examples", type=int, default=10, help="How many examples to plot.")
    p.add_argument(
        "--selection",
        type=str,
        choices=["random", "worst_error", "best_error", "first"],
        default="random",
        help="How to choose examples from test set.",
    )
    p.add_argument("--seed", type=int, default=101, help="Random seed for random selection.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <hparam-dir>/plots/cdf_examples_<run>).",
    )
    return p.parse_args()


def resolve_predictions_path(args: argparse.Namespace) -> Path:
    if args.predictions is not None:
        return args.predictions.resolve()
    if args.run_dir is None:
        raise SystemExit("Provide --predictions or --run-dir (with optional --hparam-dir).")
    return (args.hparam_dir.resolve() / args.run_dir / "predictions.npz").resolve()


def orient_samples(samples: np.ndarray, n_targets: int) -> np.ndarray:
    """Return samples as [N, S] where N=examples, S=samples per example."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 2:
        raise RuntimeError(f"test_pred_samples must be 2D; got {arr.shape}")
    if arr.shape[0] == n_targets:
        return arr
    if arr.shape[1] == n_targets:
        return arr.T
    raise RuntimeError(f"Cannot align sample shape {arr.shape} with n_targets={n_targets}")


def pick_indices(y_true: np.ndarray, y_pred: np.ndarray, n: int, mode: str, seed: int) -> np.ndarray:
    n_total = len(y_true)
    n = min(n, n_total)
    if mode == "first":
        return np.arange(n)
    err = np.abs(y_true - y_pred)
    if mode == "worst_error":
        return np.argsort(-err)[:n]
    if mode == "best_error":
        return np.argsort(err)[:n]
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=n, replace=False))


def wrap_text(s: str, width: int = 90) -> str:
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False))


def plot_one(
    out_path: Path,
    sample_vals: np.ndarray,
    y_true: float,
    y_pred: float,
    y_std: float,
    sent1: str,
    sent2: str,
    row_idx: int,
) -> None:
    xs = np.sort(sample_vals)
    s = len(xs)
    ys = np.arange(1, s + 1) / s

    grid = np.linspace(0.0, 1.0, 500)
    true_cdf = (grid >= y_true).astype(float)
    # Interpolate empirical CDF on a dense grid to visualize area error.
    pred_cdf_grid = np.searchsorted(xs, grid, side="right") / s

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 5.8))
    ax.fill_between(
        grid,
        pred_cdf_grid,
        true_cdf,
        color="tab:red",
        alpha=0.18,
        label="|CDF error| region",
    )
    ax.step(xs, ys, where="post", color="tab:blue", linewidth=2.0, label="Approx predictive CDF")
    ax.step(grid, true_cdf, where="post", color="tab:orange", linewidth=2.0, linestyle="--", label="True CDF (point mass)")
    ax.axvline(y_true, color="tab:orange", alpha=0.35, linewidth=1.2)
    ax.axvline(y_pred, color="tab:blue", alpha=0.35, linewidth=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Similarity value")
    ax.set_ylabel("CDF")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    ax.set_title(
        f"row={row_idx} | y_true={y_true:.4f} | pred_mean={y_pred:.4f} | pred_std={y_std:.4f}",
        fontsize=10,
    )

    txt = "Sentence 1: " + wrap_text(sent1, 95) + "\n\nSentence 2: " + wrap_text(sent2, 95)
    fig.text(0.02, 0.01, txt, ha="left", va="bottom", fontsize=8)

    fig.tight_layout(rect=[0, 0.20, 1, 1])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    pred_path = resolve_predictions_path(args)
    if not pred_path.is_file():
        raise RuntimeError(f"Missing predictions file: {pred_path}")

    feature_path = args.feature_file.resolve()
    if not feature_path.is_file():
        raise RuntimeError(f"Missing feature file: {feature_path}")

    with np.load(pred_path, allow_pickle=True) as pred:
        if "test_pred_samples" not in pred.files:
            raise RuntimeError(
                f"{pred_path} missing test_pred_samples. Re-run training with updated saving logic."
            )

        y_pred_mean = np.asarray(pred["test_pred_mean"], dtype=np.float64).reshape(-1)
        y_pred_std = np.asarray(pred["test_pred_std"], dtype=np.float64).reshape(-1)
        if "y_test_N" in pred.files:
            y_true = np.asarray(pred["y_test_N"], dtype=np.float64).reshape(-1)
        elif "y_test_true" in pred.files:
            y_true = np.asarray(pred["y_test_true"], dtype=np.float64).reshape(-1)
        else:
            raise RuntimeError(f"{pred_path} missing y_test_N / y_test_true")
        if "test_global_row_index" not in pred.files:
            raise RuntimeError(
                f"{pred_path} missing test_global_row_index; needed to map sentences."
            )
        test_idx = np.asarray(pred["test_global_row_index"], dtype=np.int64).reshape(-1)
        samples_NS = orient_samples(pred["test_pred_samples"], n_targets=len(y_true))

    with np.load(feature_path, allow_pickle=True) as feats:
        if "sentence1_N" not in feats.files or "sentence2_N" not in feats.files:
            raise RuntimeError(
                f"{feature_path} has no sentence1_N/sentence2_N. "
                "Regenerate embeddings or use export script with HF fallback."
            )
        s1 = feats["sentence1_N"]
        s2 = feats["sentence2_N"]

    n = min(len(y_true), len(y_pred_mean), len(y_pred_std), len(test_idx), samples_NS.shape[0])
    y_true = y_true[:n]
    y_pred_mean = y_pred_mean[:n]
    y_pred_std = y_pred_std[:n]
    test_idx = test_idx[:n]
    samples_NS = samples_NS[:n]

    chosen = pick_indices(y_true, y_pred_mean, args.n_examples, args.selection, args.seed)

    run_name = pred_path.parent.name
    out_base = (args.out_dir.resolve() if args.out_dir else (args.hparam_dir.resolve() / "plots" / f"cdf_examples_{run_name}"))
    out_base.mkdir(parents=True, exist_ok=True)

    manifest = []
    for local_i, idx in enumerate(chosen, start=1):
        global_row = int(test_idx[idx])
        out_file = out_base / f"example_{local_i:02d}_predrow_{int(idx):04d}_global_{global_row}.png"
        plot_one(
            out_path=out_file,
            sample_vals=samples_NS[idx],
            y_true=float(y_true[idx]),
            y_pred=float(y_pred_mean[idx]),
            y_std=float(y_pred_std[idx]),
            sent1=str(s1[global_row]),
            sent2=str(s2[global_row]),
            row_idx=global_row,
        )
        manifest.append(
            {
                "example_rank": local_i,
                "pred_row_in_test": int(idx),
                "global_row_index": global_row,
                "y_true": float(y_true[idx]),
                "pred_mean": float(y_pred_mean[idx]),
                "pred_std": float(y_pred_std[idx]),
                "plot_file": str(out_file),
            }
        )

    manifest_path = out_base / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(
            {
                "predictions_file": str(pred_path),
                "feature_file": str(feature_path),
                "selection": args.selection,
                "seed": args.seed,
                "n_examples": len(chosen),
                "examples": manifest,
            },
            f,
            indent=2,
        )

    print(f"Wrote {len(chosen)} CDF plots to: {out_base}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
