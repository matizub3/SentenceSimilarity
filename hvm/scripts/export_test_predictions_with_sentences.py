#!/usr/bin/env python3
"""
Join baseline test predictions to original STS17 sentence pairs for error analysis.

Rows in sts17_octen_features.npz follow the same stacking order as create_embeddings.py
(one concatenated table over language pairs). Training uses a stratified 80/10/10 split;
test predictions are stored in prediction order, which matches ``test_global_row_index``
when present in predictions.npz (written by current run_baseline.py), or the shuffled
test_ids from stratified_train_valid_test_split(..., seed=<training seed>) otherwise.

If sentence1_N / sentence2_N are missing from the feature .npz, pass
--fetch-sentences-from-hf to reload text from HuggingFace (same pair order as
create_embeddings.py — keep LANG_PAIRS in sync).

Also prints summary statistics (mean |error|, counts per language_pair in the
exported worst-k subset vs full test) and writes a companion *_pair_summary.csv
unless --skip-summary-csv.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

HVM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HVM_ROOT))

from baseline_setup import stratified_train_valid_test_split

# Must match create_embeddings.LANG_PAIRS (dataset row order).
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
        description="Export CSV of test predictions joined to sentence pairs."
    )
    p.add_argument(
        "--feature-file",
        type=Path,
        default=HVM_ROOT / "sts17_octen_features.npz",
        help="Feature .npz (same file used when training the run).",
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
        help="Used with --run-dir if --predictions is omitted.",
    )
    p.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run folder name under hparam-dir (e.g. arch_100_iters_...).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for train/valid/test split (default: read results.json next "
        "to predictions, else 101). Only used if predictions.npz has no "
        "test_global_row_index.",
    )
    p.add_argument(
        "--fetch-sentences-from-hf",
        action="store_true",
        help="Load sentence1/2 from HuggingFace if missing in the feature .npz.",
    )
    p.add_argument(
        "--top-k-worst",
        type=int,
        default=50,
        metavar="K",
        help="Keep only K rows with largest absolute error (0 = keep all, sorted by "
        "|error| descending).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: print path under cwd named from run-dir).",
    )
    p.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="CSV path for worst-subset pair-count summary "
        "(default: <out_stem>_pair_summary.csv next to --out).",
    )
    p.add_argument(
        "--skip-summary-csv",
        action="store_true",
        help="Do not write the pair-count summary CSV (still prints summary to stdout).",
    )
    return p.parse_args()


def resolve_predictions_path(args: argparse.Namespace) -> Path:
    if args.predictions is not None:
        return args.predictions.resolve()
    if args.run_dir is None:
        raise SystemExit("Provide --predictions or --run-dir (with optional --hparam-dir).")
    return (args.hparam_dir.resolve() / args.run_dir / "predictions.npz").resolve()


def read_seed_from_results(predictions_path: Path) -> Optional[int]:
    results_path = predictions_path.parent / "results.json"
    if not results_path.is_file():
        return None
    with results_path.open() as f:
        data = json.load(f)
    seed = data.get("seed")
    return int(seed) if seed is not None else None


def load_sentences_from_hf(n_expected: int) -> tuple[np.ndarray, np.ndarray]:
    from datasets import load_dataset

    sentences1: list[str] = []
    sentences2: list[str] = []
    for pair in LANG_PAIRS:
        dataset_dict = load_dataset("mteb/sts17-crosslingual-sts", pair)
        split_name = "test" if "test" in dataset_dict else list(dataset_dict.keys())[0]
        dataset = dataset_dict[split_name]
        sentences1.extend(dataset["sentence1"])
        sentences2.extend(dataset["sentence2"])
    s1 = np.array(sentences1, dtype=object)
    s2 = np.array(sentences2, dtype=object)
    if len(s1) != n_expected:
        raise RuntimeError(
            f"HF sentence count {len(s1)} != feature npz rows {n_expected}. "
            "Dataset revision may differ from when embeddings were built."
        )
    return s1, s2


def pair_and_error_summary(
    df_sorted_all_test: pd.DataFrame,
    df_worst: pd.DataFrame,
) -> pd.DataFrame:
    """Counts per language_pair in worst subset vs full test set."""
    full_counts = df_sorted_all_test["language_pair"].value_counts()
    worst_counts = df_worst["language_pair"].value_counts()
    pairs = sorted(set(full_counts.index.astype(str)) | set(worst_counts.index.astype(str)))
    nw = len(df_worst)
    out_rows = []
    for p in pairs:
        cw = int(worst_counts.get(p, 0))
        ct = int(full_counts.get(p, 0))
        out_rows.append(
            {
                "language_pair": p,
                "n_in_worst_subset": cw,
                "pct_of_worst_subset": (100.0 * cw / nw) if nw else 0.0,
                "n_in_full_test": ct,
                "frac_pair_test_in_worst": (cw / ct) if ct else 0.0,
            }
        )
    summary = pd.DataFrame(out_rows)
    summary = summary.sort_values(
        ["n_in_worst_subset", "language_pair"], ascending=[False, True]
    )
    return summary


def main() -> None:
    args = parse_args()
    feature_path = args.feature_file.resolve()
    pred_path = resolve_predictions_path(args)

    if not feature_path.is_file():
        raise RuntimeError(f"Missing feature file: {feature_path}")
    if not pred_path.is_file():
        raise RuntimeError(f"Missing predictions file: {pred_path}")

    feats = np.load(feature_path, allow_pickle=True)
    n_rows = int(feats["x_ND"].shape[0])
    pair_N = feats["pair_N"] if "pair_N" in feats.files else np.array(["unknown"] * n_rows)
    y_scaled = (feats["y_raw_N"].astype(np.float64) / 5.0).reshape(-1)

    sentence1_N = feats["sentence1_N"] if "sentence1_N" in feats.files else None
    sentence2_N = feats["sentence2_N"] if "sentence2_N" in feats.files else None

    if sentence1_N is None or sentence2_N is None:
        if args.fetch_sentences_from_hf:
            sentence1_N, sentence2_N = load_sentences_from_hf(n_rows)
        else:
            raise RuntimeError(
                f"{feature_path} has no sentence1_N/sentence2_N arrays. "
                "Regenerate with create_embeddings.py (they are saved there), or rerun "
                "this script with --fetch-sentences-from-hf."
            )

    pred = np.load(pred_path, allow_pickle=True)
    y_hat = np.asarray(pred["test_pred_mean"], dtype=np.float64).reshape(-1)
    y_sd = np.asarray(pred["test_pred_std"], dtype=np.float64).reshape(-1)
    if "y_test_N" in pred.files:
        y_obs = np.asarray(pred["y_test_N"], dtype=np.float64).reshape(-1)
    elif "y_test_true" in pred.files:
        y_obs = np.asarray(pred["y_test_true"], dtype=np.float64).reshape(-1)
    else:
        raise RuntimeError(f"No y_test_N / y_test_true in {pred_path}")

    if "test_global_row_index" in pred.files:
        global_idx = np.asarray(pred["test_global_row_index"], dtype=np.int64).reshape(-1)
    else:
        seed = args.seed if args.seed is not None else read_seed_from_results(pred_path)
        if seed is None:
            seed = 101
        _, _, test_ids = stratified_train_valid_test_split(pair_N, seed=seed)
        global_idx = np.asarray(test_ids, dtype=np.int64).reshape(-1)

    if not (len(y_hat) == len(y_sd) == len(y_obs) == len(global_idx)):
        raise RuntimeError(
            "Length mismatch between predictions and index vector: "
            f"pred_mean={len(y_hat)}, pred_std={len(y_sd)}, y_test={len(y_obs)}, "
            f"idx={len(global_idx)}"
        )

    # Sanity check labels vs feature table (optional diagnostic).
    y_from_table = y_scaled[global_idx]
    if np.max(np.abs(y_from_table - y_obs)) > 1e-4:
        raise RuntimeError(
            "y_test_N in predictions.npz does not match y_raw_N/5 at stored row indices. "
            "Wrong --feature-file, wrong split seed, or stale predictions."
        )

    abs_err = np.abs(y_obs - y_hat)
    rows = []
    for i in range(len(global_idx)):
        gid = int(global_idx[i])
        rows.append(
            {
                "pred_row_in_split": i,
                "global_row_index": gid,
                "language_pair": str(pair_N[gid]),
                "sentence1": str(sentence1_N[gid]),
                "sentence2": str(sentence2_N[gid]),
                "y_true": float(y_obs[i]),
                "y_pred_mean": float(y_hat[i]),
                "y_pred_std": float(y_sd[i]),
                "abs_error": float(abs_err[i]),
                "signed_error": float(y_obs[i] - y_hat[i]),
            }
        )

    df = pd.DataFrame(rows)
    df_sorted_full = df.sort_values("abs_error", ascending=False)

    k = args.top_k_worst
    if k > 0:
        df_worst = df_sorted_full.head(k).copy()
    else:
        df_worst = df_sorted_full.copy()

    out_path = args.out
    if out_path is None:
        stem = pred_path.parent.name
        out_path = Path.cwd() / f"test_predictions_with_sentences_{stem}.csv"
    else:
        out_path = out_path.resolve()

    df_worst.to_csv(out_path, index=False)
    print(f"Wrote {len(df_worst)} rows to {out_path}")

    n_test = len(df_sorted_full)
    worst_label = f"worst {k}" if k > 0 else "full test set"
    print("\n--- Error summaries ---")
    print(f"Full test size: {n_test}")
    print(f"Rows exported ({worst_label}): {len(df_worst)}")
    print(f"Mean |error| (full test): {df_sorted_full['abs_error'].mean():.6f}")
    print(f"Median |error| (full test): {df_sorted_full['abs_error'].median():.6f}")
    print(f"Mean |error| ({worst_label}): {df_worst['abs_error'].mean():.6f}")

    summary_df = pair_and_error_summary(df_sorted_full, df_worst)
    print(f"\nLanguage pairs in {worst_label} (count / % of subset / "
          f"n in full test / frac of that pair's test rows that hit subset):")
    for _, r in summary_df.iterrows():
        if r["n_in_worst_subset"] == 0:
            continue
        print(
            f"  {r['language_pair']}: "
            f"n={int(r['n_in_worst_subset'])} "
            f"({r['pct_of_worst_subset']:.1f}% of subset); "
            f"full_test_n={int(r['n_in_full_test'])}; "
            f"pair_frac_in_subset={r['frac_pair_test_in_worst']:.3f}"
        )

    if args.skip_summary_csv:
        return

    sum_path = args.summary_out
    if sum_path is None:
        sum_path = out_path.with_name(f"{out_path.stem}_pair_summary.csv")
    else:
        sum_path = sum_path.resolve()

    summary_df.to_csv(sum_path, index=False)
    print(f"\nWrote pair summary ({len(summary_df)} pairs) to {sum_path}")


if __name__ == "__main__":
    main()
