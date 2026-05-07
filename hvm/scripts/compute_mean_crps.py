#!/usr/bin/env python3
"""
Compute mean Continuous Ranked Probability Score (CRPS) from predictions.npz.

Uses the sample-based CRPS identity for each example y:
    CRPS(F, y) = E|X - y| - 0.5 * E|X - X'|
where X, X' are iid draws from predictive distribution F.

For finite predictive samples x_1..x_S:
    CRPS_hat = (1/S) * sum_i |x_i - y| - (1/(2*S^2)) * sum_{i,j} |x_i - x_j|
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute mean CRPS from predictions.npz")
    parser.add_argument(
        "predictions_npz",
        type=Path,
        help="Path to predictions.npz containing test_pred_samples and y_test_N/y_test_true.",
    )
    return parser.parse_args()


def load_targets(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    if "y_test_true" in npz.files:
        return np.asarray(npz["y_test_true"], dtype=np.float64).reshape(-1)
    if "y_test_N" in npz.files:
        return np.asarray(npz["y_test_N"], dtype=np.float64).reshape(-1)
    raise RuntimeError(
        "predictions.npz missing target array. Expected one of ['y_test_true', 'y_test_N']."
    )


def orient_samples(samples: np.ndarray, n_targets: int) -> np.ndarray:
    """Return samples as shape [n_examples, n_samples]."""
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 2:
        raise RuntimeError(f"test_pred_samples must be 2D, got shape {arr.shape}")

    # Stored convention is typically [n_samples, n_examples].
    if arr.shape[1] == n_targets:
        return arr.T
    if arr.shape[0] == n_targets:
        return arr
    raise RuntimeError(
        f"Cannot align test_pred_samples shape {arr.shape} with n_targets={n_targets}."
    )


def crps_per_example_from_samples(samples_NS: np.ndarray, y_N: np.ndarray) -> np.ndarray:
    """
    samples_NS: shape [N, S]
    y_N: shape [N]
    Returns: shape [N]
    """
    n_examples, n_samples = samples_NS.shape
    if n_samples < 2:
        raise RuntimeError("Need at least 2 predictive samples per example to compute CRPS.")

    # First term: E|X - y|
    term1 = np.mean(np.abs(samples_NS - y_N[:, None]), axis=1)

    # Second term: 0.5 * E|X - X'|
    # Compute pairwise absolute differences for each example.
    # shape: [N, S, S]
    diffs = np.abs(samples_NS[:, :, None] - samples_NS[:, None, :])
    term2 = 0.5 * np.mean(diffs, axis=(1, 2))

    return term1 - term2


def main() -> None:
    args = parse_args()
    pred_path = args.predictions_npz.resolve()
    if not pred_path.exists():
        raise RuntimeError(f"File not found: {pred_path}")

    with np.load(pred_path) as npz:
        if "test_pred_samples" not in npz.files:
            raise RuntimeError(
                "predictions.npz is missing 'test_pred_samples'. "
                "Re-run training with updated saving logic."
            )

        y = load_targets(npz)
        samples = orient_samples(npz["test_pred_samples"], n_targets=len(y))

    n_examples, n_samples = samples.shape
    crps_N = crps_per_example_from_samples(samples, y)
    mean_crps = float(np.mean(crps_N))

    print(f"predictions_npz: {pred_path}")
    print(f"n_examples: {n_examples}")
    print(f"n_samples_per_example: {n_samples}")
    print(f"mean_crps: {mean_crps:.10f}")


if __name__ == "__main__":
    main()
