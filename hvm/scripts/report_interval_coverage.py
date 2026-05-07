#!/usr/bin/env python3
"""
Empirical coverage of prediction intervals for continuous similarity in [0, 1].

test_pred_std in predictions.npz is the std of forward-pass outputs across
posterior weight samples (epistemic uncertainty on the mean function), not
the full predictive distribution of y (which also includes observation noise
in the training likelihood).

This script reports coverage for:
  (A) Epistemic-only:  y in [mu - k*sigma_ep, mu + k*sigma_ep]
  (B) Plug-in total:   y in [mu - k*sigma_tot, mu + k*sigma_tot]
        sigma_tot = sqrt(sigma_ep^2 + sigma_lik^2)
      where sigma_lik comes from results.json (Gaussian likelihood in training).

Nominal rates (if y | x were exactly Normal with that scale) are shown for
reference. Targets are bounded; intervals are not clipped unless --clip-intervals.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Report empirical interval coverage.")
    p.add_argument(
        "--hparam-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "hparam_results",
    )
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--leaderboard", type=str, default="leaderboard_top10.csv")
    p.add_argument("--rank", type=int, default=1)
    p.add_argument(
        "--ks",
        type=float,
        nargs="+",
        default=[1.0, 1.645, 1.96, 2.0],
        help="Half-width multipliers (sigma scale). 1.0 ~ 68% Normal central if well calibrated.",
    )
    p.add_argument(
        "--clip-intervals",
        action="store_true",
        help="Clip interval endpoints to [0, 1] before testing coverage.",
    )
    return p.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def choose_run_dir(args: argparse.Namespace, hparam_dir: Path) -> str:
    if args.run_dir:
        return args.run_dir
    rows = read_rows(hparam_dir / args.leaderboard)
    idx = args.rank - 1
    if idx < 0 or idx >= len(rows):
        raise RuntimeError(f"Invalid rank {args.rank}")
    return rows[idx]["run_dir"]


def load_y(npz) -> np.ndarray:
    if "y_test_true" in npz.files:
        return np.asarray(npz["y_test_true"], dtype=np.float64).reshape(-1)
    return np.asarray(npz["y_test_N"], dtype=np.float64).reshape(-1)


def normal_nominal_two_sided(k: float) -> float:
    """Central interval [-k, k] for standard Normal (approximate for scipy-free)."""
    # erf(k/sqrt(2)) = P(|Z|<=k) for Z~N(0,1)
    from math import erf, sqrt

    return erf(k / sqrt(2.0))


def coverage(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    k: float,
    clip: bool,
) -> float:
    lo = mu - k * sigma
    hi = mu + k * sigma
    if clip:
        lo = np.clip(lo, 0.0, 1.0)
        hi = np.clip(hi, 0.0, 1.0)
    return float(np.mean((y >= lo) & (y <= hi)))


def main() -> None:
    args = parse_args()
    hparam_dir = args.hparam_dir.resolve()
    run_dir = choose_run_dir(args, hparam_dir)
    run_path = hparam_dir / run_dir
    pred_path = run_path / "predictions.npz"
    results_path = run_path / "results.json"

    if not pred_path.exists():
        raise RuntimeError(f"Missing {pred_path}")

    with np.load(pred_path) as data:
        mu = np.asarray(data["test_pred_mean"], dtype=np.float64).reshape(-1)
        sigma_ep = np.asarray(data["test_pred_std"], dtype=np.float64).reshape(-1)
        y = load_y(data)

    n = min(len(mu), len(sigma_ep), len(y))
    mu, sigma_ep, y = mu[:n], sigma_ep[:n], y[:n]

    sigma_lik = None
    if results_path.exists():
        with open(results_path) as f:
            meta = json.load(f)
        sigma_lik = float(meta.get("likelihood_stddev", 0.0))

    sigma_tot = None
    if sigma_lik is not None and sigma_lik > 0:
        sigma_tot = np.sqrt(sigma_ep**2 + sigma_lik**2)

    print(f"run_dir: {run_dir}")
    print(f"n_test: {n}")
    print(f"likelihood_stddev (from results.json): {sigma_lik}")
    print()
    print(
        "Note: sigma_ep = std across posterior samples of f(x); "
        "not assumed Gaussian unless you treat it as a plug-in scale."
    )
    print(
        "sigma_tot = sqrt(sigma_ep^2 + sigma_lik^2) is a Gaussian plug-in for "
        "y = f(x) + noise; still approximate (bounded y, nonlinear f)."
    )
    if args.clip_intervals:
        print("Intervals clipped to [0, 1].")
    print()

    header = f"{'k':>6} {'nominal':>10} {'cov_ep':>10}"
    if sigma_tot is not None:
        header += f" {'cov_tot':>10}"
    print(header)
    print("-" * len(header))

    for k in args.ks:
        nom = normal_nominal_two_sided(k)
        c_ep = coverage(y, mu, sigma_ep, k, args.clip_intervals)
        line = f"{k:>6.3f} {nom:>10.4f} {c_ep:>10.4f}"
        if sigma_tot is not None:
            c_tot = coverage(y, mu, sigma_tot, k, args.clip_intervals)
            line += f" {c_tot:>10.4f}"
        print(line)

    print()
    print("Interpretation:")
    print("  cov_ep  ≈ nominal  → spread of f(x) across θ is consistent with errors |y-mu|")
    print("           (still not a formal Gaussian predictive; bounded y matters).")
    print("  cov_tot ≈ nominal  → stronger if y|x is roughly like mu + N(0, sigma_lik^2)")
    print("           with epistemic width added; use as a heuristic.")


if __name__ == "__main__":
    main()
