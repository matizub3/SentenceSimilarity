"""
Execution script for the STS17 mean-field Bayesian MLP baseline.

This file imports all reusable function definitions from baseline_setup.py.
Run examples:

    python run_baseline.py --feature_file sts17_octen_features.npz --hidden_sizes 100
    python run_baseline.py --hidden_sizes 100-30 --n_iters 2000 --batch_size 64 --n_mc_samples 5
    python run_baseline.py --hidden_sizes 30 --n_iters 10 --batch_size 8 --n_mc_samples 1 --print_every 1
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from baseline_setup import (
    load_feature_npz,
    prepare_train_valid_test_arrays,
    parse_bool,
    parse_hidden_sizes,
    train_mean_field_bnn_baseline,
    evaluate_rmse_with_posterior_mean,
    evaluate_rmse_with_posterior_predictive_mean,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train mean-field Bayesian MLP baseline for STS17."
    )

    # Data / paths
    parser.add_argument(
        "--feature_file",
        type=str,
        default="sts17_octen_features.npz",
        help="Path to .npz file containing x_ND, y_raw_N, and pair_N.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="baseline_results",
        help="Directory where results will be saved.",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run name. If omitted, one is created from hyperparameters.",
    )

    # Model architecture
    parser.add_argument(
        "--hidden_sizes",
        type=str,
        default="100",
        help='Hidden architecture, e.g. "100", "100-30", "100-30-15", or "linear".',
    )

    parser.add_argument(
        "--use_sigmoid_output",
        type=parse_bool,
        default=False,
        help="Whether to apply sigmoid to model output.",
    )

    # Training hyperparameters
    parser.add_argument(
        "--n_iters",
        type=int,
        default=2000,
        help="Number of training iterations.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Minibatch size.",
    )

    parser.add_argument(
        "--n_mc_samples",
        type=int,
        default=5,
        help="Number of Monte Carlo samples per ELBO estimate.",
    )

    parser.add_argument(
        "--step_size",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )

    parser.add_argument(
        "--prior_stddev",
        type=float,
        default=3.0,
        help="Prior standard deviation for all weights and biases.",
    )

    parser.add_argument(
        "--likelihood_stddev",
        type=float,
        default=0.10,
        help="Gaussian likelihood standard deviation.",
    )

    parser.add_argument(
        "--init_q_stddev",
        type=float,
        default=0.05,
        help="Initial variational posterior standard deviation.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=101,
        help="Random seed.",
    )

    parser.add_argument(
        "--print_every",
        type=int,
        default=100,
        help="How often to print training progress.",
    )

    # Evaluation
    parser.add_argument(
        "--n_test_samples",
        type=int,
        default=100,
        help="Number of posterior samples for test-time posterior predictive RMSE.",
    )

    args = parser.parse_args()

    args.hidden_sizes_list = parse_hidden_sizes(args.hidden_sizes)

    if args.run_name is None:
        sigmoid_name = "sigmoid" if args.use_sigmoid_output else "linearout"
        args.run_name = (
            f"arch_{args.hidden_sizes}_"
            f"iters_{args.n_iters}_"
            f"bs_{args.batch_size}_"
            f"mc_{args.n_mc_samples}_"
            f"lr_{args.step_size}_"
            f"prior_{args.prior_stddev}_"
            f"lik_{args.likelihood_stddev}_"
            f"qstd_{args.init_q_stddev}_"
            f"{sigmoid_name}_"
            f"seed_{args.seed}"
        )
        args.run_name = args.run_name.replace(".", "p")

    args.run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.run_dir, exist_ok=True)

    return args


def save_results(
    args,
    hist,
    valid_rmse,
    test_rmse,
    test_pred_mean,
    test_pred_std,
    test_pred_samples,
    y_test_N,
    test_ids,
):
    results = {
        "feature_file": args.feature_file,
        "hidden_sizes": args.hidden_sizes,
        "hidden_sizes_list": args.hidden_sizes_list,
        "n_iters": args.n_iters,
        "batch_size": args.batch_size,
        "n_mc_samples": args.n_mc_samples,
        "step_size": args.step_size,
        "prior_stddev": args.prior_stddev,
        "likelihood_stddev": args.likelihood_stddev,
        "init_q_stddev": args.init_q_stddev,
        "use_sigmoid_output": args.use_sigmoid_output,
        "seed": args.seed,
        "n_test_samples": args.n_test_samples,
        "valid_rmse": float(valid_rmse),
        "test_rmse": float(test_rmse),
    }

    results_json_path = os.path.join(args.run_dir, "results.json")
    history_csv_path = os.path.join(args.run_dir, "history.csv")
    predictions_npz_path = os.path.join(args.run_dir, "predictions.npz")

    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(hist).to_csv(history_csv_path, index=False)

    np.savez_compressed(
        predictions_npz_path,
        test_pred_mean=np.array(test_pred_mean),
        test_pred_std=np.array(test_pred_std),
        test_pred_samples=np.array(test_pred_samples),
        y_test_N=np.array(y_test_N),
        test_global_row_index=np.asarray(test_ids, dtype=np.int64),
    )

    print("Saved results to:", args.run_dir)
    print("results:", results_json_path)
    print("history:", history_csv_path)
    print("predictions:", predictions_npz_path)

    return results


def main():
    args = parse_args()

    print("Run args:")
    print(json.dumps(vars(args), indent=2, default=str))

    raw_x_ND, raw_y_N, pair_N = load_feature_npz(args.feature_file)

    print("x shape:", raw_x_ND.shape)
    print("y shape:", raw_y_N.shape)
    print("pair shape:", pair_N.shape)
    print("y min/max:", raw_y_N.min(), raw_y_N.max())
    print("language pairs:", np.unique(pair_N))

    arrays = prepare_train_valid_test_arrays(
        raw_x_ND=raw_x_ND,
        raw_y_N=raw_y_N,
        pair_N=pair_N,
        seed=args.seed,
    )

    print("train:", arrays["x_train_ND"].shape, arrays["y_train_N"].shape)
    print("valid:", arrays["x_valid_ND"].shape, arrays["y_valid_N"].shape)
    print("test :", arrays["x_test_ND"].shape, arrays["y_test_N"].shape)

    q_mean, q_realstddev, hist = train_mean_field_bnn_baseline(
        x_train_ND=arrays["x_train_ND"],
        y_train_N=arrays["y_train_N"],
        x_valid_ND=arrays["x_valid_ND"],
        y_valid_N=arrays["y_valid_N"],
        hidden_sizes=args.hidden_sizes_list,
        n_iters=args.n_iters,
        batch_size=args.batch_size,
        n_mc_samples=args.n_mc_samples,
        step_size=args.step_size,
        prior_stddev=args.prior_stddev,
        likelihood_stddev=args.likelihood_stddev,
        init_q_stddev=args.init_q_stddev,
        use_sigmoid_output=args.use_sigmoid_output,
        seed=args.seed,
        print_every=args.print_every,
    )

    valid_rmse = evaluate_rmse_with_posterior_mean(
        q_mean,
        arrays["x_valid_ND"],
        arrays["y_valid_N"],
        use_sigmoid_output=args.use_sigmoid_output,
    )

    test_rmse, test_pred_mean, test_pred_std, test_pred_samples = evaluate_rmse_with_posterior_predictive_mean(
        q_mean,
        q_realstddev,
        arrays["x_test_ND"],
        arrays["y_test_N"],
        n_samples=args.n_test_samples,
        use_sigmoid_output=args.use_sigmoid_output,
        seed=args.seed + 1000,
    )

    print("Validation RMSE:", float(valid_rmse))
    print("Test RMSE:", float(test_rmse))

    save_results(
        args=args,
        hist=hist,
        valid_rmse=valid_rmse,
        test_rmse=test_rmse,
        test_pred_mean=test_pred_mean,
        test_pred_std=test_pred_std,
        test_pred_samples=test_pred_samples,
        y_test_N=arrays["y_test_N"],
        test_ids=np.asarray(arrays["test_ids"]),
    )


if __name__ == "__main__":
    main()
