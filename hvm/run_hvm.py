"""
Execution script for the STS17 HVM / normalizing-flow Bayesian MLP upgrade.

Run examples:

    python run_hvm.py --feature_file sts17_octen_features.npz --hidden_sizes 30 --n_iters 10 --batch_size 8 --n_mc_samples 1 --print_every 1

    python run_hvm.py --feature_file sts17_octen_features.npz --hidden_sizes 100 --flow_length 2 --n_iters 2000 --batch_size 64 --n_mc_samples 3
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hvm_setup import (
    parse_bool,
    parse_hidden_sizes,
    load_feature_npz,
    prepare_train_valid_test_arrays,
    train_hvm_bnn_upgrade,
    evaluate_hvm_rmse,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HVM / normalizing-flow Bayesian MLP upgrade for STS17."
    )

    parser.add_argument("--feature_file", type=str, default="sts17_octen_features.npz")
    parser.add_argument("--output_dir", type=str, default="hvm_results")
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--hidden_sizes", type=str, default="100")
    parser.add_argument("--use_sigmoid_output", type=parse_bool, default=False)

    parser.add_argument("--n_iters", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_mc_samples", type=int, default=3)
    parser.add_argument("--step_size", type=float, default=1e-4)
    parser.add_argument("--prior_stddev", type=float, default=3.0)
    parser.add_argument("--likelihood_stddev", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--print_every", type=int, default=100)

    parser.add_argument("--flow_length", type=int, default=2)
    parser.add_argument("--flow_init_scale", type=float, default=1e-3)
    parser.add_argument("--aux_init_stddev", type=float, default=1.0)

    parser.add_argument("--n_valid_samples", type=int, default=20)
    parser.add_argument("--n_test_samples", type=int, default=50)

    args = parser.parse_args()
    args.hidden_sizes_list = parse_hidden_sizes(args.hidden_sizes)

    if args.run_name is None:
        sigmoid_name = "sigmoid" if args.use_sigmoid_output else "linearout"
        args.run_name = (
            f"hvm_arch_{args.hidden_sizes}_"
            f"flow_{args.flow_length}_"
            f"iters_{args.n_iters}_"
            f"bs_{args.batch_size}_"
            f"mc_{args.n_mc_samples}_"
            f"lr_{args.step_size}_"
            f"prior_{args.prior_stddev}_"
            f"lik_{args.likelihood_stddev}_"
            f"auxstd_{args.aux_init_stddev}_"
            f"{sigmoid_name}_"
            f"seed_{args.seed}"
        )
        args.run_name = args.run_name.replace(".", "p")

    args.run_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(args.run_dir, exist_ok=True)
    return args

def plot_history_csv(history_csv_path, output_dir):
    """
    Read history.csv and save plots of HVM ELBO and validation RMSE.
    """
    history_df = pd.read_csv(history_csv_path)

    # Plot validation RMSE
    if "iter" in history_df.columns and "valid_rmse" in history_df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(history_df["iter"], history_df["valid_rmse"], marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("Validation RMSE")
        plt.title("HVM Validation RMSE over Training")
        plt.grid(True)
        plt.tight_layout()

        valid_plot_path = os.path.join(output_dir, "valid_rmse_plot.png")
        plt.savefig(valid_plot_path, dpi=200)
        plt.close()

        print("valid RMSE plot:", valid_plot_path)

    # Plot HVM ELBO
    if "iter" in history_df.columns and "train_elbo" in history_df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(history_df["iter"], history_df["train_elbo"], marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("HVM ELBO / N")
        plt.title("HVM ELBO over Training")
        plt.grid(True)
        plt.tight_layout()

        elbo_plot_path = os.path.join(output_dir, "hvm_elbo_plot.png")
        plt.savefig(elbo_plot_path, dpi=200)
        plt.close()

        print("HVM ELBO plot:", elbo_plot_path)

    # In case your HVM history uses "hvm_elbo" instead of "train_elbo"
    if "iter" in history_df.columns and "hvm_elbo" in history_df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(history_df["iter"], history_df["hvm_elbo"], marker="o")
        plt.xlabel("Iteration")
        plt.ylabel("HVM ELBO / N")
        plt.title("HVM ELBO over Training")
        plt.grid(True)
        plt.tight_layout()

        elbo_plot_path = os.path.join(output_dir, "hvm_elbo_plot.png")
        plt.savefig(elbo_plot_path, dpi=200)
        plt.close()

        print("HVM ELBO plot:", elbo_plot_path)


def save_results(args, hist, valid_rmse, test_rmse, test_pred_mean, test_pred_std, y_test_N, info):
    results = {
        "method": "hvm_normalizing_flow_upgrade",
        "feature_file": args.feature_file,
        "hidden_sizes": args.hidden_sizes,
        "hidden_sizes_list": args.hidden_sizes_list,
        "n_iters": args.n_iters,
        "batch_size": args.batch_size,
        "n_mc_samples": args.n_mc_samples,
        "step_size": args.step_size,
        "prior_stddev": args.prior_stddev,
        "likelihood_stddev": args.likelihood_stddev,
        "flow_length": args.flow_length,
        "flow_init_scale": args.flow_init_scale,
        "aux_init_stddev": args.aux_init_stddev,
        "use_sigmoid_output": args.use_sigmoid_output,
        "seed": args.seed,
        "n_valid_samples": args.n_valid_samples,
        "n_test_samples": args.n_test_samples,
        "n_bnn_params": int(info["n_bnn_params"]),
        "lambda_dim": int(info["lambda_dim"]),
        "valid_rmse": float(valid_rmse),
        "test_rmse": float(test_rmse),
    }

    results_json_path = os.path.join(args.run_dir, "results.json")
    history_csv_path = os.path.join(args.run_dir, "history.csv")
    predictions_npz_path = os.path.join(args.run_dir, "predictions.npz")

    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)

    pd.DataFrame(hist).to_csv(history_csv_path, index=False)
    plot_history_csv(history_csv_path, args.run_dir)

    np.savez_compressed(
        predictions_npz_path,
        test_pred_mean=np.array(test_pred_mean),
        test_pred_std=np.array(test_pred_std),
        y_test_N=np.array(y_test_N),
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

    flow_params, aux_params, hist, info = train_hvm_bnn_upgrade(
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
        flow_length=args.flow_length,
        flow_init_scale=args.flow_init_scale,
        aux_init_stddev=args.aux_init_stddev,
        use_sigmoid_output=args.use_sigmoid_output,
        n_valid_samples=args.n_valid_samples,
        seed=args.seed,
        print_every=args.print_every,
    )

    valid_rmse, _, _ = evaluate_hvm_rmse(
        flow_params=flow_params,
        x_ND=arrays["x_valid_ND"],
        y_N=arrays["y_valid_N"],
        unflatten_fn=info["unflatten_fn"],
        n_bnn_params=info["n_bnn_params"],
        n_samples=args.n_test_samples,
        use_sigmoid_output=args.use_sigmoid_output,
        seed=args.seed + 900,
    )

    test_rmse, test_pred_mean, test_pred_std = evaluate_hvm_rmse(
        flow_params=flow_params,
        x_ND=arrays["x_test_ND"],
        y_N=arrays["y_test_N"],
        unflatten_fn=info["unflatten_fn"],
        n_bnn_params=info["n_bnn_params"],
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
        y_test_N=arrays["y_test_N"],
        info=info,
    )


if __name__ == "__main__":
    main()
