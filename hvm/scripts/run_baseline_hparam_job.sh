import subprocess
import sys
import itertools
import argparse
from pathlib import Path

GRID = {
    "n_mc_samples":       [50, 100, 200],
    "batch_size":         [32, 64, 128],
    "use_sigmoid_output": ["true"],
    "likelihood_stddev":  [0.05, 0.1, 0.2, 0.5],
    "prior_stddev":       [0.5, 1.0, 3.0, 10.0],
    "iters":              [2000, 50000, 10000]
}

def make_combinations(grid):
    keys = list(grid.keys())
    values = list(grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, required=True)
    args = parser.parse_args()

    combos = make_combinations(GRID)
    print(f"Total combinations: {len(combos)}")

    if args.task_id >= len(combos):
        print(f"Task ID {args.task_id} out of range, exiting.")
        sys.exit(0)

    params = combos[args.task_id]
    print(f"Running combo {args.task_id}: {params}")

    repo_root = Path(__file__).resolve().parents[1]
    run_baseline = repo_root / "run_baseline.py"

    cmd = [
        sys.executable, str(run_baseline),
        "--feature_file",       "sts17_octen_features.npz",
        "--hidden_sizes",       "100",
        "--n_iters",            str(params["iters"]),
        "--print_every",        "100",
        "--seed",               "101",
        "--output_dir",         "hparam_results",
        "--n_mc_samples",       str(params["n_mc_samples"]),
        "--batch_size",         str(params["batch_size"]),
        "--use_sigmoid_output", params["use_sigmoid_output"],
        "--prior_stddev",       str(params["prior_stddev"]),
        "--likelihood_stddev",  str(params["likelihood_stddev"]),
    ]

    subprocess.run(cmd, check=True, cwd=repo_root)

if __name__ == "__main__":
    main()