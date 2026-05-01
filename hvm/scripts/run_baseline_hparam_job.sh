import subprocess
import sys
import itertools
import argparse

GRID = {
    "n_mc_samples":       [5, 20, 50, 100],
    "batch_size":         [16, 32, 64, 128],
    "use_sigmoid_output": ["true", "false"],
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

    cmd = [
        "pixi", "run", "python", "run_baseline.py",
        "--feature_file",       "sts17_octen_features.npz",
        "--hidden_sizes",       "100",
        "--n_iters",            "2000",
        "--print_every",        "100",
        "--seed",               "101",
        "--output_dir",         "hparam_results",
        "--n_mc_samples",       str(params["n_mc_samples"]),
        "--batch_size",         str(params["batch_size"]),
        "--use_sigmoid_output", params["use_sigmoid_output"],
    ]

    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()