#!/usr/bin/env python3
"""
Hyperparameter Tuning Script for the NVS Pipeline.

Performs a grid search over key hyperparameters, evaluates each combination
using PSNR as the target metric, and saves results (generated image + params)
into per-combination subfolders.

Usage:
    python submission/hyper_tune.py --data_root ./data/train --output_dir ./output/hyper_tune
"""

import os
import sys
import json
import copy
import time
import argparse
import itertools

import cv2
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from submission.nvs_pipeline import load_config, process_sample


# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameter search space
# ──────────────────────────────────────────────────────────────────────────────

SEARCH_SPACE = {
    # LiDAR init parameters (were hardcoded in nvs_pipeline.py)
    "voxel_size": [0.05, 0.1, 0.15, 0.2],
    "max_points": [200000, 300000, 500000],

    # Optimization parameters
    "mapping_itr_num": [1500, 2000, 3000],
    "lambda_dssim": [0.1, 0.2, 0.4],
    "densify_grad_threshold": [0.0001, 0.0002, 0.0005],
    "densify_until_iter": [500, 1000, 1500],
    "densification_interval": [100, 200, 300],

    # Pruning parameters
    "gaussian_th": [0.001, 0.005, 0.01],
    "gaussian_extent": [30.0, 50.0, 80.0],
}


def compute_psnr(gt_path, pred_path):
    """Compute PSNR between ground truth and predicted images."""
    gt = cv2.imread(gt_path).astype(np.float64)
    pred = cv2.imread(pred_path).astype(np.float64)

    if gt is None or pred is None:
        return 0.0

    # Resize pred to match GT if needed
    if gt.shape != pred.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

    mse = np.mean((gt - pred) ** 2)
    if mse == 0:
        return 100.0
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def apply_params_to_cfg(cfg, params):
    """Apply a parameter dict to the config, returning a modified copy."""
    cfg = copy.deepcopy(cfg)

    # Map param names to config locations
    training_params = ["mapping_itr_num", "gaussian_th", "gaussian_extent"]
    opt_params = [
        "lambda_dssim", "densify_grad_threshold",
        "densify_until_iter", "densification_interval",
    ]

    for key, value in params.items():
        if key in ("voxel_size", "max_points"):
            # These are passed directly to process_sample, not in cfg
            continue
        elif key in training_params:
            cfg["mapping"]["Training"][key] = value
        elif key in opt_params:
            cfg["mapping"]["opt_params"][key] = value

    return cfg


def get_gt_path(sample_dir):
    """Get path to ground truth image for a sample."""
    meta_path = os.path.join(sample_dir, "meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    target_camera = meta["target_camera"]
    return os.path.join(sample_dir, "target", f"{target_camera}.jpg")


def params_to_folder_name(params):
    """Generate a compact folder name from parameters."""
    parts = []
    for key, value in sorted(params.items()):
        if isinstance(value, float):
            parts.append(f"{key}={value:.6g}")
        else:
            parts.append(f"{key}={value}")
    return "__".join(parts)


def run_single_trial(sample_dirs, params, base_cfg, output_base_dir, device):
    """Run the pipeline with a specific parameter combination.

    Returns:
        mean_psnr: float — average PSNR across all samples
        trial_dir: str — path to the trial output directory
    """
    # Create trial output directory
    folder_name = params_to_folder_name(params)
    trial_dir = os.path.join(output_base_dir, folder_name)
    os.makedirs(trial_dir, exist_ok=True)

    # Apply params to config
    cfg = apply_params_to_cfg(base_cfg, params)
    voxel_size = params.get("voxel_size", 0.15)
    max_points = params.get("max_points", 300000)

    # Save params
    params_path = os.path.join(trial_dir, "params.json")
    with open(params_path, "w") as f:
        json.dump(params, f, indent=2)

    # Run pipeline on all samples
    psnr_values = []
    for sample_dir in sample_dirs:
        try:
            sample_id = process_sample(
                sample_dir, trial_dir, cfg,
                device=device,
                voxel_size=voxel_size,
                max_points=max_points,
            )

            # Compute PSNR
            gt_path = get_gt_path(sample_dir)
            if os.path.exists(gt_path):
                meta_path = os.path.join(sample_dir, "meta.json")
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                pred_path = os.path.join(trial_dir, f"{meta['sample_id']}.jpg")
                psnr = compute_psnr(gt_path, pred_path)
                psnr_values.append(psnr)
        except Exception as e:
            print(f"  ERROR: {e}")
            psnr_values.append(0.0)

    mean_psnr = np.mean(psnr_values) if psnr_values else 0.0

    # Save results summary
    results = {
        "params": params,
        "mean_psnr": float(mean_psnr),
        "per_sample_psnr": [float(p) for p in psnr_values],
    }
    results_path = os.path.join(trial_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return mean_psnr, trial_dir


def generate_combinations(search_space, mode="grid", max_trials=None, seed=42):
    """Generate parameter combinations from search space.

    Args:
        search_space: dict of param_name -> list of values
        mode: "grid" for full grid search, "random" for random sampling
        max_trials: max number of trials (only used for random mode)
        seed: random seed for reproducibility

    Returns:
        list of param dicts
    """
    keys = sorted(search_space.keys())
    values = [search_space[k] for k in keys]

    if mode == "grid":
        combinations = [
            dict(zip(keys, combo))
            for combo in itertools.product(*values)
        ]
        if max_trials and len(combinations) > max_trials:
            rng = np.random.default_rng(seed)
            indices = rng.choice(len(combinations), max_trials, replace=False)
            combinations = [combinations[i] for i in sorted(indices)]
    elif mode == "random":
        rng = np.random.default_rng(seed)
        n = max_trials or 50
        combinations = []
        for _ in range(n):
            combo = {k: rng.choice(v) for k, v in zip(keys, values)}
            # Convert numpy types to Python native types
            combo = {k: v.item() if hasattr(v, 'item') else v for k, v in combo.items()}
            combinations.append(combo)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return combinations


def main():
    parser = argparse.ArgumentParser(description="NVS Hyperparameter Tuning")
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Path to dataset directory (train split with GT)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output/hyper_tune",
        help="Base directory for tuning outputs"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="CUDA device"
    )
    parser.add_argument(
        "--mode", type=str, default="random", choices=["grid", "random"],
        help="Search mode: 'grid' for full grid, 'random' for random sampling"
    )
    parser.add_argument(
        "--max_trials", type=int, default=20,
        help="Maximum number of trials to run"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--params", type=str, default=None,
        help="JSON string or file path to override search space "
             "(e.g., '{\"voxel_size\": [0.1, 0.2], \"max_points\": [300000]}')"
    )
    args = parser.parse_args()

    # Load base config
    base_cfg = load_config()

    # Load/override search space
    search_space = SEARCH_SPACE.copy()
    if args.params:
        if os.path.isfile(args.params):
            with open(args.params, "r") as f:
                custom_space = json.load(f)
        else:
            custom_space = json.loads(args.params)
        search_space.update(custom_space)

    # Find all sample directories
    sample_dirs = []
    for entry in sorted(os.listdir(args.data_root)):
        sample_dir = os.path.join(args.data_root, entry)
        if os.path.isdir(sample_dir) and os.path.exists(os.path.join(sample_dir, "meta.json")):
            sample_dirs.append(sample_dir)

    if not sample_dirs:
        print(f"No samples found in {args.data_root}")
        return

    print(f"Found {len(sample_dirs)} samples in {args.data_root}")
    print(f"Search mode: {args.mode}, max_trials: {args.max_trials}")
    print(f"Search space:")
    for k, v in sorted(search_space.items()):
        print(f"  {k}: {v}")

    # Generate parameter combinations
    combinations = generate_combinations(
        search_space, mode=args.mode,
        max_trials=args.max_trials, seed=args.seed,
    )
    total_combos = 1
    for v in search_space.values():
        total_combos *= len(v)
    print(f"\nTotal possible combinations: {total_combos}")
    print(f"Running {len(combinations)} trials\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # Run trials
    results = []
    best_psnr = -1.0
    best_params = None
    best_trial_dir = None

    for i, params in enumerate(combinations):
        print(f"\n{'='*60}")
        print(f"Trial {i+1}/{len(combinations)}")
        print(f"Params: {json.dumps(params, indent=2)}")
        print(f"{'='*60}")

        t0 = time.time()
        mean_psnr, trial_dir = run_single_trial(
            sample_dirs, params, base_cfg, args.output_dir, args.device,
        )
        elapsed = time.time() - t0

        results.append({
            "trial": i + 1,
            "params": params,
            "mean_psnr": float(mean_psnr),
            "time_seconds": elapsed,
            "trial_dir": trial_dir,
        })

        print(f"\n  Trial {i+1} result: PSNR={mean_psnr:.2f} dB (took {elapsed:.1f}s)")

        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            best_params = params
            best_trial_dir = trial_dir
            print(f"  *** New best! ***")

    # Summary
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETER TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"Trials run: {len(results)}")
    print(f"\nBest PSNR: {best_psnr:.2f} dB")
    print(f"Best params: {json.dumps(best_params, indent=2)}")
    print(f"Best trial dir: {best_trial_dir}")

    # Sort results by PSNR
    results_sorted = sorted(results, key=lambda x: x["mean_psnr"], reverse=True)
    print(f"\nTop 5 configurations:")
    for i, r in enumerate(results_sorted[:5]):
        print(f"  {i+1}. PSNR={r['mean_psnr']:.2f} dB — {r['params']}")

    # Save overall summary
    summary = {
        "best_psnr": float(best_psnr),
        "best_params": best_params,
        "best_trial_dir": best_trial_dir,
        "all_results": results_sorted,
        "search_space": {k: [v.item() if hasattr(v, 'item') else v for v in vals]
                         for k, vals in search_space.items()},
        "mode": args.mode,
        "max_trials": args.max_trials,
        "seed": args.seed,
    }
    summary_path = os.path.join(args.output_dir, "tuning_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
