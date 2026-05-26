#!/usr/bin/env python3
"""
NVS Competition — Batch Inference Entry Point

Usage:
    python submission/run.py --data_root ./data/test --output_dir ./output/submission
    python submission/run.py --data_root ./data/train --output_dir ./output/submission_train --evaluate

Processes all samples in data_root and saves predictions to output_dir.
"""

import os
import sys
import argparse
import time

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from submission.nvs_pipeline import load_config, process_sample


def evaluate_predictions(output_dir, data_root):
    """Evaluate predictions against ground truth (train split only).

    Computes PSNR per sample and the normalized competition score.
    """
    import cv2
    import numpy as np

    scores = []
    for entry in sorted(os.listdir(data_root)):
        sample_dir = os.path.join(data_root, entry)
        if not os.path.isdir(sample_dir):
            continue

        # Check if target exists
        import json
        meta_path = os.path.join(sample_dir, "meta.json")
        with open(meta_path, "r") as f:
            meta = json.load(f)

        target_camera = meta["target_camera"]
        gt_path = os.path.join(sample_dir, "target", f"{target_camera}.jpg")
        pred_path = os.path.join(output_dir, f"{meta['sample_id']}.jpg")

        if not os.path.exists(gt_path):
            print(f"  No GT for {entry}, skipping evaluation")
            continue
        if not os.path.exists(pred_path):
            print(f"  No prediction for {entry}, skipping evaluation")
            continue

        gt = cv2.imread(gt_path).astype(np.float64)
        pred = cv2.imread(pred_path).astype(np.float64)

        # Resize pred to match GT if needed
        if gt.shape != pred.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]))

        mse = np.mean((gt - pred) ** 2)
        if mse == 0:
            psnr = 100.0
        else:
            psnr = 20.0 * np.log10(255.0 / np.sqrt(mse))

        # Normalized score
        score = (np.clip(psnr, 10, 30) - 10) / 20.0 * 100.0
        scores.append(score)
        print(f"  {entry}: PSNR={psnr:.2f} dB, Score={score:.2f}")

    if scores:
        print(f"\n  Mean Score: {np.mean(scores):.2f} (over {len(scores)} samples)")
    return scores


def main():
    parser = argparse.ArgumentParser(description="NVS Competition Batch Inference")
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Path to dataset directory containing sample folders"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output/submission",
        help="Directory to save predicted images"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0",
        help="CUDA device"
    )
    parser.add_argument(
        "--evaluate", action="store_true",
        help="Evaluate predictions against GT (only for train split)"
    )
    parser.add_argument(
        "--voxel_size", type=float, default=0.1,
        help="Voxel size for LiDAR downsampling (meters)"
    )
    parser.add_argument(
        "--max_points", type=int, default=500000,
        help="Maximum number of LiDAR points to use"
    )
    parser.add_argument(
        "--n_iters", type=int, default=None,
        help="Override number of optimization iterations"
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config()

    # Override iterations if specified
    if args.n_iters is not None:
        cfg["mapping"]["Training"]["mapping_itr_num"] = args.n_iters

    # Find all sample directories
    sample_dirs = []
    for entry in sorted(os.listdir(args.data_root)):
        sample_dir = os.path.join(args.data_root, entry)
        if os.path.isdir(sample_dir) and os.path.exists(os.path.join(sample_dir, "meta.json")):
            sample_dirs.append(sample_dir)

    print(f"Found {len(sample_dirs)} samples in {args.data_root}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Process each sample
    total_time = 0
    for i, sample_dir in enumerate(sample_dirs):
        print(f"\n[{i+1}/{len(sample_dirs)}] ", end="")
        t0 = time.time()
        try:
            process_sample(
                sample_dir, args.output_dir, cfg, device=args.device,
                voxel_size=args.voxel_size, max_points=args.max_points,
            )
        except Exception as e:
            print(f"  ERROR processing {sample_dir}: {e}")
            import traceback
            traceback.print_exc()
        elapsed = time.time() - t0
        total_time += elapsed
        print(f"  Time: {elapsed:.1f}s")

    print(f"\nTotal time: {total_time:.1f}s ({total_time/max(len(sample_dirs),1):.1f}s/sample)")

    # Evaluate if requested
    if args.evaluate:
        print("\n=== Evaluation ===")
        evaluate_predictions(args.output_dir, args.data_root)


if __name__ == "__main__":
    main()
