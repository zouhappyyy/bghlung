#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one case through all MedNeXt checkpoints and save 3-view error masks.

For one preprocessed `.npz` case, this script runs inference with every saved
MedNeXtTrainerV2 checkpoint and exports axial/coronal/sagittal middle slices.
The visualization uses RGB error coding:

- green: correctly predicted foreground (true positive)
- red: false positive
- blue: false negative
- black: background / true negative
"""

from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from visualize_models_mednextv2_skip_connection import (
    CKPT_BASE,
    TASK,
    TRAINER_MODULE,
    TRAINER_NAME,
    checkpoint_tag,
    choose_case_files,
    default_dataset_directory,
    default_gt_directory,
    default_plans_path,
    find_epoch_checkpoints,
    load_case_npz,
    load_checkpoint,
)
from visualize_nnunetv2_skip_connection import get_mask_slice, resolve_target


def extract_prediction_mask(network: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        logits = network(x)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if logits.dim() != 5:
        raise RuntimeError(f"Expected 5D logits tensor, got {tuple(logits.shape)}")
    pred = torch.argmax(logits, dim=1)
    return pred[0].detach().cpu().numpy().astype(np.uint8)


def choose_slice_indices(target: Optional[np.ndarray], volume_shape: Tuple[int, int, int]) -> Dict[str, int]:
    if target is not None and np.any(target > 0):
        fg = np.argwhere(target > 0)
        return {
            "axial": int(round((fg[:, 0].min() + fg[:, 0].max()) / 2.0)),
            "coronal": int(round((fg[:, 1].min() + fg[:, 1].max()) / 2.0)),
            "sagittal": int(round((fg[:, 2].min() + fg[:, 2].max()) / 2.0)),
        }

    return {
        "axial": volume_shape[0] // 2,
        "coronal": volume_shape[1] // 2,
        "sagittal": volume_shape[2] // 2,
    }


def make_error_rgb(pred_slice: np.ndarray, gt_slice: np.ndarray) -> np.ndarray:
    pred_fg = pred_slice > 0
    gt_fg = gt_slice > 0

    rgb = np.zeros(pred_slice.shape + (3,), dtype=np.float32)
    rgb[np.logical_and(pred_fg, gt_fg)] = (0.0, 1.0, 0.0)  # TP green
    rgb[np.logical_and(pred_fg, np.logical_not(gt_fg))] = (1.0, 0.0, 0.0)  # FP red
    rgb[np.logical_and(np.logical_not(pred_fg), gt_fg)] = (0.0, 0.0, 1.0)  # FN blue
    return rgb


def save_three_view_error_masks(
    pred_mask: np.ndarray,
    target: np.ndarray,
    out_png: Path,
    title: str,
    slice_indices: Dict[str, int],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, axis in zip(axes, ("axial", "coronal", "sagittal")):
        idx = slice_indices[axis]
        pred_slice = get_mask_slice(pred_mask, axis, idx)
        gt_slice = get_mask_slice(target, axis, idx)
        rgb = make_error_rgb(pred_slice, gt_slice)
        ax.imshow(rgb, interpolation="nearest")
        ax.set_title(f"{axis.title()} ({idx})")
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_mask_montage(
    image_pairs: List[Tuple[str, Path]],
    out_png: Path,
    title: str,
    ncols: int = 4,
) -> None:
    if not image_pairs:
        return

    ncols = max(1, ncols)
    nrows = math.ceil(len(image_pairs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.0, nrows * 5.0))
    axes_list = axes.flatten().tolist() if hasattr(axes, "flatten") else [axes]

    for ax, (label, image_path) in zip(axes_list, image_pairs):
        image = plt.imread(image_path)
        ax.imshow(image)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.axis("off")

    for ax in axes_list[len(image_pairs):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one case through all MedNeXt checkpoints and save RGB 3-view error masks",
    )
    parser.add_argument("--fold", type=int, default=2, help="Fold number")
    parser.add_argument(
        "--case-dir",
        default="/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/nnUNetData2D_plans_v2.1_trgSp_1x1x1_stage0",
        help="Directory containing input .npz files",
    )
    parser.add_argument("--case-id", default=None, help="Specific case ID; defaults to the first case")
    parser.add_argument(
        "--gt-directory",
        default="/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/gt_segmentations",
        help="Directory containing GT .nii.gz files",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory containing model_ep_*.model checkpoints",
    )
    parser.add_argument(
        "--dataset-directory",
        default=None,
        help="Custom dataset directory for trainer initialization",
    )
    parser.add_argument(
        "--plans-file",
        default=None,
        help="Custom plans file",
    )
    parser.add_argument(
        "--include-final",
        action="store_true",
        help="Also process model_final_checkpoint.model if it exists",
    )
    parser.add_argument(
        "--output-dir",
        default="mednext_middle_slice_masks",
        help="Output directory root",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else CKPT_BASE / f"fold_{args.fold}"
    case_dir = Path(args.case_dir)
    gt_directory = Path(args.gt_directory) if args.gt_directory else default_gt_directory(TASK)
    plans_file = args.plans_file or str(default_plans_path())
    dataset_directory = args.dataset_directory or str(default_dataset_directory(TASK))
    output_folder = str(CKPT_BASE / f"fold_{args.fold}")

    print(f"\n{'=' * 72}")
    print("MedNeXt Three-View RGB Error Masks")
    print(f"{'=' * 72}")
    print(f"Task:               {TASK}")
    print(f"Trainer:            {TRAINER_NAME}")
    print(f"Fold:               {args.fold}")
    print(f"Device:             {args.device}")
    print(f"Checkpoint dir:     {checkpoint_dir}")
    print(f"Case dir:           {case_dir}")
    print(f"GT dir:             {gt_directory}")
    print(f"Output dir:         {args.output_dir}")
    print(f"{'=' * 72}\n")

    trainer_module = importlib.import_module(TRAINER_MODULE)
    TrainerClass = getattr(trainer_module, TRAINER_NAME)
    trainer = TrainerClass(
        plans_file,
        args.fold,
        output_folder=output_folder,
        dataset_directory=dataset_directory,
    )
    trainer.initialize(training=False)
    trainer.network.eval().to(args.device)
    print("Initialized trainer")

    checkpoints = find_epoch_checkpoints(checkpoint_dir, args.include_final)
    case_files, resolved_case_dir = choose_case_files([case_dir], args.case_id, max_cases=1)
    case_file = case_files[0]
    case_id = case_file.stem
    expected_in = getattr(trainer.network, "num_input_channels", None) or 1
    data, npz_target = load_case_npz(case_file, expected_in)
    target = resolve_target(
        case_id,
        npz_target,
        gt_directory,
        expected_shape=(int(data.shape[2]), int(data.shape[3]), int(data.shape[4])),
    )
    if target is None:
        raise RuntimeError("GT target is required for RGB error-mask visualization but was not found.")

    x = torch.from_numpy(data).float().to(args.device)
    slice_indices = choose_slice_indices(target, (int(data.shape[2]), int(data.shape[3]), int(data.shape[4])))

    print(f"Resolved case dir:  {resolved_case_dir}")
    print(f"Selected case:      {case_id}")
    print(f"Slice indices:      {slice_indices}")
    print(f"Found {len(checkpoints)} checkpoints\n")

    outdir = Path(args.output_dir) / TASK / TRAINER_NAME / f"fold_{args.fold}" / case_id
    outdir.mkdir(parents=True, exist_ok=True)

    montage_pairs: List[Tuple[str, Path]] = []

    for idx, checkpoint_path in enumerate(checkpoints, start=1):
        ckpt_name = checkpoint_tag(checkpoint_path)
        print(f"[{idx}/{len(checkpoints)}] {ckpt_name}")
        load_checkpoint(trainer, checkpoint_path)
        pred_mask = extract_prediction_mask(trainer.network, x)

        out_png = outdir / f"{ckpt_name}_3views_error_mask.png"
        save_three_view_error_masks(
            pred_mask,
            target,
            out_png,
            title=f"{case_id} | {ckpt_name} | TP green / FP red / FN blue",
            slice_indices=slice_indices,
        )
        montage_pairs.append((ckpt_name, out_png))

    montage_png = outdir / "all_models_3views_error_mask_montage.png"
    save_mask_montage(
        montage_pairs,
        montage_png,
        title=f"{case_id} | MedNeXtTrainerV2 | TP green / FP red / FN blue",
        ncols=4,
    )

    print(f"\nSaved outputs to: {outdir}")
    print(f"Saved montage:    {montage_png}\n")


if __name__ == "__main__":
    main()
