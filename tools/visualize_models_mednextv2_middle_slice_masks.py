#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run one case through all MedNeXt checkpoints and save middle-slice masks.

This script loads a single preprocessed `.npz` case, runs inference with every
saved MedNeXtTrainerV2 checkpoint, and exports the middle axial slice of the
predicted segmentation as an RGB mask image.
"""

from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.colors as mcolors
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
    default_plans_path,
    find_epoch_checkpoints,
    load_case_npz,
    load_checkpoint,
)


RGB_MASK_COLORS = [
    "#000000",  # background
    "#ff0000",  # class 1: red
    "#00ff00",  # class 2: green
    "#0000ff",  # class 3: blue
]


def get_mask_cmap(num_classes: int) -> mcolors.ListedColormap:
    colors = list(RGB_MASK_COLORS)
    if num_classes > len(colors):
        extra = plt.cm.tab10(np.linspace(0, 1, num_classes - len(colors)))
        colors.extend(extra)
    return mcolors.ListedColormap(colors[:num_classes])


def extract_prediction_mask(network: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        logits = network(x)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if logits.dim() != 5:
        raise RuntimeError(f"Expected 5D logits tensor, got {tuple(logits.shape)}")

    pred = torch.argmax(logits, dim=1)
    return pred[0].detach().cpu().numpy().astype(np.uint8)


def save_middle_slice_mask(
    pred_mask: np.ndarray,
    out_png: Path,
    title: str,
    num_classes: int,
) -> int:
    axial_idx = pred_mask.shape[0] // 2
    mask_slice = pred_mask[axial_idx]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(
        mask_slice,
        cmap=get_mask_cmap(max(num_classes, int(mask_slice.max()) + 1)),
        interpolation="nearest",
        vmin=0,
        vmax=max(num_classes - 1, int(mask_slice.max())),
    )
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return axial_idx


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
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 4.5))
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
        description="Run one case through all MedNeXt checkpoints and save middle-slice RGB masks",
    )
    parser.add_argument("--fold", type=int, default=2, help="Fold number")
    parser.add_argument(
        "--case-dir",
        default="/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/nnUNetData2D_plans_v2.1_trgSp_1x1x1_stage0",
        help="Directory containing input .npz files",
    )
    parser.add_argument("--case-id", default=None, help="Specific case ID; defaults to the first case")
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
    plans_file = args.plans_file or str(default_plans_path())
    dataset_directory = args.dataset_directory or str(default_dataset_directory(TASK))
    output_folder = str(CKPT_BASE / f"fold_{args.fold}")

    print(f"\n{'=' * 72}")
    print("MedNeXt Middle Slice Segmentation Masks")
    print(f"{'=' * 72}")
    print(f"Task:               {TASK}")
    print(f"Trainer:            {TRAINER_NAME}")
    print(f"Fold:               {args.fold}")
    print(f"Device:             {args.device}")
    print(f"Checkpoint dir:     {checkpoint_dir}")
    print(f"Case dir:           {case_dir}")
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
    data, _ = load_case_npz(case_file, expected_in)
    x = torch.from_numpy(data).float().to(args.device)

    print(f"Resolved case dir:  {resolved_case_dir}")
    print(f"Selected case:      {case_id}")
    print(f"Found {len(checkpoints)} checkpoints\n")

    outdir = Path(args.output_dir) / TASK / TRAINER_NAME / f"fold_{args.fold}" / case_id
    outdir.mkdir(parents=True, exist_ok=True)

    montage_pairs: List[Tuple[str, Path]] = []
    num_classes = getattr(trainer.network, "num_classes", 2)

    for idx, checkpoint_path in enumerate(checkpoints, start=1):
        ckpt_name = checkpoint_tag(checkpoint_path)
        print(f"[{idx}/{len(checkpoints)}] {ckpt_name}")
        load_checkpoint(trainer, checkpoint_path)
        pred_mask = extract_prediction_mask(trainer.network, x)

        out_png = outdir / f"{ckpt_name}_middle_mask.png"
        axial_idx = save_middle_slice_mask(
            pred_mask,
            out_png,
            title=f"{case_id} | {ckpt_name} | axial",
            num_classes=num_classes,
        )
        montage_pairs.append((ckpt_name, out_png))

    montage_png = outdir / "all_models_middle_mask_montage.png"
    save_mask_montage(
        montage_pairs,
        montage_png,
        title=f"{case_id} | MedNeXtTrainerV2 | axial slice {axial_idx}",
        ncols=4,
    )

    print(f"\nSaved outputs to: {outdir}")
    print(f"Saved montage:    {montage_png}\n")


if __name__ == "__main__":
    main()
