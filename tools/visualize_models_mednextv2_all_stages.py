#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch-visualize MedNeXtTrainerV2 features for all major stages and epochs.

This script is designed for MedNeXtTrainerV2 models trained on
Task530_EsoTJ_30pct. For every checkpoint and every input case, it extracts
heatmaps from multiple MedNeXt stages so that feature evolution can be compared
across both depth and training epoch.

Default stages:
- enc_block_0
- enc_block_1
- enc_block_2
- enc_block_3
- bottleneck
- dec_block_3
- dec_block_2
- dec_block_1
- dec_block_0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from feature_heatmap_montage import build_all_case_montages
from visualize_models_mednextv2_skip_connection import (
    CKPT_BASE,
    TASK,
    TRAINER_MODULE,
    TRAINER_NAME,
    checkpoint_tag,
    choose_case_files,
    default_dataset_directory,
    default_plans_path,
    extract_heatmap,
    find_epoch_checkpoints,
    load_case_npz,
    load_checkpoint,
    save_outputs,
)
import importlib

from visualize_nnunetv2_skip_connection import resize_to_original


CORE_STAGE_LAYERS = [
    "enc_block_0",
    "enc_block_1",
    "enc_block_2",
    "enc_block_3",
    "bottleneck",
    "dec_block_3",
    "dec_block_2",
    "dec_block_1",
    "dec_block_0",
]

TRANSITION_STAGE_LAYERS = [
    "down_0",
    "down_1",
    "down_2",
    "down_3",
    "up_3",
    "up_2",
    "up_1",
    "up_0",
]


def parse_stage_layers(stage_mode: str, custom_layers: str | None) -> List[str]:
    if custom_layers:
        return [layer.strip() for layer in custom_layers.split(",") if layer.strip()]

    if stage_mode == "core":
        return CORE_STAGE_LAYERS
    if stage_mode == "all":
        return CORE_STAGE_LAYERS + TRANSITION_STAGE_LAYERS
    raise ValueError(f"Unknown stage mode: {stage_mode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-generate MedNeXt heatmaps for all major stages and saved epochs",
    )
    parser.add_argument("--fold", type=int, default=2, help="Fold number")
    parser.add_argument(
        "--backend",
        choices=["activation", "gradcam"],
        default="activation",
        help="Heatmap generation backend",
    )
    parser.add_argument(
        "--stage-mode",
        choices=["core", "all"],
        default="core",
        help="core: enc/dec/bottleneck only; all: also include up/down transitions",
    )
    parser.add_argument(
        "--target-layers",
        default=None,
        help="Comma-separated custom layer list. Overrides --stage-mode",
    )
    parser.add_argument(
        "--output-dir",
        default="heatmap_output_mednext_all_stages",
        help="Output directory root",
    )
    parser.add_argument(
        "--normalize",
        choices=["quantile", "none"],
        default="quantile",
        help="Heatmap normalization mode",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Specific case ID to process; defaults to all available cases",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional limit for the number of cases",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--dataset-directory",
        default=None,
        help="Custom dataset directory",
    )
    parser.add_argument(
        "--plans-file",
        default=None,
        help="Custom plans file",
    )
    parser.add_argument(
        "--case-dir",
        default=None,
        help="Directory containing input .npz files",
    )
    parser.add_argument(
        "--include-final",
        action="store_true",
        help="Also process model_final_checkpoint.model if it exists",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--skip-montage",
        action="store_true",
        help="Do not create per-stage montage images after batch processing",
    )
    args = parser.parse_args()

    stage_layers = parse_stage_layers(args.stage_mode, args.target_layers)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else CKPT_BASE / f"fold_{args.fold}"
    plans_file = args.plans_file or str(default_plans_path())
    dataset_directory = args.dataset_directory or str(default_dataset_directory(TASK))
    dataset_directory_path = Path(dataset_directory)
    case_dir = Path(args.case_dir) if args.case_dir else checkpoint_dir / "validation_raw"
    output_folder = str(CKPT_BASE / f"fold_{args.fold}")

    print(f"\n{'=' * 80}")
    print("Batch MedNeXtTrainerV2 Multi-Stage Heatmap Generation")
    print(f"{'=' * 80}")
    print(f"Task:               {TASK}")
    print(f"Trainer:            {TRAINER_NAME}")
    print(f"Fold:               {args.fold}")
    print(f"Backend:            {args.backend}")
    print(f"Stage layers:       {', '.join(stage_layers)}")
    print(f"Device:             {args.device}")
    print(f"Checkpoint dir:     {checkpoint_dir}")
    print(f"Requested case dir: {case_dir}")
    print(f"Dataset directory:  {dataset_directory_path}")
    print(f"Output dir:         {args.output_dir}")
    print(f"{'=' * 80}\n")

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
    case_files, resolved_case_dir = choose_case_files(
        [case_dir, dataset_directory_path],
        args.case_id,
        args.max_cases,
    )
    total_jobs = len(checkpoints) * len(case_files) * len(stage_layers)

    print(f"Found {len(checkpoints)} checkpoints")
    print(f"Found {len(case_files)} cases")
    print(f"Resolved case dir:  {resolved_case_dir}")
    print(f"Total jobs:         {total_jobs}\n")

    expected_in = getattr(trainer.network, "num_input_channels", None) or 1
    case_output_dirs: List[Path] = []
    job_idx = 0

    for checkpoint_path in checkpoints:
        ckpt_name = checkpoint_tag(checkpoint_path)
        print(f"Loading checkpoint: {ckpt_name}")
        load_checkpoint(trainer, checkpoint_path)

        for case_file in case_files:
            case_id = case_file.stem
            data, target = load_case_npz(case_file, expected_in)
            original_shape = (int(data.shape[2]), int(data.shape[3]), int(data.shape[4]))
            outdir = (
                Path(args.output_dir)
                / TASK
                / TRAINER_NAME
                / f"fold_{args.fold}"
                / case_id
            )
            if outdir not in case_output_dirs:
                case_output_dirs.append(outdir)

            for layer_name in stage_layers:
                job_idx += 1
                print(f"[{job_idx}/{total_jobs}] {ckpt_name} -> {case_id} -> {layer_name}")

                x = torch.from_numpy(data).float().to(args.device)
                if args.backend == "gradcam":
                    x.requires_grad_(True)

                heatmap, resolved_layer_name, feature_shape = extract_heatmap(
                    trainer.network,
                    x,
                    args.backend,
                    layer_name,
                    args.normalize,
                )
                heatmap_resized = resize_to_original(heatmap, original_shape)

                save_outputs(
                    outdir=outdir,
                    case_id=case_id,
                    backend=args.backend,
                    normalize=args.normalize,
                    target=target,
                    checkpoint_path=checkpoint_path,
                    checkpoint_name=ckpt_name,
                    layer_name=resolved_layer_name,
                    feature_shape=feature_shape,
                    data=data,
                    heatmap_resized=heatmap_resized,
                )

    if not args.skip_montage:
        total_montages = 0
        for layer_name in stage_layers:
            marker = f"_{args.backend}_{args.normalize}_{layer_name.replace('.', '_')}_3views.png"
            out_name = f"{args.backend}_{args.normalize}_{layer_name.replace('.', '_')}_montage.png"
            montage_outputs = build_all_case_montages(
                case_dirs=case_output_dirs,
                marker=marker,
                out_name=out_name,
                title_prefix=f"{TASK} | {TRAINER_NAME} | fold_{args.fold} | {layer_name}",
                ncols=4,
            )
            total_montages += len(montage_outputs)
        print(f"Created {total_montages} montage image(s)")

    print(f"\n{'=' * 80}")
    print("Batch multi-stage visualization complete")
    print(
        "Generated outputs under: "
        f"{Path(args.output_dir) / TASK / TRAINER_NAME / f'fold_{args.fold}'}"
    )
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
