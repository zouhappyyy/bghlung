#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch-visualize BGHNetV4Trainer features for all major stages and epochs.

This script is specialized for BGHNetV4Trainer models trained on
Task530_EsoTJ_30pct. For every checkpoint and every case, it extracts heatmaps
from encoder, downsampling, bottleneck, segmentation decoder, boundary decoder,
upsampling, and deep-supervision output stages.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from feature_heatmap_montage import build_all_case_montages
from visualize_models_nnunetv2_skip_connection import (
    choose_case_files,
    checkpoint_tag,
    find_epoch_checkpoints,
    load_case_npz,
    load_checkpoint,
)
from visualize_nnunetv2_skip_connection import (
    _backward_target_from_logits,
    default_dataset_directory,
    default_gt_directory,
    logits_to_volume,
    make_heatmap,
    pick_middle_slices,
    plot_image_and_target,
    plot_logits_slice,
    plot_multi_view_overlay,
    plot_single_view_overlay,
    resolve_target,
    resize_logits_to_original,
    resize_to_original,
)


TASK = "Task530_EsoTJ_30pct"
TRAINER_NAME = "BGHNetV4Trainer"
TRAINER_MODULE = "nnunet.training.network_training.BGHNetV4Trainer"
CKPT_BASE = (
    Path("/home/fangzheng/zoule/ESO_nnUNet_dataset")
    / TASK
    / f"{TRAINER_NAME}__nnUNetPlansv2.1"
)
NETWORK = "3d_fullres"


def default_plans_path() -> Path:
    return CKPT_BASE / "plans.pkl"


def default_validation_raw_dir(fold: int) -> Path:
    return CKPT_BASE / f"fold_{fold}" / "validation_raw"


def _selector_first(output):
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _selector_second(output):
    if isinstance(output, (tuple, list)):
        return output[1]
    raise RuntimeError("Expected tuple/list output for selector_second")


def discover_stage_specs(network: torch.nn.Module, stage_mode: str) -> List[Tuple[str, torch.nn.Module, Callable]]:
    specs: List[Tuple[str, torch.nn.Module, Callable]] = []
    num_pool = network.num_pool

    for idx, layer in enumerate(network.down_layers):
        encoder_module = getattr(layer, "conv_blocks", None)
        if encoder_module is None:
            encoder_module = getattr(layer, "convnext_blocks", None)
        if encoder_module is None:
            raise RuntimeError(f"Could not find encoder feature module for down_layers[{idx}]")
        specs.append((f"encoder_stage{idx}", encoder_module, _selector_first))
        if stage_mode == "all" and layer.down_or_upsample is not None:
            specs.append((f"downsample_stage{idx}", layer.down_or_upsample, _selector_first))

    specs.append(("bottleneck", network.down_layers[-1], _selector_second))

    stage_ids = list(range(num_pool, -1, -1))
    for idx, stage_id in enumerate(stage_ids):
        seg_layer = network.up_layers[idx]
        bdr_layer = network.up_bdr_layers[idx]

        specs.append((f"seg_decoder_stage{stage_id}", seg_layer.conv_blocks, _selector_first))
        specs.append((f"bdr_decoder_stage{stage_id}", bdr_layer.conv_blocks, _selector_first))

        if stage_mode == "all":
            if seg_layer.down_or_upsample is not None:
                specs.append((f"seg_upsample_stage{stage_id}", seg_layer.down_or_upsample, _selector_first))
            if bdr_layer.down_or_upsample is not None:
                specs.append((f"bdr_upsample_stage{stage_id}", bdr_layer.down_or_upsample, _selector_first))

        if seg_layer.deep_supervision is not None:
            specs.append((f"seg_output_stage{stage_id}", seg_layer.deep_supervision, _selector_first))
        if bdr_layer.deep_supervision is not None:
            specs.append((f"bdr_output_stage{stage_id}", bdr_layer.deep_supervision, _selector_first))

    return specs


def parse_stage_specs(
    network: torch.nn.Module,
    stage_mode: str,
    target_layers: Optional[str],
) -> List[Tuple[str, torch.nn.Module, Callable]]:
    specs = discover_stage_specs(network, stage_mode)
    if not target_layers:
        return specs

    requested = [item.strip() for item in target_layers.split(",") if item.strip()]
    spec_map = {name: (name, module, selector) for name, module, selector in specs}
    missing = [name for name in requested if name not in spec_map]
    if missing:
        available = ", ".join(spec_map.keys())
        raise RuntimeError(f"Unknown target layer(s): {missing}. Available layers: {available}")
    return [spec_map[name] for name in requested]


def extract_stage_heatmap(
    network: torch.nn.Module,
    x: torch.Tensor,
    backend: str,
    stage_name: str,
    stage_module: torch.nn.Module,
    selector: Callable,
    normalize: str,
) -> Tuple[np.ndarray, np.ndarray, str, List[int]]:
    captured: Dict[str, torch.Tensor] = {}
    grad_holder: Dict[str, torch.Tensor] = {}

    def hook_fn(_module, _inputs, output):
        tensor = selector(output)
        if isinstance(tensor, (tuple, list)):
            tensor = tensor[0]
        if not torch.is_tensor(tensor):
            raise RuntimeError(f"Stage '{stage_name}' did not produce a tensor")
        captured["feat"] = tensor
        if backend == "gradcam":
            tensor.retain_grad()
            grad_holder["tensor"] = tensor

    hook = stage_module.register_forward_hook(hook_fn)
    try:
        if backend == "gradcam":
            network.zero_grad(set_to_none=True)
            logits = network(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            score = _backward_target_from_logits(logits)
            score.backward()
        else:
            with torch.no_grad():
                logits = network(x)
    finally:
        hook.remove()

    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    feat_map = captured.get("feat")
    if feat_map is None:
        raise RuntimeError(f"Failed to capture features from '{stage_name}'")

    if backend == "gradcam":
        grad = grad_holder.get("tensor").grad if grad_holder.get("tensor") is not None else None
        if grad is None:
            raise RuntimeError(f"Failed to capture gradients from '{stage_name}'")
        heatmap = make_heatmap(feat_map, "gradcam", grad=grad, normalize=normalize)
    else:
        heatmap = make_heatmap(feat_map, "activation", normalize=normalize)

    logits_volume = logits_to_volume(logits)
    return heatmap, logits_volume, stage_name, list(feat_map.shape)


def save_outputs(
    *,
    outdir: Path,
    case_id: str,
    backend: str,
    normalize: str,
    target: Optional[np.ndarray],
    checkpoint_path: Path,
    checkpoint_name: str,
    layer_name: str,
    feature_shape: List[int],
    data: np.ndarray,
    heatmap_resized: np.ndarray,
    logits_resized: np.ndarray,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    slices = pick_middle_slices(
        (int(data.shape[2]), int(data.shape[3]), int(data.shape[4]))
    )
    stem = f"{checkpoint_name}_{backend}_{normalize}_{layer_name}"

    axial_idx = slices["axial"]
    target_slice = None if target is None else target[axial_idx]
    plot_logits_slice(
        logits_resized[axial_idx],
        outdir / f"{stem}_logits.png",
        f"{case_id} | {checkpoint_name} | {layer_name} | Foreground Logits",
    )
    plot_image_and_target(
        data[0, 0, axial_idx],
        outdir / f"{stem}_image_target.png",
        f"{case_id} | {checkpoint_name} | {layer_name} | Image + Target",
        target_slice,
    )
    plot_single_view_overlay(
        data[0, 0, axial_idx],
        heatmap_resized[axial_idx],
        outdir / f"{stem}_axial.png",
        f"{case_id} | {checkpoint_name} | {backend} | {layer_name}",
        target_slice,
    )
    plot_multi_view_overlay(
        data,
        heatmap_resized,
        outdir / f"{stem}_3views.png",
        f"{case_id} | {checkpoint_name} | {backend} | {layer_name}",
        target,
    )

    meta = {
        "task": TASK,
        "trainer": TRAINER_NAME,
        "network": NETWORK,
        "case_id": case_id,
        "backend": backend,
        "normalize": normalize,
        "layer_name": layer_name,
        "target_feature": "bghnetv4_stage_feature",
        "checkpoint": str(checkpoint_path),
        "checkpoint_name": checkpoint_name,
        "slice_indices": slices,
        "original_shape": [
            int(data.shape[2]),
            int(data.shape[3]),
            int(data.shape[4]),
        ],
        "feature_shape": feature_shape,
        "heatmap_shape": list(heatmap_resized.shape),
        "logits_shape": list(logits_resized.shape),
    }
    (outdir / f"{stem}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-generate BGHNetV4 feature heatmaps for all major stages and saved epochs",
    )
    parser.add_argument("--fold", type=int, default=1, help="Fold number")
    parser.add_argument(
        "--backend",
        choices=["activation", "gradcam"],
        default="activation",
        help="Heatmap generation backend",
    )
    parser.add_argument(
        "--stage-mode",
        choices=["core", "all"],
        default="all",
        help="core: encoder/bottleneck/decoder/output; all: also include down/up sampling",
    )
    parser.add_argument(
        "--target-layers",
        default=None,
        help="Comma-separated custom layer list. Overrides --stage-mode",
    )
    parser.add_argument(
        "--output-dir",
        default="heatmap_output_bghnetv4_all_stages",
        help="Output directory root",
    )
    parser.add_argument(
        "--normalize",
        choices=["quantile", "none"],
        default="quantile",
        help="Heatmap normalization mode",
    )
    parser.add_argument("--case-id", default=None, help="Specific case ID to process")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional case limit")
    parser.add_argument("--checkpoint-dir", default=None, help="Directory containing model checkpoints")
    parser.add_argument("--dataset-directory", default=None, help="Custom dataset directory")
    parser.add_argument("--gt-directory", default=None, help="Directory containing GT .nii.gz files")
    parser.add_argument("--plans-file", default=None, help="Custom plans file")
    parser.add_argument("--validation-raw-dir", default=None, help="Directory containing input .npz files")
    parser.add_argument("--case-dir", default=None, help="Alias for --validation-raw-dir")
    parser.add_argument("--include-final", action="store_true", help="Also process model_final_checkpoint.model")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--skip-montage", action="store_true", help="Do not create per-stage montage images")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else CKPT_BASE / f"fold_{args.fold}"
    case_dir_arg = args.case_dir or args.validation_raw_dir
    case_dir = Path(case_dir_arg) if case_dir_arg else Path(default_validation_raw_dir(args.fold))
    plans_file = args.plans_file or str(default_plans_path())
    dataset_directory = args.dataset_directory or str(default_dataset_directory(TASK))
    dataset_directory_path = Path(dataset_directory)
    gt_directory = Path(args.gt_directory) if args.gt_directory else default_gt_directory(TASK)
    output_folder = str(CKPT_BASE / f"fold_{args.fold}")

    print(f"\n{'=' * 80}")
    print("Batch BGHNetV4Trainer Multi-Stage Heatmap Generation")
    print(f"{'=' * 80}")
    print(f"Task:               {TASK}")
    print(f"Trainer:            {TRAINER_NAME}")
    print(f"Fold:               {args.fold}")
    print(f"Backend:            {args.backend}")
    print(f"Normalize:          {args.normalize}")
    print(f"Stage mode:         {args.stage_mode}")
    print(f"Device:             {args.device}")
    print(f"Checkpoint dir:     {checkpoint_dir}")
    print(f"Requested case dir: {case_dir}")
    print(f"Dataset directory:  {dataset_directory_path}")
    print(f"GT directory:       {gt_directory}")
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

    stage_specs = parse_stage_specs(trainer.network, args.stage_mode, args.target_layers)
    print(f"Stage layers:       {', '.join(name for name, _, _ in stage_specs)}")

    checkpoints = find_epoch_checkpoints(checkpoint_dir, args.include_final)
    case_files, resolved_case_dir = choose_case_files(
        [case_dir, dataset_directory_path],
        args.case_id,
        args.max_cases,
    )
    total_jobs = len(checkpoints) * len(case_files) * len(stage_specs)

    print(f"Found {len(checkpoints)} checkpoints")
    print(f"Found {len(case_files)} cases")
    print(f"Resolved case dir:  {resolved_case_dir}")
    print(f"Total jobs:         {total_jobs}\n")

    expected_in = getattr(trainer.network, "num_input_channels", None)
    case_output_dirs: List[Path] = []
    job_idx = 0

    for checkpoint_path in checkpoints:
        ckpt_name = checkpoint_tag(checkpoint_path)
        print(f"Loading checkpoint: {ckpt_name}")
        load_checkpoint(trainer, checkpoint_path)

        for case_file in case_files:
            case_id = case_file.stem
            data, npz_target = load_case_npz(case_file, expected_in)
            original_shape = (int(data.shape[2]), int(data.shape[3]), int(data.shape[4]))
            target = resolve_target(case_id, npz_target, gt_directory, expected_shape=original_shape)

            outdir = (
                Path(args.output_dir)
                / TASK
                / TRAINER_NAME
                / f"fold_{args.fold}"
                / case_id
            )
            if outdir not in case_output_dirs:
                case_output_dirs.append(outdir)

            for stage_name, stage_module, selector in stage_specs:
                job_idx += 1
                print(f"[{job_idx}/{total_jobs}] {ckpt_name} -> {case_id} -> {stage_name}")

                x = torch.from_numpy(data).float().to(args.device)
                if args.backend == "gradcam":
                    x.requires_grad_(True)

                heatmap, logits_volume, resolved_layer_name, feature_shape = extract_stage_heatmap(
                    trainer.network,
                    x,
                    args.backend,
                    stage_name,
                    stage_module,
                    selector,
                    args.normalize,
                )
                heatmap_resized = resize_to_original(heatmap, original_shape)
                logits_resized = resize_logits_to_original(logits_volume, original_shape)

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
                    logits_resized=logits_resized,
                )

    if not args.skip_montage:
        total_montages = 0
        for stage_name, _, _ in stage_specs:
            marker = f"_{args.backend}_{args.normalize}_{stage_name}_3views.png"
            out_name = f"{args.backend}_{args.normalize}_{stage_name}_montage.png"
            montage_outputs = build_all_case_montages(
                case_dirs=case_output_dirs,
                marker=marker,
                out_name=out_name,
                title_prefix=f"{TASK} | {TRAINER_NAME} | fold_{args.fold} | {stage_name}",
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
