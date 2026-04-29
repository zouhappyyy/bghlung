#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch-visualize nnUNetTrainerV2 features for all major stages and epochs.

This script is specialized for nnUNetTrainerV2 models trained on
Task530_EsoTJ_30pct. For every checkpoint and every case, it extracts heatmaps
from multiple Generic_UNet stages so feature evolution can be compared across
both network depth and training epoch.

Default stages:
- conv_blocks_context.0 ... encoder stages
- conv_blocks_context.<last> ... bottleneck
- conv_blocks_localization.0 ... decoder stages

Optional transition stages include:
- td.* ... downsampling
- tu.* ... upsampling
"""

from __future__ import annotations

import argparse
import json
import importlib
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from feature_heatmap_montage import build_all_case_montages
from visualize_models_nnunetv2_skip_connection import (
    CKPT_BASE,
    TASK,
    TRAINER_NAME,
    checkpoint_tag,
    choose_case_files,
    default_dataset_directory,
    default_gt_directory,
    default_plans_path,
    default_validation_raw_dir,
    find_epoch_checkpoints,
    load_case_npz,
    load_checkpoint,
    resolve_target,
)
from visualize_nnunetv2_skip_connection import (
    _backward_target_from_logits,
    _unwrap_module,
    logits_to_volume,
    make_heatmap,
    plot_image_and_target,
    plot_logits_slice,
    plot_multi_view_overlay,
    plot_single_view_overlay,
    pick_middle_slices,
    resize_logits_to_original,
    resize_to_original,
)


TRAINER_MODULE = "nnunet.training.network_training.nnUNetTrainerV2"


def discover_stage_layers(network: torch.nn.Module, stage_mode: str) -> List[str]:
    net = _unwrap_module(network)
    num_context = len(net.conv_blocks_context)
    num_localization = len(net.conv_blocks_localization)

    layers: List[str] = []
    for idx in range(num_context):
        layers.append(f"conv_blocks_context.{idx}")

    for idx in range(num_localization):
        layers.append(f"conv_blocks_localization.{idx}")

    if stage_mode == "all":
        for idx in range(len(net.td)):
            layers.append(f"td.{idx}")
        for idx in range(len(net.tu)):
            layers.append(f"tu.{idx}")

    return layers


def parse_stage_layers(network: torch.nn.Module, stage_mode: str, custom_layers: str | None) -> List[str]:
    if custom_layers:
        return [layer.strip() for layer in custom_layers.split(",") if layer.strip()]
    return discover_stage_layers(network, stage_mode)


def find_target_layer(network: torch.nn.Module, target_layer: str) -> Tuple[str, torch.nn.Module]:
    net = _unwrap_module(network)
    module_dict = dict(net.named_modules())

    if target_layer in module_dict:
        return target_layer, module_dict[target_layer]

    if hasattr(net, target_layer):
        return target_layer, getattr(net, target_layer)

    preview = ", ".join(list(module_dict.keys())[:30])
    raise RuntimeError(
        f"Could not find target layer '{target_layer}'. "
        f"Available module examples: {preview}"
    )


def extract_heatmap(
    network: torch.nn.Module,
    x: torch.Tensor,
    backend: str,
    target_layer: str,
    normalize: str,
) -> Tuple[torch.Tensor, torch.Tensor, str, List[int]]:
    net = _unwrap_module(network)
    layer_name, target_module = find_target_layer(net, target_layer)

    if backend == "gradcam":
        feat_holder: Dict[str, torch.Tensor] = {}
        grad_holder: Dict[str, torch.Tensor] = {}

        def fwd_hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            if torch.is_tensor(output):
                feat_holder["feat"] = output
                output.retain_grad()
                grad_holder["tensor"] = output

        hook = target_module.register_forward_hook(fwd_hook)
        try:
            network.zero_grad(set_to_none=True)
            logits = net(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            score = _backward_target_from_logits(logits)
            score.backward()
        finally:
            hook.remove()

        feat_map = feat_holder.get("feat")
        grad = grad_holder.get("tensor").grad if grad_holder.get("tensor") is not None else None
        if feat_map is None:
            raise RuntimeError(f"Failed to capture features from '{layer_name}'")
        if grad is None:
            raise RuntimeError(f"Failed to capture gradients from '{layer_name}'")

        heatmap = make_heatmap(feat_map, "gradcam", grad=grad, normalize=normalize)
        logits_volume = logits_to_volume(logits)
        return heatmap, logits_volume, layer_name, list(feat_map.shape)

    captured: Dict[str, torch.Tensor] = {}

    def hook_fn(_module, _inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        if torch.is_tensor(output):
            captured["feat"] = output

    hook = target_module.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            logits = net(x)
    finally:
        hook.remove()

    feat_map = captured.get("feat")
    if feat_map is None:
        raise RuntimeError(f"Failed to capture features from '{layer_name}'")

    heatmap = make_heatmap(feat_map, "activation", normalize=normalize)
    logits_volume = logits_to_volume(logits)
    return heatmap, logits_volume, layer_name, list(feat_map.shape)


def save_outputs(
    *,
    outdir: Path,
    case_id: str,
    backend: str,
    normalize: str,
    target: torch.Tensor | None,
    checkpoint_path: Path,
    checkpoint_name: str,
    layer_name: str,
    feature_shape: List[int],
    data,
    heatmap_resized,
    logits_resized,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    slices = pick_middle_slices(
        (int(data.shape[2]), int(data.shape[3]), int(data.shape[4]))
    )
    layer_tag = layer_name.replace(".", "_")
    stem = f"{checkpoint_name}_{backend}_{normalize}_{layer_tag}"

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
        "network": "3d_fullres",
        "case_id": case_id,
        "backend": backend,
        "normalize": normalize,
        "layer_name": layer_name,
        "target_feature": "nnunet_stage_feature",
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
        description="Batch-generate nnUNet feature heatmaps for all major stages and saved epochs",
    )
    parser.add_argument("--fold", type=int, default=1, help="Fold number (0-4)")
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
        help="core: encoder/bottleneck/decoder only; all: also include td/tu transitions",
    )
    parser.add_argument(
        "--target-layers",
        default=None,
        help="Comma-separated custom layer list. Overrides --stage-mode",
    )
    parser.add_argument(
        "--output-dir",
        default="heatmap_output_nnunet_all_stages",
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
    print("Batch nnUNetTrainerV2 Multi-Stage Heatmap Generation")
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

    stage_layers = parse_stage_layers(trainer.network, args.stage_mode, args.target_layers)
    print(f"Stage layers:       {', '.join(stage_layers)}")

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

            for layer_name in stage_layers:
                job_idx += 1
                print(f"[{job_idx}/{total_jobs}] {ckpt_name} -> {case_id} -> {layer_name}")

                x = torch.from_numpy(data).float().to(args.device)
                if args.backend == "gradcam":
                    x.requires_grad_(True)

                heatmap, logits_volume, resolved_layer_name, feature_shape = extract_heatmap(
                    trainer.network,
                    x,
                    args.backend,
                    layer_name,
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
