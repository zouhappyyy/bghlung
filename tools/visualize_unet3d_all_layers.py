#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize UNet3D features for every major layer on a single case.

This script is specialized for the UNet3DTrainer checkpoint layout used in
Task530_EsoTJ_30pct. It loads one case, runs one UNet3D checkpoint, extracts
feature maps from every encoder / pooling / bottleneck / upsampling / decoder
stage, and saves heatmap visualizations for each layer.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from feature_heatmap_montage import build_case_montage
from visualize_models_nnunetv2_skip_connection import (
    choose_case_files,
    load_case_npz,
    load_checkpoint,
)
from visualize_nnunetv2_skip_connection import (
    _backward_target_from_logits,
    _unwrap_module,
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
TRAINER_NAME = "UNet3DTrainer"
TRAINER_MODULE = "nnunet.training.network_training.UNet3DTrainer"
REPO_ROOT = Path(__file__).resolve().parents[1]
CKPT_BASE = REPO_ROOT / "ckpt" / "nnUNet" / "3d_fullres" / TASK / f"{TRAINER_NAME}__nnUNetPlansv2.1"
DEFAULT_CASE_DIR = Path(
    "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/"
    "nnUNetData2D_plans_v2.1_trgSp_1x1x1_stage0"
)
DEFAULT_GT_DIR = Path(
    "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task530_EsoTJ_30pct/"
    "gt_segmentations"
)


def default_plans_path() -> Path:
    return CKPT_BASE / "plans.pkl"


def default_checkpoint_dir(fold: int) -> Path:
    return CKPT_BASE / f"fold_{fold}"


def default_checkpoint_path(fold: int) -> Path:
    checkpoint_dir = default_checkpoint_dir(fold)
    best_ckpt = checkpoint_dir / "model_best.model"
    if best_ckpt.is_file():
        return best_ckpt

    final_ckpt = checkpoint_dir / "model_final_checkpoint.model"
    if final_ckpt.is_file():
        return final_ckpt

    epoch_ckpts = sorted(checkpoint_dir.glob("model_ep_*.model"))
    if epoch_ckpts:
        return epoch_ckpts[-1]

    raise RuntimeError(f"No checkpoint found in {checkpoint_dir}")


def discover_stage_layers(network: torch.nn.Module) -> List[str]:
    net = _unwrap_module(network)
    ordered_layers = [
        "encoder1",
        "pool1",
        "encoder2",
        "pool2",
        "encoder3",
        "pool3",
        "encoder4",
        "pool4",
        "bottleneck",
        "upconv4",
        "decoder4",
        "upconv3",
        "decoder3",
        "upconv2",
        "decoder2",
        "upconv1",
        "decoder1",
        "conv",
    ]
    missing = [name for name in ordered_layers if not hasattr(net, name)]
    if missing:
        raise RuntimeError(f"UNet3D is missing expected layers: {missing}")
    return ordered_layers


def find_target_layer(network: torch.nn.Module, target_layer: str) -> Tuple[str, torch.nn.Module]:
    net = _unwrap_module(network)
    module_dict = dict(net.named_modules())

    if target_layer in module_dict:
        return target_layer, module_dict[target_layer]

    if hasattr(net, target_layer):
        return target_layer, getattr(net, target_layer)

    preview = ", ".join(list(module_dict.keys())[:40])
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


def save_layer_outputs(
    *,
    outdir: Path,
    case_id: str,
    checkpoint_name: str,
    backend: str,
    normalize: str,
    layer_name: str,
    feature_shape: List[int],
    data,
    heatmap_resized,
    target,
) -> None:
    slices = pick_middle_slices(
        (int(data.shape[2]), int(data.shape[3]), int(data.shape[4]))
    )
    axial_idx = slices["axial"]
    layer_tag = layer_name.replace(".", "_")
    stem = f"{layer_tag}_{backend}_{normalize}"
    target_slice = None if target is None else target[axial_idx]

    plot_single_view_overlay(
        data[0, 0, axial_idx],
        heatmap_resized[axial_idx],
        outdir / f"{stem}_axial.png",
        f"{case_id} | {checkpoint_name} | {layer_name}",
        target_slice,
    )
    plot_multi_view_overlay(
        data,
        heatmap_resized,
        outdir / f"{stem}_3views.png",
        f"{case_id} | {checkpoint_name} | {layer_name}",
        target,
    )

    meta = {
        "task": TASK,
        "trainer": TRAINER_NAME,
        "case_id": case_id,
        "backend": backend,
        "normalize": normalize,
        "layer_name": layer_name,
        "checkpoint_name": checkpoint_name,
        "feature_shape": feature_shape,
        "heatmap_shape": list(heatmap_resized.shape),
        "slice_indices": slices,
        "original_shape": [
            int(data.shape[2]),
            int(data.shape[3]),
            int(data.shape[4]),
        ],
    }
    (outdir / f"{stem}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize every UNet3D stage on a single case"
    )
    parser.add_argument("--fold", type=int, default=1, help="Fold number")
    parser.add_argument(
        "--backend",
        choices=["activation", "gradcam"],
        default="activation",
        help="Heatmap backend",
    )
    parser.add_argument(
        "--normalize",
        choices=["quantile", "none"],
        default="quantile",
        help="Heatmap normalization mode",
    )
    parser.add_argument("--case-id", default=None, help="Specific case ID to process")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path. Defaults to model_best.model if available",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Optional checkpoint directory override",
    )
    parser.add_argument(
        "--plans-file",
        default=None,
        help="Custom plans.pkl path",
    )
    parser.add_argument(
        "--dataset-directory",
        default=str(DEFAULT_CASE_DIR.parent),
        help="Dataset root used by trainer",
    )
    parser.add_argument(
        "--case-dir",
        default=str(DEFAULT_CASE_DIR),
        help="Directory containing input .npz files",
    )
    parser.add_argument(
        "--gt-directory",
        default=str(DEFAULT_GT_DIR),
        help="Directory containing GT .nii.gz files",
    )
    parser.add_argument(
        "--output-dir",
        default="heatmap_output_unet3d_all_layers",
        help="Output directory root",
    )
    parser.add_argument(
        "--target-layers",
        default=None,
        help="Comma-separated custom layer list. Defaults to all UNet3D stages",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else default_checkpoint_dir(args.fold)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint_path(args.fold)
    plans_file = args.plans_file or str(default_plans_path())
    dataset_directory = args.dataset_directory
    case_dir = Path(args.case_dir)
    gt_directory = Path(args.gt_directory) if args.gt_directory else None

    print(f"\n{'=' * 80}")
    print("UNet3D All-Layer Heatmap Visualization")
    print(f"{'=' * 80}")
    print(f"Task:               {TASK}")
    print(f"Trainer:            {TRAINER_NAME}")
    print(f"Fold:               {args.fold}")
    print(f"Backend:            {args.backend}")
    print(f"Normalize:          {args.normalize}")
    print(f"Checkpoint dir:     {checkpoint_dir}")
    print(f"Checkpoint:         {checkpoint_path}")
    print(f"Case dir:           {case_dir}")
    print(f"GT directory:       {gt_directory}")
    print(f"Output dir:         {args.output_dir}")
    print(f"{'=' * 80}\n")

    trainer_module = importlib.import_module(TRAINER_MODULE)
    TrainerClass = getattr(trainer_module, TRAINER_NAME)
    trainer = TrainerClass(
        plans_file=plans_file,
        fold=args.fold,
        output_folder=str(checkpoint_dir),
        dataset_directory=dataset_directory,
        batch_dice=True,
        stage=None,
        unpack_data=False,
        deterministic=True,
        fp16=False,
    )
    trainer.initialize(training=False)
    load_checkpoint(trainer, checkpoint_path)

    device = torch.device(args.device)
    trainer.network.to(device)
    trainer.network.eval()

    stage_layers = (
        [layer.strip() for layer in args.target_layers.split(",") if layer.strip()]
        if args.target_layers
        else discover_stage_layers(trainer.network)
    )
    print(f"Resolved layers ({len(stage_layers)}): {', '.join(stage_layers)}")

    case_files, resolved_case_dir = choose_case_files([case_dir], args.case_id, max_cases=1)
    case_file = case_files[0]
    case_id = case_file.stem
    print(f"Using case:         {case_id}")
    print(f"Resolved case dir:  {resolved_case_dir}")

    encoder1 = getattr(_unwrap_module(trainer.network), "encoder1")
    expected_in = getattr(encoder1[0], "in_channels", None)
    data, npz_target = load_case_npz(case_file, expected_in)
    target = resolve_target(
        case_id=case_id,
        npz_target=npz_target,
        gt_directory=gt_directory,
        expected_shape=(int(data.shape[2]), int(data.shape[3]), int(data.shape[4])),
    )

    x = torch.from_numpy(data).to(device=device, dtype=torch.float32)

    common_outdir = Path(args.output_dir) / TASK / TRAINER_NAME / f"fold_{args.fold}" / case_id
    common_outdir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        logits = trainer.network(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
    logits_volume = logits_to_volume(logits)
    logits_resized = resize_logits_to_original(
        logits_volume,
        (int(data.shape[2]), int(data.shape[3]), int(data.shape[4])),
    )
    axial_idx = pick_middle_slices(
        (int(data.shape[2]), int(data.shape[3]), int(data.shape[4]))
    )["axial"]
    target_slice = None if target is None else target[axial_idx]
    checkpoint_name = checkpoint_path.stem.replace(".model", "")

    plot_logits_slice(
        logits_resized[axial_idx],
        common_outdir / "case_logits.png",
        f"{case_id} | {checkpoint_name} | Foreground Logits",
    )
    plot_image_and_target(
        data[0, 0, axial_idx],
        common_outdir / "case_image_target.png",
        f"{case_id} | {checkpoint_name} | Image + Target",
        target_slice,
    )

    for idx, layer_name in enumerate(stage_layers, start=1):
        print(f"[{idx}/{len(stage_layers)}] {layer_name}")
        heatmap, _logits_volume, resolved_layer_name, feature_shape = extract_heatmap(
            trainer.network,
            x,
            backend=args.backend,
            target_layer=layer_name,
            normalize=args.normalize,
        )
        heatmap_resized = resize_to_original(
            heatmap,
            (int(data.shape[2]), int(data.shape[3]), int(data.shape[4])),
        )
        save_layer_outputs(
            outdir=common_outdir,
            case_id=case_id,
            checkpoint_name=checkpoint_name,
            backend=args.backend,
            normalize=args.normalize,
            layer_name=resolved_layer_name,
            feature_shape=feature_shape,
            data=data,
            heatmap_resized=heatmap_resized,
            target=target,
        )

    marker = f"_{args.backend}_{args.normalize}_3views.png"
    montage_path = build_case_montage(
        case_dir=common_outdir,
        marker=marker,
        out_name=f"all_layers_{args.backend}_{args.normalize}_montage.png",
        title=f"{TRAINER_NAME} all layers | {case_id} | {checkpoint_name}",
        ncols=4,
    )
    if montage_path is not None:
        print(f"Saved montage:      {montage_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
