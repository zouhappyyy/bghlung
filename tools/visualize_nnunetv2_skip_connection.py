#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nnUNetTrainerV2 first skip connection feature visualization tool.

This script is specialized for nnUNetTrainerV2 models trained on Task530_EsoTJ_30pct.
It extracts features from the first skip connection layer and generates multi-view
(axial, coronal, sagittal) heatmap visualizations.

Features:
- Automatic model loading for nnUNetTrainerV2
- First skip connection feature heatmap (conv_blocks_localization[0])
- Multi-view heatmap generation (axial, coronal, sagittal)
- Support for activation-based and gradient-based (Grad-CAM) visualizations
- Automatic checkpoint discovery
- Detailed metadata logging

Outputs:
- multi-view overlay .png (all three views)
- single-view overlay .png (axial view)
- meta .json (metadata and configuration)
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================================
# Configuration and Setup
# ============================================================================

TASK = "Task530_EsoTJ_30pct"
TRAINER_NAME = "nnUNetTrainerV2"
NETWORK = "3d_fullres"
HEATMAP_IMSHOW_KWARGS = {"cmap": "jet", "vmin": 0, "vmax": 1, "interpolation": "bilinear"}
ACTIVATION_TOPK = 8
HEATMAP_CLIP_QUANTILE = 0.99
HEATMAP_DISPLAY_QUANTILE = 0.99

# Hardcoded paths for this specific trainer/task setup
CKPT_BASE = Path("/home/fangzheng/zoule/ESO_nnUNet_dataset") / TASK / f"{TRAINER_NAME}__nnUNetPlansv2.1"


# ============================================================================
# Utilities
# ============================================================================

def default_task_dir(task: str) -> str:
    return task if task.startswith("Task") else f"Task{task}"


def default_checkpoint_path(fold: int) -> Path:
    return CKPT_BASE / f"fold_{fold}" / "model_final_checkpoint.model"


def default_plans_path() -> Path:
    return CKPT_BASE / "plans.pkl"

def find_all_checkpoints(ckpt_dir: Path):
    ckpts = sorted(ckpt_dir.glob("model_ep_*.model"))
    if not ckpts:
        raise RuntimeError(f"No checkpoint found in {ckpt_dir}")
    return ckpts

def default_dataset_directory(task: str) -> Path:
    from nnunet.paths import preprocessing_output_dir
    if preprocessing_output_dir is None:
        raise RuntimeError("nnUNet preprocessing_output_dir is not configured in nnunet.paths")
    return Path(preprocessing_output_dir) / default_task_dir(task)


def default_validation_raw_dir(fold: int) -> Path:
    return CKPT_BASE / f"fold_{fold}" / "validation_raw"


def load_checkpoint_if_available(trainer, checkpoint_path: Optional[str]) -> None:
    if checkpoint_path is None:
        return
    ckpt = Path(checkpoint_path)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(str(ckpt), map_location=torch.device("cpu"))
    trainer.network.load_state_dict(state["state_dict"], strict=False)
    print(f"鉁?Loaded checkpoint: {ckpt}")


def choose_case_file(validation_raw_dir: Path, case_id: Optional[str]) -> Path:
    files = sorted(validation_raw_dir.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No npz files found in {validation_raw_dir}")
    if case_id is None:
        return files[0]
    for f in files:
        if f.stem == case_id:
            return f
    raise FileNotFoundError(f"Case {case_id} not found in {validation_raw_dir}")


def load_case_npz(case_file: Path, expected_in: Optional[int]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load case from npz file."""
    sample = np.load(case_file, allow_pickle=True)
    data_key = "data" if "data" in sample.files else sample.files[0]
    target_key = "target" if "target" in sample.files else None

    data = sample[data_key]
    if data.ndim == 4:
        data = data[None]
    if expected_in is not None:
        if data.shape[1] < expected_in:
            raise RuntimeError(f"Input channels in {case_file} are {data.shape[1]}, but model expects {expected_in}")
        data = data[:, :expected_in]
    else:
        data = data[:, :1]

    target = None
    if target_key is not None:
        target = sample[target_key]
        if target.ndim == 4:
            target = target[0]
        if target.ndim != 3:
            raise RuntimeError(f"Expected target to be 3D, got {target.shape} from {case_file}")
    return data.astype(np.float32), None if target is None else target.astype(np.int64)


def _backward_target_from_logits(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() < 2:
        raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
    target_channel = 1 if logits.shape[1] > 1 else 0
    return logits[:, target_channel].mean()


# ==========================================================================
# Feature Extraction
# ==========================================================================

def _unwrap_module(network: torch.nn.Module) -> torch.nn.Module:
    return network.module if hasattr(network, "module") else network


def _first_module_from_container(container: torch.nn.Module) -> Tuple[str, torch.nn.Module]:
    # if isinstance(container, torch.nn.ModuleList):
    #     if len(container) == 0:
    #         raise RuntimeError("conv_blocks_localization is empty")
    #     return "0", container[0]
    # if isinstance(container, torch.nn.Sequential):
    #     if len(container) == 0:
    #         raise RuntimeError("conv_blocks_localization is empty")
    #     return "0", container[0]
    # if isinstance(container, torch.nn.Module):
    #     for child_name, child in container.named_children():
    #         return child_name, child
    if isinstance(container, torch.nn.ModuleList):
        if len(container) == 0:
            raise RuntimeError("conv_blocks_localization is empty")
        idx = len(container) - 1  # 鉁?鍙栨渶鍚庝竴涓?
        return str(idx), container[idx]

    if isinstance(container, torch.nn.Sequential):
        if len(container) == 0:
            raise RuntimeError("conv_blocks_localization is empty")
        idx = len(container) - 1
        return str(idx), container[idx]

    if isinstance(container, torch.nn.Module):
        children = list(container.named_children())
        if not children:
            raise RuntimeError("No children in localization container")
        return children[-1]  # 鉁?鏈€鍚庝竴涓?
    raise RuntimeError(f"Unsupported localization container type: {type(container).__name__}")


def find_first_skip_connection(network: torch.nn.Module) -> Tuple[str, torch.nn.Module]:
    """Find the first decoder/localization block for nnUNetTrainerV2.

    nnUNet's `conv_blocks_localization` is usually a ModuleList. We want the
    first real child module rather than the container itself.
    """
    net = _unwrap_module(network)

    if hasattr(net, "conv_blocks_localization"):
        container = getattr(net, "conv_blocks_localization")
        child_name, child = _first_module_from_container(container)
        full_name = f"conv_blocks_localization.{child_name}"
        print(f"鉁?Found first skip layer: {full_name}")
        return full_name, child

    for name, module in net.named_modules():
        if "conv_blocks_localization" in name and name.split(".")[-1].isdigit():
            print(f"鉁?Found first skip layer (fallback): {name}")
            return name, module

    raise RuntimeError("Could not find first skip connection layer in network")


def extract_skip_connection_features(
    network: torch.nn.Module,
    x: torch.Tensor,
    backend: str = "activation"
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """Extract features from the first skip connection layer."""
    captured: Dict[str, torch.Tensor] = {}
    net = _unwrap_module(network)
    layer_name, skip_module = find_first_skip_connection(net)

    def hook_fn(name: str):
        def _hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            if torch.is_tensor(output):
                captured[name] = output
        return _hook

    hook = skip_module.register_forward_hook(hook_fn(layer_name))
    try:
        logits = net(x)
    finally:
        hook.remove()

    feat = captured.get(layer_name)
    if feat is None:
        raise RuntimeError(f"Failed to capture features from '{layer_name}'")
    if feat.dim() != 5:
        raise RuntimeError(f"Expected 5D feature map, got shape {tuple(feat.shape)}")
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    return logits, feat, layer_name


# ============================================================================
# Heatmap Generation
# ============================================================================

def make_heatmap(
    feature_map: torch.Tensor,
    backend: str,
    grad: Optional[torch.Tensor] = None,
    normalize: str = "quantile",
) -> np.ndarray:
    """Generate heatmap from feature map.

    Args:
        feature_map: (B, C, D, H, W) tensor
        backend: "activation" or "gradcam"
        grad: gradients for grad-CAM (B, C, D, H, W)
        normalize: "quantile" or "none"

    Returns:
        (D, H, W) normalized heatmap as numpy array
    """
    if backend == "activation":
        # Averaging only the strongest channels keeps salient regions while
        # avoiding the over-sparse maps produced by a pure channel-wise max.
        k = min(ACTIVATION_TOPK, feature_map.shape[1])
        topk = torch.topk(feature_map.abs(), k=k, dim=1).values
        cam = topk.mean(dim=1)
    elif backend == "gradcam":
        if grad is None:
            raise RuntimeError("Grad-CAM requires gradients")
        # Weighted by gradient
        weights = grad.mean(dim=(2, 3, 4), keepdim=True)
        cam = torch.relu((weights * feature_map).sum(dim=1))
    else:
        raise ValueError(f"Unknown backend: {backend}")

    cam = cam[0]
    cam = cam - cam.min()
    if normalize == "quantile":
        clip_value = torch.quantile(cam.reshape(-1), HEATMAP_CLIP_QUANTILE)
        cam = torch.clamp(cam / (clip_value + 1e-8), 0, 1)
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization mode: {normalize}")
    return cam.detach().cpu().numpy()


def resolve_heatmap_imshow_kwargs(heatmap: np.ndarray) -> Dict[str, float | str]:
    if heatmap.size == 0:
        return dict(HEATMAP_IMSHOW_KWARGS)

    max_value = float(np.max(heatmap))
    if max_value <= 1.0 + 1e-6:
        return dict(HEATMAP_IMSHOW_KWARGS)

    vmax = float(np.quantile(heatmap.reshape(-1), HEATMAP_DISPLAY_QUANTILE))
    if vmax <= 0:
        vmax = max_value if max_value > 0 else 1.0
    return {
        "cmap": HEATMAP_IMSHOW_KWARGS["cmap"],
        "vmin": 0,
        "vmax": vmax,
        "interpolation": HEATMAP_IMSHOW_KWARGS["interpolation"],
    }


def resize_to_original(arr: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    """Resize heatmap to original volume dimensions using nearest neighbor."""
    t = torch.from_numpy(arr[None, None].astype(np.float32))
    t = F.interpolate(t, size=target_shape, mode="nearest")
    return t[0, 0].cpu().numpy()


# ============================================================================
# Visualization
# ============================================================================

def get_view_slices(
    data: np.ndarray,
    heatmap: np.ndarray,
    axis: str,
    idx: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract 2D slices from 3D volumes along a given axis.

    Args:
        data: (1, 1, D, H, W)
        heatmap: (D, H, W)
        axis: "axial", "coronal", or "sagittal"
        idx: slice index

    Returns:
        (image_slice, heatmap_slice)
    """
    if axis == "axial":
        img = data[0, 0, idx]
        hm = heatmap[idx]
    elif axis == "coronal":
        img = data[0, 0, :, idx, :]
        hm = heatmap[:, idx, :]
    elif axis == "sagittal":
        img = data[0, 0, :, :, idx]
        hm = heatmap[:, :, idx]
    else:
        raise ValueError(f"Invalid axis: {axis}")
    return img, hm


def pick_middle_slices(volume_shape: Tuple[int, int, int]) -> Dict[str, int]:
    """Pick middle slice indices for each view."""
    return {
        "axial": volume_shape[0] // 2,
        "coronal": volume_shape[1] // 2,
        "sagittal": volume_shape[2] // 2,
    }


def plot_single_view_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    out_png: Path,
    title: str
) -> None:
    """Plot original image and heatmap-only view for a single slice."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    heatmap_kwargs = resolve_heatmap_imshow_kwargs(heatmap)

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("CT Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap, **heatmap_kwargs)
    axes[1].set_title("Feature Heatmap")
    axes[1].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"鉁?Saved single-view overlay: {out_png}")


def plot_multi_view_overlay(
    data: np.ndarray,
    heatmap: np.ndarray,
    out_png: Path,
    title: str
) -> None:
    """Plot three-view image/heatmap pairs (axial, coronal, sagittal)."""
    volume_shape = (int(data.shape[2]), int(data.shape[3]), int(data.shape[4]))
    slices = pick_middle_slices(volume_shape)

    fig, axes = plt.subplots(3, 2, figsize=(12, 14))

    for row, axis in enumerate(("axial", "coronal", "sagittal")):
        idx = slices[axis]
        img, hm = get_view_slices(data, heatmap, axis, idx)
        heatmap_kwargs = resolve_heatmap_imshow_kwargs(hm)

        # Left: CT image only
        ax = axes[row, 0]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{axis.title()} CT Image (slice {idx})")
        ax.axis("off")

        # Right: heatmap only
        ax = axes[row, 1]
        ax.imshow(hm, **heatmap_kwargs)
        ax.set_title(f"{axis.title()} Feature Heatmap (slice {idx})")
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"鉁?Saved multi-view overlay: {out_png}")


# ============================================================================
# Main Pipeline
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="nnUNetTrainerV2 first skip connection feature heatmap visualization"
    )
    parser.add_argument("--fold", type=int, default=1, help="Fold number (0-4)")
    parser.add_argument("--case-id", default=None, help="Case ID; defaults to first case")
    parser.add_argument("--backend", choices=["activation", "gradcam"], default="activation",
                        help="Heatmap generation backend")
    parser.add_argument("--normalize", choices=["quantile", "none"], default="quantile",
                        help="Heatmap normalization mode")
    parser.add_argument("--output-dir", default="heatmap_output",
                        help="Output directory")
    parser.add_argument("--checkpoint", default=None, help="Custom checkpoint path")
    parser.add_argument("--dataset-directory", default=None, help="Custom dataset directory")
    parser.add_argument("--plans-file", default=None, help="Custom plans file")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--print-structure", action="store_true",
                        help="Print network structure and exit")
    parser.add_argument("--debug-stats", action="store_true",
                        help="Print feature statistics and exit")

    args = parser.parse_args()

    # ========================================================================
    # Setup paths
    # ========================================================================

    plans_file = args.plans_file or str(default_plans_path())
    dataset_directory = args.dataset_directory or str(default_dataset_directory(TASK))
    output_folder = str(CKPT_BASE / f"fold_{args.fold}")
    checkpoint = args.checkpoint or str(default_checkpoint_path(args.fold))
    validation_raw_dir = str(default_validation_raw_dir(args.fold))

    print(f"\n{'='*70}")
    print(f"nnUNetTrainerV2 First Skip Connection Feature Heatmap")
    print(f"{'='*70}")
    print(f"Task:               {TASK}")
    print(f"Trainer:            {TRAINER_NAME}")
    print(f"Fold:               {args.fold}")
    print(f"Backend:            {args.backend}")
    print(f"Normalize:          {args.normalize}")
    print(f"Device:             {args.device}")
    print(f"Checkpoint:         {checkpoint}")
    print(f"Plans file:         {plans_file}")
    print(f"Dataset directory:  {dataset_directory}")
    print(f"Validation raw dir: {validation_raw_dir}")
    print(f"{'='*70}\n")

    # ========================================================================
    # Load trainer and model
    # ========================================================================

    trainer_module = importlib.import_module("nnunet.training.network_training.nnUNetTrainerV2")
    TrainerClass = getattr(trainer_module, TRAINER_NAME)

    trainer = TrainerClass(plans_file, args.fold, output_folder=output_folder,
                          dataset_directory=dataset_directory)
    trainer.initialize(training=False)
    load_checkpoint_if_available(trainer, checkpoint)
    trainer.network.eval()

    print(f"鉁?Initialized {TRAINER_NAME}")

    if args.print_structure:
        print(f"\nNetwork structure:")
        for name, module in trainer.network.named_modules():
            if name:
                print(f"  {name}: {module.__class__.__name__}")
        return

    # ========================================================================
    # Load data
    # ========================================================================

    case_file = choose_case_file(Path(validation_raw_dir), args.case_id)
    print(f"鉁?Selected case: {case_file.stem}")

    expected_in = getattr(trainer.network, "num_input_channels", None)
    data, target = load_case_npz(case_file, expected_in)
    print(f"鉁?Loaded data shape: {data.shape}")
    if target is not None:
        print(f"鉁?Target shape: {target.shape}")

    x = torch.from_numpy(data).float().to(args.device)
    if args.backend == "gradcam":
        x.requires_grad_(True)

    # ========================================================================
    # Extract features
    # ========================================================================

    print(f"\nExtracting features...")
    logits, feat_map, layer_name = extract_skip_connection_features(trainer.network, x, args.backend)

    feat_np = feat_map.detach().cpu().numpy()
    print(f"鉁?Feature layer: {layer_name}")
    print(f"  Feature shape:    {tuple(feat_np.shape)}")
    print(f"  Feature min:      {feat_np.min():.6g}")
    print(f"  Feature max:      {feat_np.max():.6g}")
    print(f"  Feature mean:     {feat_np.mean():.6g}")
    print(f"  Feature std:      {feat_np.std():.6g}")

    if args.debug_stats:
        return

    # ========================================================================
    # Generate heatmap
    # ========================================================================

    if args.backend == "gradcam":
        net = _unwrap_module(trainer.network)
        feat_holder: Dict[str, torch.Tensor] = {}
        grad_holder: Dict[str, torch.Tensor] = {}
        layer_name, skip_module = find_first_skip_connection(net)

        def fwd_hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            if torch.is_tensor(output):
                feat_holder["feat"] = output
                output.retain_grad()
                grad_holder["tensor"] = output

        hook = skip_module.register_forward_hook(fwd_hook)
        try:
            trainer.network.zero_grad(set_to_none=True)
            logits = net(x)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            score = _backward_target_from_logits(logits)
            score.backward()
        finally:
            hook.remove()

        feat_map = feat_holder.get("feat")
        if feat_map is None:
            raise RuntimeError(f"Failed to capture features from '{layer_name}'")
        grad = grad_holder.get("tensor").grad if grad_holder.get("tensor") is not None else None
        if grad is None:
            raise RuntimeError(f"Failed to capture gradients from '{layer_name}'")
        heatmap = make_heatmap(feat_map, "gradcam", grad=grad, normalize=args.normalize)
    else:
        heatmap = make_heatmap(feat_map, "activation", normalize=args.normalize)

    print(f"\nGenerated heatmap:")
    print(f"  Heatmap shape:    {tuple(heatmap.shape)}")
    print(f"  Heatmap min:      {heatmap.min():.6g}")
    print(f"  Heatmap max:      {heatmap.max():.6g}")
    print(f"  Heatmap mean:     {heatmap.mean():.6g}")
    print(f"  Heatmap std:      {heatmap.std():.6g}")

    # ========================================================================
    # Resize and visualize
    # ========================================================================

    original_shape = (int(data.shape[2]), int(data.shape[3]), int(data.shape[4]))
    heatmap_resized = resize_to_original(heatmap, original_shape)
    slices = pick_middle_slices(original_shape)

    print(f"\nResized heatmap to original shape: {tuple(heatmap_resized.shape)}")

    # ========================================================================
    # Save outputs
    # ========================================================================

    outdir = Path(args.output_dir) / TASK / TRAINER_NAME / f"fold_{args.fold}"
    outdir.mkdir(parents=True, exist_ok=True)

    stem = f"{case_file.stem}_{args.backend}_{args.normalize}_first_skip"

    # Save single-view overlay (axial)
    axial_idx = slices["axial"]
    image_slice = data[0, 0, axial_idx]
    heat_slice = heatmap_resized[axial_idx]
    plot_single_view_overlay(
        image_slice, heat_slice,
        outdir / f"{stem}_axial.png",
        f"{TASK} | {TRAINER_NAME} | {case_file.stem} | First Skip Connection | Axial View"
    )

    # Save multi-view overlay
    plot_multi_view_overlay(
        data, heatmap_resized,
        outdir / f"{stem}_3views.png",
        f"{TASK} | {TRAINER_NAME} | {case_file.stem} | First Skip Connection | Multi-View"
    )

    # Save metadata
    meta = {
        "task": TASK,
        "trainer": TRAINER_NAME,
        "network": NETWORK,
        "fold": args.fold,
        "case_id": case_file.stem,
        "backend": args.backend,
        "normalize": args.normalize,
        "layer_name": layer_name,
        "target_feature": "first_skip_connection",
        "checkpoint": str(checkpoint),
        "slice_indices": slices,
        "original_shape": original_shape,
        "feature_shape": list(feat_np.shape),
        "heatmap_shape": list(heatmap_resized.shape),
    }
    (outdir / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"鉁?Saved: {outdir / f'{stem}.json'}")

    print(f"\n{'='*70}")
    print(f"鉁?Visualization complete!")
    print(f"  Output directory: {outdir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

