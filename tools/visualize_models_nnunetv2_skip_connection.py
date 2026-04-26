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
- heatmap .npy (resized to original volume dimensions)
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
    print(f"✓ Loaded checkpoint: {ckpt}")


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
        idx = len(container) - 1  # ✅ 取最后一个
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
        return children[-1]  # ✅ 最后一个
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
        print(f"✓ Found first skip layer: {full_name}")
        return full_name, child

    for name, module in net.named_modules():
        if "conv_blocks_localization" in name and name.split(".")[-1].isdigit():
            print(f"✓ Found first skip layer (fallback): {name}")
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
    grad: Optional[torch.Tensor] = None
) -> np.ndarray:
    """Generate heatmap from feature map.

    Args:
        feature_map: (B, C, D, H, W) tensor
        backend: "activation" or "gradcam"
        grad: gradients for grad-CAM (B, C, D, H, W)

    Returns:
        (D, H, W) normalized heatmap as numpy array
    """
    if backend == "activation":
        # Channel-average absolute activation
        cam = feature_map.abs().mean(dim=1, keepdim=False)
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
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()


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
    """Plot image and heatmap overlay for a single view."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("CT Image")
    axes[0].axis("off")

    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Heatmap Overlay")
    axes[1].axis("off")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved single-view overlay: {out_png}")


def plot_multi_view_overlay(
    data: np.ndarray,
    heatmap: np.ndarray,
    out_png: Path,
    title: str
) -> None:
    """Plot three-view heatmap overlays (axial, coronal, sagittal)."""
    volume_shape = (int(data.shape[2]), int(data.shape[3]), int(data.shape[4]))
    slices = pick_middle_slices(volume_shape)

    fig, axes = plt.subplots(3, 2, figsize=(12, 14))

    for row, axis in enumerate(("axial", "coronal", "sagittal")):
        idx = slices[axis]
        img, hm = get_view_slices(data, heatmap, axis, idx)

        # Left: CT image only
        ax = axes[row, 0]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{axis.title()} CT Image (slice {idx})")
        ax.axis("off")

        # Right: CT with heatmap overlay
        ax = axes[row, 1]
        ax.imshow(img, cmap="gray")
        ax.imshow(hm, cmap="jet", vmin=0, vmax=1)
        ax.set_title(f"{axis.title()} Heatmap Overlay (slice {idx})")
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved multi-view overlay: {out_png}")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=3)
    parser.add_argument("--backend", choices=["activation", "gradcam"], default="activation")
    parser.add_argument("--output-dir", default="heatmap_output")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-cases", type=int, default=5)  # 控制数量（建议）
    args = parser.parse_args()

    # ===== 路径 =====
    plans_file = str(default_plans_path())
    dataset_directory = str(default_dataset_directory(TASK))
    output_folder = str(CKPT_BASE / f"fold_{args.fold}")
    validation_raw_dir = Path(default_validation_raw_dir(args.fold))

    # ===== 初始化 trainer（只做一次！）=====
    trainer_module = importlib.import_module(
        "nnunet.training.network_training.nnUNetTrainerV2"
    )
    TrainerClass = getattr(trainer_module, TRAINER_NAME)

    trainer = TrainerClass(plans_file, args.fold,
                           output_folder=output_folder,
                           dataset_directory=dataset_directory)
    trainer.initialize(training=False)
    trainer.network.eval().to(args.device)

    print("✓ Trainer initialized")

    # ===== 找 checkpoint =====
    ckpt_list = find_all_checkpoints(CKPT_BASE / f"fold_{args.fold}")
    print(f"✓ Found {len(ckpt_list)} checkpoints")

    # ===== 找 case =====
    case_files = sorted(validation_raw_dir.glob("*.npz"))
    case_files = case_files[:args.max_cases]  # 控制数量
    print(f"✓ Using {len(case_files)} cases")

    # ===== 固定 slice（关键！）=====
    slice_cache = {}

    for ckpt_path in ckpt_list:
        epoch = ckpt_path.stem.split("_")[-1]
        print(f"\n===== Epoch {epoch} =====")

        # 加载模型权重
        state = torch.load(str(ckpt_path), map_location="cpu")
        trainer.network.load_state_dict(state["state_dict"], strict=False)

        for case_file in case_files:
            print(f"→ Case: {case_file.stem}")

            data, _ = load_case_npz(case_file, trainer.network.num_input_channels)
            x = torch.from_numpy(data).float().to(args.device)

            with torch.no_grad():
                logits, feat_map, layer_name = extract_skip_connection_features(
                    trainer.network, x, args.backend
                )
                heatmap = make_heatmap(feat_map, args.backend)

            # ===== resize =====
            original_shape = (data.shape[2], data.shape[3], data.shape[4])
            heatmap_resized = resize_to_original(heatmap, original_shape)

            # ===== 固定 slice =====
            if case_file.stem not in slice_cache:
                slice_cache[case_file.stem] = pick_middle_slices(original_shape)

            slices = slice_cache[case_file.stem]
            axial_idx = slices["axial"]

            # ===== 保存 =====
            outdir = Path(args.output_dir) / case_file.stem
            outdir.mkdir(parents=True, exist_ok=True)

            # npy
            np.save(outdir / f"ep_{epoch}.npy", heatmap_resized)

            # png
            plot_single_view_overlay(
                data[0, 0, axial_idx],
                heatmap_resized[axial_idx],
                outdir / f"ep_{epoch}.png",
                f"{case_file.stem} | Epoch {epoch}"
            )

    print("\n✅ Done!")

if __name__ == "__main__":
    main()
