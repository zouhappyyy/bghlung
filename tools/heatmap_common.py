#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for trainer-specific heatmap scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def default_task_dir(task: str) -> str:
    return task if task.startswith("Task") else f"Task{task}"


def default_dataset_directory(task: str) -> Path:
    from nnunet.paths import preprocessing_output_dir
    if preprocessing_output_dir is None:
        raise RuntimeError("nnUNet preprocessing_output_dir is not configured in nnunet.paths")
    return Path(preprocessing_output_dir) / default_task_dir(task)


def default_checkpoint_path(task: str, trainer_name: str, fold: int) -> Path:
    return Path("ckpt") / "nnUNet" / "3d_fullres" / default_task_dir(task) / f"{trainer_name}__nnUNetPlansv2.1" / f"fold_{fold}" / "model_final_checkpoint.model"


def default_plans_path(task: str, trainer_name: str) -> Path:
    return Path("ckpt") / "nnUNet" / "3d_fullres" / default_task_dir(task) / f"{trainer_name}__nnUNetPlansv2.1" / "plans.pkl"


def default_validation_raw_dir(task: str, trainer_name: str, fold: int) -> Path:
    return Path("ckpt") / "nnUNet" / "3d_fullres" / default_task_dir(task) / f"{trainer_name}__nnUNetPlansv2.1" / f"fold_{fold}" / "validation_raw"


def load_checkpoint_if_available(trainer, checkpoint_path: Optional[str]) -> None:
    if checkpoint_path is None:
        return
    ckpt = Path(checkpoint_path)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(str(ckpt), map_location=torch.device("cpu"))
    trainer.network.load_state_dict(state["state_dict"], strict=False)


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


def resize_to_feature(arr: np.ndarray, feature_shape: Tuple[int, int, int]) -> np.ndarray:
    t = torch.from_numpy(arr[None, None].astype(np.float32))
    t = F.interpolate(t, size=feature_shape, mode="nearest")
    return t[0, 0].cpu().numpy()


def make_heatmap(feature_map: torch.Tensor, backend: str, grad: Optional[torch.Tensor] = None) -> np.ndarray:
    feat = feature_map
    if backend == "activation":
        cam = feat.abs().mean(dim=1, keepdim=False)
    elif backend == "gradcam":
        if grad is None:
            raise RuntimeError("Grad-CAM backend requires gradients")
        weights = grad.mean(dim=(2, 3, 4), keepdim=True)
        cam = torch.relu((weights * feat).sum(dim=1))
    else:
        raise ValueError(f"Unknown backend: {backend}")

    cam = cam[0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()


def pick_axis_slices(volume_shape: Tuple[int, int, int]) -> Dict[str, int]:
    return {"axial": volume_shape[0] // 2, "coronal": volume_shape[1] // 2, "sagittal": volume_shape[2] // 2}


def get_view_slices(data: np.ndarray, heatmap: np.ndarray, axis: str, idx: int):
    if axis == "axial":
        return data[0, 0, idx], heatmap[idx]
    if axis == "coronal":
        return data[0, 0, :, idx, :], heatmap[:, idx, :]
    if axis == "sagittal":
        return data[0, 0, :, :, idx], heatmap[:, :, idx]
    raise ValueError(axis)


def save_single_view(image: np.ndarray, heatmap: np.ndarray, out_png: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("CT slice")
    axes[0].axis("off")

    axes[1].imshow(image, cmap="gray")
    axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Heatmap")
    axes[1].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def save_multi_view(data: np.ndarray, heatmap: np.ndarray, out_png: Path, title: str) -> None:
    volume_shape = (int(data.shape[2]), int(data.shape[3]), int(data.shape[4]))
    slices = pick_axis_slices(volume_shape)
    fig, axes = plt.subplots(3, 2, figsize=(10, 14))
    for row, axis in enumerate(("axial", "coronal", "sagittal")):
        idx = slices[axis]
        img, hm = get_view_slices(data, heatmap, axis, idx)
        views = [(img, f"{axis.title()} CT"), (hm, f"{axis.title()} Heatmap")]
        for col, (overlay, ttl) in enumerate(views):
            ax = axes[row, col]
            if col == 0:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img, cmap="gray")
                ax.imshow(overlay, cmap="jet", vmin=0, vmax=1)
            ax.set_title(f"{ttl} (idx={idx})")
            ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
