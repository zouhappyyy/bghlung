#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize all BANetV2 stages with jet heatmaps using validation_raw cases.

This script:
- loads BANetTrainerV2 with its nnUNet plans
- reads validation cases from validation_raw/summary.json
- loads the matching preprocessed stage0 .npy image
- crops/pads to the trainer patch size
- captures encoder, bottleneck, segmentation decoder, boundary decoder,
  segmentation outputs, and boundary outputs
- saves per-stage overlays and grouped montages with jet colormap
"""

from __future__ import annotations

import argparse
import importlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


DISPLAY_CROP_SIZE = 64
SUMMARY_STAGE_NAMES = (
    "encoder_stage0",
    "bottleneck",
    "seg_decoder_stage0",
    "bdr_decoder_stage0",
    "seg_output_stage0",
    "bdr_output_stage0",
)


def _default_task_dir(task: str) -> str:
    return task if task.startswith("Task") else f"Task{task}"


def _default_checkpoint(task: str, fold: int) -> Path:
    return (
        Path("ckpt")
        / "nnUNet"
        / "3d_fullres"
        / _default_task_dir(task)
        / "BANetTrainerV2__nnUNetPlansv2.1"
        / f"fold_{fold}"
        / "model_final_checkpoint.model"
    )


def _default_plans_file(task: str) -> Path:
    return (
        Path("ckpt")
        / "nnUNet"
        / "3d_fullres"
        / _default_task_dir(task)
        / "BANetTrainerV2__nnUNetPlansv2.1"
        / "plans.pkl"
    )


def _default_output_folder(task: str, fold: int) -> Path:
    return (
        Path("ckpt")
        / "nnUNet"
        / "3d_fullres"
        / _default_task_dir(task)
        / "BANetTrainerV2__nnUNetPlansv2.1"
        / f"fold_{fold}"
    )


def _default_validation_raw_dir(task: str, fold: int) -> Path:
    return _default_output_folder(task, fold) / "validation_raw"


def _default_dataset_dir(task: str) -> Path:
    from nnunet.paths import preprocessing_output_dir

    if preprocessing_output_dir is None:
        raise RuntimeError("nnUNet preprocessing_output_dir is not configured in nnunet.paths")
    return Path(preprocessing_output_dir) / _default_task_dir(task)


def _default_stage0_dir(task: str) -> Path:
    return _default_dataset_dir(task) / "nnUNetData_plans_v2.1_stage0"


def _load_trainer_cls():
    module = importlib.import_module("nnunet.training.network_training.BANetTrainerV2")
    return getattr(module, "BANetTrainerV2")


def _load_validation_case_ids(validation_raw_dir: Path) -> List[str]:
    summary_file = validation_raw_dir / "summary.json"
    if not summary_file.is_file():
        return []

    try:
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
    except Exception:
        return []

    case_ids: List[str] = []
    for item in summary.get("results", {}).get("all", []):
        chosen = item.get("test") or item.get("reference")
        if not chosen:
            continue
        name = Path(chosen).name
        case_id = name[:-7] if name.endswith(".nii.gz") else Path(name).stem
        if case_id and case_id not in case_ids:
            case_ids.append(case_id)
    return case_ids


def _resolve_case_id(case_id: Optional[str], validation_raw_dir: Path, data_root: Path) -> Optional[str]:
    if case_id is not None:
        return case_id

    for candidate in _load_validation_case_ids(validation_raw_dir):
        if (data_root / f"{candidate}.npy").is_file():
            return candidate
        if sorted(data_root.glob(f"{candidate}*.npy")):
            return candidate
    return None


def _resolve_case_npy(data_root: Path, case_id: Optional[str]) -> Path:
    if case_id is not None:
        exact = data_root / f"{case_id}.npy"
        if exact.is_file():
            return exact
        candidates = sorted(data_root.glob(f"{case_id}*.npy"))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"Could not find case '{case_id}' under {data_root}")

    files = sorted(data_root.glob("*.npy"))
    if not files:
        raise RuntimeError(f"No .npy cases found under {data_root}")
    return files[0]


def _load_case_data(data_root: Path, dataset_dir: Path, case_id: Optional[str]) -> Tuple[str, np.ndarray, Optional[np.ndarray]]:
    case_file = _resolve_case_npy(data_root, case_id)
    arr = np.load(case_file)
    if arr.ndim == 4:
        data = arr.astype(np.float32)
    elif arr.ndim == 3:
        data = arr[None].astype(np.float32)
    else:
        raise RuntimeError(f"Unexpected case shape {arr.shape} in {case_file}")

    data = data[:1]
    case_name = case_file.stem
    seg = None
    gt_dir = dataset_dir / "gt_segmentations"
    if gt_dir.is_dir():
        try:
            import nibabel as nib

            gt_path = None
            for ext in (".nii.gz", ".nii"):
                candidate = gt_dir / f"{case_name}{ext}"
                if candidate.is_file():
                    gt_path = candidate
                    break
            if gt_path is not None:
                gt = nib.load(str(gt_path)).get_fdata()
                gt = np.transpose(gt, (2, 1, 0))
                if gt.shape != data.shape[1:]:
                    min_shape = tuple(min(a, b) for a, b in zip(gt.shape, data.shape[1:]))
                    gt = gt[: min_shape[0], : min_shape[1], : min_shape[2]]
                    data = data[:, : min_shape[0], : min_shape[1], : min_shape[2]]
                seg = gt.astype(np.uint8)
        except Exception as e:
            print(f"[WARN] Failed to load GT for {case_name}: {e}")

    return case_name, data, seg


def _compute_crop_bounds(center: int, size: int, limit: int) -> Tuple[int, int]:
    start = center - size // 2
    end = start + size
    if start < 0:
        end -= start
        start = 0
    if end > limit:
        start -= end - limit
        end = limit
    start = max(0, start)
    end = min(limit, end)
    return start, end


def _crop_or_pad_to_patch(
    data: np.ndarray,
    seg: Optional[np.ndarray],
    patch_size: Tuple[int, int, int],
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, List[int]]]:
    spatial_shape = tuple(int(i) for i in data.shape[1:])
    if seg is not None and np.any(seg > 0):
        fg = np.argwhere(seg > 0)
        center = [int(round((fg[:, i].min() + fg[:, i].max()) / 2.0)) for i in range(3)]
    else:
        center = [s // 2 for s in spatial_shape]

    slices = []
    crop_bbox = []
    for c, size, limit in zip(center, patch_size, spatial_shape):
        start, end = _compute_crop_bounds(c, size, limit)
        slices.append(slice(start, end))
        crop_bbox.append([int(start), int(end)])

    cropped_data = data[:, slices[0], slices[1], slices[2]]
    cropped_seg = None if seg is None else seg[slices[0], slices[1], slices[2]]

    pad_width_data = [(0, 0)]
    pad_width_seg = []
    final_shape = tuple(int(i) for i in cropped_data.shape[1:])
    for size, current in zip(patch_size, final_shape):
        total_pad = max(0, size - current)
        before = total_pad // 2
        after = total_pad - before
        pad_width_data.append((before, after))
        pad_width_seg.append((before, after))

    if any(pad != (0, 0) for pad in pad_width_data[1:]):
        cropped_data = np.pad(cropped_data, pad_width_data, mode="constant", constant_values=0)
        if cropped_seg is not None:
            cropped_seg = np.pad(cropped_seg, pad_width_seg, mode="constant", constant_values=0)

    crop_info = {
        "original_shape": list(spatial_shape),
        "patch_size": list(patch_size),
        "crop_bbox_zyx": crop_bbox,
        "crop_center_zyx": [int(i) for i in center],
        "cropped_shape_before_pad": list(final_shape),
        "pad_width_zyx": [[int(a), int(b)] for a, b in pad_width_seg],
        "final_patch_shape": list(cropped_data.shape[1:]),
    }
    return cropped_data, cropped_seg, crop_info


def _mask_center_index(mask: np.ndarray, axis: str) -> Optional[int]:
    fg = np.argwhere(mask > 0)
    if fg.size == 0:
        return None
    axis_map = {"axial": 0, "coronal": 1, "sagittal": 2}
    return int(round((fg[:, axis_map[axis]].min() + fg[:, axis_map[axis]].max()) / 2.0))


def _choose_slice_index(seg: Optional[np.ndarray], axis: str, volume_shape: Tuple[int, int, int], explicit_index: Optional[int]) -> int:
    if explicit_index is not None:
        return explicit_index
    if seg is not None:
        idx = _mask_center_index(seg, axis)
        if idx is not None:
            return idx
    axis_map = {"axial": 0, "coronal": 1, "sagittal": 2}
    return volume_shape[axis_map[axis]] // 2


def _slice_2d(volume: np.ndarray, axis: str, idx: int) -> np.ndarray:
    if axis == "axial":
        return volume[idx]
    if axis == "coronal":
        return volume[:, idx, :]
    if axis == "sagittal":
        return volume[:, :, idx]
    raise ValueError(axis)


def _crop_2d_center(image: np.ndarray, crop_size: int) -> Tuple[np.ndarray, Dict[str, List[int]]]:
    h, w = image.shape
    target_h = min(crop_size, h)
    target_w = min(crop_size, w)
    cy = h // 2
    cx = w // 2

    y1 = cy - target_h // 2
    x1 = cx - target_w // 2
    y2 = y1 + target_h
    x2 = x1 + target_w

    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y2 > h:
        y1 -= y2 - h
        y2 = h
    if x2 > w:
        x1 -= x2 - w
        x2 = w

    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = min(h, y2)
    x2 = min(w, x2)

    cropped = image[y1:y2, x1:x2]
    return cropped, {
        "y_range": [int(y1), int(y2)],
        "x_range": [int(x1), int(x2)],
        "shape": [int(cropped.shape[0]), int(cropped.shape[1])],
    }


def _save_overlay_png(image: np.ndarray, heatmap: np.ndarray, out_file: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("CT")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Heatmap")
    axes[1].axis("off")

    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(heatmap, cmap="jet", vmin=0, vmax=1, alpha=0.55)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_montage(image_2d: np.ndarray, heatmaps: "OrderedDict[str, np.ndarray]", out_file: Path, title: str) -> None:
    names = list(heatmaps.keys())
    cols = 3
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.4 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for ax in axes.ravel():
        ax.axis("off")

    for ax, name in zip(axes.ravel(), names):
        ax.imshow(image_2d, cmap="gray")
        ax.imshow(heatmaps[name], cmap="jet", vmin=0, vmax=1, alpha=0.55)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _make_activation_heatmap(tensor: torch.Tensor, out_shape: Tuple[int, int, int]) -> np.ndarray:
    feat = tensor
    if feat.dim() == 4:
        feat = feat.unsqueeze(0)
    if feat.dim() != 5:
        raise RuntimeError(f"Expected 5D tensor, got shape {tuple(feat.shape)}")

    cam = feat.abs().mean(dim=1, keepdim=True)
    cam = F.interpolate(cam, size=out_shape, mode="trilinear", align_corners=False)
    cam = cam[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam.detach().cpu().numpy()


def _collect_stage_tensors(network: torch.nn.Module, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
    captures: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    handles = []

    def _hook_factory(name: str):
        def _hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(tensor):
                captures[name] = tensor.detach()
        return _hook

    base = network.module if hasattr(network, "module") else network

    for idx, module in enumerate(base.conv_blocks_context[:-1]):
        handles.append(module.register_forward_hook(_hook_factory(f"encoder_stage{idx}")))
    handles.append(base.conv_blocks_context[-1].register_forward_hook(_hook_factory("bottleneck")))

    for idx, module in enumerate(base.conv_blocks_localization):
        handles.append(module.register_forward_hook(_hook_factory(f"seg_decoder_stage{idx}")))
    for idx, module in enumerate(base.conv_blocks_boundary):
        handles.append(module.register_forward_hook(_hook_factory(f"bdr_decoder_stage{idx}")))

    with torch.no_grad():
        seg_outputs = base(x)
        bdr_outputs = base.get_bdr_outputs

    for h in handles:
        h.remove()

    if isinstance(seg_outputs, torch.Tensor):
        seg_outputs = [seg_outputs]
    if isinstance(bdr_outputs, torch.Tensor):
        bdr_outputs = [bdr_outputs]

    for idx, tensor in enumerate(seg_outputs):
        captures[f"seg_output_stage{idx}"] = tensor.detach()
    for idx, tensor in enumerate(bdr_outputs):
        captures[f"bdr_output_stage{idx}"] = tensor.detach()

    return captures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize BANetTrainerV2 multi-stage features with jet colormap.")
    parser.add_argument("--task", default="Task530_EsoTJ_30pct")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--plans-file", default=None)
    parser.add_argument("--output-folder", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--validation-raw-dir", default=None)
    parser.add_argument("--dataset-directory", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--axis", choices=["axial", "coronal", "sagittal"], default="axial")
    parser.add_argument("--slice-index", type=int, default=None)
    parser.add_argument("--no-crop", action="store_true", help="Disable patch-size crop/pad before forward")
    parser.add_argument("--display-crop-size", type=int, default=DISPLAY_CROP_SIZE)
    parser.add_argument("--output-dir", default="feature_vis_output")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_dir = _default_task_dir(args.task)
    plans_file = Path(args.plans_file) if args.plans_file else _default_plans_file(args.task)
    output_folder = Path(args.output_folder) if args.output_folder else _default_output_folder(args.task, args.fold)
    checkpoint = Path(args.checkpoint) if args.checkpoint else _default_checkpoint(args.task, args.fold)
    validation_raw_dir = Path(args.validation_raw_dir) if args.validation_raw_dir else _default_validation_raw_dir(args.task, args.fold)
    dataset_dir = Path(args.dataset_directory) if args.dataset_directory else _default_dataset_dir(args.task)
    data_root = Path(args.data_root) if args.data_root else _default_stage0_dir(args.task)
    resolved_case_id = _resolve_case_id(args.case_id, validation_raw_dir, data_root)

    print(f"[INFO] task={task_dir}")
    print(f"[INFO] plans_file={plans_file}")
    print(f"[INFO] output_folder={output_folder}")
    print(f"[INFO] checkpoint={checkpoint}")
    print(f"[INFO] validation_raw_dir={validation_raw_dir}")
    print(f"[INFO] dataset_dir={dataset_dir}")
    print(f"[INFO] data_root={data_root}")
    print(f"[INFO] requested_case_id={args.case_id}")
    print(f"[INFO] resolved_case_id={resolved_case_id}")

    trainer_cls = _load_trainer_cls()
    trainer = trainer_cls(str(plans_file), args.fold, output_folder=str(output_folder), dataset_directory=str(dataset_dir))
    trainer.initialize(training=False)

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Please pass --checkpoint with the correct model_final_checkpoint.model path."
        )

    state = torch.load(str(checkpoint), map_location=torch.device(args.device))
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    trainer.network.load_state_dict(state_dict, strict=False)
    trainer.network.to(args.device)
    trainer.network.eval()

    patch_size = tuple(int(i) for i in trainer.patch_size)
    case_name, data, seg = _load_case_data(data_root, dataset_dir, resolved_case_id)

    crop_info = None
    if args.no_crop:
        patch_data = data
        patch_seg = seg
    else:
        patch_data, patch_seg, crop_info = _crop_or_pad_to_patch(data, seg, patch_size)

    x = torch.from_numpy(patch_data[None]).float().to(args.device)
    stage_tensors = _collect_stage_tensors(trainer.network, x)

    volume_shape = (int(patch_data.shape[1]), int(patch_data.shape[2]), int(patch_data.shape[3]))
    slice_index = _choose_slice_index(patch_seg, args.axis, volume_shape, args.slice_index)

    image_2d_full = _slice_2d(patch_data[0], args.axis, slice_index)
    image_2d, display_crop_info = _crop_2d_center(image_2d_full, args.display_crop_size)

    out_base = Path(args.output_dir) / task_dir / "BANetTrainerV2" / f"fold_{args.fold}" / case_name
    heatmap_dir = out_base / "heatmaps_npy"
    overlay_dir = out_base / "overlays"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    groups: Dict[str, "OrderedDict[str, np.ndarray]"] = {
        "encoder": OrderedDict(),
        "seg_decoder": OrderedDict(),
        "bdr_decoder": OrderedDict(),
        "seg_output": OrderedDict(),
        "bdr_output": OrderedDict(),
        "other": OrderedDict(),
    }
    tensor_shapes: Dict[str, List[int]] = {}

    for name, tensor in stage_tensors.items():
        tensor_shapes[name] = list(tensor.shape)
        heatmap_3d = _make_activation_heatmap(tensor, volume_shape)
        np.save(heatmap_dir / f"{name}.npy", heatmap_3d)
        heatmap_2d_full = _slice_2d(heatmap_3d, args.axis, slice_index)
        heatmap_2d, _ = _crop_2d_center(heatmap_2d_full, args.display_crop_size)
        _save_overlay_png(
            image_2d,
            heatmap_2d,
            overlay_dir / f"{name}_{args.axis}_{slice_index:03d}.png",
            title=f"{case_name} | {name} | {args.axis}={slice_index}",
        )

        if name.startswith("encoder_"):
            groups["encoder"][name] = heatmap_2d
        elif name.startswith("seg_decoder_"):
            groups["seg_decoder"][name] = heatmap_2d
        elif name.startswith("bdr_decoder_"):
            groups["bdr_decoder"][name] = heatmap_2d
        elif name.startswith("seg_output_"):
            groups["seg_output"][name] = heatmap_2d
        elif name.startswith("bdr_output_"):
            groups["bdr_output"][name] = heatmap_2d
        else:
            groups["other"][name] = heatmap_2d

    for group_name, group_maps in groups.items():
        if not group_maps:
            continue
        _save_montage(
            image_2d,
            group_maps,
            out_base / f"{group_name}_montage_{args.axis}_{slice_index:03d}.png",
            title=f"{case_name} | {group_name} | {args.axis}={slice_index}",
        )

    summary_stage_names = [name for name in SUMMARY_STAGE_NAMES if name in tensor_shapes]
    meta = {
        "task": task_dir,
        "fold": args.fold,
        "case_id": case_name,
        "plans_file": str(plans_file),
        "output_folder": str(output_folder),
        "checkpoint": str(checkpoint),
        "validation_raw_dir": str(validation_raw_dir),
        "dataset_directory": str(dataset_dir),
        "data_root": str(data_root),
        "requested_case_id": args.case_id,
        "resolved_case_id": resolved_case_id,
        "patch_size": list(patch_size),
        "auto_crop_enabled": not args.no_crop,
        "crop_info": crop_info,
        "axis": args.axis,
        "slice_index": slice_index,
        "volume_shape": list(volume_shape),
        "display_crop_size": args.display_crop_size,
        "display_crop_info": display_crop_info,
        "stage_tensor_shapes": tensor_shapes,
        "summary_stage_names": summary_stage_names,
        "saved_groups": [name for name, group in groups.items() if group],
        "colormap": "jet",
    }
    (out_base / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Case: {case_name}")
    print(f"[OK] Saved directory: {out_base}")
    print(f"[OK] Saved {len(stage_tensors)} stage heatmaps with jet colormap.")


if __name__ == "__main__":
    main()
