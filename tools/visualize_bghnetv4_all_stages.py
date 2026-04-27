#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize all BGHNetV4 stages, including the boundary branch, with jet heatmaps.

This script loads a trained BGHNetV4 checkpoint, runs a single 3D case through the
network, captures multi-stage features from:

- encoder stages
- bottleneck
- segmentation decoder stages
- boundary decoder stages
- segmentation deep supervision outputs
- boundary deep supervision outputs

Each captured tensor is reduced to a 3D activation heatmap, resized to input size,
and saved as both:

- per-stage overlay PNG
- grouped montage PNG
- raw heatmap NPY

The visualizations use the `jet` colormap as requested.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from nnunet.network_architecture.BGHNetV4 import BGHNetV4


PATCH_SIZE = (80, 80, 80)
DISPLAY_CROP_SIZE = 64


def _default_task_dir(task: str) -> str:
    return task if task.startswith("Task") else f"Task{task}"


def _default_checkpoint(task: str, fold: int) -> Path:
    return (
        Path("ckpt")
        / "nnUNet"
        / "3d_fullres"
        / _default_task_dir(task)
        / "BGHNetV4Trainer__nnUNetPlansv2.1"
        / f"fold_{fold}"
        / "model_final_checkpoint.model"
    )


def _default_validation_raw_dir(task: str, fold: int) -> Path:
    return (
        Path("ckpt")
        / "nnUNet"
        / "3d_fullres"
        / _default_task_dir(task)
        / "BGHNetV4Trainer__nnUNetPlansv2.1"
        / f"fold_{fold}"
        / "validation_raw"
    )


def _default_dataset_dir(task: str) -> Path:
    from nnunet.paths import preprocessing_output_dir

    if preprocessing_output_dir is None:
        raise RuntimeError("nnUNet preprocessing_output_dir is not configured in nnunet.paths")
    return Path(preprocessing_output_dir) / _default_task_dir(task)


def _default_stage0_dir(task: str) -> Path:
    return _default_dataset_dir(task) / "nnUNetData_plans_v2.1_stage0"


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
        test_path = item.get("test")
        reference_path = item.get("reference")
        chosen = test_path or reference_path
        if not chosen:
            continue

        path_obj = Path(chosen)
        name = path_obj.name
        if name.endswith(".nii.gz"):
            case_id = name[:-7]
        else:
            case_id = path_obj.stem
        if case_id and case_id not in case_ids:
            case_ids.append(case_id)
    return case_ids


def _resolve_case_id(case_id: Optional[str], validation_raw_dir: Path, data_root: Path) -> Optional[str]:
    if case_id is not None:
        return case_id

    case_ids = _load_validation_case_ids(validation_raw_dir)
    for candidate in case_ids:
        npy_exact = data_root / f"{candidate}.npy"
        npy_candidates = sorted(data_root.glob(f"{candidate}*.npy"))
        if npy_exact.is_file() or npy_candidates:
            return candidate
    return None


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
                if gt.ndim != 3:
                    raise RuntimeError(f"Unexpected GT shape {gt.shape} in {gt_path}")
                gt = np.transpose(gt, (2, 1, 0))
                if gt.shape != data.shape[1:]:
                    min_shape = tuple(min(a, b) for a, b in zip(gt.shape, data.shape[1:]))
                    gt = gt[: min_shape[0], : min_shape[1], : min_shape[2]]
                    data = data[:, : min_shape[0], : min_shape[1], : min_shape[2]]
                seg = gt.astype(np.uint8)
        except Exception as e:
            print(f"[WARN] Failed to load GT for {case_name}: {e}")

    return case_name, data, seg


def _build_model(device: str) -> BGHNetV4:
    patch_size = np.array(PATCH_SIZE)
    pool_op_kernel_sizes = [
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
    ]
    conv_kernel_sizes = [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
    ]

    model = BGHNetV4(
        img_size=patch_size,
        base_num_features=24,
        image_channels=1,
        num_classes=2,
        num_pool=len(pool_op_kernel_sizes),
        pool_op_kernel_sizes=pool_op_kernel_sizes,
        conv_kernel_sizes=conv_kernel_sizes,
        num_only_conv_stage=3,
        num_conv_per_stage=2,
        deep_supervision=True,
        max_num_features=24 * 13,
        depths=[2, 2, 2, 2],
        num_heads=[6, 12, 24, 48],
        window_size=[5, 5, 5],
        drop_path_rate=0,
    )
    model.to(device)
    model.eval()
    return model


def _load_checkpoint(model: torch.nn.Module, checkpoint: Path, device: str) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Please pass --checkpoint with the correct model_final_checkpoint.model path."
        )
    state = torch.load(str(checkpoint), map_location=device)
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading checkpoint: {len(unexpected)}")


def _selector_first(x):
    if isinstance(x, (tuple, list)):
        return x[0]
    return x


def _selector_second(x):
    if isinstance(x, (tuple, list)):
        return x[1]
    raise RuntimeError("Expected tuple/list output for second selector")


def _register_hook(
    module: torch.nn.Module,
    store: "OrderedDict[str, torch.Tensor]",
    name: str,
    selector: Callable[[object], torch.Tensor],
):
    def _hook(_module, _inputs, output):
        tensor = selector(output)
        if isinstance(tensor, (tuple, list)):
            tensor = tensor[0]
        if not torch.is_tensor(tensor):
            raise RuntimeError(f"Hook '{name}' did not receive a tensor")
        store[name] = tensor.detach()

    return module.register_forward_hook(_hook)


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
    return cam.cpu().numpy()


def _mask_center_index(mask: np.ndarray, axis: str) -> Optional[int]:
    fg = np.argwhere(mask > 0)
    if fg.size == 0:
        return None
    axis_map = {"axial": 0, "coronal": 1, "sagittal": 2}
    col = axis_map[axis]
    return int(round((fg[:, col].min() + fg[:, col].max()) / 2.0))


def _choose_slice_index(seg: Optional[np.ndarray], axis: str, volume_shape: Tuple[int, int, int], explicit_index: Optional[int]) -> int:
    if explicit_index is not None:
        return explicit_index
    if seg is not None:
        idx = _mask_center_index(seg, axis)
        if idx is not None:
            return idx
    axis_map = {"axial": 0, "coronal": 1, "sagittal": 2}
    return volume_shape[axis_map[axis]] // 2


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
    crop_info = {
        "y_range": [int(y1), int(y2)],
        "x_range": [int(x1), int(x2)],
        "shape": [int(cropped.shape[0]), int(cropped.shape[1])],
    }
    return cropped, crop_info


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


def _save_montage(
    image_2d: np.ndarray,
    heatmaps: "OrderedDict[str, np.ndarray]",
    out_file: Path,
    title: str,
) -> None:
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


def _collect_stage_tensors(model: BGHNetV4, x: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
    captures: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    handles = []

    for i, layer in enumerate(model.down_layers):
        handles.append(_register_hook(layer, captures, f"encoder_stage{i}", _selector_first))
        if i == len(model.down_layers) - 1:
            handles.append(_register_hook(layer, captures, "bottleneck", _selector_second))

    decoder_stage_ids = list(range(model.num_pool, -1, -1))
    for idx, stage_id in enumerate(decoder_stage_ids):
        handles.append(
            _register_hook(
                model.up_layers[idx].conv_blocks,
                captures,
                f"seg_decoder_stage{stage_id}",
                _selector_first,
            )
        )
        handles.append(
            _register_hook(
                model.up_bdr_layers[idx].conv_blocks,
                captures,
                f"bdr_decoder_stage{stage_id}",
                _selector_first,
            )
        )

    with torch.no_grad():
        seg_outputs = model(x)
        bdr_outputs = model.get_bdr_outputs

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
    parser = argparse.ArgumentParser(description="Visualize BGHNetV4 multi-stage features with jet colormap.")
    parser.add_argument("--task", default="Task530_EsoTJ_30pct")
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--checkpoint", default=None, help="Path to model_final_checkpoint.model")
    parser.add_argument("--validation-raw-dir", default=None, help="validation_raw directory; summary.json is used to auto-pick validation cases")
    parser.add_argument("--dataset-directory", default=None, help="Task root that contains gt_segmentations")
    parser.add_argument("--data-root", default=None, help="Stage0 preprocessed .npy directory")
    parser.add_argument("--case-id", default=None, help="Case id without suffix; defaults to the first .npy case")
    parser.add_argument("--axis", choices=["axial", "coronal", "sagittal"], default="axial")
    parser.add_argument("--slice-index", type=int, default=None, help="Override slice index")
    parser.add_argument("--no-crop", action="store_true", help="Disable automatic 80x80x80 patch cropping")
    parser.add_argument("--display-crop-size", type=int, default=DISPLAY_CROP_SIZE, help="Center crop size for saved 2D visualization PNGs")
    parser.add_argument("--output-dir", default="feature_vis_output")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    task_dir = _default_task_dir(args.task)
    checkpoint = Path(args.checkpoint) if args.checkpoint else _default_checkpoint(args.task, args.fold)
    validation_raw_dir = Path(args.validation_raw_dir) if args.validation_raw_dir else _default_validation_raw_dir(args.task, args.fold)
    dataset_dir = Path(args.dataset_directory) if args.dataset_directory else _default_dataset_dir(args.task)
    data_root = Path(args.data_root) if args.data_root else _default_stage0_dir(args.task)
    resolved_case_id = _resolve_case_id(args.case_id, validation_raw_dir, data_root)

    print(f"[INFO] task={task_dir}")
    print(f"[INFO] checkpoint={checkpoint}")
    print(f"[INFO] validation_raw_dir={validation_raw_dir}")
    print(f"[INFO] dataset_dir={dataset_dir}")
    print(f"[INFO] data_root={data_root}")
    print(f"[INFO] requested_case_id={args.case_id}")
    print(f"[INFO] resolved_case_id={resolved_case_id}")

    model = _build_model(args.device)
    _load_checkpoint(model, checkpoint, args.device)

    case_name, data, seg = _load_case_data(data_root, dataset_dir, resolved_case_id)
    crop_info = None
    if args.no_crop:
        patch_data = data
        patch_seg = seg
    else:
        patch_data, patch_seg, crop_info = _crop_or_pad_to_patch(data, seg, PATCH_SIZE)

    x = torch.from_numpy(patch_data[None]).to(args.device)

    if x.shape != (1, 1, patch_data.shape[1], patch_data.shape[2], patch_data.shape[3]):
        raise RuntimeError(f"Unexpected input tensor shape {tuple(x.shape)}")

    stage_tensors = _collect_stage_tensors(model, x)
    volume_shape = (int(patch_data.shape[1]), int(patch_data.shape[2]), int(patch_data.shape[3]))
    slice_index = _choose_slice_index(patch_seg, args.axis, volume_shape, args.slice_index)

    image_2d_full = _slice_2d(patch_data[0], args.axis, slice_index)
    image_2d, display_crop_info = _crop_2d_center(image_2d_full, args.display_crop_size)

    out_base = Path(args.output_dir) / task_dir / "BGHNetV4Trainer" / f"fold_{args.fold}" / case_name
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

    meta = {
        "task": task_dir,
        "fold": args.fold,
        "case_id": case_name,
        "checkpoint": str(checkpoint),
        "validation_raw_dir": str(validation_raw_dir),
        "dataset_directory": str(dataset_dir),
        "data_root": str(data_root),
        "requested_case_id": args.case_id,
        "resolved_case_id": resolved_case_id,
        "auto_crop_enabled": not args.no_crop,
        "crop_info": crop_info,
        "axis": args.axis,
        "slice_index": slice_index,
        "volume_shape": list(volume_shape),
        "display_crop_size": args.display_crop_size,
        "display_crop_info": display_crop_info,
        "stage_tensor_shapes": tensor_shapes,
        "saved_groups": [name for name, group in groups.items() if group],
        "colormap": "jet",
    }
    (out_base / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Case: {case_name}")
    print(f"[OK] Saved directory: {out_base}")
    print(f"[OK] Saved {len(stage_tensors)} stage heatmaps with jet colormap.")


if __name__ == "__main__":
    main()
