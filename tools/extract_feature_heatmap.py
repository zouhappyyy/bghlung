#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""3D segmentation feature heatmap / Grad-CAM style visualization.

This tool reuses the nnUNet trainer loading pattern in this repo and produces
2D slice overlays for a selected validation case.

Supported backends:
- activation: channel-averaged absolute activation map from an intermediate layer
- gradcam:    gradient-weighted activation map from an intermediate layer

Outputs:
- heatmap .npy
- heatmap overlay .png
- meta .json
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def _load_trainer(trainer_name: str):
    trainer_map = {
        "BGHNetV4Trainer": "nnunet.training.network_training.BGHNetV4Trainer",
        "MedNeXtTrainerV2": "nnunet.training.network_training.MedNeXt.MedNeXtTrainerV2",
        "BANetTrainerV2": "nnunet.training.network_training.BANetTrainerV2",
        "nnUNetTrainerV2": "nnunet.training.network_training.nnUNetTrainerV2",
    }
    if trainer_name not in trainer_map:
        raise ValueError(f"Unsupported trainer: {trainer_name}")
    module = importlib.import_module(trainer_map[trainer_name])
    return getattr(module, trainer_name)


def _default_task_dir(task: str) -> str:
    return task if task.startswith("Task") else f"Task{task}"


def _default_checkpoint_path(task: str, trainer_name: str, fold: int) -> Path:
    return Path("ckpt") / "nnUNet" / "3d_fullres" / _default_task_dir(task) / f"{trainer_name}__nnUNetPlansv2.1" / f"fold_{fold}" / "model_final_checkpoint.model"


def _default_plans_path(task: str, trainer_name: str) -> Path:
    return Path("ckpt") / "nnUNet" / "3d_fullres" / _default_task_dir(task) / f"{trainer_name}__nnUNetPlansv2.1" / "plans.pkl"


def _default_dataset_directory(task: str) -> Path:
    from nnunet.paths import preprocessing_output_dir
    if preprocessing_output_dir is None:
        raise RuntimeError("nnUNet preprocessing_output_dir is not configured in nnunet.paths")
    return Path(preprocessing_output_dir) / _default_task_dir(task)


def _default_validation_raw_dir(task: str, trainer_name: str, fold: int) -> Path:
    return Path("ckpt") / "nnUNet" / "3d_fullres" / _default_task_dir(task) / f"{trainer_name}__nnUNetPlansv2.1" / f"fold_{fold}" / "validation_raw"


def _load_checkpoint_if_available(trainer, checkpoint_path: Optional[str]) -> None:
    if checkpoint_path is None:
        return
    ckpt = Path(checkpoint_path)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(str(ckpt), map_location=torch.device("cpu"))
    trainer.network.load_state_dict(state["state_dict"], strict=False)


def _mednext_enc_block0_map(network: torch.nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, str]:
    """Capture MedNeXt's enc_block_0 output for heatmap visualization."""
    mednext = network.module if hasattr(network, "module") else network
    captured: Dict[str, torch.Tensor] = {}

    def _save(name: str):
        def _hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            if torch.is_tensor(output):
                captured[name] = output
        return _hook

    x0 = mednext.stem(x)
    h1 = mednext.enc_block_0.register_forward_hook(_save("enc_block_0"))
    feat = mednext.enc_block_0(x0)
    h1.remove()

    feat = captured.get("enc_block_0", feat)
    if feat is None:
        raise RuntimeError("Could not capture MedNeXt enc_block_0 activations")

    logits = mednext.down_0(feat)
    logits = mednext.enc_block_1(logits)
    logits = mednext.down_1(logits)
    logits = mednext.enc_block_2(logits)
    logits = mednext.down_2(logits)
    logits = mednext.enc_block_3(logits)
    logits = mednext.down_3(logits)
    logits = mednext.bottleneck(logits)
    logits = mednext.up_3(logits)
    logits = mednext.dec_block_3(logits)
    logits = mednext.up_2(logits)
    logits = mednext.dec_block_2(logits)
    logits = mednext.up_1(logits)
    logits = mednext.dec_block_1(logits)
    logits = mednext.up_0(logits)
    logits = mednext.dec_block_0(logits)
    logits = mednext.out_0(logits)

    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits, feat, "enc_block_0"


def _preferred_feature_layers(trainer_name: str) -> Tuple[str, ...]:
    if trainer_name == "BGHNetV4Trainer":
        # Prefer the earliest decoder-side skip features first.
        return ("up_layers", "decoder", "seg_decoder", "bdr_decoder", "conv_blocks_context", "conv_blocks")
    if trainer_name == "BANetTrainerV2":
        return ("conv_blocks_localization", "decoder", "up", "conv_blocks_context", "conv_blocks")
    if trainer_name == "nnUNetTrainerV2":
        return ("conv_blocks_localization", "decoder", "up", "conv_blocks_context", "conv_blocks")
    return ("conv_blocks_localization", "decoder", "up", "conv_blocks_context", "conv_blocks")


def _first_skip_layers(trainer_name: str) -> Tuple[str, ...]:
    if trainer_name == "BGHNetV4Trainer":
        return ("up_layers", "decoder", "seg_decoder", "bdr_decoder", "conv_blocks_context", "conv_blocks")
    if trainer_name in {"BANetTrainerV2", "nnUNetTrainerV2"}:
        return ("conv_blocks_localization", "decoder", "up", "conv_blocks_context", "conv_blocks")
    return ("decoder", "up", "conv_blocks_localization", "conv_blocks_context", "conv_blocks")


def _candidate_layers_for_trainer(trainer_name: str, explicit_layer: str) -> List[str]:
    if explicit_layer == "first_skip" and trainer_name == "MedNeXtTrainerV2":
        return ["enc_block_0"]
    if explicit_layer == "first_skip" and trainer_name == "nnUNetTrainerV2":
        return ["conv_blocks_localization", "up", "decoder", "conv_blocks_context", "conv_blocks"]
    if explicit_layer == "first_skip":
        return list(_first_skip_layers(trainer_name))
    if explicit_layer != "auto":
        return [explicit_layer]
    return list(_preferred_feature_layers(trainer_name))


def _normalize_requested_feature_layer(args_layer: str, trainer_name: str) -> str:
    if trainer_name == "MedNeXtTrainerV2" and args_layer == "first_skip":
        return "enc_block_0"
    if trainer_name == "nnUNetTrainerV2" and args_layer == "first_skip":
        return "conv_blocks_localization"
    return args_layer


def _find_feature_module(network: torch.nn.Module, feature_layer: str, trainer_name: Optional[str] = None):
    named = list(network.named_modules())
    matches = [(n, m) for n, m in named if feature_layer in n]
    if matches:
        return matches[-1]
    if feature_layer == "auto":
        # Prefer architecture-specific decoder-like modules.
        for key in _preferred_feature_layers(trainer_name or ""):
            matches = [(n, m) for n, m in named if key in n]
            if matches:
                return matches[-1]
    if feature_layer == "first_skip":
        for key in _first_skip_layers(trainer_name or ""):
            matches = [(n, m) for n, m in named if key in n]
            if matches:
                return matches[-1]
    raise RuntimeError(f"Could not find a module matching feature_layer='{feature_layer}'")


def _activation_map(network: torch.nn.Module, x: torch.Tensor, feature_layer: str, trainer_name: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor, str]:
    if trainer_name == "MedNeXtTrainerV2" and feature_layer == "first_skip":
        return _mednext_enc_block0_map(network, x)

    captured: Dict[str, torch.Tensor] = {}

    def hook_fn(name):
        def _hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            if torch.is_tensor(output):
                captured[name] = output
        return _hook

    name, module = _find_feature_module(network, feature_layer, trainer_name=trainer_name)
    hook = module.register_forward_hook(hook_fn(name))
    logits = network(x)
    hook.remove()
    feat = captured.get(name)
    if feat is None:
        raise RuntimeError(f"No activations captured from '{name}'")
    if feat.dim() != 5:
        raise RuntimeError(f"Expected 5D feature map, got {tuple(feat.shape)}")
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits, feat, name


def _try_activation_map(network: torch.nn.Module, x: torch.Tensor, trainer_name: str, feature_layer: str) -> Tuple[torch.Tensor, torch.Tensor, str]:
    if trainer_name == "MedNeXtTrainerV2" and feature_layer == "first_skip":
        return _activation_map(network, x, "first_skip", trainer_name=trainer_name)
    last_error: Optional[Exception] = None
    for candidate in _candidate_layers_for_trainer(trainer_name, feature_layer):
        try:
            return _activation_map(network, x, candidate, trainer_name=trainer_name)
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"No usable feature layer found for {trainer_name}. Last error: {last_error}")


def _choose_case_file(validation_raw_dir: Path, case_id: Optional[str]) -> Path:
    files = sorted(validation_raw_dir.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No npz files found in {validation_raw_dir}")
    if case_id is None:
        return files[0]
    for f in files:
        if f.stem == case_id:
            return f
    raise FileNotFoundError(f"Case {case_id} not found in {validation_raw_dir}")


def _load_case_npz(case_file: Path, expected_in: Optional[int]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
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


def _resize_to_feature(arr: np.ndarray, feature_shape: Tuple[int, int, int]) -> np.ndarray:
    t = torch.from_numpy(arr[None, None].astype(np.float32))
    t = F.interpolate(t, size=feature_shape, mode="nearest")
    return t[0, 0].cpu().numpy()


def _pick_strongest_axial_slice(heatmap: np.ndarray) -> int:
    # Choose the slice with the largest mean activation so the overlay is not
    # accidentally drawn on a nearly empty slice.
    scores = heatmap.mean(axis=(1, 2))
    return int(np.argmax(scores))


def _make_heatmap(feature_map: torch.Tensor, backend: str, grad: Optional[torch.Tensor] = None) -> np.ndarray:
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


def _pick_slice_index(volume_shape: Tuple[int, int, int]) -> int:
    # Keep the default slice selection independent from the target mask so the
    # visualization always represents the whole volume.
    return volume_shape[0] // 2


def _pick_axis_slices(volume_shape: Tuple[int, int, int]) -> Dict[str, int]:
    # Keep the default slice selection independent from the target mask so the
    # visualization always represents the whole volume.
    return {"axial": volume_shape[0] // 2, "coronal": volume_shape[1] // 2, "sagittal": volume_shape[2] // 2}


def _get_view_slices(data: np.ndarray, heatmap: np.ndarray, axis: str, idx: int):
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
        raise ValueError(axis)
    return img, hm


def _overlay_slice(image: np.ndarray, heatmap: np.ndarray, out_png: Path, title: str) -> None:
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


def _overlay_multi_view(data: np.ndarray, heatmap: np.ndarray, out_png: Path, title: str) -> None:
    # Use the same middle slice policy as the single-view output for consistency.
    volume_shape = (int(data.shape[2]), int(data.shape[3]), int(data.shape[4]))
    slices = _pick_axis_slices(volume_shape)
    fig, axes = plt.subplots(3, 2, figsize=(10, 14))
    for row, axis in enumerate(("axial", "coronal", "sagittal")):
        idx = slices[axis]
        img, hm = _get_view_slices(data, heatmap, axis, idx)
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate feature heatmaps for nnUNet 3D segmentation models.")
    p.add_argument("--trainer", required=False, choices=["BGHNetV4Trainer", "MedNeXtTrainerV2", "BANetTrainerV2", "nnUNetTrainerV2"])
    p.add_argument("--trainers", nargs="+", choices=["BGHNetV4Trainer", "MedNeXtTrainerV2", "BANetTrainerV2", "nnUNetTrainerV2"], help="Run multiple trainers in one command and save the same layout for each.")
    p.add_argument("--task", required=True)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--print-structure", action="store_true", help="Print the trainer network structure and exit.")
    p.add_argument("--debug-stats", action="store_true", help="Print feature statistics and exit after capturing the feature tensor.")
    p.add_argument("--case-id", default=None, help="Case stem in validation_raw; defaults to the first case")
    p.add_argument("--feature-layer", default="first_skip", help="Use 'first_skip' to visualize the first skip connection feature map.")
    p.add_argument("--backend", choices=["activation", "gradcam"], default="activation")
    p.add_argument("--output-dir", default="heatmap_output")
    p.add_argument("--plans-file", default=None)
    p.add_argument("--output-folder", default=None)
    p.add_argument("--dataset-directory", default=None)
    p.add_argument("--validation-raw-dir", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--auto-checkpoint", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def _run_one_trainer(args: argparse.Namespace, trainer_name: str) -> None:
    trainer_cls = _load_trainer(trainer_name)

    plans_file = args.plans_file or str(_default_plans_path(args.task, trainer_name))
    dataset_directory = args.dataset_directory or str(_default_dataset_directory(args.task))
    output_folder = args.output_folder or str(_default_checkpoint_path(args.task, trainer_name, args.fold).parent)
    checkpoint = args.checkpoint
    if args.auto_checkpoint and checkpoint is None:
        checkpoint = str(_default_checkpoint_path(args.task, trainer_name, args.fold))
    validation_raw_dir = args.validation_raw_dir or str(_default_validation_raw_dir(args.task, trainer_name, args.fold))

    trainer = trainer_cls(plans_file, args.fold, output_folder=output_folder, dataset_directory=dataset_directory)
    trainer.initialize(training=False)
    _load_checkpoint_if_available(trainer, checkpoint)
    trainer.network.eval()

    if args.print_structure:
        print(f"\n[{trainer_name}] Network structure:")
        for name, module in trainer.network.named_modules():
            if name:
                print(f"{name}: {module.__class__.__name__}")
        return

    resolved_feature_layer = _normalize_requested_feature_layer(args.feature_layer, trainer_name)
    if trainer_name in {"MedNeXtTrainerV2", "nnUNetTrainerV2"}:
        print(f"[{trainer_name}] Using feature layer: {resolved_feature_layer}")

    case_file = _choose_case_file(Path(validation_raw_dir), args.case_id)
    expected_in = getattr(trainer.network, "num_input_channels", None)
    data, target = _load_case_npz(case_file, expected_in)
    x = torch.from_numpy(data).float().to(args.device)
    x.requires_grad_(args.backend == "gradcam")

    if args.feature_layer == "first_skip":
        print(f"[{trainer_name}] Using first-skip feature layer candidates")

    if args.backend == "gradcam":
        logits, feat_map, layer_name = _try_activation_map(trainer.network, x, trainer_name, resolved_feature_layer)
        tumor_channel = 1 if logits.shape[1] > 1 else 0
        score = logits[:, tumor_channel].mean()
        trainer.network.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        score.backward(retain_graph=True)
        grads: List[torch.Tensor] = []

        def _grad_hook(_module, _grad_in, grad_out):
            g = grad_out[0] if isinstance(grad_out, (tuple, list)) else grad_out
            grads.append(g.detach())

        hook_module = dict(trainer.network.named_modules())[layer_name]
        ghook = hook_module.register_full_backward_hook(_grad_hook)
        trainer.network.zero_grad(set_to_none=True)
        logits, feat_map, _ = _activation_map(trainer.network, x, resolved_feature_layer)
        score = logits[:, tumor_channel].mean()
        score.backward()
        ghook.remove()
        grad = grads[-1] if grads else None
        if grad is None:
            raise RuntimeError("Could not capture gradients for Grad-CAM")
        heatmap = _make_heatmap(feat_map, "gradcam", grad=grad)
    else:
        logits, feat_map, layer_name = _try_activation_map(trainer.network, x, trainer_name, resolved_feature_layer)
        heatmap = _make_heatmap(feat_map, "activation")

    feat_np = feat_map.detach().cpu().numpy()
    print(f"[{trainer_name}] resolved feature layer = {layer_name}")
    print(
        f"[{trainer_name}] feature stats | layer={layer_name} | shape={tuple(feat_np.shape)} | "
        f"min={feat_np.min():.6g} max={feat_np.max():.6g} mean={feat_np.mean():.6g} std={feat_np.std():.6g}"
    )
    print(
        f"[{trainer_name}] heatmap stats | shape={tuple(heatmap.shape)} | "
        f"min={heatmap.min():.6g} max={heatmap.max():.6g} mean={heatmap.mean():.6g} std={heatmap.std():.6g}"
    )
    if args.debug_stats:
        return

    heatmap_resized = _resize_to_feature(heatmap, (int(data.shape[2]), int(data.shape[3]), int(data.shape[4])))
    slice_axes = _pick_axis_slices((int(data.shape[2]), int(data.shape[3]), int(data.shape[4])))
    axial_idx = slice_axes["axial"]
    image_slice = data[0, 0, axial_idx]
    heat_slice = heatmap_resized[axial_idx]

    outdir = Path(args.output_dir) / _default_task_dir(args.task) / trainer_name / f"fold_{args.fold}"
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{case_file.stem}_{args.backend}_{args.feature_layer}"
    np.save(outdir / f"{stem}.npy", heatmap_resized)
    _overlay_slice(image_slice, heat_slice, outdir / f"{stem}.png", title=f"{args.task} | {trainer_name} | {case_file.stem}")
    _overlay_multi_view(data, heatmap_resized, outdir / f"{stem}_multi.png", title=f"{args.task} | {trainer_name} | {case_file.stem}")
    meta = {
        "trainer": trainer_name,
        "task": args.task,
        "fold": args.fold,
        "case_id": case_file.stem,
        "backend": args.backend,
        "feature_layer": args.feature_layer,
        "resolved_feature_layer": resolved_feature_layer,
        "checkpoint": checkpoint,
        "validation_raw_dir": validation_raw_dir,
        "slice_index": axial_idx,
        "slice_axes": slice_axes,
        "layer_name": layer_name,
    }
    (outdir / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{trainer_name}] Saved: {outdir / f'{stem}.npy'}")
    print(f"[{trainer_name}] Saved: {outdir / f'{stem}.png'}")
    print(f"[{trainer_name}] Saved: {outdir / f'{stem}_multi.png'}")
    print(f"[{trainer_name}] Saved: {outdir / f'{stem}.json'}")


def main() -> None:
    args = parse_args()
    trainers = args.trainers or ([args.trainer] if args.trainer else None)
    if not trainers:
        raise ValueError("Please provide --trainer or --trainers")
    for trainer_name in trainers:
        _run_one_trainer(args, trainer_name)


if __name__ == "__main__":
    main()
