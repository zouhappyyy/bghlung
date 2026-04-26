#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Voxel-level feature visualization for nnUNet trainers.

This script samples intermediate voxel features from validation cases,
balances background/tumor voxels, and visualizes them with t-SNE/UMAP.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def _lazy_import_optional(name: str):
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _load_trainer(trainer_name: str):
    trainer_map = {
        "BGHNetV4Trainer": "nnunet.training.network_training.BGHNetV4Trainer",
        "MedNeXtTrainerV2": "nnunet.training.network_training.MedNeXt.MedNeXtTrainerV2",
        "BANetTrainerV2": "nnunet.training.network_training.BANetTrainerV2",
        "nnUNetTrainerV2": "nnunet.training.network_training.nnUNetTrainerV2",
    }
    if trainer_name not in trainer_map:
        raise ValueError(f"Unsupported trainer: {trainer_name}. Choose from {sorted(trainer_map)}")
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


def _feature_layer_candidates(trainer_name: str) -> List[str]:
    if trainer_name == "BGHNetV4Trainer":
        return ["conv_blocks_context", "conv_blocks_localization", "conv_blocks"]
    if trainer_name == "BANetTrainerV2":
        return ["conv_blocks_context", "conv_blocks_localization", "conv_blocks"]
    if trainer_name == "nnUNetTrainerV2":
        return ["conv_blocks_context", "conv_blocks_localization", "conv_blocks"]
    return ["conv_blocks_context", "conv_blocks_localization", "conv_blocks"]


def _collect_matching_modules(network: torch.nn.Module, feature_layer: str):
    named_modules = list(network.named_modules())
    matches = [(name, module) for name, module in named_modules if feature_layer in name]
    if matches:
        return matches
    if feature_layer == "auto":
        for key in _feature_layer_candidates("auto"):
            matches = [(name, module) for name, module in named_modules if key in name]
            if matches:
                return matches
    return []


@torch.no_grad()
def _capture_voxel_features(network: torch.nn.Module, data: torch.Tensor, feature_layer: str) -> torch.Tensor:
    captured: Dict[str, torch.Tensor] = {}

    def _hook_factory(tag):
        def _hook(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            if torch.is_tensor(output):
                captured[tag] = output.detach()
        return _hook

    matches = _collect_matching_modules(network, feature_layer)
    if not matches:
        raise RuntimeError(f"Could not find any module containing '{feature_layer}'")

    tag, module = matches[-1]
    hook = module.register_forward_hook(_hook_factory(tag))
    _ = network(data)
    hook.remove()

    if not captured:
        raise RuntimeError(f"No features captured for layer '{feature_layer}'")
    feat = next(iter(captured.values()))
    if feat.dim() != 5:
        raise RuntimeError(f"Expected voxel feature map with 5 dims, got shape {tuple(feat.shape)}")
    return feat


def _load_checkpoint_if_available(trainer, checkpoint_path: Optional[str]) -> None:
    if checkpoint_path is None:
        return
    ckpt = Path(checkpoint_path)
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(str(ckpt), map_location=torch.device("cpu"))
    trainer.network.load_state_dict(state["state_dict"], strict=False)


def _downsample_indices(idxs: np.ndarray, max_keep: int, rng: np.random.Generator) -> np.ndarray:
    if len(idxs) <= max_keep:
        return idxs
    return rng.choice(idxs, size=max_keep, replace=False)


def _resize_target_to_feature(target: np.ndarray, feature_shape: Tuple[int, int, int]) -> np.ndarray:
    # target: [D, H, W] -> resized target aligned with feature_map spatial size
    t = torch.from_numpy(target[None, None].astype(np.float32))
    t = F.interpolate(t, size=feature_shape, mode="nearest")
    return t[0, 0].cpu().numpy().astype(np.int64)


def _boundary_band_mask(mask: np.ndarray, band_width: int) -> np.ndarray:
    """Return a binary mask for voxels within a band around the tumor boundary.

    The band contains:
      - tumor voxels that touch background after dilation/erosion behavior
      - nearby background voxels within `band_width`
    """
    mask_t = torch.from_numpy(mask[None, None].astype(np.float32))
    tumor = (mask_t > 0).float()
    kernel = torch.ones((1, 1, 3, 3, 3), dtype=torch.float32)
    pad = 1
    dil = (F.conv3d(tumor, kernel, padding=pad) > 0)
    ero = (F.conv3d(tumor, kernel, padding=pad) == kernel.numel())
    boundary = (dil & ~ero)
    if band_width <= 1:
        band = boundary
    else:
        band = boundary.clone()
        cur = boundary.clone()
        for _ in range(band_width - 1):
            cur = (F.conv3d(cur.float(), kernel, padding=pad) > 0)
            band = band | cur
    return band[0, 0].cpu().numpy().astype(bool)


def _adaptive_boundary_mask(target: np.ndarray, band_width: int) -> np.ndarray:
    # Try progressively wider bands until we get a non-empty mask.
    for bw in range(max(1, band_width), 6):
        band = _boundary_band_mask(target, band_width=bw)
        if band.any():
            return band
    # Final fallback: full non-zero tumor mask, so sampling never collapses to 0.
    return target > 0


def _sample_voxels(feature_map: torch.Tensor, target: np.ndarray, max_bg: int, max_fg: int, seed: int, band_width: int) -> Tuple[np.ndarray, np.ndarray]:
    # feature_map: [1, C, D, H, W]
    feat = feature_map[0].permute(1, 2, 3, 0).reshape(-1, feature_map.shape[1]).detach().cpu().numpy()
    target = _resize_target_to_feature(target, (int(feature_map.shape[2]), int(feature_map.shape[3]), int(feature_map.shape[4])))
    lbl = target.reshape(-1).astype(np.int64)

    band_mask = _adaptive_boundary_mask(target, band_width=band_width).reshape(-1)
    bg_idx = np.where((lbl == 0) & band_mask)[0]
    fg_idx = np.where((lbl > 0) & band_mask)[0]
    if len(bg_idx) == 0 and len(fg_idx) == 0:
        # absolute fallback: sample from the whole resized mask to avoid empty outputs
        bg_idx = np.where(lbl == 0)[0]
        fg_idx = np.where(lbl > 0)[0]
    rng = np.random.default_rng(seed)
    bg_sel = _downsample_indices(bg_idx, max_bg, rng)
    fg_sel = _downsample_indices(fg_idx, max_fg, rng)
    sel = np.concatenate([bg_sel, fg_sel])
    sel = rng.permutation(sel)
    return feat[sel], (lbl[sel] > 0).astype(np.int64)


def _collect_voxel_samples(trainer, validation_raw_dir: str, feature_layer: str, checkpoint_path: Optional[str], device: str, num_cases: int, max_bg_per_case: int, max_fg_per_case: int, seed: int, band_width: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    trainer.initialize(training=False)
    _load_checkpoint_if_available(trainer, checkpoint_path)
    trainer.network.eval()

    validation_raw_dir = Path(validation_raw_dir)
    case_files = sorted(validation_raw_dir.glob("*.npz"))
    if not case_files:
        raise RuntimeError(f"No npz files found in {validation_raw_dir}")

    expected_in = getattr(trainer.network, "num_input_channels", None)
    Xs, ys, case_ids = [], [], []

    for case_file in case_files[:num_cases]:
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
        target = sample[target_key] if target_key is not None else np.zeros(data.shape[2:], dtype=np.int64)
        if target.ndim == 4:
            target = target[0]
        if target.ndim == 3:
            pass
        else:
            raise RuntimeError(f"Expected target to be 3D, got {target.shape} from {case_file}")

        data_t = torch.from_numpy(data).float().to(device)
        feat_map = _capture_voxel_features(trainer.network, data_t, feature_layer)
        X_case, y_case = _sample_voxels(feat_map, target, max_bg_per_case, max_fg_per_case, seed=seed, band_width=band_width)
        if X_case.shape[0] == 0:
            continue
        Xs.append(X_case)
        ys.append(y_case)
        case_ids.extend([case_file.stem] * len(y_case))

    if not Xs:
        raise RuntimeError("No voxel samples were collected. Try increasing --band-width or using a larger --num-cases.")
    return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0), case_ids


def _reduce_2d(features: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == "tsne":
        from sklearn.manifold import TSNE
        n = int(features.shape[0])
        if n < 3:
            raise ValueError(f"Need at least 3 samples for t-SNE, got {n}")
        perplexity = max(1, min(30, max(2, n // 20), n - 1))
        return TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca", learning_rate="auto").fit_transform(features)
    umap_mod = _lazy_import_optional("umap")
    if umap_mod is None:
        raise ImportError("UMAP is not installed. Install package 'umap-learn' to use --method umap")
    return umap_mod.UMAP(n_components=2, random_state=seed).fit_transform(features)


def _plot_embedding(embedding: np.ndarray, labels: np.ndarray, case_ids: Sequence[str], title: str, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    masks = [(labels == 0), (labels == 1)]
    titles = ["Background voxels", "Tumor voxels"]
    colors = ["tab:blue", "tab:red"]

    for ax, mask, panel_title, color in zip(axes, masks, titles, colors):
        pts = embedding[mask]
        if pts.shape[0] == 0:
            ax.set_title(f"{panel_title} (empty)")
            ax.axis("off")
            continue
        ax.scatter(pts[:, 0], pts[:, 1], c=color, s=14, alpha=0.85, edgecolors="none")
        ax.set_title(f"{panel_title} (n={pts.shape[0]})")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.grid(alpha=0.2, linestyle="--")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voxel-level latent feature visualization for nnUNet trainers.")
    parser.add_argument("--trainer", required=True, choices=["BGHNetV4Trainer", "MedNeXtTrainerV2", "BANetTrainerV2", "nnUNetTrainerV2"])
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--task", required=True)
    parser.add_argument("--plans-file", default=None)
    parser.add_argument("--output-folder", default=None)
    parser.add_argument("--dataset-directory", default=None)
    parser.add_argument("--validation-raw-dir", default=None)
    parser.add_argument("--feature-layer", default="auto")
    parser.add_argument("--num-cases", type=int, default=8)
    parser.add_argument("--max-bg-per-case", type=int, default=1000)
    parser.add_argument("--max-fg-per-case", type=int, default=1000)
    parser.add_argument("--band-width", type=int, default=2)
    parser.add_argument("--method", choices=["tsne", "umap"], default="tsne")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="feature_vis_output")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--auto-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trainer_cls = _load_trainer(args.trainer)

    if args.plans_file is None:
        args.plans_file = str(_default_plans_path(args.task, args.trainer))
    if args.dataset_directory is None:
        args.dataset_directory = str(_default_dataset_directory(args.task))
    if args.output_folder is None:
        args.output_folder = str(_default_checkpoint_path(args.task, args.trainer, args.fold).parent)
    if args.auto_checkpoint and args.checkpoint is None:
        args.checkpoint = str(_default_checkpoint_path(args.task, args.trainer, args.fold))
    if args.validation_raw_dir is None:
        args.validation_raw_dir = str(_default_validation_raw_dir(args.task, args.trainer, args.fold))

    trainer = trainer_cls(args.plans_file, args.fold, output_folder=args.output_folder, dataset_directory=args.dataset_directory)
    features, labels, case_ids = _collect_voxel_samples(
        trainer=trainer,
        validation_raw_dir=args.validation_raw_dir,
        feature_layer=args.feature_layer,
        checkpoint_path=args.checkpoint,
        device=args.device,
        num_cases=args.num_cases,
        max_bg_per_case=args.max_bg_per_case,
        max_fg_per_case=args.max_fg_per_case,
        seed=args.seed,
        band_width=args.band_width,
    )

    embedding = _reduce_2d(features, args.method, args.seed)
    outdir = Path(args.outdir) / _default_task_dir(args.task) / args.trainer / f"fold_{args.fold}"
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.method}_{args.feature_layer}_voxel"
    npz_path = outdir / f"{stem}.npz"
    png_path = outdir / f"{stem}.png"
    meta_path = outdir / f"{stem}.json"

    np.savez_compressed(npz_path, features=features, embedding=embedding, labels=labels, case_ids=np.asarray(case_ids))
    _plot_embedding(embedding, labels, case_ids, f"{args.task} | {args.trainer} | voxel features", png_path)
    meta = {
        "trainer": args.trainer,
        "task": args.task,
        "fold": args.fold,
        "method": args.method,
        "feature_layer": args.feature_layer,
        "num_cases": args.num_cases,
        "max_bg_per_case": args.max_bg_per_case,
        "max_fg_per_case": args.max_fg_per_case,
        "band_width": args.band_width,
        "checkpoint": args.checkpoint,
        "validation_raw_dir": args.validation_raw_dir,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {npz_path}")
    print(f"Saved: {png_path}")
    print(f"Saved: {meta_path}")


if __name__ == "__main__":
    main()
