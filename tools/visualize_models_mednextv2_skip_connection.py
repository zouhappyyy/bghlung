#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch-visualize MedNeXtTrainerV2 decoder features for all saved epochs.

This script is specialized for MedNeXtTrainerV2 models trained on
Task530_EsoTJ_30pct. It iterates through every ``model_ep_*.model`` checkpoint
and every available ``.npz`` case, then extracts a target MedNeXt decoder
feature map and generates heatmap visualizations.

By default, the script captures ``dec_block_0``, which is the shallowest decoder
block after the final skip fusion and is the closest MedNeXt analogue to the
"first skip connection" visualization used in the nnUNet script.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from feature_heatmap_montage import build_all_case_montages
from visualize_nnunetv2_skip_connection import (
    NETWORK,
    TASK,
    _backward_target_from_logits,
    _unwrap_module,
    default_dataset_directory,
    load_case_npz,
    make_heatmap,
    pick_middle_slices,
    plot_multi_view_overlay,
    plot_single_view_overlay,
    resize_to_original,
)


TRAINER_NAME = "MedNeXtTrainerV2"
TRAINER_MODULE = "nnunet.training.network_training.MedNeXt.MedNeXtTrainerV2"
CKPT_BASE = (
    Path("/home/fangzheng/zoule/ESO_nnUNet_dataset")
    / TASK
    / f"{TRAINER_NAME}__nnUNetPlansv2.1"
)
DEFAULT_TARGET_LAYER = "dec_block_0"


def default_plans_path() -> Path:
    return CKPT_BASE / "plans.pkl"


def find_epoch_checkpoints(checkpoint_dir: Path, include_final: bool) -> List[Path]:
    checkpoints = sorted(checkpoint_dir.glob("model_ep_*.model"))
    best_ckpt = checkpoint_dir / "model_best.model"
    if best_ckpt.is_file():
        checkpoints.append(best_ckpt)
    if include_final:
        final_ckpt = checkpoint_dir / "model_final_checkpoint.model"
        if final_ckpt.is_file():
            checkpoints.append(final_ckpt)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {checkpoint_dir}")
    return checkpoints


def _find_npz_files(directory: Path, recursive: bool) -> List[Path]:
    pattern = "**/*.npz" if recursive else "*.npz"
    return sorted(directory.glob(pattern))


def choose_case_files(
    candidate_dirs: Sequence[Path],
    case_id: Optional[str],
    max_cases: Optional[int],
) -> Tuple[List[Path], Path]:
    searched_dirs: List[Path] = []
    case_files: List[Path] = []

    for idx, candidate_dir in enumerate(candidate_dirs):
        if candidate_dir in searched_dirs:
            continue
        searched_dirs.append(candidate_dir)
        recursive = idx > 0
        if not candidate_dir.is_dir():
            continue
        case_files = _find_npz_files(candidate_dir, recursive=recursive)
        if case_files:
            source_dir = candidate_dir
            break
    else:
        searched_text = "\n".join(f"  - {d}" for d in searched_dirs)
        raise RuntimeError(
            "No npz files found. Checked these directories:\n"
            f"{searched_text}\n"
            "Use --case-dir to point to the folder containing input .npz files."
        )

    if case_id is not None:
        case_files = [f for f in case_files if f.stem == case_id]
        if not case_files:
            raise FileNotFoundError(f"Case {case_id} not found under {source_dir}")

    if max_cases is not None:
        case_files = case_files[:max_cases]
    return case_files, source_dir


def checkpoint_tag(checkpoint_path: Path) -> str:
    if checkpoint_path.name == "model_best.model":
        return "best"
    if checkpoint_path.name == "model_final_checkpoint.model":
        return "final"

    match = re.fullmatch(r"model_ep_(.+)\.model", checkpoint_path.name)
    if match:
        return f"ep_{match.group(1)}"
    return checkpoint_path.stem


def load_checkpoint(trainer, checkpoint_path: Path) -> None:
    state = torch.load(str(checkpoint_path), map_location=torch.device("cpu"))
    trainer.network.load_state_dict(state["state_dict"], strict=False)
    trainer.network.eval()


def find_target_layer(network: torch.nn.Module, target_layer: str) -> Tuple[str, torch.nn.Module]:
    net = _unwrap_module(network)
    module_dict = dict(net.named_modules())

    if target_layer in module_dict:
        return target_layer, module_dict[target_layer]

    if hasattr(net, target_layer):
        return target_layer, getattr(net, target_layer)

    available = [
        name
        for name in module_dict
        if name.startswith("enc_block_")
        or name.startswith("down_")
        or name.startswith("bottleneck")
        or name.startswith("up_")
        or name.startswith("dec_block_")
        or name.startswith("out_")
    ]
    preview = ", ".join(available[:20])
    raise RuntimeError(
        f"Could not find target layer '{target_layer}'. "
        f"Available MedNeXt layers include: {preview}"
    )


def extract_heatmap(
    network: torch.nn.Module,
    x: torch.Tensor,
    backend: str,
    target_layer: str,
    normalize: str,
) -> Tuple[np.ndarray, str, List[int]]:
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
        return heatmap, layer_name, list(feat_map.shape)

    captured: Dict[str, torch.Tensor] = {}

    def hook_fn(_module, _inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        if torch.is_tensor(output):
            captured["feat"] = output

    hook = target_module.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            net(x)
    finally:
        hook.remove()

    feat_map = captured.get("feat")
    if feat_map is None:
        raise RuntimeError(f"Failed to capture features from '{layer_name}'")

    heatmap = make_heatmap(feat_map, "activation", normalize=normalize)
    return heatmap, layer_name, list(feat_map.shape)


def save_outputs(
    *,
    outdir: Path,
    case_id: str,
    backend: str,
    normalize: str,
    checkpoint_path: Path,
    checkpoint_name: str,
    layer_name: str,
    feature_shape: List[int],
    data: np.ndarray,
    heatmap_resized: np.ndarray,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    slices = pick_middle_slices(
        (int(data.shape[2]), int(data.shape[3]), int(data.shape[4]))
    )
    stem = f"{checkpoint_name}_{backend}_{normalize}_{layer_name.replace('.', '_')}"

    axial_idx = slices["axial"]
    plot_single_view_overlay(
        data[0, 0, axial_idx],
        heatmap_resized[axial_idx],
        outdir / f"{stem}_axial.png",
        f"{case_id} | {checkpoint_name} | {backend} | {layer_name}",
    )

    plot_multi_view_overlay(
        data,
        heatmap_resized,
        outdir / f"{stem}_3views.png",
        f"{case_id} | {checkpoint_name} | {backend} | {layer_name}",
    )

    meta = {
        "task": TASK,
        "trainer": TRAINER_NAME,
        "network": NETWORK,
        "case_id": case_id,
        "backend": backend,
        "normalize": normalize,
        "layer_name": layer_name,
        "target_feature": "mednext_decoder_feature",
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
    }
    (outdir / f"{stem}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-generate MedNeXt feature heatmaps for all saved epochs",
    )
    parser.add_argument("--fold", type=int, default=2, help="Fold number")
    parser.add_argument(
        "--backend",
        choices=["activation", "gradcam"],
        default="activation",
        help="Heatmap generation backend",
    )
    parser.add_argument(
        "--target-layer",
        default=DEFAULT_TARGET_LAYER,
        help="MedNeXt layer to visualize, default: dec_block_0",
    )
    parser.add_argument(
        "--output-dir",
        default="heatmap_output_mednext_all_epochs",
        help="Output directory root",
    )
    parser.add_argument(
        "--normalize",
        choices=["quantile", "none"],
        default="quantile",
        help="Heatmap normalization mode",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Specific case ID to process; defaults to all available cases",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional limit for the number of cases",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory containing model_ep_*.model checkpoints",
    )
    parser.add_argument(
        "--dataset-directory",
        default=None,
        help="Custom dataset directory",
    )
    parser.add_argument(
        "--plans-file",
        default=None,
        help="Custom plans file",
    )
    parser.add_argument(
        "--case-dir",
        default=None,
        help="Directory containing input .npz files",
    )
    parser.add_argument(
        "--include-final",
        action="store_true",
        help="Also process model_final_checkpoint.model if it exists",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--skip-montage",
        action="store_true",
        help="Do not create per-case montage images after batch processing",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else CKPT_BASE / f"fold_{args.fold}"
    plans_file = args.plans_file or str(default_plans_path())
    dataset_directory = args.dataset_directory or str(default_dataset_directory(TASK))
    dataset_directory_path = Path(dataset_directory)
    case_dir = Path(args.case_dir) if args.case_dir else checkpoint_dir / "validation_raw"
    output_folder = str(CKPT_BASE / f"fold_{args.fold}")

    print(f"\n{'=' * 72}")
    print("Batch MedNeXtTrainerV2 Feature Heatmap Generation")
    print(f"{'=' * 72}")
    print(f"Task:               {TASK}")
    print(f"Trainer:            {TRAINER_NAME}")
    print(f"Fold:               {args.fold}")
    print(f"Backend:            {args.backend}")
    print(f"Target layer:       {args.target_layer}")
    print(f"Device:             {args.device}")
    print(f"Checkpoint dir:     {checkpoint_dir}")
    print(f"Requested case dir: {case_dir}")
    print(f"Dataset directory:  {dataset_directory_path}")
    print(f"Output dir:         {args.output_dir}")
    print(f"{'=' * 72}\n")

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

    checkpoints = find_epoch_checkpoints(checkpoint_dir, args.include_final)
    case_files, resolved_case_dir = choose_case_files(
        [case_dir, dataset_directory_path],
        args.case_id,
        args.max_cases,
    )
    total_jobs = len(checkpoints) * len(case_files)

    print(f"Found {len(checkpoints)} checkpoints")
    print(f"Found {len(case_files)} cases")
    print(f"Resolved case dir:  {resolved_case_dir}")
    print(f"Total jobs:         {total_jobs}\n")

    expected_in = getattr(trainer.network, "num_input_channels", None)
    if expected_in is None:
        expected_in = 1
    case_output_dirs: List[Path] = []

    job_idx = 0
    for checkpoint_path in checkpoints:
        ckpt_name = checkpoint_tag(checkpoint_path)
        print(f"Loading checkpoint: {ckpt_name}")
        load_checkpoint(trainer, checkpoint_path)

        for case_file in case_files:
            job_idx += 1
            case_id = case_file.stem
            print(f"[{job_idx}/{total_jobs}] {ckpt_name} -> {case_id}")

            data, _ = load_case_npz(case_file, expected_in)
            x = torch.from_numpy(data).float().to(args.device)
            if args.backend == "gradcam":
                x.requires_grad_(True)

            heatmap, layer_name, feature_shape = extract_heatmap(
                trainer.network,
                x,
                args.backend,
                args.target_layer,
                args.normalize,
            )
            original_shape = (int(data.shape[2]), int(data.shape[3]), int(data.shape[4]))
            heatmap_resized = resize_to_original(heatmap, original_shape)

            outdir = (
                Path(args.output_dir)
                / TASK
                / TRAINER_NAME
                / f"fold_{args.fold}"
                / case_id
            )
            if outdir not in case_output_dirs:
                case_output_dirs.append(outdir)
            save_outputs(
                outdir=outdir,
                case_id=case_id,
                backend=args.backend,
                normalize=args.normalize,
                checkpoint_path=checkpoint_path,
                checkpoint_name=ckpt_name,
                layer_name=layer_name,
                feature_shape=feature_shape,
                data=data,
                heatmap_resized=heatmap_resized,
            )

    if not args.skip_montage:
        montage_outputs = build_all_case_montages(
            case_dirs=case_output_dirs,
            marker=f"_{args.backend}_{args.normalize}_{args.target_layer.replace('.', '_')}_3views.png",
            out_name=f"{args.backend}_{args.normalize}_{args.target_layer.replace('.', '_')}_montage.png",
            title_prefix=f"{TASK} | {TRAINER_NAME} | fold_{args.fold}",
            ncols=4,
        )
        print(f"Created {len(montage_outputs)} montage image(s)")

    print(f"\n{'=' * 72}")
    print("Batch visualization complete")
    print(
        "Generated outputs under: "
        f"{Path(args.output_dir) / TASK / TRAINER_NAME / f'fold_{args.fold}'}"
    )
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()

# python tools/visualize_models_mednextv2_skip_connection.py --fold 2
