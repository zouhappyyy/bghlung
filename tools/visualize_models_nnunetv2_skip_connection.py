#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch-visualize skip-connection heatmaps for all saved nnUNet epochs.

This script is designed for nnUNetTrainerV2 models trained on
Task530_EsoTJ_30pct. It iterates over every checkpoint matching
``model_ep_*.model`` and every validation ``.npz`` case, then generates
feature heatmaps from the target skip-connection layer.

Outputs are organized by case and checkpoint so that heatmaps from different
epochs can be compared directly.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from visualize_nnunetv2_skip_connection import (
    CKPT_BASE,
    NETWORK,
    TASK,
    TRAINER_NAME,
    _backward_target_from_logits,
    _unwrap_module,
    default_dataset_directory,
    default_plans_path,
    default_validation_raw_dir,
    find_first_skip_connection,
    load_case_npz,
    make_heatmap,
    pick_middle_slices,
    plot_multi_view_overlay,
    plot_single_view_overlay,
    resize_to_original,
)


def find_epoch_checkpoints(checkpoint_dir: Path, include_final: bool) -> List[Path]:
    checkpoints = sorted(checkpoint_dir.glob("model_ep_*.model"))
    if include_final:
        final_ckpt = checkpoint_dir / "model_final_checkpoint.model"
        if final_ckpt.is_file():
            checkpoints.append(final_ckpt)
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {checkpoint_dir}")
    return checkpoints


def choose_case_files(
    validation_raw_dir: Path,
    case_id: Optional[str],
    max_cases: Optional[int],
) -> List[Path]:
    case_files = sorted(validation_raw_dir.glob("*.npz"))
    if not case_files:
        raise RuntimeError(f"No npz files found in {validation_raw_dir}")

    if case_id is not None:
        case_files = [f for f in case_files if f.stem == case_id]
        if not case_files:
            raise FileNotFoundError(f"Case {case_id} not found in {validation_raw_dir}")

    if max_cases is not None:
        case_files = case_files[:max_cases]
    return case_files


def checkpoint_tag(checkpoint_path: Path) -> str:
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


def extract_heatmap(
    network: torch.nn.Module,
    x: torch.Tensor,
    backend: str,
) -> Tuple[np.ndarray, str, List[int]]:
    if backend == "gradcam":
        net = _unwrap_module(network)
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

        heatmap = make_heatmap(feat_map, "gradcam", grad=grad)
        feature_shape = list(feat_map.shape)
        return heatmap, layer_name, feature_shape

    captured: Dict[str, torch.Tensor] = {}
    net = _unwrap_module(network)
    layer_name, skip_module = find_first_skip_connection(net)

    def hook_fn(_module, _inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        if torch.is_tensor(output):
            captured["feat"] = output

    hook = skip_module.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            net(x)
    finally:
        hook.remove()

    feat_map = captured.get("feat")
    if feat_map is None:
        raise RuntimeError(f"Failed to capture features from '{layer_name}'")

    heatmap = make_heatmap(feat_map, "activation")
    feature_shape = list(feat_map.shape)
    return heatmap, layer_name, feature_shape


def save_outputs(
    *,
    outdir: Path,
    case_id: str,
    backend: str,
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
    stem = f"{checkpoint_name}_{backend}_first_skip"

    np.save(outdir / f"{stem}.npy", heatmap_resized)

    axial_idx = slices["axial"]
    plot_single_view_overlay(
        data[0, 0, axial_idx],
        heatmap_resized[axial_idx],
        outdir / f"{stem}_axial.png",
        f"{case_id} | {checkpoint_name} | {backend} | First Skip Connection",
    )

    plot_multi_view_overlay(
        data,
        heatmap_resized,
        outdir / f"{stem}_3views.png",
        f"{case_id} | {checkpoint_name} | {backend} | First Skip Connection",
    )

    meta = {
        "task": TASK,
        "trainer": TRAINER_NAME,
        "network": NETWORK,
        "case_id": case_id,
        "backend": backend,
        "layer_name": layer_name,
        "target_feature": "first_skip_connection",
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
        description="Batch-generate nnUNet skip-connection heatmaps for all saved epochs",
    )
    parser.add_argument("--fold", type=int, default=1, help="Fold number (0-4)")
    parser.add_argument(
        "--backend",
        choices=["activation", "gradcam"],
        default="activation",
        help="Heatmap generation backend",
    )
    parser.add_argument(
        "--output-dir",
        default="heatmap_output_all_epochs",
        help="Output directory root",
    )
    parser.add_argument(
        "--case-id",
        default=None,
        help="Specific case ID to process; defaults to all validation cases",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional limit for the number of validation cases",
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
        "--validation-raw-dir",
        default=None,
        help="Custom validation_raw directory",
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
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else CKPT_BASE / f"fold_{args.fold}"
    validation_raw_dir = (
        Path(args.validation_raw_dir)
        if args.validation_raw_dir
        else Path(default_validation_raw_dir(args.fold))
    )
    plans_file = args.plans_file or str(default_plans_path())
    dataset_directory = args.dataset_directory or str(default_dataset_directory(TASK))
    output_folder = str(CKPT_BASE / f"fold_{args.fold}")

    print(f"\n{'=' * 72}")
    print("Batch nnUNetTrainerV2 Skip Connection Heatmap Generation")
    print(f"{'=' * 72}")
    print(f"Task:               {TASK}")
    print(f"Trainer:            {TRAINER_NAME}")
    print(f"Fold:               {args.fold}")
    print(f"Backend:            {args.backend}")
    print(f"Device:             {args.device}")
    print(f"Checkpoint dir:     {checkpoint_dir}")
    print(f"Validation raw dir: {validation_raw_dir}")
    print(f"Output dir:         {args.output_dir}")
    print(f"{'=' * 72}\n")

    trainer_module = importlib.import_module(
        "nnunet.training.network_training.nnUNetTrainerV2"
    )
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
    case_files = choose_case_files(validation_raw_dir, args.case_id, args.max_cases)
    total_jobs = len(checkpoints) * len(case_files)

    print(f"Found {len(checkpoints)} checkpoints")
    print(f"Found {len(case_files)} cases")
    print(f"Total jobs: {total_jobs}\n")

    expected_in = getattr(trainer.network, "num_input_channels", None)

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
            save_outputs(
                outdir=outdir,
                case_id=case_id,
                backend=args.backend,
                checkpoint_path=checkpoint_path,
                checkpoint_name=ckpt_name,
                layer_name=layer_name,
                feature_shape=feature_shape,
                data=data,
                heatmap_resized=heatmap_resized,
            )

    print(f"\n{'=' * 72}")
    print("Batch visualization complete")
    print(f"Generated outputs under: {Path(args.output_dir) / TASK / TRAINER_NAME / f'fold_{args.fold}'}")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()


# python tools/visualize_models_nnunetv2_skip_connection.py --fold 2
