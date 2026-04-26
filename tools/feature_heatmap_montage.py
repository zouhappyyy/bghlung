#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for building montage images from batch heatmap visualizations."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def _checkpoint_rank(label: str) -> Tuple[int, int]:
    if label == "best":
        return (1, 0)
    if label == "final":
        return (2, 0)

    match = re.fullmatch(r"ep_(\d+)", label)
    if match:
        return (0, int(match.group(1)))
    return (3, 0)


def _extract_checkpoint_label(filename: str, marker: str) -> str:
    if marker in filename:
        return filename.split(marker, 1)[0]
    return Path(filename).stem


def collect_montage_images(case_dir: Path, marker: str) -> List[Tuple[str, Path]]:
    pairs: List[Tuple[str, Path]] = []
    for image_path in sorted(case_dir.glob(f"*{marker}")):
        label = _extract_checkpoint_label(image_path.name, marker)
        pairs.append((label, image_path))
    pairs.sort(key=lambda item: (_checkpoint_rank(item[0]), item[0]))
    return pairs


def create_montage(
    image_pairs: Sequence[Tuple[str, Path]],
    out_png: Path,
    title: str,
    ncols: int = 4,
) -> None:
    if not image_pairs:
        raise RuntimeError("No images were provided for montage creation")

    ncols = max(1, ncols)
    nrows = math.ceil(len(image_pairs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 7))
    axes_list = axes.flatten().tolist() if hasattr(axes, "flatten") else [axes]

    for ax, (label, image_path) in zip(axes_list, image_pairs):
        image = mpimg.imread(image_path)
        ax.imshow(image)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.axis("off")

    for ax in axes_list[len(image_pairs):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close(fig)


def build_case_montage(
    case_dir: Path,
    marker: str,
    out_name: str,
    title: str,
    ncols: int = 4,
) -> Path | None:
    image_pairs = collect_montage_images(case_dir, marker)
    if not image_pairs:
        return None

    out_png = case_dir / out_name
    create_montage(image_pairs, out_png, title=title, ncols=ncols)
    return out_png


def build_all_case_montages(
    case_dirs: Iterable[Path],
    marker: str,
    out_name: str,
    title_prefix: str,
    ncols: int = 4,
) -> List[Path]:
    outputs: List[Path] = []
    for case_dir in case_dirs:
        out_png = build_case_montage(
            case_dir=case_dir,
            marker=marker,
            out_name=out_name,
            title=f"{title_prefix} | {case_dir.name}",
            ncols=ncols,
        )
        if out_png is not None:
            outputs.append(out_png)
    return outputs
