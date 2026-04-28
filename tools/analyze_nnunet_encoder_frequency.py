#!/usr/bin/env python3
"""Analyze frequency-spectrum changes across model encoder stages.

This script compares the input spectrum with the spectra after the first two
encoder stages of a supported segmentation model. It currently supports:
- `nnUNetTrainerV2`: `conv_blocks_context[0]`, `conv_blocks_context[1]`
- `MedNeXtTrainerV2`: `enc_block_0`, `enc_block_1`

It is designed for Task530_EsoTJ_30pct and defaults to the paths provided for
the nnUNet baseline model, while still allowing custom paths via CLI flags.

Outputs:
- `summary.json`: aggregate statistics and configuration
- `case_metrics.csv`: per-case spectrum statistics for each layer
- `radial_profiles.csv`: mean normalized radial spectra
- `mean_spectrum_slices.png`: central spectrum slices for each layer
- `radial_profiles.png`: line plot of radial spectra
- `band_energy_ratios.png`: low/mid/high frequency energy ratios
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from nnunet.training.model_restore import restore_model


MODEL_CONFIGS = {
    "nnUNetTrainerV2": {
        "trainer_module": "nnunet.training.network_training.nnUNetTrainerV2",
        "trainer_class": "nnUNetTrainerV2",
        "default_fold_dir": Path(
            "ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/"
            "nnUNetTrainerV2__nnUNetPlansv2.1/fold_1"
        ),
        "default_output_dir": Path(
            "feature_vis_output/frequency_spectrum/Task530_EsoTJ_30pct/nnUNetTrainerV2/fold_1"
        ),
        "layers": (
            ("encoder_stage_0", "conv_blocks_context", 0, "Encoder Stage 0"),
            ("encoder_stage_1", "conv_blocks_context", 1, "Encoder Stage 1"),
        ),
    },
    "MedNeXtTrainerV2": {
        "trainer_module": "nnunet.training.network_training.MedNeXt.MedNeXtTrainerV2",
        "trainer_class": "MedNeXtTrainerV2",
        "default_fold_dir": Path(
            "ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/"
            "MedNeXtTrainerV2__nnUNetPlansv2.1/fold_2"
        ),
        "default_output_dir": Path(
            "feature_vis_output/frequency_spectrum/Task530_EsoTJ_30pct/MedNeXtTrainerV2/fold_2"
        ),
        "layers": (
            ("encoder_stage_0", "enc_block_0", None, "Encoder Stage 0"),
            ("encoder_stage_1", "enc_block_1", None, "Encoder Stage 1"),
        ),
    },
}
DEFAULT_DATA_DIR = Path(
    "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/"
    "Task530_EsoTJ_30pct/nnUNetData2D_plans_v2.1_trgSp_1x1x1_stage0"
)
LAYER_ORDER = ("input", "encoder_stage_0", "encoder_stage_1")
LAYER_TITLES_BASE = {"input": "Input"}
RADIAL_CACHE: Dict[Tuple[int, int, int], np.ndarray] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze 3D frequency-spectrum changes for the first two encoder stages."
    )
    parser.add_argument(
        "--trainer",
        choices=sorted(MODEL_CONFIGS.keys()),
        default="nnUNetTrainerV2",
        help="Trainer/model family to analyze.",
    )
    parser.add_argument("--fold-dir", default=None, help="Fold directory containing checkpoint files.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory containing preprocessed .npz cases.")
    parser.add_argument("--output-dir", default=None, help="Directory to save outputs.")
    parser.add_argument(
        "--checkpoint-name",
        default="model_final_checkpoint",
        help="Checkpoint stem inside fold-dir, without suffix. Example: model_final_checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional full checkpoint path. Overrides --checkpoint-name.",
    )
    parser.add_argument(
        "--plans-file",
        default=None,
        help="Optional plans.pkl path used only when checkpoint .pkl metadata is unavailable.",
    )
    parser.add_argument("--fold", type=int, default=None, help="Fold index used by the trainer fallback path.")
    parser.add_argument("--case-id", default=None, help="Analyze a single case id only.")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit the number of analyzed cases.")
    parser.add_argument("--num-bins", type=int, default=96, help="Number of radial frequency bins.")
    parser.add_argument(
        "--spectrum-grid-size",
        type=int,
        default=96,
        help="Common cube size used when averaging 3D spectra across cases.",
    )
    parser.add_argument(
        "--remove-dc",
        action="store_true",
        default=True,
        help="Subtract the spatial mean before FFT. Enabled by default.",
    )
    parser.add_argument(
        "--keep-dc",
        dest="remove_dc",
        action="store_false",
        help="Keep the DC component before FFT.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to run inference on.",
    )
    return parser.parse_args()


def resolve_checkpoint_path(fold_dir: Path, checkpoint_name: str, checkpoint: Optional[str]) -> Path:
    if checkpoint is not None:
        return Path(checkpoint)
    return fold_dir / f"{checkpoint_name}.model"


def infer_fold_from_dir(fold_dir: Path, fallback: int = 0) -> int:
    name = fold_dir.name
    if name.startswith("fold_"):
        suffix = name.split("_", 1)[1]
        if suffix.isdigit():
            return int(suffix)
    return fallback


def find_case_files(data_dir: Path, case_id: Optional[str], max_cases: Optional[int]) -> List[Path]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = sorted(data_dir.rglob("*.npz"))
    if case_id is not None:
        files = [path for path in files if path.stem == case_id]
        if not files:
            raise FileNotFoundError(f"Case '{case_id}' not found under {data_dir}")
    if max_cases is not None:
        files = files[:max_cases]
    if not files:
        raise RuntimeError(f"No .npz cases found under {data_dir}")
    return files


def load_case_npz(case_file: Path, expected_in: Optional[int]) -> np.ndarray:
    with np.load(case_file, allow_pickle=True) as sample:
        data_key = "data" if "data" in sample.files else sample.files[0]
        data = sample[data_key]
    if data.ndim == 4:
        data = data[None]
    if data.ndim != 5:
        raise RuntimeError(f"Expected 5D data after batching, got {data.shape} from {case_file}")

    if expected_in is not None:
        if data.shape[1] < expected_in:
            raise RuntimeError(
                f"Input channels in {case_file} are {data.shape[1]}, but model expects {expected_in}"
            )
        data = data[:, :expected_in]
    return data.astype(np.float32)


def restore_trainer(
    trainer_name: str,
    fold_dir: Path,
    checkpoint_path: Path,
    plans_file: Optional[str],
    data_dir: Path,
    fold: int,
):
    companion_pkl = Path(f"{checkpoint_path}.pkl")
    if companion_pkl.is_file():
        trainer = restore_model(str(companion_pkl), checkpoint=str(checkpoint_path), train=False, fp16=False)
    else:
        if plans_file is None:
            plans_path = fold_dir.parent / "plans.pkl"
        else:
            plans_path = Path(plans_file)
        if not plans_path.is_file():
            raise FileNotFoundError(
                "Could not restore trainer: checkpoint metadata .pkl is missing and no valid plans.pkl was found. "
                f"Checked checkpoint companion '{companion_pkl}' and plans '{plans_path}'."
            )

        config = MODEL_CONFIGS[trainer_name]
        trainer_module = __import__(config["trainer_module"], fromlist=[config["trainer_class"]])
        TrainerClass = getattr(trainer_module, config["trainer_class"])
        trainer = TrainerClass(
            str(plans_path),
            fold,
            output_folder=str(fold_dir),
            dataset_directory=str(data_dir.parent if data_dir.parent != data_dir else data_dir),
        )
        trainer.load_checkpoint(str(checkpoint_path), train=False)

    trainer.network.to(torch.device("cpu"))
    trainer.network.eval()
    base = unwrap_module(trainer.network)
    if hasattr(base, "do_ds"):
        base.do_ds = False
    return trainer


def unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    return module.module if hasattr(module, "module") else module


def get_layer_titles(trainer_name: str) -> Dict[str, str]:
    titles = dict(LAYER_TITLES_BASE)
    for layer_key, _attr_name, _index, title in MODEL_CONFIGS[trainer_name]["layers"]:
        titles[layer_key] = title
    return titles


def get_encoder_modules(trainer_name: str, network: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    base = unwrap_module(network)
    modules: List[Tuple[str, torch.nn.Module]] = []
    for layer_key, attr_name, attr_index, _title in MODEL_CONFIGS[trainer_name]["layers"]:
        if not hasattr(base, attr_name):
            raise RuntimeError(
                f"The loaded {trainer_name} network has no attribute '{attr_name}'."
            )
        target = getattr(base, attr_name)
        if attr_index is not None:
            if len(target) <= attr_index:
                raise RuntimeError(
                    f"Expected at least {attr_index + 1} elements in '{attr_name}', got {len(target)}"
                )
            target = target[attr_index]
        modules.append((layer_key, target))
    return modules


def collect_stage_outputs(trainer_name: str, network: torch.nn.Module, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    captured: Dict[str, torch.Tensor] = {}
    handles = []
    for name, module in get_encoder_modules(trainer_name, network):
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, layer_name=name: captured.__setitem__(
                    layer_name, output[0] if isinstance(output, (tuple, list)) else output
                )
            )
        )

    try:
        with torch.no_grad():
            _ = network(x)
    finally:
        for handle in handles:
            handle.remove()

    missing = [name for name in ("encoder_stage_0", "encoder_stage_1") if name not in captured]
    if missing:
        raise RuntimeError(f"Failed to capture encoder outputs: {missing}")
    captured["input"] = x.detach()
    return captured


def compute_power_spectrum(volume: torch.Tensor, remove_dc: bool) -> np.ndarray:
    if volume.ndim != 5:
        raise RuntimeError(f"Expected 5D tensor (B,C,D,H,W), got {tuple(volume.shape)}")
    work = volume.detach().float()
    if remove_dc:
        work = work - work.mean(dim=(-3, -2, -1), keepdim=True)
    fft = torch.fft.fftn(work, dim=(-3, -2, -1))
    power = torch.abs(torch.fft.fftshift(fft, dim=(-3, -2, -1))) ** 2
    return power.mean(dim=(0, 1)).cpu().numpy().astype(np.float64)


def get_normalized_radius(shape: Tuple[int, int, int]) -> np.ndarray:
    cached = RADIAL_CACHE.get(shape)
    if cached is not None:
        return cached

    axes = [np.fft.fftshift(np.fft.fftfreq(length)) for length in shape]
    zz, yy, xx = np.meshgrid(*axes, indexing="ij")
    radius = np.sqrt(zz ** 2 + yy ** 2 + xx ** 2)
    max_radius = float(radius.max())
    if max_radius > 0:
        radius = radius / max_radius
    RADIAL_CACHE[shape] = radius.astype(np.float32)
    return RADIAL_CACHE[shape]


def compute_radial_profile(spectrum: np.ndarray, num_bins: int) -> np.ndarray:
    radius = get_normalized_radius(tuple(int(v) for v in spectrum.shape))
    bins = np.linspace(0.0, 1.0, num_bins + 1, dtype=np.float32)
    flat_radius = radius.reshape(-1)
    flat_spec = spectrum.reshape(-1)
    bin_ids = np.clip(np.digitize(flat_radius, bins, right=False) - 1, 0, num_bins - 1)
    weighted = np.bincount(bin_ids, weights=flat_spec, minlength=num_bins)
    counts = np.bincount(bin_ids, minlength=num_bins).astype(np.float64)
    return weighted / np.maximum(counts, 1.0)


def compute_band_energy_ratios(spectrum: np.ndarray) -> Dict[str, float]:
    radius = get_normalized_radius(tuple(int(v) for v in spectrum.shape))
    total = float(spectrum.sum())
    if total <= 0:
        return {"low": 0.0, "mid": 0.0, "high": 0.0}

    low = float(spectrum[radius < 0.15].sum()) / total
    mid = float(spectrum[(radius >= 0.15) & (radius < 0.5)].sum()) / total
    high = float(spectrum[radius >= 0.5].sum()) / total
    return {"low": low, "mid": mid, "high": high}


def resample_for_average(spectrum: np.ndarray, grid_size: int) -> np.ndarray:
    log_spectrum = np.log1p(spectrum.astype(np.float32))
    if float(log_spectrum.max()) > 0:
        log_spectrum = log_spectrum / float(log_spectrum.max())
    tensor = torch.from_numpy(log_spectrum)[None, None]
    resized = F.interpolate(
        tensor,
        size=(grid_size, grid_size, grid_size),
        mode="trilinear",
        align_corners=False,
    )
    return resized[0, 0].cpu().numpy().astype(np.float64)


def central_slices(volume: np.ndarray) -> Dict[str, np.ndarray]:
    dz, dy, dx = (dim // 2 for dim in volume.shape)
    return {
        "axial": volume[dz],
        "coronal": volume[:, dy, :],
        "sagittal": volume[:, :, dx],
    }


class LayerAccumulator:
    def __init__(self, num_bins: int, grid_size: int):
        self.num_bins = num_bins
        self.grid_size = grid_size
        self.case_count = 0
        self.raw_power_mean_sum = 0.0
        self.raw_power_mean_sq_sum = 0.0
        self.raw_power_std_sum = 0.0
        self.total_power_sum = 0.0
        self.total_power_sq_sum = 0.0
        self.radial_sum = np.zeros(num_bins, dtype=np.float64)
        self.band_sum = {"low": 0.0, "mid": 0.0, "high": 0.0}
        self.resized_spectrum_sum = np.zeros((grid_size, grid_size, grid_size), dtype=np.float64)
        self.shape_counter: Counter[str] = Counter()
        self.channel_counter: Counter[int] = Counter()

    def update(self, tensor: torch.Tensor, num_channels: int, remove_dc: bool) -> Dict[str, float]:
        spectrum = compute_power_spectrum(tensor, remove_dc=remove_dc)
        raw_power_mean = float(spectrum.mean())
        raw_power_std = float(spectrum.std())
        total_power = float(spectrum.sum())

        normalized = spectrum / max(total_power, 1e-12)
        radial_profile = compute_radial_profile(normalized, self.num_bins)
        band_ratios = compute_band_energy_ratios(normalized)

        self.case_count += 1
        self.raw_power_mean_sum += raw_power_mean
        self.raw_power_mean_sq_sum += raw_power_mean ** 2
        self.raw_power_std_sum += raw_power_std
        self.total_power_sum += total_power
        self.total_power_sq_sum += total_power ** 2
        self.radial_sum += radial_profile
        for key, value in band_ratios.items():
            self.band_sum[key] += float(value)
        self.resized_spectrum_sum += resample_for_average(spectrum, self.grid_size)
        self.shape_counter["x".join(str(int(v)) for v in spectrum.shape)] += 1
        self.channel_counter[int(num_channels)] += 1

        return {
            "raw_power_mean": raw_power_mean,
            "raw_power_std": raw_power_std,
            "total_power": total_power,
            "band_low": band_ratios["low"],
            "band_mid": band_ratios["mid"],
            "band_high": band_ratios["high"],
            "depth": int(spectrum.shape[0]),
            "height": int(spectrum.shape[1]),
            "width": int(spectrum.shape[2]),
        }

    def summary(self) -> Dict[str, object]:
        if self.case_count == 0:
            raise RuntimeError("Cannot summarize an empty accumulator.")

        mean_profile = self.radial_sum / self.case_count
        raw_power_mean = self.raw_power_mean_sum / self.case_count
        raw_power_var = max(self.raw_power_mean_sq_sum / self.case_count - raw_power_mean ** 2, 0.0)
        total_power_mean = self.total_power_sum / self.case_count
        total_power_var = max(self.total_power_sq_sum / self.case_count - total_power_mean ** 2, 0.0)

        return {
            "num_cases": self.case_count,
            "raw_power_mean": raw_power_mean,
            "raw_power_mean_std_over_cases": raw_power_var ** 0.5,
            "raw_power_std_mean_over_cases": self.raw_power_std_sum / self.case_count,
            "total_power_mean": total_power_mean,
            "total_power_std_over_cases": total_power_var ** 0.5,
            "band_energy_ratio_mean": {
                key: value / self.case_count for key, value in self.band_sum.items()
            },
            "most_common_spectrum_shape": self.shape_counter.most_common(3),
            "most_common_num_channels": self.channel_counter.most_common(3),
            "mean_radial_profile": mean_profile.tolist(),
            "mean_resized_spectrum": (
                self.resized_spectrum_sum / self.case_count
            ).astype(np.float32),
        }


def plot_mean_spectrum_slices(
    layer_summaries: Dict[str, Dict[str, object]],
    layer_titles: Dict[str, str],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, len(LAYER_ORDER), figsize=(5 * len(LAYER_ORDER), 12))
    for col, layer_name in enumerate(LAYER_ORDER):
        mean_spectrum = layer_summaries[layer_name]["mean_resized_spectrum"]
        slices = central_slices(mean_spectrum)
        high_ratio = float(layer_summaries[layer_name]["band_energy_ratio_mean"]["high"])
        for row, axis_name in enumerate(("axial", "coronal", "sagittal")):
            ax = axes[row, col]
            ax.imshow(slices[axis_name], cmap="magma", interpolation="nearest")
            if row == 0:
                ax.set_title(
                    f"{layer_titles[layer_name]}\n{axis_name} | High={high_ratio:.4f} ({high_ratio * 100:.2f}%)"
                )
            else:
                ax.set_title(f"{axis_name} | High={high_ratio:.4f} ({high_ratio * 100:.2f}%)")
            ax.axis("off")
    fig.suptitle("Mean Log-Spectrum Slices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_radial_profiles(
    layer_summaries: Dict[str, Dict[str, object]],
    layer_titles: Dict[str, str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    radius = np.linspace(0.0, 1.0, len(next(iter(layer_summaries.values()))["mean_radial_profile"]))
    for layer_name in LAYER_ORDER:
        profile = np.asarray(layer_summaries[layer_name]["mean_radial_profile"], dtype=np.float32)
        high_ratio = float(layer_summaries[layer_name]["band_energy_ratio_mean"]["high"])
        ax.plot(
            radius,
            profile,
            linewidth=2.0,
            label=f"{layer_titles[layer_name]} | High={high_ratio:.4f} ({high_ratio * 100:.2f}%)",
        )
    ax.set_xlabel("Normalized Radius")
    ax.set_ylabel("Mean Normalized Spectral Energy")
    ax.set_title("Radial Frequency Profiles")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_band_energy_ratios(
    layer_summaries: Dict[str, Dict[str, object]],
    layer_titles: Dict[str, str],
    output_path: Path,
) -> None:
    labels = ["low", "mid", "high"]
    x = np.arange(len(labels))
    width = 0.22

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for idx, layer_name in enumerate(LAYER_ORDER):
        ratios = layer_summaries[layer_name]["band_energy_ratio_mean"]
        values = [ratios[label] for label in labels]
        bars = ax.bar(x + idx * width - width, values, width=width, label=layer_titles[layer_name])
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.01,
                f"{value:.4f}\n({value * 100:.2f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([label.title() for label in labels])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Energy Ratio")
    ax.set_title("Low / Mid / High Frequency Energy Ratios")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def write_case_metrics_csv(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "case_id",
        "layer",
        "channels",
        "depth",
        "height",
        "width",
        "raw_power_mean",
        "raw_power_std",
        "total_power",
        "band_low",
        "band_mid",
        "band_high",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_radial_profiles_csv(layer_summaries: Dict[str, Dict[str, object]], output_path: Path) -> None:
    num_bins = len(next(iter(layer_summaries.values()))["mean_radial_profile"])
    radius = np.linspace(0.0, 1.0, num_bins)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["radius", *LAYER_ORDER])
        for idx in range(num_bins):
            writer.writerow(
                [
                    f"{radius[idx]:.6f}",
                    *[
                        f"{layer_summaries[layer_name]['mean_radial_profile'][idx]:.10f}"
                        for layer_name in LAYER_ORDER
                    ],
                ]
            )


def write_high_frequency_summary_csv(
    layer_summaries: Dict[str, Dict[str, object]],
    layer_titles: Dict[str, str],
    output_path: Path,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer", "layer_title", "high_freq_ratio", "high_freq_percent"])
        for layer_name in LAYER_ORDER:
            high_ratio = float(layer_summaries[layer_name]["band_energy_ratio_mean"]["high"])
            writer.writerow([layer_name, layer_titles[layer_name], f"{high_ratio:.10f}", f"{high_ratio * 100:.4f}"])


def print_run_header(args: argparse.Namespace, checkpoint_path: Path, case_files: Iterable[Path]) -> None:
    print("=" * 72)
    print(f"{args.trainer} Encoder Frequency Spectrum Analysis")
    print("=" * 72)
    print(f"Fold dir:           {checkpoint_path.parent}")
    print(f"Checkpoint:         {checkpoint_path}")
    print(f"Data dir:           {Path(args.data_dir)}")
    resolved_output_dir = Path(args.output_dir) if args.output_dir is not None else MODEL_CONFIGS[args.trainer]["default_output_dir"]
    print(f"Output dir:         {resolved_output_dir}")
    print(f"Device:             {args.device}")
    print(f"Remove DC:          {args.remove_dc}")
    print(f"Radial bins:        {args.num_bins}")
    print(f"Spectrum grid size: {args.spectrum_grid_size}")
    print(f"Cases selected:     {len(list(case_files))}")
    print("=" * 72)


def main() -> None:
    args = parse_args()
    config = MODEL_CONFIGS[args.trainer]
    fold_dir = Path(args.fold_dir) if args.fold_dir is not None else config["default_fold_dir"]
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir is not None else config["default_output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_titles = get_layer_titles(args.trainer)
    fold = args.fold if args.fold is not None else infer_fold_from_dir(fold_dir)

    checkpoint_path = resolve_checkpoint_path(fold_dir, args.checkpoint_name, args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    case_files = find_case_files(data_dir, args.case_id, args.max_cases)
    print_run_header(args, checkpoint_path, case_files)

    trainer = restore_trainer(
        trainer_name=args.trainer,
        fold_dir=fold_dir,
        checkpoint_path=checkpoint_path,
        plans_file=args.plans_file,
        data_dir=data_dir,
        fold=fold,
    )
    device = torch.device(args.device)
    trainer.network.to(device)
    trainer.network.eval()

    expected_in = getattr(trainer, "num_input_channels", None)
    if expected_in is None:
        expected_in = getattr(unwrap_module(trainer.network), "input_channels", None)
    if expected_in is None:
        expected_in = getattr(unwrap_module(trainer.network), "num_input_channels", None)

    accumulators = {
        layer_name: LayerAccumulator(args.num_bins, args.spectrum_grid_size)
        for layer_name in LAYER_ORDER
    }
    case_metric_rows: List[Dict[str, object]] = []

    for idx, case_file in enumerate(case_files, start=1):
        data = load_case_npz(case_file, expected_in=expected_in)
        x = torch.from_numpy(data).to(device=device, dtype=torch.float32)
        outputs = collect_stage_outputs(args.trainer, trainer.network, x)

        print(f"[{idx:03d}/{len(case_files):03d}] {case_file.stem}")
        for layer_name in LAYER_ORDER:
            tensor = outputs[layer_name]
            metrics = accumulators[layer_name].update(
                tensor=tensor,
                num_channels=int(tensor.shape[1]),
                remove_dc=args.remove_dc,
            )
            case_metric_rows.append(
                {
                    "case_id": case_file.stem,
                    "layer": layer_name,
                    "channels": int(tensor.shape[1]),
                    **metrics,
                }
            )

        del outputs
        del x
        if device.type == "cuda":
            torch.cuda.empty_cache()

    layer_summaries: Dict[str, Dict[str, object]] = {}
    summary_json: Dict[str, object] = {
        "config": {
            "fold_dir": str(fold_dir),
            "checkpoint": str(checkpoint_path),
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "trainer": args.trainer,
            "fold": fold,
            "num_cases": len(case_files),
            "num_bins": args.num_bins,
            "spectrum_grid_size": args.spectrum_grid_size,
            "remove_dc": args.remove_dc,
            "device": args.device,
        },
        "layers": {},
    }

    for layer_name in LAYER_ORDER:
        layer_summary = accumulators[layer_name].summary()
        layer_summaries[layer_name] = layer_summary
        summary_json["layers"][layer_name] = {
            key: value
            for key, value in layer_summary.items()
            if key != "mean_resized_spectrum"
        }

    write_case_metrics_csv(case_metric_rows, output_dir / "case_metrics.csv")
    write_radial_profiles_csv(layer_summaries, output_dir / "radial_profiles.csv")
    write_high_frequency_summary_csv(layer_summaries, layer_titles, output_dir / "high_frequency_summary.csv")
    plot_mean_spectrum_slices(layer_summaries, layer_titles, output_dir / "mean_spectrum_slices.png")
    plot_radial_profiles(layer_summaries, layer_titles, output_dir / "radial_profiles.png")
    plot_band_energy_ratios(layer_summaries, layer_titles, output_dir / "band_energy_ratios.png")

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary_json, handle, indent=2, ensure_ascii=False)

    print("\nHigh-frequency ratios:")
    for layer_name in LAYER_ORDER:
        high_ratio = float(layer_summaries[layer_name]["band_energy_ratio_mean"]["high"])
        print(f"  - {layer_titles[layer_name]}: {high_ratio:.6f} ({high_ratio * 100:.2f}%)")

    print("\nSaved outputs:")
    for name in (
        "summary.json",
        "case_metrics.csv",
        "radial_profiles.csv",
        "high_frequency_summary.csv",
        "mean_spectrum_slices.png",
        "radial_profiles.png",
        "band_energy_ratios.png",
    ):
        print(f"  - {output_dir / name}")


if __name__ == "__main__":
    main()
