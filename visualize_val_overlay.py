import os
import argparse
from typing import List, Optional, Tuple, Dict

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from nnunet_mednext.utilities.overlay_plots import generate_overlay
from visualize_fd_edge_and_ds import _extract_case_data


def _load_pred_nifti(pred_path: str) -> np.ndarray:
    """Load prediction nifti and return array in [Z, Y, X] order.

    nnUNet usually stores NIfTI as [X, Y, Z]. We transpose to [Z, Y, X]
    to match the convention used in `_extract_case_data` for GT.
    """
    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")
    img = nib.load(pred_path)
    arr = img.get_fdata()
    if arr.ndim != 3:
        raise RuntimeError(f"Expected 3D prediction, got shape {arr.shape} from {pred_path}")
    # [X, Y, Z] -> [Z, Y, X]
    arr = np.transpose(arr, (2, 1, 0))
    return arr.astype(np.int16)


def _match_shapes(image: np.ndarray, seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Center-crop seg to match image spatial size if needed.

    image: [C, D, H, W]
    seg: [D, H, W]
    """
    d_i, h_i, w_i = image.shape[1:]
    d_s, h_s, w_s = seg.shape
    d = min(d_i, d_s)
    h = min(h_i, h_s)
    w = min(w_i, w_s)
    image_c = image[:, :d, :h, :w]
    seg_c = seg[:d, :h, :w]
    return image_c, seg_c


def visualize_case_overlay(
    data_root: str,
    dataset_directory: str,
    pred_folder: str,
    case_id: str,
    output_dir: str,
    alpha: float = 0.6,
    mode: str = "slice",
    axis: str = "z",
    slices: Optional[List[int]] = None,
    name_prefix: str = "",
    save_gt: bool = False,
) -> List[Dict[str, object]]:
    """Visualize overlay of original image, prediction, and optional GT for a single case.

    data_root: directory with preprocessed stage0 .npy volumes
    dataset_directory: nnUNet dataset directory (contains gt_segmentations)
    pred_folder: folder containing prediction nifti files (validation_raw or validation_raw_postprocessed)
    case_id: case identifier (e.g. ESO_TJ_60011222468)
    output_dir: where to save pngs
    mode: "slice" (single or multiple 2D slices) or "mip" (3D MIP views)
    axis: for slice mode, one of {"z", "y", "x"}
    slices: optional list of slice indices; if None, use middle slice
    name_prefix: optional string prefix to prepend in filename (e.g. fold-modelname)
    save_gt: if True, also export GT overlay images alongside predictions.

    Returns a list of dicts with information about generated images.
    """
    os.makedirs(output_dir, exist_ok=True)
    results: List[Dict[str, object]] = []

    image, gt = _extract_case_data(data_root, case_id, dataset_directory)
    gt = gt[0]

    pred_path = None
    for ext in (".nii.gz", ".nii"):
        cand = os.path.join(pred_folder, f"{case_id}{ext}")
        if os.path.isfile(cand):
            pred_path = cand
            break
    if pred_path is None:
        raise FileNotFoundError(f"Could not find prediction for case {case_id} in {pred_folder}")

    pred = _load_pred_nifti(pred_path)

    image, pred = _match_shapes(image, pred)
    _, gt = _match_shapes(image, gt)

    _, D, H, W = image.shape

    if mode == "slice":
        if axis not in ("z", "y", "x"):
            raise ValueError("axis must be one of 'z', 'y', 'x'")
        if slices is None or len(slices) == 0:
            if axis == "z":
                slices = [D // 2]
            elif axis == "y":
                slices = [H // 2]
            else:
                slices = [W // 2]

        for s in slices:
            # --- build slices for pred & gt ---
            if axis == "z":
                if not (0 <= s < D):
                    continue
                img_slice = image[0, s]
                pred_slice = pred[s]
                gt_slice = gt[s]
            elif axis == "y":
                if not (0 <= s < H):
                    continue
                img_slice = image[0, :, s, :]
                pred_slice = pred[:, s, :]
                gt_slice = gt[:, s, :]
            else:  # x
                if not (0 <= s < W):
                    continue
                img_slice = image[0, :, :, s]
                pred_slice = pred[:, :, s]
                gt_slice = gt[:, :, s]

            # --- prediction overlay ---
            overlay_pred = generate_overlay(img_slice, pred_slice, overlay_intensity=alpha)

            base_name_pred = f"{case_id}_axis-{axis}_slice-{s}_pred.png"
            if name_prefix:
                out_name_pred = f"{name_prefix}_{base_name_pred}"
            else:
                out_name_pred = base_name_pred

            out_path_pred = os.path.join(output_dir, out_name_pred)
            plt.imsave(out_path_pred, overlay_pred)
            print(f"Saved pred overlay to: {out_path_pred}")
            results.append(
                {
                    "case_id": case_id,
                    "axis": axis,
                    "slice": s,
                    "path": out_path_pred,
                    "pred_folder": pred_folder,
                    "type": "pred",
                }
            )

            # --- GT overlay ---
            if save_gt:
                overlay_gt = generate_overlay(img_slice, gt_slice, overlay_intensity=alpha)

                base_name_gt = f"{case_id}_axis-{axis}_slice-{s}_gt.png"
                if name_prefix:
                    out_name_gt = f"{name_prefix}_{base_name_gt}"
                else:
                    out_name_gt = base_name_gt

                out_path_gt = os.path.join(output_dir, out_name_gt)
                plt.imsave(out_path_gt, overlay_gt)
                print(f"Saved GT overlay to: {out_path_gt}")
                results.append(
                    {
                        "case_id": case_id,
                        "axis": axis,
                        "slice": s,
                        "path": out_path_gt,
                        "pred_folder": pred_folder,
                        "type": "gt",
                    }
                )

    elif mode == "mip":
        # Simple 3-view MIP overlay
        vol = image[0]
        # Axial (Z): max over depth
        axial_img = vol.max(axis=0)
        axial_pred = (pred > 0).max(axis=0).astype(np.int16)
        axial_overlay = generate_overlay(axial_img, axial_pred, overlay_intensity=alpha)

        # Coronal (Y): max over height
        cor_img = vol.max(axis=1)
        cor_pred = (pred > 0).max(axis=1).astype(np.int16)
        cor_overlay = generate_overlay(cor_img, cor_pred, overlay_intensity=alpha)

        # Sagittal (X): max over width
        sag_img = vol.max(axis=2)
        sag_pred = (pred > 0).max(axis=2).astype(np.int16)
        sag_overlay = generate_overlay(sag_img, sag_pred, overlay_intensity=alpha)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(axial_overlay)
        axes[0].set_title("Axial MIP")
        axes[0].axis("off")

        axes[1].imshow(cor_overlay)
        axes[1].set_title("Coronal MIP")
        axes[1].axis("off")

        axes[2].imshow(sag_overlay)
        axes[2].set_title("Sagittal MIP")
        axes[2].axis("off")

        plt.tight_layout()
        base_name = f"{case_id}_mip.png"
        if name_prefix:
            out_name = f"{name_prefix}_{base_name}"
        else:
            out_name = base_name
        out_path = os.path.join(output_dir, out_name)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        print(f"Saved MIP overlays to: {out_path}")
        results.append(
            {
                "case_id": case_id,
                "axis": "mip",
                "slice": None,
                "path": out_path,
                "pred_folder": pred_folder,
            }
        )

    else:
        raise ValueError("mode must be 'slice' or 'mip'")

    return results


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize overlay of validation predictions and original images (single case or batch over models)",
    )
    subparsers = parser.add_subparsers(dest="command", required=False)

    # single-case mode (backwards compatible)
    p_single = subparsers.add_parser("single", help="Visualize one case")
    p_single.add_argument("--data_root", type=str, required=True, help="Folder with preprocessed stage0 .npy volumes")
    p_single.add_argument(
        "--dataset_directory",
        type=str,
        required=True,
        help="nnUNet dataset directory (contains gt_segmentations)",
    )
    p_single.add_argument(
        "--pred_folder",
        type=str,
        required=True,
        help="Folder with prediction NIfTI files (e.g. validation_raw_postprocessed)",
    )
    p_single.add_argument("--case_id", type=str, required=True, help="Case id to visualize (e.g. ESO_TJ_60011222468)")
    p_single.add_argument("--output_dir", type=str, required=True, help="Where to save output PNGs")
    p_single.add_argument("--alpha", type=float, default=0.6, help="Overlay intensity (alpha) for segmentation")
    p_single.add_argument(
        "--mode",
        type=str,
        choices=["slice", "mip"],
        default="slice",
        help="Visualization mode: slice or mip",
    )
    p_single.add_argument(
        "--axis",
        type=str,
        choices=["z", "y", "x"],
        default="z",
        help="Slice axis for slice mode",
    )
    p_single.add_argument(
        "--slices",
        type=int,
        nargs="*",
        default=None,
        help="Slice indices to visualize (if empty, use middle slice)",
    )

    # batch mode: multiple models, all cases, all three axes
    p_batch = subparsers.add_parser("batch", help="Batch visualize all cases for multiple models")
    p_batch.add_argument("--data_root", type=str, required=True, help="Folder with preprocessed stage0 .npy volumes")
    p_batch.add_argument(
        "--dataset_directory",
        type=str,
        required=True,
        help="nnUNet dataset directory (contains gt_segmentations)",
    )
    p_batch.add_argument(
        "--pred_folders",
        type=str,
        nargs="+",
        required=True,
        help=(
            "List of prediction folders (e.g. different models' validation_raw_postprocessed). "
            "Model name will be inferred from the folder path unless --model_names is given."
        ),
    )
    p_batch.add_argument(
        "--model_names",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional explicit model names aligned with pred_folders. "
            "If omitted, will take the trainer directory name from each pred_folder path."
        ),
    )
    p_batch.add_argument(
        "--fold", type=int, required=True, help="Fold index used in filename prefix (e.g. 0/1/2/3/4)",
    )
    p_batch.add_argument(
        "--output_dir", type=str, required=True, help="Base folder where visualization PNGs will be stored",
    )
    p_batch.add_argument(
        "--alpha", type=float, default=0.6, help="Overlay intensity (alpha) for segmentation",
    )
    p_batch.add_argument(
        "--slices",
        type=int,
        nargs="*",
        default=None,
        help="Slice indices to visualize. If omitted, middle slice along each axis is used.",
    )
    p_batch.add_argument("--save_gt", action="store_true", help="Also export GT overlays for each slice")

    # default: preserve old behavior if no subcommand is specified
    parser.add_argument("--data_root", type=str, help="[legacy] Folder with preprocessed stage0 .npy volumes")
    parser.add_argument(
        "--dataset_directory",
        type=str,
        help="[legacy] nnUNet dataset directory (contains gt_segmentations)",
    )
    parser.add_argument(
        "--pred_folder",
        type=str,
        help="[legacy] Folder with prediction NIfTI files (e.g. validation_raw_postprocessed)",
    )
    parser.add_argument("--case_id", type=str, help="[legacy] Case id to visualize (e.g. ESO_TJ_60011222468)")
    parser.add_argument("--output_dir", type=str, help="[legacy] Where to save output PNGs")
    parser.add_argument("--alpha", type=float, default=0.6, help="[legacy] Overlay intensity (alpha) for segmentation")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["slice", "mip"],
        default="slice",
        help="[legacy] Visualization mode: slice or mip",
    )
    parser.add_argument(
        "--axis",
        type=str,
        choices=["z", "y", "x"],
        default="z",
        help="[legacy] Slice axis for slice mode",
    )
    parser.add_argument(
        "--slices",
        type=int,
        nargs="*",
        default=None,
        help="[legacy] Slice indices to visualize (if empty, use middle slice)",
    )

    return parser


def _infer_model_name_from_pred_folder(pred_folder: str) -> str:
    """Infer a model name from the prediction folder path.

    For typical nnUNet layout, pred_folder is something like:
      .../TaskXXX/TrainerName__Plans/fold_1/validation_raw_postprocessed
    We take the TrainerName__Plans part as model name by default.
    """
    parts = os.path.normpath(pred_folder).split(os.sep)
    if len(parts) < 3:
        return os.path.basename(pred_folder)
    # ... fold_x / validation_raw_postprocessed / Trainer__Plans / Task...
    # assume trainer dir is two levels above pred_folder
    trainer_dir = parts[-3]
    return trainer_dir


def _collect_case_ids_from_pred_folder(pred_folder: str) -> List[str]:
    """List unique case_ids from NIfTI files in prediction folder."""
    if not os.path.isdir(pred_folder):
        raise FileNotFoundError(f"Prediction folder not found: {pred_folder}")
    ids = []
    for fn in os.listdir(pred_folder):
        if fn.endswith(".nii") or fn.endswith(".nii.gz"):
            case_id = fn.split(".")[0]
            if case_id not in ids:
                ids.append(case_id)
    ids.sort()
    return ids


def _run_batch(
    data_root: str,
    dataset_directory: str,
    pred_folders: List[str],
    model_names: Optional[List[str]],
    fold: int,
    output_dir: str,
    alpha: float,
    slices: Optional[List[int]],
    save_gt: bool,
) -> None:
    # normalize model names
    if model_names is not None and len(model_names) != len(pred_folders):
        raise ValueError("If model_names is provided, its length must match pred_folders")

    if model_names is None:
        model_names = [_infer_model_name_from_pred_folder(p) for p in pred_folders]

    os.makedirs(output_dir, exist_ok=True)

    for pred_folder, model_name in zip(pred_folders, model_names):
        case_ids = _collect_case_ids_from_pred_folder(pred_folder)
        if not case_ids:
            print(f"[WARN] No prediction files found in {pred_folder}, skip")
            continue

        print(f"[INFO] Processing model {model_name} (fold {fold}) with {len(case_ids)} cases from {pred_folder}")

        for case_id in case_ids:
            for axis in ("z", "y", "x"):
                try:
                    results = visualize_case_overlay(
                        data_root=data_root,
                        dataset_directory=dataset_directory,
                        pred_folder=pred_folder,
                        case_id=case_id,
                        output_dir=output_dir,
                        alpha=alpha,
                        mode="slice",
                        axis=axis,
                        slices=slices,
                        name_prefix=None,
                        save_gt=save_gt,
                    )
                    # rename files to pattern: fold-id-axis-sliceindex-modelname[-type]
                    for item in results:
                        slice_idx = item["slice"]
                        img_type = item.get("type", "pred")
                        old_path = item["path"]
                        ext = os.path.splitext(old_path)[1]
                        new_name = f"{fold}-{case_id}-{axis}-{slice_idx}-{model_name}-{img_type}{ext}"
                        new_path = os.path.join(output_dir, new_name)
                        os.replace(old_path, new_path)
                except Exception as e:
                    print(f"[ERROR] Failed on case {case_id}, axis {axis}, model {model_name}: {e}")


if __name__ == "__main__":
    parser = _build_argparser()
    args = parser.parse_args()

    # prioritize subcommands if given
    if getattr(args, "command", None) == "single":
        visualize_case_overlay(
            data_root=args.data_root,
            dataset_directory=args.dataset_directory,
            pred_folder=args.pred_folder,
            case_id=args.case_id,
            output_dir=args.output_dir,
            alpha=args.alpha,
            mode=args.mode,
            axis=args.axis,
            slices=args.slices,
        )
    elif getattr(args, "command", None) == "batch":
        _run_batch(
            data_root=args.data_root,
            dataset_directory=args.dataset_directory,
            pred_folders=args.pred_folders,
            model_names=args.model_names,
            fold=args.fold,
            output_dir=args.output_dir,
            alpha=args.alpha,
            slices=args.slices,
            save_gt=getattr(args, "save_gt", False),
        )
    else:
        # legacy behavior: single-case mode with top-level args
        if not (args.data_root and args.dataset_directory and args.pred_folder and args.case_id and args.output_dir):
            parser.error(
                "For legacy single-case mode, --data_root, --dataset_directory, "
                "--pred_folder, --case_id and --output_dir must be provided.",
            )
        visualize_case_overlay(
            data_root=args.data_root,
            dataset_directory=args.dataset_directory,
            pred_folder=args.pred_folder,
            case_id=args.case_id,
            output_dir=args.output_dir,
            alpha=args.alpha,
            mode=args.mode,
            axis=args.axis,
            slices=args.slices,
        )

