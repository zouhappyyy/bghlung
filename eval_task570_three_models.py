import os
import nibabel as nib
import numpy as np

from nnunet_mednext.evaluation.evaluator import Evaluator
from nnunet_mednext.evaluation.metrics import dice


GT_DIR = "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_raw_data/Task570_EsoTJ83/labelsTr"
PRED_ROOT = "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task570_EsoTJ83"

MODELS = {
    "Double_CCA_UPSam_fd_RWKV": os.path.join(PRED_ROOT, "Double_CCA_UPSam_fd_RWKV", "preds"),
    "MedNeXt_S_kernel3": os.path.join(PRED_ROOT, "MedNeXt_S_kernel3", "preds"),
    "Double_CCA_UPSam_fd_loss_RWKV": os.path.join(PRED_ROOT, "Double_CCA_UPSam_fd_loss_RWKV", "preds"),
}


def list_case_ids(gt_dir):
    ids = []
    for f in os.listdir(gt_dir):
        if f.endswith(".nii.gz") or f.endswith(".nii"):
            cid = f.split(".")[0]
            ids.append(cid)
    ids.sort()
    return ids


def load_nii_int(path):
    img = nib.load(path)
    arr = img.get_fdata()
    return arr.astype(np.int16)


def main():
    case_ids = list_case_ids(GT_DIR)
    print(f"Found {len(case_ids)} GT cases in {GT_DIR}")

    model_dice = {m: [] for m in MODELS.keys()}

    for cid in case_ids:
        gt_path = os.path.join(GT_DIR, f"{cid}.nii.gz")
        if not os.path.isfile(gt_path):
            gt_path = os.path.join(GT_DIR, f"{cid}.nii")
            if not os.path.isfile(gt_path):
                print(f"[WARN] GT not found for {cid}, skip")
                continue

        gt = load_nii_int(gt_path)

        print(f"\nCase {cid}:")
        for model_name, pred_dir in MODELS.items():
            pred_path = os.path.join(pred_dir, f"{cid}.nii.gz")
            if not os.path.isfile(pred_path):
                pred_path = os.path.join(pred_dir, f"{cid}.nii")
            if not os.path.isfile(pred_path):
                print(f"  [WARN] {model_name}: no prediction for {cid}")
                continue

            pred = load_nii_int(pred_path)

            if pred.shape != gt.shape:
                print(
                    f"  [WARN] {model_name}: shape mismatch for {cid}: pred {pred.shape}, gt {gt.shape} (no auto-fix)",
                )

            evaluator = Evaluator(test=pred, reference=gt, metrics=[dice])
            evaluator.evaluate()
            res_dict = evaluator.to_dict()
            dice_fg = res_dict.get("1", {}).get("dice", np.nan)
            model_dice[model_name].append(dice_fg)
            print(f"  {model_name}: Dice = {dice_fg:.4f}")

    print("\n=== Mean Dice over all cases ===")
    for model_name, vals in model_dice.items():
        if not vals:
            print(f"  {model_name}: no valid cases")
            continue
        mean_d = float(np.nanmean(vals))
        print(f"  {model_name}: mean Dice = {mean_d:.4f} (N={len(vals)})")


if __name__ == "__main__":
    main()

