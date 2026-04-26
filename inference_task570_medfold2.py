#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对 Task570_EsoTJ83 使用 MedNeXtTrainerV2 的 fold_2 模型进行预测，
调用 nnunet.inference.predict.predict_from_folder，
并将结果保存到：
  /home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task570_EsoTJ83/medfold2/preds
"""

from pathlib import Path

from nnunet.inference.predict import predict_from_folder


def main():
    input_folder = Path("/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_raw_data/Task570_EsoTJ83/imagesTr")
    output_folder = Path(
        "/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task570_EsoTJ83/medfold2/preds",
    )
    model_dir = Path(
        "./ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/MedNeXtTrainerV2__nnUNetPlansv2.1",
    )

    folds = (2,)
    checkpoint_name = "model_final_checkpoint"

    num_threads_preprocessing = 2
    num_threads_nifti_save = 2
    step_size = 0.5
    tta = True
    mixed_precision = True

    if not input_folder.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    output_folder.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Start inference for Task570_EsoTJ83 with MedNeXtTrainerV2 fold_2")
    print(f"Model directory: {model_dir}")
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Folds: {folds}, checkpoint: {checkpoint_name}")
    print("=" * 80)

    predict_from_folder(
        model=str(model_dir),
        input_folder=str(input_folder),
        output_folder=str(output_folder),
        folds=folds,
        save_npz=False,
        num_threads_preprocessing=num_threads_preprocessing,
        num_threads_nifti_save=num_threads_nifti_save,
        lowres_segmentations=None,
        part_id=0,
        num_parts=1,
        tta=tta,
        overwrite_existing=False,
        mode="normal",
        overwrite_all_in_gpu=None,
        mixed_precision=mixed_precision,
        step_size=step_size,
        checkpoint_name=checkpoint_name,
    )

    print(f"Inference finished. Predictions saved to: {output_folder}")


if __name__ == "__main__":
    main()
