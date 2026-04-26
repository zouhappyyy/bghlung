#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对 ESO 测试集 Task570_EsoTJ83 使用多个 nnUNet 模型进行预测，
调用 nnunet.inference.predict.predict_from_folder，
将结果保存到：
  /home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task570_EsoTJ83/<TrainerName>/preds
"""

import os
from pathlib import Path

from nnunet.inference.predict import predict_from_folder


def main():
    # ========== 路径配置 ==========
    # 测试集 raw imagesTr 路径
    task570_imagesTr = Path("/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_raw_data/Task602_ls/imagesTr")

    # 预测结果根目录
    pred_root = Path("/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_predictions/Task602_ls")

    # 你训练好的模型根目录（根据你实际情况修改）
    # 假设结构类似：
    #   ./ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/UNet3DTrainer__nnUNetPlansv2.1/
    #   ./ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2__nnUNetPlansv2.1/
    #   ./ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/BANetTrainerV2__nnUNetPlansv2.1/
    #   ./ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/BGHNetV4Trainer__nnUNetPlansv2.1/
    model_root = Path("./ckpt/nnUNet/3d_fullres/Task601_lctsc40")

    # Trainer 名称到模型目录的映射（按你的实际目录来改）
    model_dir_map = {
        # "UNet3DTrainer": model_root / "UNet3DTrainer__nnUNetPlansv2.1",
        # "nnUNetTrainerV2": model_root / "nnUNetTrainerV2__nnUNetPlansv2.1",
        # "BANetTrainerV2": model_root / "BANetTrainerV2__nnUNetPlansv2.1",
        "BGHNetV4Trainer": model_root / "BGHNetV4Trainer__nnUNetPlansv2.1",
    }

    # 要使用的 folds（你之前训练是 fold=1，可以改成 ['all'] 看实际情况）
    folds = (0,)  # 或者 ('all',)

    # checkpoint 名称（不带扩展名，和你训练时保持一致）
    # 如果你的文件名是 model_final_checkpoint.model，则这里写 'model_final_checkpoint'
    checkpoint_name = "model_final_checkpoint"

    # 一些 nnUNet 推理参数
    num_threads_preprocessing = 2
    num_threads_nifti_save = 2
    step_size = 0.5
    tta = True  # 测试时是否启用 test-time augmentation
    mixed_precision = True

    # ========== 开始预测 ==========
    if not task570_imagesTr.is_dir():
        raise FileNotFoundError(f"Task570 imagesTr not found: {task570_imagesTr}")

    print(f"Found Task570 imagesTr folder: {task570_imagesTr}")

    for trainer_name, model_dir in model_dir_map.items():
        if not model_dir.is_dir():
            print(f"[警告] 模型目录不存在: {model_dir}，跳过 {trainer_name}")
            continue

        output_folder = pred_root / trainer_name / "preds"
        output_folder.mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print(f"开始预测: Trainer = {trainer_name}")
        print(f"  模型目录: {model_dir}")
        print(f"  输入目录: {task570_imagesTr}")
        print(f"  输出目录: {output_folder}")
        print(f"  folds: {folds}, checkpoint: {checkpoint_name}")
        print("=" * 80)

        # 调用 nnunet.inference.predict.predict_from_folder
        predict_from_folder(
            model=str(model_dir),
            input_folder=str(task570_imagesTr),
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

        print(f"完成 {trainer_name} 在 Task570 上的预测，结果保存在: {output_folder}")

    print("所有模型预测结束。")


if __name__ == "__main__":
    # 建议在调用前已激活 nnUNet 训练时同一个环境
    # 例如： conda activate zl_nnunetv1
    main()
