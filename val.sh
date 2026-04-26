#!/usr/bin/env bash


cd "$(dirname "$0")"  # 进入脚本所在目录，也就是 BGHNet_lung_500

#python ./nnunet/run/run_training.py 3d_fullres UNet3DTrainer     Task530_EsoTJ_30pct 1 -val
#python ./nnunet/run/run_training.py 3d_fullres nnUNetTrainerV2  Task530_EsoTJ_30pct 1 -val
#python ./nnunet/run/run_training.py 3d_fullres BANetTrainerV2   Task530_EsoTJ_30pct 1 -val
#python ./nnunet/run/run_training.py 3d_fullres BGHNetV4Trainer  Task530_EsoTJ_30pct 1 -val


python extract_metrics_from_summary.py \
  --summary ./ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/BANetTrainerV2__nnUNetPlansv2.1/fold_1/validation_raw_postprocessed/summary.json \
python extract_metrics_from_summary.py \
  --summary ./ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/BGHNetV4Trainer__nnUNetPlansv2.1/fold_1/validation_raw_postprocessed/summary.json \
python extract_metrics_from_summary.py \
  --summary ./ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/validation_raw_postprocessed/summary.json \
python extract_metrics_from_summary.py \
  --summary ./ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/UNet3DTrainer__nnUNetPlansv2.1/fold_1/validation_raw_postprocessed/summary.json \
