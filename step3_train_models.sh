# preprocess, copy Task530_EsoTJ_30pct to nnUNet_raw_data
# install
pip install -e .

# make dataset
python ./nnunet/experiment_planning/nnUNet_plan_and_preprocess.py -t 602 --verify_dataset_integrity


export PYTHONPATH=$(pwd):$PYTHONPATH
# UNETR 0.81
CUDA_VISIBLE_DEVICES=0 nohup python ./nnunet/run/run_training.py 3d_fullres UNETRTrainer Task530_EsoTJ_30pct 1 > ./log/UNETRTrainer_530_Fold1.log 2>&1 &

# 111UNet3D 0.84
CUDA_VISIBLE_DEVICES=0 nohup python ./nnunet/run/run_training.py 3d_fullres UNet3DTrainer Task530_EsoTJ_30pct 1 > ./log/UNet3DTrainer_530_Fold1.log 2>&1 &

# 11nnUNetV2 0.849
CUDA_VISIBLE_DEVICES=0 nohup python ./nnunet/run/run_training.py 3d_fullres nnUNetTrainerV2 Task530_EsoTJ_30pct 1 > ./log/nnUNetV2_530_Fold2.log 2>&1 &

# 423 BANet 0.857
CUDA_VISIBLE_DEVICES=0 nohup python ./nnunet/run/run_training.py 3d_fullres BANetTrainerV2 Task530_EsoTJ_30pct 1 > ./log/BANet_530_Fold1.log 2>&1 &

# MedNeXtTrainerV2 0.8558
CUDA_VISIBLE_DEVICES=0 nohup python ./nnunet/run/run_training.py 3d_fullres MedNeXtTrainerV2 Task530_EsoTJ_30pct 2 > ./log/MedNeXtTrainerV2_530_Fold2.log 2>&1 &

# phtrans 0
CUDA_VISIBLE_DEVICES=0 nohup python ./nnunet/run/run_training.py 3d_fullres PHTransTrainerV2 Task505_EsoTJ_10pct 0 > ./log/PHTransTrainerV2_505_Fold0.log 2>&1 &


# BAPCSwinNextV4 0
CUDA_VISIBLE_DEVICES=1 nohup python ./nnunet/run/run_training.py 3d_fullres BAPCSwinNextTrainerV4 Task530_EsoTJ_30pct 0 > ./log/BAPCSwinNextTrainerV4_530_Fold0.log 2>&1 &

# BGHNetV4 0.844
CUDA_VISIBLE_DEVICES=1 nohup python ./nnunet/run/run_training.py 3d_fullres BGHNetV4Trainer 601 0 > ./log/BGHNetV4Trainer_601_Fold0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python ./nnunet/run/run_training.py 3d_fullres BGHNetV4Trainer 500 0 > ./log/BGHNetV4Trainer_530_Fold0.log 2>&1 &
export CUDA_VISIBLE_DEVICES=6 && python ./nnunet/run/run_training.py 3d_fullres BGHNetV4Trainer 500 0


# eval
CUDA_VISIBLE_DEVICES=1 nohup python ./step4_pred_val_results_and_eval.py  > ./log/step4_pred_val_results_and_eval.log 2>&1 &
