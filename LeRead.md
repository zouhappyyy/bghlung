# 1. 安装环境
先创建一个新的conda环境，安装pytorch，依次安装各种包
运行
```
pip install -e .
```
将所有包集成为nnUNet环境
实验环境为nnUNetv1_py3.10
当出现numpy包不兼容问题是，将np.bool替换成np.bool_，保证numpy环境在1.20以上，避免产生其他包不兼容的情况

# 2. 整理数据集
```
python step2_make_nnunet_set_supervision.py
```
准备train和test数据集，生成nnUNet_raw_data数据集，test不参与五折交叉验证

# 3. 训练模型
```bash
CUDA_VISIBLE_DEVICES=1 nohup python ./nnunet/run/run_training.py 3d_fullres BGHNetV4Trainer Task500_LungTumor500 0 > ./BGHNetV4Trainer_Fold0.log 2>&1 &
```
nohup 、、、 > ./BGHNetV4Trainer_Fold0.log 2>&1 & 
2>&1表示标准错误输出重定向到标准输出，即日志也记载错误信息
BGHNetV4Trainer是模型名，会找到模型并返回，路径为 nnunet/training/network_training/BGHNetV4Trainer.py

训练入口为 ./nnunet/run/run_training.py
之前一直停在pin_memory上，怀疑是多线程的包出错了。重新安装了个没有opencv的环境，成功！

# 4. 评估模型
```bash
python step4_pred_val_results_and_eval.py
```
test数据集的预测结果在pred_results/
分割结果可视化在vis_imgs/
出现评估指标全为0的情况，是numpy的bool弃用原因，已更改为bool_

# 5. 预测没有Mask的图像
```bash
python step5_20240401_pred_test_lung_data.py
python step5_20240511_pred_test_lung_data.py
```

python tools/visualize_bghnetv4_all_stages.py \
  --task Task530_EsoTJ_30pct \
  --fold 1 \
  --checkpoint ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/BGHNetV4Trainer__nnUNetPlansv2.1/fold_1/model_final_checkpoint.model \
  --axis axial \
  --output-dir feature_vis_output

python tools/visualize_bghnetv4_all_stages.py \
  --task Task530_EsoTJ_30pct \
  --fold 1 \
  --validation-raw-dir ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/BGHNetV4Trainer__nnUNetPlansv2.1/fold_1/validation_raw \
  --all-cases \
  --axis axial \
  --display-crop-size 64 \
  --output-dir feature_vis_output


python tools/visualize_banetv2_all_stages.py \
  --task Task530_EsoTJ_30pct \
  --fold 1 \
  --validation-raw-dir ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/BANetTrainerV2__nnUNetPlansv2.1/fold_1/validation_raw \
  --axis axial \
  --display-crop-size 64 \
  --output-dir feature_vis_output