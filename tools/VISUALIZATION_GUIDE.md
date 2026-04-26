# nnUNetTrainerV2 Skip Connection Feature Visualization

## 概述

这是一个专门为 `nnUNetTrainerV2` 模型设计的特征可视化工具。它针对 `Task530_EsoTJ_30pct` 数据集上训练的模型，从第一个跳跃连接层提取特征，并生成三视角（轴向、冠状、矢状）的热力图可视化。

## 功能特性

- ✅ **自动模型加载**：自动发现并加载 nnUNetTrainerV2 模型检查点
- ✅ **第一跳跃连接层检测**：自动定位第一个跳跃连接层 (`conv_blocks_localization[0]`)
- ✅ **多视角热力图**：生成轴向、冠状、矢状三个方向的热力图叠加
- ✅ **多种可视化后端**：支持激活值和 Grad-CAM 两种热力图生成方式
- ✅ **详细元数据日志**：保存完整的配置和统计信息
- ✅ **自动检查点发现**：无需手动指定检查点路径

## 输出文件

对于每个案例，脚本生成以下文件（以 `CASEID_activation_skip_connection` 为例）：

1. **`CASEID_activation_skip_connection.npy`**
   - 调整到原始体积尺寸的热力图
   - 数据格式：numpy array，形状 (D, H, W)
   - 值范围：[0, 1]（归一化）

2. **`CASEID_activation_skip_connection_axial.png`**
   - 单视角（轴向）可视化
   - 左图：CT原始切片
   - 右图：CT + 热力图叠加

3. **`CASEID_activation_skip_connection_3views.png`**
   - 三视角完整可视化（3×2 网格）
   - 第1行：轴向视角（上下方向）
   - 第2行：冠状视角（前后方向）
   - 第3行：矢状视角（左右方向）
   - 每行左列为原始CT，右列为热力图叠加

4. **`CASEID_activation_skip_connection.json`**
   - 元数据文件，包含：
     - 模型配置信息
     - 特征层名称
     - 检查点路径
     - 切片索引
     - 特征和热力图统计信息

## 安装依赖

确保已安装以下Python包：

```bash
pip install torch matplotlib numpy scipy
```

## 使用方法

### 基础用法

默认参数运行（fold 1，第一个案例，activation 模式）：

```bash
python tools/visualize_nnunetv2_skip_connection.py
```

### 常见命令

#### 1. 指定 Fold 和 Case

```bash
# Fold 2，特定案例
python tools/visualize_nnunetv2_skip_connection.py --fold 2 --case-id ESO_TJ_60011222468-x-64
```

#### 2. 使用 Grad-CAM 后端

```bash
# 生成 Grad-CAM 热力图（考虑梯度权重）
python tools/visualize_nnunetv2_skip_connection.py --backend gradcam
```

#### 3. 自定义输出目录

```bash
python tools/visualize_nnunetv2_skip_connection.py \
  --output-dir /path/to/custom/output
```

#### 4. 查看网络结构

```bash
# 打印模型结构以了解层名称
python tools/visualize_nnunetv2_skip_connection.py --print-structure | head -50
```

#### 5. 调试模式

```bash
# 只提取特征并打印统计信息，不生成可视化
python tools/visualize_nnunetv2_skip_connection.py --debug-stats
```

#### 6. 指定 GPU/CPU

```bash
# 使用 CPU
python tools/visualize_nnunetv2_skip_connection.py --device cpu

# 使用特定 GPU
CUDA_VISIBLE_DEVICES=0 python tools/visualize_nnunetv2_skip_connection.py --device cuda
```

### 完整参数列表

```
usage: visualize_nnunetv2_skip_connection.py [-h] [--fold FOLD] 
    [--case-id CASE_ID] [--backend {activation,gradcam}]
    [--output-dir OUTPUT_DIR] [--checkpoint CHECKPOINT]
    [--dataset-directory DATASET_DIRECTORY] [--plans-file PLANS_FILE]
    [--device DEVICE] [--print-structure] [--debug-stats]

optional arguments:
  -h, --help                      show this help message and exit
  --fold FOLD                     Fold number (0-4), default: 1
  --case-id CASE_ID               Case ID; defaults to first case
  --backend {activation,gradcam}  Heatmap generation backend, default: activation
  --output-dir OUTPUT_DIR         Output directory, default: heatmap_output
  --checkpoint CHECKPOINT         Custom checkpoint path
  --dataset-directory DATASET_DIR Custom dataset directory
  --plans-file PLANS_FILE         Custom plans file
  --device DEVICE                 Device to use (cuda/cpu), default: auto
  --print-structure               Print network structure and exit
  --debug-stats                   Print feature statistics and exit
```

## 输出示例

```
======================================================================
nnUNetTrainerV2 Skip Connection Feature Visualization
======================================================================
Task:               Task530_EsoTJ_30pct
Trainer:            nnUNetTrainerV2
Fold:               1
Backend:            activation
Device:             cuda
Checkpoint:         ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/model_final_checkpoint.model
Plans file:         ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2__nnUNetPlansv2.1/plans.pkl
Dataset directory:  /data/nnUNet_preprocessed/Task530_EsoTJ_30pct
Validation raw dir: ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/validation_raw
======================================================================

✓ Loaded checkpoint: ckpt/nnUNet/.../model_final_checkpoint.model
✓ Initialized nnUNetTrainerV2
✓ Selected case: ESO_TJ_60011222468-x-64
✓ Loaded data shape: (1, 1, 64, 128, 128)
✓ Found first skip layer: conv_blocks_localization.0

Extracting features...
✓ Feature layer: conv_blocks_localization.0
  Feature shape:    (1, 64, 32, 64, 64)
  Feature min:      -5.234156
  Feature max:      8.945623
  Feature mean:     0.234567
  Feature std:      1.123456

Generated heatmap:
  Heatmap shape:    (32, 64, 64)
  Heatmap min:      0.0
  Heatmap max:      1.0
  Heatmap mean:     0.234567
  Heatmap std:      0.234567

Resized heatmap to original shape: (64, 128, 128)

✓ Saved: heatmap_output/Task530_EsoTJ_30pct/nnUNetTrainerV2/fold_1/ESO_TJ_60011222468-x-64_activation_skip_connection.npy
✓ Saved: heatmap_output/Task530_EsoTJ_30pct/nnUNetTrainerV2/fold_1/ESO_TJ_60011222468-x-64_activation_skip_connection_axial.png
✓ Saved: heatmap_output/Task530_EsoTJ_30pct/nnUNetTrainerV2/fold_1/ESO_TJ_60011222468-x-64_activation_skip_connection_3views.png
✓ Saved: heatmap_output/Task530_EsoTJ_30pct/nnUNetTrainerV2/fold_1/ESO_TJ_60011222468-x-64_activation_skip_connection.json

======================================================================
✓ Visualization complete!
  Output directory: heatmap_output/Task530_EsoTJ_30pct/nnUNetTrainerV2/fold_1
======================================================================
```

## 热力图类型说明

### 1. Activation（默认）

**原理**：对所有通道的特征图取绝对值，然后计算通道平均值。

$$\text{CAM}_{activation} = \frac{1}{C} \sum_{c=1}^{C} |F_c|$$

**优点**：
- 快速计算，无需梯度
- 反映网络激活情况
- 适合快速探索

**缺点**：
- 不考虑对输出的影响
- 可能存在噪声

### 2. Grad-CAM

**原理**：使用梯度权重加权特征图，突出对模型输出影响最大的区域。

$$\text{CAM}_{gradcam} = \text{ReLU}\left(\sum_{c=1}^{C} w_c \cdot F_c\right)$$

其中 $w_c$ 是特征图 $c$ 对肿瘤分类的平均梯度。

**优点**：
- 反映特征对输出决策的影响
- 通常更有针对性
- 更适合解释模型决策

**缺点**：
- 计算复杂，需要两次前向和一次反向传播
- 速度较慢

## 三视角说明

### 轴向 (Axial)
- **方向**：垂直于脊柱，从上往下看
- **常用于**：肺部和腹部扫描
- **解剖学**：左右对称

### 冠状 (Coronal)
- **方向**：从前往后看，正面视图
- **常用于**：观察前后方向的病变
- **解剖学**：可以看到胸腔的前后深度

### 矢状 (Sagittal)
- **方向**：从侧面看，侧视图
- **常用于**：观察脊柱和中线结构
- **解剖学**：可以看到左右对称性

## 故障排除

### 问题1：找不到检查点文件

```
FileNotFoundError: Checkpoint not found: ckpt/nnUNet/.../model_final_checkpoint.model
```

**解决方案**：
```bash
# 检查检查点是否存在
ls ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/

# 或使用自定义路径
python tools/visualize_nnunetv2_skip_connection.py \
  --checkpoint /path/to/your/checkpoint.model
```

### 问题2：找不到验证数据

```
RuntimeError: No npz files found in validation_raw
```

**解决方案**：
```bash
# 检查验证数据目录
ls ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/validation_raw/

# 运行验证以生成验证数据
python step4_pred_val_results_and_eval.py
```

### 问题3：CUDA 内存不足

```bash
# 使用 CPU
python tools/visualize_nnunetv2_skip_connection.py --device cpu

# 或减小 batch size（通过修改脚本）
```

### 问题4：找不到特征层

```
RuntimeError: Could not find first skip connection layer in network
```

**解决方案**：
```bash
# 查看网络结构找到正确的层名
python tools/visualize_nnunetv2_skip_connection.py --print-structure

# 然后修改脚本中的 find_first_skip_connection() 函数
```

## 批量处理

处理多个 fold 和案例的脚本：

```bash
#!/bin/bash

# 处理所有 fold
for fold in 0 1 2 3 4; do
    echo "Processing fold $fold..."
    python tools/visualize_nnunetv2_skip_connection.py --fold $fold
done

# 或使用 Grad-CAM
for fold in 0 1 2; do
    python tools/visualize_nnunetv2_skip_connection.py \
      --fold $fold \
      --backend gradcam
done
```

## 与现有工具的比较

### vs. `extract_feature_heatmap.py`

| 特性 | 本脚本 | extract_feature_heatmap.py |
|-----|--------|---------------------------|
| 目标模型 | nnUNetTrainerV2（Task530） | 多种模型 |
| 配置复杂度 | 低（硬编码配置） | 高（需要指定trainer） |
| 三视角支持 | ✅ 完整 | ✅ 完整 |
| 使用难度 | 简单 | 中等 |
| 灵活性 | 低 | 高 |

## 文件结构

```
tools/
├── visualize_nnunetv2_skip_connection.py  ← 新脚本
├── extract_feature_heatmap.py              ← 通用工具
├── heatmap_common.py                       ← 共享工具函数
└── README.md                               ← 本文档
```

## 参考资源

- nnUNet 官方文档：https://github.com/MIC-DKFZ/nnUNet
- Grad-CAM 论文：https://arxiv.org/abs/1610.02055
- PyTorch Hook 教程：https://pytorch.org/tutorials/

## License

Same as parent project

## 作者备注

该脚本专门针对以下配置优化：
- **任务**：Task530_EsoTJ_30pct
- **模型**：nnUNetTrainerV2
- **网络**：3d_fullres
- **特征层**：conv_blocks_localization（第一个解码器跳跃连接）

若要用于其他模型或任务，建议使用通用的 `extract_feature_heatmap.py` 脚本。
