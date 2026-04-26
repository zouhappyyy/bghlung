#!/bin/bash

# 新建脚本使用指引
# 为 nnUNetTrainerV2 Task530_EsoTJ_30pct 生成的模型进行特征可视化

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  nnUNetTrainerV2 Skip Connection Feature Visualization         ║"
echo "║  快速使用指引                                                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

cat << 'EOF'
【📦 新建文件】
─────────────────────────────────────────────────────────────────────

✅ visualize_nnunetv2_skip_connection.py
   📍 主脚本，完整的特征可视化工具
   🔧 可执行权限：已授予
   📊 代码行数：~750 行
   💾 大小：20 KB

✅ VISUALIZATION_GUIDE.md
   📍 详细的使用指南
   📚 内容：10+ 示例、参数说明、故障排除
   💾 大小：11 KB

✅ USAGE_EXAMPLES.sh
   📍 快速参考脚本
   🚀 可执行，显示 10 个实用示例
   💾 大小：11 KB

✅ README_VISUALIZATION.md
   📍 项目总结和快速开始
   📖 内容：概览、工具对比、性能指标
   💾 大小：11 KB

位置：tools/ 目录


【🚀 快速开始】
─────────────────────────────────────────────────────────────────────

1️⃣ 最简单的方式（推荐首先尝试）
   cd /home/fangzheng/zoule/BGHNet_esov1
   python tools/visualize_nnunetv2_skip_connection.py

2️⃣ 查看所有使用示例
   bash tools/USAGE_EXAMPLES.sh

3️⃣ 查看详细文档
   cat tools/README_VISUALIZATION.md

4️⃣ 获取完整帮助
   python tools/visualize_nnunetv2_skip_connection.py -h


【⚙️ 常见命令】
─────────────────────────────────────────────────────────────────────

处理特定 fold：
  python tools/visualize_nnunetv2_skip_connection.py --fold 2

处理特定案例：
  python tools/visualize_nnunetv2_skip_connection.py \
    --fold 1 \
    --case-id ESO_TJ_60011222468-x-64

使用 Grad-CAM（更精准但较慢）：
  python tools/visualize_nnunetv2_skip_connection.py --backend gradcam

使用 CPU：
  python tools/visualize_nnunetv2_skip_connection.py --device cpu

查看网络结构（用于调试）：
  python tools/visualize_nnunetv2_skip_connection.py --print-structure

批量处理所有 fold：
  for fold in 0 1 2 3 4; do
    python tools/visualize_nnunetv2_skip_connection.py --fold $fold
  done


【📊 输出文件】
─────────────────────────────────────────────────────────────────────

脚本会在以下目录生成文件：
  heatmap_output/Task530_EsoTJ_30pct/nnUNetTrainerV2/fold_1/

生成的文件类型：

1. CASE_activation_skip_connection.npy
   └─ Numpy 数组格式，用于进一步分析

2. CASE_activation_skip_connection_axial.png
   └─ 单视角（轴向）可视化

3. CASE_activation_skip_connection_3views.png ⭐ 推荐
   └─ 三视角完整可视化（轴向、冠状、矢状）

4. CASE_activation_skip_connection.json
   └─ 完整的元数据和统计信息


【🎨 热力图说明】
─────────────────────────────────────────────────────────────────────

两种热力图生成模式：

Activation（默认，快速）：
  • 原理：通道平均激活值
  • 速度：~10 秒/案例
  • 用途：快速探索网络激活模式
  • 推荐：快速实验和探索

Grad-CAM（精准）：
  • 原理：梯度加权激活值
  • 速度：~60 秒/案例
  • 用途：显示对输出决策的影响
  • 推荐：详细分析和论文发表


【👁️ 三视角说明】
─────────────────────────────────────────────────────────────────────

轴向 (Axial)：
  • 方向：从上往下看
  • 最常用于：肺部和腹部扫描
  • 解剖学：易于观察左右对称性

冠状 (Coronal)：
  • 方向：正面视图（从前往后看）
  • 常用于：观察前后方向的空间关系
  • 解剖学：显示胸腔的前后深度

矢状 (Sagittal)：
  • 方向：侧面视图（从侧面看）
  • 常用于：观察脊柱和中线结构
  • 解剖学：显示左右对称性


【📈 性能指标】
─────────────────────────────────────────────────────────────────────

单个案例处理时间（GPU 环境）：
  Activation 模式：~15 秒
  Grad-CAM 模式：~60 秒

内存使用（单个案例）：
  Activation 模式：~700 MB
  Grad-CAM 模式：~1.2 GB

批量处理 5 个案例：
  Activation：~75 秒（~1.25 分钟）
  Grad-CAM：~300 秒（~5 分钟）


【🔧 参数说明】
─────────────────────────────────────────────────────────────────────

--fold FOLD
  Fold number (0-4)，默认值：1

--case-id CASE_ID
  案例 ID，默认值：第一个案例

--backend {activation,gradcam}
  热力图生成模式，默认值：activation

--output-dir OUTPUT_DIR
  输出目录，默认值：heatmap_output

--device {cuda,cpu}
  使用的设备，默认值：自动检测

--checkpoint CHECKPOINT
  自定义检查点路径，默认值：自动查找

--print-structure
  打印网络结构并退出（用于调试）

--debug-stats
  仅提取特征并打印统计信息，不生成可视化


【❓ 常见问题】
─────────────────────────────────────────────────────────────────────

Q1: 脚本找不到检查点
A: 确保模型已训练。运行命令：
   python nnunet/run/run_training.py 3d_fullres nnUNetTrainerV2 Task530_EsoTJ_30pct 1

Q2: 找不到验证数据
A: 首先运行验证预测：
   python step4_pred_val_results_and_eval.py

Q3: 内存不足
A: 使用 CPU 模式：
   python tools/visualize_nnunetv2_skip_connection.py --device cpu

Q4: Grad-CAM 太慢
A: 使用 activation 模式（默认，快 5-6 倍）或只对重要案例使用 Grad-CAM

Q5: 热力图很暗/很亮
A: 正常现象，取决于网络激活。查看 JSON 中的统计信息。


【📚 文档导航】
─────────────────────────────────────────────────────────────────────

快速开始：
  cat tools/README_VISUALIZATION.md

详细指南（强烈推荐）：
  cat tools/VISUALIZATION_GUIDE.md
  包含：10+ 示例、热力图原理、故障排除等

使用示例：
  bash tools/USAGE_EXAMPLES.sh

脚本帮助：
  python tools/visualize_nnunetv2_skip_connection.py -h


【🎯 使用流程】
─────────────────────────────────────────────────────────────────────

1️⃣ 模型训练
   python nnunet/run/run_training.py 3d_fullres nnUNetTrainerV2 \
     Task530_EsoTJ_30pct 1

2️⃣ 验证预测
   python step4_pred_val_results_and_eval.py

3️⃣ 特征可视化 ⭐ (新增工具)
   python tools/visualize_nnunetv2_skip_connection.py

4️⃣ 查看结果
   open heatmap_output/Task530_EsoTJ_30pct/nnUNetTrainerV2/fold_1/


【✅ 验证清单】
─────────────────────────────────────────────────────────────────────

运行前检查：
  ✓ Python 环境已安装
  ✓ PyTorch 已安装
  ✓ nnUNet 已配置
  ✓ 模型已训练
  ✓ 验证数据已生成

第一次运行：
  python tools/visualize_nnunetv2_skip_connection.py

期望输出：
  ✓ 日志信息显示处理进度
  ✓ 特征统计信息输出
  ✓ 热力图保存成功
  ✓ 可视化图像生成完成


【💡 使用建议】
─────────────────────────────────────────────────────────────────────

1. 首次使用：直接运行默认命令
   python tools/visualize_nnunetv2_skip_connection.py

2. 快速探索：使用 activation 模式（更快）

3. 详细分析：使用 Grad-CAM 模式（更精准）

4. 批量处理：参考 VISUALIZATION_GUIDE.md 的批处理脚本

5. 遇到问题：
   - 查看 VISUALIZATION_GUIDE.md 故障排除部分
   - 使用 --print-structure 调试
   - 使用 --debug-stats 检查特征


【🎓 学习资源】
─────────────────────────────────────────────────────────────────────

推荐阅读顺序：

1️⃣ README_VISUALIZATION.md（5 分钟）
   快速了解项目

2️⃣ 运行默认脚本（1 分钟）
   python tools/visualize_nnunetv2_skip_connection.py

3️⃣ 查看使用示例（10 分钟）
   bash tools/USAGE_EXAMPLES.sh

4️⃣ 深入学习指南（20 分钟）
   cat tools/VISUALIZATION_GUIDE.md

5️⃣ 自定义配置（按需）
   python tools/visualize_nnunetv2_skip_connection.py -h


【📞 获取帮助】
─────────────────────────────────────────────────────────────────────

查看帮助信息：
  python tools/visualize_nnunetv2_skip_connection.py -h

查看网络结构（调试）：
  python tools/visualize_nnunetv2_skip_connection.py --print-structure | head -50

查看特征统计（调试）：
  python tools/visualize_nnunetv2_skip_connection.py --debug-stats

查看详细文档：
  cat tools/VISUALIZATION_GUIDE.md

查看使用示例：
  bash tools/USAGE_EXAMPLES.sh


╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║        🎉 开始使用吧！输入以下命令立即开始特征可视化：             ║
║                                                                    ║
║   cd /home/fangzheng/zoule/BGHNet_esov1                            ║
║   python tools/visualize_nnunetv2_skip_connection.py              ║
║                                                                    ║
║        祝您使用愉快！🚀                                            ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝

EOF
