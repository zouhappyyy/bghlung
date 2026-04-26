#!/bin/bash
# Quick reference guide for visualize_nnunetv2_skip_connection.py
# This script shows common usage examples

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=========================================================================="
echo "nnUNetTrainerV2 Skip Connection Feature Visualization - Quick Reference"
echo "=========================================================================="
echo ""

# ============================================================================
# Example 1: Default usage (Fold 1, first case, activation mode)
# ============================================================================
cat << 'EOF'
【Example 1】Basic usage - default settings
Command:
    python tools/visualize_nnunetv2_skip_connection.py

Description:
    - Processes fold 1 (default)
    - Uses first validation case
    - Generates activation-based heatmap
    - Outputs to heatmap_output/

Expected output:
    - heatmap_output/Task530_EsoTJ_30pct/nnUNetTrainerV2/fold_1/
      ├── CASEID_activation_skip_connection.npy
      ├── CASEID_activation_skip_connection_axial.png
      ├── CASEID_activation_skip_connection_3views.png
      └── CASEID_activation_skip_connection.json

EOF

# ============================================================================
# Example 2: Process specific fold
# ============================================================================
cat << 'EOF'
【Example 2】Process different folds
Commands:
    # Fold 0
    python tools/visualize_nnunetv2_skip_connection.py --fold 0

    # Fold 2
    python tools/visualize_nnunetv2_skip_connection.py --fold 2

    # Fold 3
    python tools/visualize_nnunetv2_skip_connection.py --fold 3

Note:
    Valid folds: 0, 1, 2, 3, 4

EOF

# ============================================================================
# Example 3: Process specific case
# ============================================================================
cat << 'EOF'
【Example 3】Process specific case
Commands:
    # List available cases
    ls ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/validation_raw/*.npz

    # Process specific case
    python tools/visualize_nnunetv2_skip_connection.py \
      --fold 1 \
      --case-id ESO_TJ_60011222468-x-64

Note:
    - case-id is the stem of the .npz filename (without extension)
    - If not specified, processes the first case

EOF

# ============================================================================
# Example 4: Grad-CAM mode
# ============================================================================
cat << 'EOF'
【Example 4】Generate Grad-CAM heatmap
Commands:
    # Grad-CAM for fold 1
    python tools/visualize_nnunetv2_skip_connection.py --backend gradcam

    # Grad-CAM for fold 2, specific case
    python tools/visualize_nnunetv2_skip_connection.py \
      --fold 2 \
      --case-id YOUR_CASE_ID \
      --backend gradcam

Differences:
    - activation (default): Fast, shows channel-averaged activation
    - gradcam: Slower, shows gradient-weighted importance

Performance:
    - activation: ~10 seconds per case
    - gradcam: ~30-60 seconds per case

EOF

# ============================================================================
# Example 5: Batch processing
# ============================================================================
cat << 'EOF'
【Example 5】Batch processing - process all folds
Save as: run_all_folds.sh

#!/bin/bash
for fold in 0 1 2 3 4; do
    echo "Processing fold $\{fold}..."
    python tools/visualize_nnunetv2_skip_connection.py --fold $fold
done

Usage:
    bash run_all_folds.sh

EOF

# ============================================================================
# Example 6: Process with Grad-CAM for multiple folds
# ============================================================================
cat << 'EOF'
【Example 6】Batch processing with Grad-CAM
Save as: run_all_folds_gradcam.sh

#!/bin/bash
for fold in 0 1 2 3; do
    echo "Processing fold $\{fold} with Grad-CAM..."
    python tools/visualize_nnunetv2_skip_connection.py \
      --fold $fold \
      --backend gradcam
done

Usage:
    bash run_all_folds_gradcam.sh

EOF

# ============================================================================
# Example 7: Debug mode
# ============================================================================
cat << 'EOF'
【Example 7】Debug and troubleshooting
Commands:
    # Print network structure (shows all layer names)
    python tools/visualize_nnunetv2_skip_connection.py --print-structure

    # Extract and print feature statistics only (no visualization)
    python tools/visualize_nnunetv2_skip_connection.py --debug-stats

    # Check available checkpoints
    ls -la ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/

    # Check available validation data
    ls ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/validation_raw/

EOF

# ============================================================================
# Example 8: Custom paths and devices
# ============================================================================
cat << 'EOF'
【Example 8】Custom configurations
Commands:
    # Use CPU instead of CUDA
    python tools/visualize_nnunetv2_skip_connection.py --device cpu

    # Custom output directory
    python tools/visualize_nnunetv2_skip_connection.py \
      --output-dir /custom/path/heatmap_output

    # Specific GPU (if multiple GPUs available)
    CUDA_VISIBLE_DEVICES=0 python tools/visualize_nnunetv2_skip_connection.py
    CUDA_VISIBLE_DEVICES=1 python tools/visualize_nnunetv2_skip_connection.py

EOF

# ============================================================================
# Example 9: Output file structure
# ============================================================================
cat << 'EOF'
【Example 9】Understanding output files
Directory structure:
    heatmap_output/
    └── Task530_EsoTJ_30pct/
        └── nnUNetTrainerV2/
            └── fold_1/
                └── {CASEID}_activation_skip_connection/
                    ├── {CASEID}_activation_skip_connection.npy
                    │   └── Numpy array (D, H, W) with values in [0, 1]
                    │
                    ├── {CASEID}_activation_skip_connection_axial.png
                    │   └── Single view (axial) with CT + heatmap overlay
                    │
                    ├── {CASEID}_activation_skip_connection_3views.png
                    │   └── Three-view visualization (3 rows × 2 columns)
                    │       Row 1: Axial (Top/Bottom view)
                    │       Row 2: Coronal (Front/Back view)
                    │       Row 3: Sagittal (Left/Right view)
                    │       Column 1: Original CT
                    │       Column 2: CT + Heatmap overlay
                    │
                    └── {CASEID}_activation_skip_connection.json
                        └── Metadata:
                            {
                              "task": "Task530_EsoTJ_30pct",
                              "trainer": "nnUNetTrainerV2",
                              "fold": 1,
                              "case_id": "CASEID",
                              "backend": "activation",
                              "layer_name": "conv_blocks_localization.0",
                              "checkpoint": "...",
                              "slice_indices": {...},
                              "original_shape": [64, 128, 128],
                              "feature_shape": [1, 64, 32, 64, 64],
                              ...
                            }

EOF

# ============================================================================
# Example 10: Common issues and solutions
# ============================================================================
cat << 'EOF'
【Example 10】Troubleshooting
Issue: FileNotFoundError: Checkpoint not found
Solution:
    ls ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/
    # If model_final_checkpoint.model not found, train the model first:
    python nnunet/run/run_training.py 3d_fullres nnUNetTrainerV2 Task530_EsoTJ_30pct 1

Issue: RuntimeError: No npz files found in validation_raw
Solution:
    # Run validation prediction first:
    python step4_pred_val_results_and_eval.py

Issue: CUDA out of memory
Solution:
    # Use CPU instead:
    python tools/visualize_nnunetv2_skip_connection.py --device cpu

Issue: Could not find first skip connection layer
Solution:
    # View network structure:
    python tools/visualize_nnunetv2_skip_connection.py --print-structure | head -100

EOF

# ============================================================================
# Summary
# ============================================================================
cat << 'EOF'
========================================================================
QUICK START
========================================================================

1. Default (activation mode, fold 1, first case):
   python tools/visualize_nnunetv2_skip_connection.py

2. Grad-CAM (gradient-weighted, often more interpretable):
   python tools/visualize_nnunetv2_skip_connection.py --backend gradcam

3. Specific fold and case:
   python tools/visualize_nnunetv2_skip_connection.py \
     --fold 2 \
     --case-id YOUR_CASE_ID

4. Batch process all folds:
   for fold in 0 1 2 3 4; do
     python tools/visualize_nnunetv2_skip_connection.py --fold $fold
   done

========================================================================
OUTPUT FORMATS
========================================================================

Three-view PNG (recommended):
  - Shows all three anatomical views (axial, coronal, sagittal)
  - Each row: CT image (left) + Heatmap overlay (right)
  - High resolution (300 DPI)

Single-view PNG:
  - Shows only axial view for quick inspection
  - Smaller file size

Numpy NPY:
  - Raw heatmap data for further analysis
  - Can be loaded: heatmap = np.load('file.npy')

JSON metadata:
  - Configuration and statistics
  - Useful for documentation and batch processing

========================================================================
For detailed documentation, see: tools/VISUALIZATION_GUIDE.md
========================================================================
EOF
