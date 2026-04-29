#!/usr/bin/env bash
set -euo pipefail

# One-click frequency-spectrum analysis for two Task570 cases on two models.
#
# It runs 4 jobs:
# 1) ESO_TJ_1010895787 -> nnUNetTrainerV2
# 2) ESO_TJ_1010895787 -> MedNeXtTrainerV2
# 3) ESO_TJ_60012747242 -> nnUNetTrainerV2
# 4) ESO_TJ_60012747242 -> MedNeXtTrainerV2
#
# Optional overrides:
#   PYTHON_BIN=python
#   DEVICE=cuda
#   DATA_DIR=/path/to/Task570_preprocessed_dir
#   NNUNET_FOLD_DIR=...
#   MEDNEXT_FOLD_DIR=...
#   NNUNET_PLANS_FILE=...
#   MEDNEXT_PLANS_FILE=...

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
DATA_DIR="${DATA_DIR:-/home/fangzheng/zoule/ESO_nnUNet_dataset/nnUNet_preprocessed/Task570_EsoTJ83/nnUNetData_plans_v2.1_trgSp_1x1x1_stage0}"

NNUNET_FOLD_DIR="${NNUNET_FOLD_DIR:-${REPO_ROOT}/ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1}"
MEDNEXT_FOLD_DIR="${MEDNEXT_FOLD_DIR:-${REPO_ROOT}/ckpt/nnUNet/3d_fullres/Task530_EsoTJ_30pct/MedNeXtTrainerV2__nnUNetPlansv2.1/fold_2}"

NNUNET_PLANS_FILE="${NNUNET_PLANS_FILE:-}"
MEDNEXT_PLANS_FILE="${MEDNEXT_PLANS_FILE:-}"

ANALYZER="${REPO_ROOT}/tools/analyze_nnunet_encoder_frequency.py"
OUTPUT_ROOT="${REPO_ROOT}/feature_vis_output/frequency_spectrum/Task570_EsoTJ83"

CASES=(
  "ESO_TJ_1010895787"
  "ESO_TJ_60012747242"
)

run_one() {
  local trainer="$1"
  local fold_dir="$2"
  local plans_file="$3"
  local case_id="$4"
  local output_dir="${OUTPUT_ROOT}/${case_id}/${trainer}"

  echo "========================================================================"
  echo "Running ${trainer} for case ${case_id}"
  echo "Fold dir:   ${fold_dir}"
  echo "Data dir:   ${DATA_DIR}"
  echo "Output dir: ${output_dir}"
  echo "========================================================================"

  local cmd=(
    "${PYTHON_BIN}" "${ANALYZER}"
    --trainer "${trainer}"
    --fold-dir "${fold_dir}"
    --data-dir "${DATA_DIR}"
    --case-id "${case_id}"
    --output-dir "${output_dir}"
    --checkpoint-name model_final_checkpoint
    --device "${DEVICE}"
  )

  if [[ -n "${plans_file}" ]]; then
    cmd+=(--plans-file "${plans_file}")
  fi

  "${cmd[@]}"
}

for case_id in "${CASES[@]}"; do
  run_one "nnUNetTrainerV2" "${NNUNET_FOLD_DIR}" "${NNUNET_PLANS_FILE}" "${case_id}"
  run_one "MedNeXtTrainerV2" "${MEDNEXT_FOLD_DIR}" "${MEDNEXT_PLANS_FILE}" "${case_id}"
done

echo
echo "All jobs completed."
echo "Results saved under: ${OUTPUT_ROOT}"
