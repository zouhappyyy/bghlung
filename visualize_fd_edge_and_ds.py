import os
import argparse
from typing import List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F



def _extract_case_data(data_root: str, case_id: str, dataset_directory: str) -> Tuple[np.ndarray, np.ndarray]:
    """从预处理数据目录和 gt_segmentations 中读取某病例的 data 和 seg.

    对于 Task530_EsoTJ_30pct，stage0 预处理数据每个病例对应单个 .npy 文件，
    其 shape 为 (C, D, H, W)。因此这里：
      - 直接加载 <case_id>.npy（或以 case_id 开头的 .npy），并规范为 data[C, D, H, W]
      - 从 dataset_directory/gt_segmentations 读取对应的 NIfTI 标签作为 seg[D, H, W]
    """
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data_root {data_root} does not exist")

    # 1) 找到单个 .npy 文件（例如 ESO_TJ_60011222468.npy）
    exact_npy = os.path.join(data_root, f"{case_id}.npy")
    if os.path.isfile(exact_npy):
        npy_candidates = [exact_npy]
    else:
        npy_files = [
            os.path.join(data_root, f) for f in os.listdir(data_root)
            if f.startswith(case_id) and f.endswith(".npy")
        ]
        if not npy_files:
            raise FileNotFoundError(
                f"No preprocessed .npy file found for case {case_id} under {data_root}. "
                f"Expected a file like {case_id}.npy."
            )
        if len(npy_files) > 1:
            print(f"[WARN] Multiple .npy files match case_id {case_id}: {[os.path.basename(f) for f in npy_files]}. Using {os.path.basename(npy_files[0])}")
        npy_candidates = [npy_files[0]]

    arr = np.load(npy_candidates[0])
    # 规范化为 [C, D, H, W]
    if arr.ndim == 4:
        # 常见情况: (C, D, H, W)
        data = arr.astype(np.float32)
    elif arr.ndim == 3:
        # 单通道 (D, H, W)
        data = arr[None].astype(np.float32)
    else:
        raise RuntimeError(f"Unexpected array shape {arr.shape} in {npy_candidates[0]}, expected (C,D,H,W) or (D,H,W)")

    # 当前 Task530 模型以单通道训练，这里仅保留第一个通道，避免 conv3d 输入通道数不匹配
    data = data[:1]

    # 2) 从 gt_segmentations 中读取 NIfTI 标签
    import nibabel as nib

    gt_dir = os.path.join(dataset_directory, "gt_segmentations")
    if not os.path.isdir(gt_dir):
        raise FileNotFoundError(f"gt_segmentations folder not found at {gt_dir}")

    gt_path = None
    for ext in (".nii.gz", ".nii"):
        p = os.path.join(gt_dir, f"{case_id}{ext}")
        if os.path.isfile(p):
            gt_path = p
            break
    if gt_path is None:
        raise FileNotFoundError(f"Could not find GT NIfTI for case {case_id} in {gt_dir}")

    gt_img = nib.load(gt_path)
    gt_arr = gt_img.get_fdata()

    # === 关键：将 NIfTI 标签从 [X, Y, Z] 调整到与预处理 .npy 一致的 [Z, Y, X] 坐标系，并可选做左右翻转 ===
    if gt_arr.ndim != 3:
        raise RuntimeError(f"Unexpected GT ndim {gt_arr.ndim} for {gt_path}, expected 3D array")

    # 步骤 1：轴重排，假设 NIfTI 为 (X, Y, Z)，转为 (Z, Y, X)
    gt_arr = np.transpose(gt_arr, (2, 1, 0))

    # 步骤 2（可选）：如果发现仍然左右颠倒，可以取消下一行注释做左右翻转（axis=2 对应 X 方向）
    # gt_arr = np.flip(gt_arr, axis=2)

    gt = gt_arr.astype(np.int16)
    if gt.ndim == 4 and gt.shape[-1] == 1:
        gt = gt[..., 0]

    # 对齐 data 与 gt 的空间维度（简单裁剪到最小形状）
    if gt.shape != data.shape[1:]:
        min_shape = tuple(min(g, d) for g, d in zip(gt.shape, data.shape[1:]))
        gt = gt[:min_shape[0], :min_shape[1], :min_shape[2]]
        data = data[:, :min_shape[0], :min_shape[1], :min_shape[2]]

    seg = gt[None].astype(np.uint8)
    return data, seg


