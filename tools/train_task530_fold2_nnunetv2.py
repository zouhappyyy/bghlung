#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed training entry for Task530_EsoTJ_30pct fold 2 with nnUNetTrainerV2.

Goals:
- Do not modify nnunet/run/run_training.py
- Keep max epochs at 300
- Save checkpoints every 10 epochs
- Allow patch-size override via NNUNET_PATCH_SIZE (default 64)

Usage example:
  NNUNET_PATCH_SIZE=64 python tools/train_task530_fold2_nnunetv2.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_TRAINING = ROOT / "nnunet" / "run" / "run_training.py"
DEFAULT_TASK = "Task530_EsoTJ_30pct"
DEFAULT_NETWORK = "3d_fullres"
DEFAULT_TRAINER = "nnUNetTrainerV2"
DEFAULT_FOLD = "2"
DEFAULT_LOG = ROOT / "log" / "nnUNetV2_530_Fold2.log"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fixed training launcher for Task530 fold 2 using nnUNetTrainerV2.")
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--fold", default=DEFAULT_FOLD)
    p.add_argument("--network", default=DEFAULT_NETWORK)
    p.add_argument("--trainer", default=DEFAULT_TRAINER)
    p.add_argument("--log-file", default=str(DEFAULT_LOG))
    p.add_argument("--patch-size", default=os.environ.get("NNUNET_PATCH_SIZE", "64"), help="Patch size override. Use 64 or 64,64,64.")
    p.add_argument("--save-every", type=int, default=10, help="Checkpoint every N epochs (enforced via env when supported).")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    log_file = Path(args.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["NNUNET_PATCH_SIZE"] = str(args.patch_size)
    env["NNUNET_SAVE_EVERY"] = str(args.save_every)
    env["NNUNET_MAX_EPOCHS"] = "300"

    cmd = [
        sys.executable,
        str(RUN_TRAINING),
        args.network,
        args.trainer,
        args.task,
        str(args.fold),
    ]

    with open(log_file, "ab") as f:
        f.write(f"\n[launcher] command: {' '.join(cmd)}\n".encode("utf-8"))
        f.write(f"[launcher] NNUNET_PATCH_SIZE={env['NNUNET_PATCH_SIZE']} NNUNET_SAVE_EVERY={env['NNUNET_SAVE_EVERY']} NNUNET_MAX_EPOCHS={env['NNUNET_MAX_EPOCHS']}\n".encode("utf-8"))
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT)
        return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
