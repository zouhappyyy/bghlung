#!/usr/bin/env python
import os
import json
import argparse


def extract_metrics(summary_path, label="1"):
    """
    从 nnUNet/MedNeXt 的 summary.json 中提取指定标签的指标：
    Dice, Jaccard, Recall(Sensitivity), Precision, Specificity, Hausdorff Distance 95, ASSD
    """
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"summary 文件不存在: {summary_path}")

    with open(summary_path, "r") as f:
        data = json.load(f)

    # nnUNet 结构: data["results"]["mean"][label] 是该标签在验证集上的平均指标
    try:
        mean_metrics = data["results"]["mean"][str(label)]
    except KeyError:
        raise KeyError(
            f"在 summary.json 中找不到标签 {label} 的 mean 结果，"
            f"可用的标签有: {list(data['results']['mean'].keys())}"
        )

    # 必备重叠/分类指标
    dice = mean_metrics.get("Dice", None)
    iou = mean_metrics.get("Jaccard", None)
    sensitivity = mean_metrics.get("Recall", None)      # = Sensitivity
    precision = mean_metrics.get("Precision", None)

    # Specificity: 一般对应 evaluator 中的 "True Negative Rate"
    specificity = None
    for key, val in mean_metrics.items():
        k = key.lower().replace(" ", "")
        if k in ("truenegativerate", "specificity"):
            specificity = val
            break

    # D95 可能叫 "Hausdorff Distance 95" 或 "Hausdorff_95" 之类，这里做多种兼容
    d95 = None
    for key, val in mean_metrics.items():
        k = key.lower().replace(" ", "")
        if k.startswith("hausdorffdistance95"):
            d95 = val
            break
        if k.startswith("hausdorff95"):
            d95 = val
            break
        if k in ("hd95", "hausdorffdistance95mm"):
            d95 = val
            break

    # ASSD: Avg. Symmetric Surface Distance，可能有多种写法
    assd = None
    for key, val in mean_metrics.items():
        k = key.lower().replace(" ", "")
        k = k.lower().replace(".", "")
        # 常见写法: "Avg. Symmetric Surface Distance" / "Average Symmetric Surface Distance" / "assd"
        if k.startswith("avgsymmetricsurfacedistance"):
            assd = val
            break
        if k.startswith("averagesymmetricsurfacedistance"):
            assd = val
            break
        if k in ("assd",):
            assd = val
            break

    return {
        "label": str(label),
        "Dice": dice,
        "IoU": iou,
        "Sensitivity": sensitivity,
        "Precision": precision,
        "Specificity": specificity,
        "D95": d95,
        "ASSD": assd,
    }


def main():
    parser = argparse.ArgumentParser(
        description="从 nnUNet/MedNeXt summary.json 中提取指标: "
                    "Dice, IoU, Sensitivity, Precision, Specificity, D95, ASSD"
    )
    parser.add_argument(
        "--summary",
        "-s",
        type=str,
        required=True,
        help="summary.json 文件路径，例如: fold_1/validation_raw_postprocessed/summary.json",
    )
    parser.add_argument(
        "--label",
        "-l",
        type=str,
        default="1",
        help="要提取的标签 ID，默认为 '1'",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        default="val_metrics.csv",
        help="可选：输出结果保存到的 txt/csv 文件路径；若不指定则只打印",
    )

    args = parser.parse_args()

    metrics = extract_metrics(args.summary, label=args.label)

    print(f"\n=== 指标提取自: {args.summary} ===")
    print(f"标签: {metrics['label']}")
    print(f"Dice (DSC):          {metrics['Dice']}")
    print(f"IoU (Jaccard):       {metrics['IoU']}")
    print(f"Sensitivity (Recall):{metrics['Sensitivity']}")
    print(f"Precision:           {metrics['Precision']}")
    print(f"Specificity (TNR):   {metrics['Specificity']}")
    print(f"95% Hausdorff (D95): {metrics['D95']}")
    print(f"ASSD:                {metrics['ASSD']}")

    if args.out:
        # 只保留 validation 上两级：plans 目录 和 fold 目录
        abs_summary = os.path.abspath(args.summary)
        val_dir = os.path.dirname(abs_summary)          # .../validation_raw_postprocessed
        fold_dir = os.path.dirname(val_dir)             # .../fold_1
        plans_dir = os.path.dirname(fold_dir)           # .../nnUNetPlans...

        fold_name = os.path.basename(fold_dir)          # e.g. fold_1
        plans_name = os.path.basename(plans_dir)        # e.g. nnUNetPlansv2.1_trgSp_1x1x1_rwkv

        source_str = f"{plans_name}/{fold_name}"

        # 简单写成一行 CSV
        header = "label,Dice,IoU,Sensitivity,Precision,Specificity,D95,ASSD,Source\n"
        line = "{label},{Dice},{IoU},{Sensitivity},{Precision},{Specificity},{D95},{ASSD},{source}\n".format(
            source=source_str,
            **metrics,
        )

        write_header = not os.path.exists(args.out)
        with open(args.out, "a", encoding="utf-8") as f:
            if write_header:
                f.write(header)
            f.write(line)

        print(f"\n结果已追加写入: {args.out}")


if __name__ == "__main__":
    main()