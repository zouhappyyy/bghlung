import os
import json
import yaml
import random
import shutil


def load_config(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def extract_case_id(img_path):
    """
    nnU-Net 标准 case_id 提取
    xxx_0000.nii.gz -> xxx
    """
    base = os.path.basename(img_path)
    return base.replace("_0000.nii.gz", "")


def main(cfg_path):
    cfg = load_config(cfg_path)

    # ===============================
    # 读取 config
    # ===============================
    raw_root = cfg["data"]["nnunet_raw_root"]
    src_task = cfg["data"]["source_task"]
    dst_task = cfg["data"]["target_task"]
    sample_ratio = cfg["data"]["sample_ratio"]
    train_ratio =cfg["data"]["train_ratio"]
    centers = cfg["data"].get("centers", [])
    use_pre_only = cfg["data"].get("use_pre_only", True)

    seed = cfg["experiment"]["random_seed"]
    random.seed(seed)

    ds_cfg = cfg["dataset_json"]
    keep_keys = ds_cfg.get("keep_keys", [])
    inherit = ds_cfg.get("inherit_from_source", True)
    auto_update_num = ds_cfg.get("auto_update_num_training", True)

    src_task_dir = os.path.join(raw_root, src_task)
    dst_task_dir = os.path.join(raw_root, dst_task)

    # ===============================
    # 创建目标 raw 目录
    # ===============================
    ensure_dir(os.path.join(dst_task_dir, "imagesTr"))
    ensure_dir(os.path.join(dst_task_dir, "labelsTr"))

    # ===============================
    # 读取源 dataset.json
    # ===============================
    with open(os.path.join(src_task_dir, "dataset.json"), "r") as f:
        src_ds = json.load(f)

    # ===============================
    # 收集所有 case_id
    # ===============================
    all_items = src_ds["training"]

    all_cases = []
    case_to_item = {}

    for item in all_items:
        cid = extract_case_id(item["image"])
        all_cases.append(cid)
        case_to_item[cid] = item

    # ===============================
    # 中心过滤
    # ===============================
    if centers:
        all_cases = [
            c for c in all_cases
            if any(c.startswith(f"ESO_{center}_") for center in centers)
        ]

    if len(all_cases) == 0:
        raise RuntimeError("❌ 中心过滤后无病例，请检查 centers 配置")

    # ===============================
    # 抽样
    # ===============================
    sample_num = max(1, int(len(all_cases) * sample_ratio))
    sampled_cases = sorted(random.sample(all_cases, sample_num))

    print(f"源病例数: {len(case_to_item)}")
    print(f"可用病例数: {len(all_cases)}")
    print(f"抽取病例数: {len(sampled_cases)}")

    # ===============================
    # 复制文件 + 构建 training
    # ===============================
    new_training = []

    for cid in sampled_cases:
        item = case_to_item[cid]

        img_rel = item["image"].replace("./", "")
        lbl_rel = item["label"].replace("./", "")
        img_rel_new = img_rel.replace(".nii.gz", "_0000.nii.gz")


        shutil.copy(
            os.path.join(src_task_dir, img_rel_new),
            os.path.join(dst_task_dir, img_rel_new)
        )
        shutil.copy(
            os.path.join(src_task_dir, lbl_rel),
            os.path.join(dst_task_dir, lbl_rel)
        )

        new_training.append({
            "image": f"./{img_rel}",
            "label": f"./{lbl_rel}"
        })

    # ===============================
    # 构建新的 dataset.json
    # ===============================
    if inherit:
        new_ds = {k: src_ds[k] for k in keep_keys if k in src_ds}
    else:
        new_ds = {}

    new_ds["training"] = new_training
    # --- v1 必需字段（关键） ---
    new_ds["test"] = []  # ← 必须有
    new_ds["numTest"] = 0  # ← 强烈建议

    if auto_update_num:
        new_ds["numTraining"] = len(new_training)

    with open(os.path.join(dst_task_dir, "dataset.json"), "w") as f:
        json.dump(new_ds, f, indent=4)

    print(f"✅ raw 子集构建完成: {dst_task}")
    print(f"📁 位置: {dst_task_dir}")


if __name__ == "__main__":
    import sys
    assert len(sys.argv) == 2, "用法: python build_raw_subset.py config/xxx.yaml"
    main(sys.argv[1])


# python step2_generate_small_splits.py config/eso_pre_small_exp.yaml