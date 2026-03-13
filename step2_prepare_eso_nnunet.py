import os
import json
import yaml
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm

# ===============================
# 1. 读取 YAML 配置
# ===============================
CONFIG_PATH = "config/eso_data.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

task_id = cfg["task"]["task_id"]
task_name = cfg["task"]["task_name"]
description = cfg["task"]["description"]

nnunet_base = cfg["nnunet"]["raw_data_base"]
task_dir = os.path.join(nnunet_base, task_name)

centers = cfg["centers"]

prefix = cfg["naming"]["prefix"]
suffix = cfg["naming"]["modality_suffix"]
start_idx = cfg["naming"]["start_index"]

# ===============================
# 2. 创建 nnU-Net 目录
# ===============================
imagesTr = os.path.join(task_dir, "imagesTr")
labelsTr = os.path.join(task_dir, "labelsTr")
imagesTs = os.path.join(task_dir, "imagesTs")
os.makedirs(imagesTs, exist_ok=True)


os.makedirs(imagesTr, exist_ok=True)
os.makedirs(labelsTr, exist_ok=True)

# ===============================
# 3. 初始化 dataset.json
# ===============================
dataset = {
    "name": "ESO_PreSeg",
    "description": description,
    "tensorImageSize": "3D",
    "modality": cfg["dataset_json"]["modality"],
    "labels": cfg["dataset_json"]["labels"],
    "numTraining": 0,
    "numTest": 0,
    "training": [],
    "test": []
}

# ===============================
# 4. 遍历所有中心，只处理 pre 数据
# ===============================
idx = start_idx

for center in centers:
    center_name = center["name"]
    img_dir = center["pre_path"]
    mask_dir = center["mask_path"]

    print(f"\n📌 Processing center: {center_name}")

    img_paths = sorted(glob(os.path.join(img_dir, "*.nii*")))

    for img_path in tqdm(img_paths, desc=center_name):
        fname = os.path.basename(img_path)
        mask_path = os.path.join(mask_dir, fname)

        if not os.path.exists(mask_path):
            print(f"[WARN] Mask not found: {mask_path}")
            continue

        img = sitk.ReadImage(img_path)
        mask = sitk.ReadImage(mask_path)

        if img.GetSize() != mask.GetSize():
            print(f"[SKIP] Size mismatch: {fname}")
            continue

        # 构造 case_id: ESO_SY_001
        pid = fname.split(".")[0]  # 去掉扩展名
        case_id = f"{prefix}_{pid}"

        img_out = os.path.join(imagesTr, f"{case_id}{suffix}.nii.gz")
        mask_out = os.path.join(labelsTr, f"{case_id}.nii.gz")

        sitk.WriteImage(img, img_out)
        sitk.WriteImage(mask, mask_out)

        dataset["training"].append({
            "image": f"./imagesTr/{case_id}.nii.gz",
            "label": f"./labelsTr/{case_id}.nii.gz"
        })

        idx += 1

# ===============================
# 5. 更新 dataset.json
# ===============================
dataset["numTraining"] = len(dataset["training"])

# ===============================
# 6. 处理 test 数据（无 label）
# ===============================
test_centers = cfg.get("test", [])

for center in test_centers:
    center_name = center["name"]
    img_dir = center["image_path"]

    print(f"\n🧪 Processing TEST center: {center_name}")

    img_paths = sorted(glob(os.path.join(img_dir, "*.nii*")))

    for img_path in tqdm(img_paths, desc=f"TEST-{center_name}"):
        fname = os.path.basename(img_path)

        img = sitk.ReadImage(img_path)

        pid = fname.split(".")[0]
        case_id = f"{prefix}_{pid}"

        img_out = os.path.join(imagesTs, f"{case_id}{suffix}.nii.gz")
        sitk.WriteImage(img, img_out)

        dataset["test"].append(f"./imagesTs/{case_id}.nii.gz")

dataset["numTest"] = len(dataset["test"])

with open(os.path.join(task_dir, "dataset.json"), "w") as f:
    json.dump(dataset, f, indent=4)

print("\n✅ nnU-Net Pre-therapy 数据集整理完成！")
print(f"Total pre cases: {dataset['numTraining']}")
print(f"Output dir: {task_dir}")
