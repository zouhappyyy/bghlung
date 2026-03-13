import os
import json
import SimpleITK as sitk
from glob import glob
import shutil
import numpy as np
import yaml
from tqdm import tqdm
import pandas as pd

def lung_normalization(image_narray):
    MAX_BOUND = 400.0
    MIN_BOUND = -1000.0

    image_narray[image_narray > MAX_BOUND] = MAX_BOUND
    image_narray[image_narray < MIN_BOUND] = MIN_BOUND
    norm_array = (image_narray - MIN_BOUND)/(MAX_BOUND-MIN_BOUND)
    return norm_array

def find_duplicates(lst):
    duplicates = {}
    result = []

    for item in lst:
        if item in duplicates:
            duplicates[item] += 1
        else:
            duplicates[item] = 1

    for item, count in duplicates.items():
        if count > 1:
            result.append(item)

    return result


with open('config/data.yaml', 'r') as f:
    config = yaml.safe_load(f)

base = os.path.join(config['nnunet']['base'], 'nnUNet_raw_data', 'Task500_LungTumor500')

data_dict = {
    "name": "LungTumorSeg",
    "description": "",
    "author": "",
    "reference": "",
    "licence": "",
    "release": "",
    "tensorImageSize": "3D",
    "modality": {"0": "CT"},
    "labels": {"0": "background", "1": "LungTumor"},
    "numTraining": 22, "numTest": 4,
    "training": [],
    "test": []
}

imagesTr = os.path.join(base, 'imagesTr')
labelsTr = os.path.join(base, 'labelsTr')
imagesTs = os.path.join(base, 'imagesTs')

if not os.path.exists(imagesTr):
    os.makedirs(imagesTr)
if not os.path.exists(labelsTr):
    os.makedirs(labelsTr)
if not os.path.exists(imagesTs):
    os.makedirs(imagesTs)

# sup

all_ct_nii_csv = pd.read_csv(config['all_ct_resample_nii_csv_path'])
# all_id_list = all_ct_nii_csv['pid'].tolist()
# print(find_duplicates(all_id_list))
# assert False

df1 = pd.read_csv(config['test_id_pre_csv'])
df2 = pd.read_csv(config['test_id_post_csv'])
eval_id_list = list(df1['pid']) + list(df2['pid'])

print(eval_id_list,len(eval_id_list))
test_set = all_ct_nii_csv[all_ct_nii_csv['pid'].isin(eval_id_list)].reset_index()
train_set = all_ct_nii_csv[~all_ct_nii_csv['pid'].isin(eval_id_list)].reset_index()

train_set.to_csv('./csv/train_set.csv',index=False)
test_set.to_csv('./csv/test_set.csv',index=False)
print(len(test_set))


train_idx = 0
for pid_idx, current_row in tqdm(train_set.iterrows(),total=len(train_set),desc='train ct'):

    origin_img_path = current_row['ct_path']
    origin_label_path = current_row['mask_path']

    img = sitk.ReadImage(origin_img_path)
    mask = sitk.ReadImage(origin_label_path)

    if img.GetSize() != mask.GetSize():
        print('skip', origin_img_path)
        continue

    no_f = train_idx
    data_dict["training"].append({
        "image": "./imagesTr/lungtumor_{:04d}.nii.gz".format(no_f),
        "label": "./labelsTr/lungtumor_{:04d}.nii.gz".format(no_f),
    })

    target_img_path = os.path.join(base, "imagesTr/lungtumor_{:04d}_0000.nii.gz".format(no_f))
    target_label_path = os.path.join(base, "labelsTr/lungtumor_{:04d}.nii.gz".format(no_f))

    arr = sitk.GetArrayFromImage(img)
    arr = lung_normalization(arr)
    img_new = sitk.GetImageFromArray(arr)
    img_new.CopyInformation(img)
    sitk.WriteImage(img_new, target_img_path)


    #
    mask.SetOrigin(img_new.GetOrigin())
    sitk.WriteImage(mask, target_label_path)
    # shutil.copyfile(origin_label_path, target_label_path)

    train_idx += 1

test_idx = 0
for pid_idx, current_row in tqdm(test_set.iterrows(),total=len(test_set),desc='test ct'):
    no_f = test_idx
    no_f += 1000
    data_dict["test"].append(
        "./imagesTs/lungtumor_{:04d}.nii.gz".format(no_f)
    )
    origin_img_path = current_row['ct_path']
    target_img_path = os.path.join(base, "imagesTs/lungtumor_{:04d}_0000.nii.gz".format(no_f))

    img = sitk.ReadImage(origin_img_path)
    arr = sitk.GetArrayFromImage(img)
    arr = lung_normalization(arr)
    img_new = sitk.GetImageFromArray(arr)
    img_new.CopyInformation(img)
    sitk.WriteImage(img_new, target_img_path)

    test_idx += 1

with open(os.path.join(base, 'dataset.json'), 'w+') as f:
    f.write(json.dumps(data_dict))
