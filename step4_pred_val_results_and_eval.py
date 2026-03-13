#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# # # # # # # # # # # # # # # # # # # # # # # #
# @Author: ZhuangYuZhou
# @E-mail: 605540375@qq.com
# @Desc:
# # # # # # # # # # # # # # # # # # # # # # # #

import cv2
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import SimpleITK as sitk
import shutil
from scipy.ndimage import label
from tqdm import tqdm
import pandas as pd
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.lung_prep.preprocess import get_volume_orientation, resampleDCM, reorient, lung_normalization,padding_array_by_3D_cropping_window,reverse_padding_array_to_3D_origin_array
from nnunet.inference.predict import predict_from_folder
from lib.eval.lung_tumor_metric import lung_tumor_metric

from skimage import measure
def post_process(input, cal_class_list):
    """
    分割结果的后处理：保留最大连通区域:largest connected component
    :param input: whole volume （D, W, H）
    :return: （D, W, H）
    """
    num_organ = len(cal_class_list)
    s = input.shape
    output = np.zeros((num_organ+1, s[0], s[1], s[2]))
    for id in range(1, num_organ+1):
        org_seg = (input == id) + .0
        if (org_seg == .0).all():
            continue
        labels, num = measure.label(org_seg, return_num=True)
        regions = measure.regionprops(labels)
        regions_area = [regions[i].area for i in range(num)]

        # omit_region_id = []
        # for rid, area in enumerate(regions_area):
        #     if area < 0.1 * organs_size[organs_index[id - 1]]:  # 记录区域面积小于10%*器官尺寸的区域的id
        #         omit_region_id.append(rid)
        # for idx in omit_region_id:
        #     org_seg[labels == (idx+1)] = 0

        region_num = regions_area.index(max(regions_area)) + 1  # 记录面积最大的区域，不会计算background(0)
        org_seg[labels == region_num] = 1
        org_seg[labels != region_num] = 0

        output[id, :, :, :] = org_seg

    return np.argmax(output, axis=0)

def draw_img_list_to_one_png(img_list, save_path='result.png', small_pic_size=(128, 128)):
    # 定义显示的行数
    row = 20
    # 定义显示的列数
    col = 20

    assert len(img_list) < row * col

    # 生成一张空白大图
    big_picture = np.full((int(row * small_pic_size[0]), int(col * small_pic_size[1]), 3), 255, dtype=np.uint8)

    for slice_idx, pic in enumerate(img_list):
        block_x = int(small_pic_size[0] * (slice_idx // row))
        block_y = int(small_pic_size[1] * (slice_idx % col))
        current_resize_pic = cv2.resize(pic, dsize=small_pic_size)
        current_resize_pic = cv2.putText(current_resize_pic, 'Slice:%s' % (slice_idx), (10, 10),
                                         cv2.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255))
        current_resize_pic[0, :, :] = 255
        current_resize_pic[-1, :, :] = 255
        current_resize_pic[:, 0, :] = 255
        current_resize_pic[:, -1, :] = 255
        big_picture[block_x:block_x + small_pic_size[0], block_y:block_y + small_pic_size[1], :] = current_resize_pic
    # cv2.imshow('big',big_picture)
    # cv2.waitKey()
    cv2.imwrite(save_path, cv2.cvtColor(big_picture, cv2.COLOR_BGR2RGB))

def define_color_masks(image, label_list=[1]):
    color_dict = {'tumor': [255, 0, 0]}

    label_to_region_dict = {'1': 'tumor'}

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for idx, label in enumerate(label_list):
        b[image == label], g[image == label], r[image == label] = color_dict[label_to_region_dict[str(label)]]

    # # PIL
    # r[image == label], g[image == label], b[image == label] = colors[idx]
    # opencv
    coloured_mask = np.stack([b, g, r], axis=2)

    # # PIL
    # coloured_mask = np.stack([r, g, b], axis=2)

    return coloured_mask

current_time = 'split_train_and_test'

test_data_csv = pd.read_csv('./csv/test_set.csv')


model_name = 'BGHNetV4'
model_dir = './ckpt/nnUNet/3d_fullres/Task500_LungTumor500/BGHNetV4Trainer__nnUNetPlansv2.1/'

spacing = [1.0, 1.0, 2.0]
folds = ['all']
# folds = [0]
patch_size = (80, 160, 160)

# lung_mask_output_dir = './pred_eval_data_mask_and_metric_%s/%s_folds_%s/'%(current_time, model_name, len(folds))
lung_mask_output_dir = './pred_eval_data_mask_and_metric_BAPCSwinNextTrainerV4/BAPCSwinNextTrainerV4_folds_all/'
lung_vis_mask_output_dir = os.path.join(lung_mask_output_dir,'vis_imgs')

if not os.path.exists(lung_mask_output_dir): os.makedirs(lung_mask_output_dir)
if not os.path.exists(lung_vis_mask_output_dir): os.makedirs(lung_vis_mask_output_dir)

lung_mask_eval_csv_path = os.path.join(lung_mask_output_dir, 'eval_metric.csv')
lung_mask_eval_data_list = []

output_dir = os.path.join(lung_mask_output_dir, 'pred_results')
if not os.path.exists(output_dir): os.makedirs(output_dir)
tmp_dir = os.path.join(lung_mask_output_dir,'infer_tmp')
if not os.path.exists(tmp_dir): os.makedirs(tmp_dir)


tmp_in_dir = os.path.join(tmp_dir, 'Task500_LungTumor500/imagesTs')
tmp_out_dir = os.path.join(tmp_dir, 'output_500')

if not os.path.exists(output_dir): os.makedirs(output_dir)
if not os.path.exists(tmp_in_dir): os.makedirs(tmp_in_dir)


for case_idx, current_row in tqdm(test_data_csv.iterrows(),total=len(test_data_csv), desc='pred test CT'):
    # if case_idx > 1:
    #     break

    current_patient_id = current_row['pid']
    current_lung_type = current_row['type']
    current_input_ct_path = current_row['ct_path']
    current_input_mask_path = current_row['mask_path']
    mask_file_name = os.path.basename(current_input_mask_path)

    assert os.path.exists(current_input_ct_path) and os.path.exists(current_input_mask_path)

    print('>>>' * 30)


    current_output_mask_path = os.path.join(output_dir, mask_file_name)

    if not os.path.exists(current_output_mask_path):
        # print('skip',current_output_mask_path)
        # continue

        print('pred',current_input_ct_path)
        ct_img = sitk.ReadImage(current_input_ct_path)

        # if case_idx > 0:
        #     assert False
        ##
        # your logic here. Below we do binary thresholding as a demo
        ##
        """ preprocess """
        ### Step 1: resampling
        ct_img_resample, _, _ = resampleDCM(ct_img, new_spacing=spacing, is_label=False)

        ### Step 2: norm
        ct_img_origin_arr = sitk.GetArrayFromImage(ct_img_resample)

        ct_img_norm_origin_arr = lung_normalization(ct_img_origin_arr)

        # else:
        ct_img_norm_pad_arr, ct_img_norm_arr_pad_list = padding_array_by_3D_cropping_window(
            ct_img_norm_origin_arr,
            crop_size=patch_size,
            is_sample=True,
            constant_values=0)

        if np.min(ct_img_norm_pad_arr) == np.max(ct_img_norm_pad_arr) == 0: assert False



        # scale to [0,1], norm
        # real t2 1500
        ct_img_crop = sitk.GetImageFromArray(ct_img_norm_pad_arr)
        ct_img_crop.SetSpacing(ct_img_resample.GetSpacing())
        sitk.WriteImage(ct_img_crop, os.path.join(tmp_in_dir, 'cmda_0001_0000.nii.gz'))

        """ segmentation """
        if os.path.exists(tmp_out_dir):
            shutil.rmtree(tmp_out_dir)
        if not os.path.exists(tmp_out_dir):
            os.makedirs(tmp_out_dir)

        # nnunet
        predict_from_folder(model=model_dir,
                            input_folder=tmp_in_dir,
                            output_folder=tmp_out_dir,
                            folds=folds,
                            save_npz=False,
                            num_threads_preprocessing=2,
                            num_threads_nifti_save=2,
                            lowres_segmentations=None,
                            part_id=0,
                            num_parts=1,
                            tta=True,
                            overwrite_existing=False,
                            mode='normal',
                            overwrite_all_in_gpu=None,
                            mixed_precision=True,
                            step_size=0.5,
                            # checkpoint_name='model_best',
                            checkpoint_name='model_final_checkpoint'
                            )

        # post-process
        print('post-process')
        current_prediction_volume = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(tmp_out_dir, 'cmda_0001.nii.gz')))


        ct_pred_mask_origin_arr= reverse_padding_array_to_3D_origin_array(current_prediction_volume,
                                                                                  ct_img_norm_arr_pad_list)

        # post-process
        ct_pred_mask_origin_arr = post_process(input=ct_pred_mask_origin_arr, cal_class_list=[1])

        assert ct_pred_mask_origin_arr.shape == ct_img_norm_origin_arr.shape == ct_img_origin_arr.shape

        # visual preds
        slice_num = ct_pred_mask_origin_arr.shape[0]
        current_patient_ct_255 = np.uint8(ct_img_norm_origin_arr * 255)

        current_patient_visual_ct_and_mask_img_list = []
        for current_slice_idx in range(slice_num):
            # visual
            # ct img
            ct_uint255_img = cv2.cvtColor(current_patient_ct_255[current_slice_idx], cv2.COLOR_GRAY2BGR)
            current_mask_centre_axial_img = ct_pred_mask_origin_arr[current_slice_idx]

            lbvs = np.unique(current_mask_centre_axial_img)

            if len(lbvs) == 1:
                continue

            # color mask
            centre_axial_rgb_mask_img = define_color_masks(current_mask_centre_axial_img, label_list=[1])
            # ct+mask
            seg_axial_mask_img = cv2.addWeighted(ct_uint255_img, 1, centre_axial_rgb_mask_img, 0.9, 0)

            current_patient_visual_ct_and_mask_img_list.append(seg_axial_mask_img)

        # save visual result
        draw_img_list_to_one_png(img_list=current_patient_visual_ct_and_mask_img_list,
                                 save_path=os.path.join(lung_vis_mask_output_dir, '%s_t2_and_mask.png' % (
                                 current_patient_id)))




        result = sitk.GetImageFromArray(ct_pred_mask_origin_arr)
        result.CopyInformation(ct_img_resample)

        result = resampleDCM(result, ct_img.GetSpacing(), is_label=True, new_size=ct_img.GetSize(), new_origin=ct_img.GetOrigin())[0]
        result.CopyInformation(ct_img)
        result = sitk.Cast(result, sitk.sitkInt32)
        sitk.WriteImage(result, current_output_mask_path)


    # eval
    print('eval!')
    # Numpy arrays
    gt_array = nib.load(current_input_mask_path).get_fdata().astype(np.int32)
    pred_array = nib.load(current_output_mask_path).get_fdata().astype(np.int32)

    # Voxel spacing
    affine = nib.load(current_input_mask_path).affine
    vxlspacing = [abs(affine[k, k]) for k in range(3)]

    current_dsc_dict, current_sens_dict, current_ppv_dict, current_assd_dict = lung_tumor_metric(gt=gt_array, pred=pred_array, vxlspacing=vxlspacing)

    current_dsc = round(current_dsc_dict['LungTumor'],4)
    current_sens = round(current_sens_dict['LungTumor'], 4)
    current_ppv = round(current_ppv_dict['LungTumor'], 4)
    current_assd = round(current_assd_dict['LungTumor'], 4)

    print('patient_id', current_patient_id, 'current_dsc', str(current_dsc))

    lung_mask_eval_data_list.append([current_patient_id, current_lung_type, current_dsc, current_sens, current_ppv, current_assd])
    print([current_patient_id, current_lung_type, current_dsc, current_sens, current_ppv, current_assd], '============')


lung_mask_eval_csv = pd.DataFrame(columns=['patient_id', 'type', 'DSC', 'SENS', 'PPV', 'ASSD'],data=lung_mask_eval_data_list)
lung_mask_eval_csv.to_csv(lung_mask_eval_csv_path, index=False)

lung_mask_eval_text = open(os.path.join(lung_mask_output_dir, 'eval_mean_metric.txt'), mode='w+', encoding='utf-8')
lung_mask_eval_text.write('Mean DSC: %.4f+%.4f, Mean SENS: %.4f+%.4f, Mean PPV: %.4f+%.4f, Mean ASSD: %.4f+%.4f'%(lung_mask_eval_csv['DSC'].mean(),lung_mask_eval_csv['DSC'].std(),
                                                                              lung_mask_eval_csv['SENS'].mean(),lung_mask_eval_csv['SENS'].std(),
                                                                              lung_mask_eval_csv['PPV'].mean(),lung_mask_eval_csv['PPV'].std(),
                                                                              lung_mask_eval_csv['ASSD'].mean(),lung_mask_eval_csv['ASSD'].std()))
lung_mask_eval_text.close()