# import pandas as pd
#
# # 读取CSV文件
# csv_file = './csv/all_ct_resample_nii_data.csv'
# df = pd.read_csv(csv_file)
#
# # 替换路径
# df = df.replace('/mnt/data2/zoule/dataset/prep_data/', '/data/zoule/dataset/', regex=True)
# df_cleaned = df.drop_duplicates(subset='pid', keep='first')
# # 保存替换后的CSV文件
# df_cleaned.to_csv('./csv/all_ct_resample_nii_data_new.csv', index=False)

#
# import pandas as pd
# from sklearn.model_selection import train_test_split
#
# # 读取CSV数据集
# df = pd.read_csv('./csv/all_ct_resample_nii_data_new.csv')
#
# # 按类别划分为pre_therapy和post_therapy
# pre_therapy_df = df[df['type'] == 'pre_therapy']
# post_therapy_df = df[df['type'] == 'post_therapy']
#
# # 按4:1比例划分pre_therapy的训练集和测试集
# pre_train, pre_test = train_test_split(pre_therapy_df, test_size=0.2, random_state=42)
#
# # 按4:1比例划分post_therapy的训练集和测试集
# post_train, post_test = train_test_split(post_therapy_df, test_size=0.2, random_state=42)
#
# # 提取pid列并保存为CSV文件
# pre_train[['pid']].to_csv('./csv/pre_therapy_train.csv', index=False)
# pre_test[['pid']].to_csv('./csv/pre_therapy_test.csv', index=False)
# post_train[['pid']].to_csv('./csv/post_therapy_train.csv', index=False)
# post_test[['pid']].to_csv('./csv/post_therapy_test.csv', index=False)

#
# import os
#
#
# def get_npy_files_size(directory):
#     # 遍历目录，获取所有 .npy 文件的大小
#     npy_files = [f for f in os.listdir(directory) if f.endswith('.npy')]
#
#     file_sizes = {}
#     for npy_file in npy_files:
#         file_path = os.path.join(directory, npy_file)
#         file_size = os.path.getsize(file_path)
#         file_sizes[npy_file] = file_size
#
#     return file_sizes
#
#
# # 指定目录
# directory = '/data/zoule/dataset/s4_nnunet_data/nnUNet_preprocessed/Task500_LungTumor500/nnUNetData_plans_v2.1_stage1/'
#
# # 获取目录下所有 .npy 文件的大小
# npy_file_sizes = get_npy_files_size(directory)
#
# # 打印文件大小信息
# for file_name, size in npy_file_sizes.items():
#     print(f"{file_name}: {size} bytes")

