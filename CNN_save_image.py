import os
import torch
from torchvision.utils import save_image
import h5py

# 폴더 생성 함수
def dir_create(dir_name):
    try:
        if not (os.path.isdir(dir_name)):
            os.makedirs(os.path.join(dir_name))
    except OSError as e:
        print("Failed to create directory!!!!!")
        raise

def save_images(dir_path, HDF5_file_path):
    # 최상단 파일 생성
    dir_create(dir_path)
    file_object = h5py.File(HDF5_file_path, 'r')
    LV1_list = list(file_object.keys())
    for LV1_num in range(len(LV1_list)):
        LV1_group = file_object[LV1_list[LV1_num]]
        LV1_dir_path = dir_path + "/" + LV1_list[LV1_num]
        dir_create(LV1_dir_path)
        LV2_list = list(LV1_group.keys())
        for LV2_num in range(len(LV2_list)):
            LV2_group = LV1_group[LV2_list[LV2_num]]
            LV2_dir_path = LV1_dir_path + "/" + LV2_list[LV2_num]
            dir_create(LV2_dir_path)
            LV3_list = list(LV2_group.keys())
            for LV3_num in range(len(LV3_list)):
                img = LV2_group[LV3_list[LV3_num]]
                img = torch.FloatTensor(img) / 255
                img = img.transpose(0,2)
                img = img.transpose(1,2)
                LV3_file_path = LV2_dir_path + "/" + str(LV2_num) + "_" + str(LV3_num) +".jpeg"
                print(LV3_file_path)
                save_image(img, LV3_file_path)

dir_path = "./data/Eiric/TrainType4/test"
HDF5_fil_path = './data/Eiric/TrainType4/TrainType4_trans.h5'
save_images(dir_path, HDF5_fil_path)


