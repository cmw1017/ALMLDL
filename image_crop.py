import cv2
from matplotlib import pyplot as plt
from os import read
import os
import pandas as pd
import numpy as np


def crop_YOLO_img(data_file_pandas, img_file, save_path, img_name, padding=0.5,
    width_min=0, width_max=0, height_min=0, height_max=0, fx_set=1, fy_set=1):

    for i in range(len(data_file_pandas)):
        x= np.float32(data_file_pandas.loc[i][0]) * (img_file.shape[1])
        y= np.float32(data_file_pandas.loc[i][1]) * (img_file.shape[0])
        width = np.float32(data_file_pandas.loc[i][2]) * (img_file.shape[1])
        height = np.float32(data_file_pandas.loc[i][3]) * (img_file.shape[0])

        x_ret_1 = int(x - width * padding)
        x_ret_2 = int(x + width * padding)
        y_ret_1 = int(y - height * padding)
        y_ret_2 = int(y + height * padding)

        if width_min != 0 and width <= width_min:
            continue
        if width_max != 0 and width >= width_max:
            continue
        if height_min != 0 and height <= height_min:
            continue
        if height_max != 0 and height >= height_max:
            continue

        dsize =(0,0) 
        crop_image = img_file[y_ret_1 : y_ret_2, x_ret_1:x_ret_2]
        disp_resize1 = cv2.resize(crop_image,dsize,fx=fx_set,fy=fy_set)
        # 저장파일 이름
        cv2.imwrite(save_path + "\\" + img_name + "_" + str(i) + ".png", disp_resize1)

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory.' + directory)

# 이전 text_area_select의 결과가 저장된 폴더
select_dir = os.getcwd() + '\\data\PNID\PDFs\selected'

# png 이미지가 있는 폴더
png_dir = "data\\PNID\\PDFs\\divided"

# crop된 이미지가 저장되는 위치 
result_path = "data\\PNID\\PDFs\\results"
file_list = os.listdir(select_dir)
for i, name in enumerate(file_list):
    print(name)
    
    # 각종 경로 설정
    position_path_file = select_dir + "\\" + name + "\\modified\\" + name + "_modified.txt"
    image_path = png_dir + "\\" + name + ".png"
    createFolder(result_path + "\\" + name)
    image_save_path = result_path + "\\" + name + "\\crops"
    createFolder(image_save_path)

    # 이미지 불러오기
    img = cv2.imread(image_path, 0)

    # YOLO 형식 데이터를 불러와서 dataframe 형태로 저장
    postion = open(position_path_file,"r")
    postion = postion.read().split('\n')
    postion_pd = pd.DataFrame(columns=["x_center","y_center","width",'hight'])
    for i in range(len(postion)-1):
        postion_pd.loc[i] = postion[i].split(' ')
    postion_pd = postion_pd.reset_index(drop = True)
    crop_YOLO_img(postion_pd, img, image_save_path, name, 0.6, 50, 150)




