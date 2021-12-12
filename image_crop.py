import cv2
from matplotlib import pyplot as plt
from os import read
import pandas as pd
import numpy as np


def crop_YOLO_img(data_file_pandas,img_file,save_path,fx_set=1,fy_set=1 ):

    for i in range(len(data_file_pandas)):
        x= np.float32(data_file_pandas.loc[i][1]) * (img.shape[1])
        y= np.float32(data_file_pandas.loc[i][2]) * (img.shape[0])
        width =np.float32(data_file_pandas.loc[i][3]) * (img.shape[1])
        hight= np.float32(data_file_pandas.loc[i][4]) * (img.shape[0])

        x_ret_1 = int(x - width *0.55)
        x_ret_2 = int(x + width *0.55)
        y_ret_1 = int(y - hight *0.55)
        y_ret_2 = int(y + hight *0.55)


        dsize =(0,0) 
        crop_image = img_file[y_ret_1 : y_ret_2, x_ret_1:x_ret_2]
        disp_resize1 = cv2.resize(crop_image,dsize,fx=fx_set,fy=fy_set)
        # 저장파일 이름
        cv2.imwrite(save_path + "acid_1_" + str(i) + ".png",disp_resize1)


# YOLO 결과 데이터
position_path_file = "det/labels/acid_1.pdf0.txt"
# 이미지 데이터
image_path = "det/acid_1.pdf0.png"
# 저장 경로
image_save_path = "crop_img/"


img = cv2.imread(image_path,0) 
postion = open(position_path_file,"r")
postion = postion.read().split('\n')


postion_pd = pd.DataFrame(columns=["clss","x_center","y_center","width",'hight'])

for i in range(len(postion)-1):
    if postion[i].split(' ')[0] == "0":
        postion_pd.loc[i] = postion[i].split(' ')

postion_pd = postion_pd.reset_index(drop = True)



crop_YOLO_img(postion_pd,img,image_save_path)



