import enum
from tkinter.tix import Tree
from turtle import pos
import cv2
import os
import matplotlib as plt
from matplotlib.pyplot import get
from numpy import imag
import pandas as pd



def cal_IOU_x(now,plus):
    x1_now = now['x_pos'] - now['width']*0.6
    x2_now = now['x_pos'] + now['width']*0.6
    y1_now = now['y_pos'] - now['hight']*0.51
    y2_now = now['y_pos'] + now['hight']*0.51

    x1_plus = plus['x_pos'] - plus['width']*0.6
    x2_plus = plus['x_pos'] + plus['width']*0.6
    y1_plus = plus['y_pos'] - plus['hight']*0.51
    y2_plus = plus['y_pos'] + plus['hight']*0.51

    x1 = max(x1_now,x1_plus)
    y1 = max(y1_now,y1_plus)
    x2 = min(x2_now,x2_plus)
    y2 = min(y2_now,y2_plus)

    del_x = (x2-x1)
    del_y = (y2-y1)
    resize_x1 = 0
    resize_y1 = 0
    resize_x2 = 0
    resize_y2 = 0

    if del_x >0 and del_y > 0:

        resize_x1 = min(x1_now,x1_plus)
        resize_y1 = min(y1_now,y1_plus)
        resize_x2 = max(x2_now,x2_plus)
        resize_y2 = max(y2_now,y2_plus)

        return del_x, del_y, resize_x1, resize_y1, resize_x2, resize_y2
    else:
        return del_x, del_y, resize_x1, resize_y1, resize_x2, resize_y2

def cal_IOU_y(now,plus):
    x1_now = now['x_pos'] - now['width']*0.51
    x2_now = now['x_pos'] + now['width']*0.51
    y1_now = now['y_pos'] - now['hight']*0.6
    y2_now = now['y_pos'] + now['hight']*0.6

    x1_plus = plus['x_pos'] - plus['width']*0.51
    x2_plus = plus['x_pos'] + plus['width']*0.51
    y1_plus = plus['y_pos'] - plus['hight']*0.6
    y2_plus = plus['y_pos'] + plus['hight']*0.6

    x1 = max(x1_now,x1_plus)
    y1 = max(y1_now,y1_plus)
    x2 = min(x2_now,x2_plus)
    y2 = min(y2_now,y2_plus)

    del_x = (x2-x1)
    del_y = (y2-y1)
    resize_x1 = 0
    resize_y1 = 0
    resize_x2 = 0
    resize_y2 = 0

    if del_x >0 and del_y > 0:

        resize_x1 = min(x1_now,x1_plus)
        resize_y1 = min(y1_now,y1_plus)
        resize_x2 = max(x2_now,x2_plus)
        resize_y2 = max(y2_now,y2_plus)

        return del_x, del_y, resize_x1, resize_y1, resize_x2, resize_y2
    else:
        return del_x, del_y, resize_x1, resize_y1, resize_x2, resize_y2
       
def Change_small2Yolo(x1,x2,y1,y2,pic_x_pos,pic_y_pos,division,image):

    width = (abs(int(x1)-int(x2))/image.shape[1])
    width = round(width,5)
    hight = (abs(int(y1)-int(y2))/image.shape[0])
    hight = round(hight,5)
    y_pos = (1/division) * pic_y_pos + (((int(y1)+int(y2)) * 0.5)/image.shape[0])
    y_pos = round(y_pos,5)
    x_pos = (1/division) * pic_x_pos + (((int(x1)+int(x2)) * 0.5)/image.shape[1])
    x_pos = round(x_pos,5)

    return y_pos, x_pos, hight, width


def Change_big2Yolo(x1,x2,y1,y2,image):

    width = (abs(int(x1)-int(x2))/image.shape[1])
    width = round(width,5)
    hight = (abs(int(y1)-int(y2))/image.shape[0])
    hight = round(hight,5)
    y_pos = (((int(y1)+int(y2)) * 0.5)/image.shape[0])
    y_pos = round(y_pos,5)
    x_pos = (((int(x1)+int(x2)) * 0.5)/image.shape[1])
    x_pos = round(x_pos,5)

    return y_pos, x_pos, hight, width


# text_area_select에서 나눠진 이미지 영역을 곂치는 부분을 합쳐서 출력
def merge_selected_img(png_dir, detect_dir, file_name, save_dir, division):
    file_list = os.listdir(detect_dir)
    img = cv2.imread(png_dir + "\\" + file_name + ".png")



    text_list =[]
    for i in file_list:
        if i.split(".")[-1] == "txt":
            data = open(detect_dir + "\\" + i, 'r')
            read_data =[]
            for j in data:
                read_data.append(j.strip())
            data.close()
            text_list.append(read_data)


    yolo_format =[]
    for num,i in enumerate(text_list):
        if len(i)> 0:
            for j in i:
                if len(j)>0:
                    count_y, count_x = divmod(num,division)
                    pos_pic= j.split(',')

                    y_pos, x_pos,hight,width = Change_small2Yolo(pos_pic[0],pos_pic[2],pos_pic[3],pos_pic[5],count_x,count_y,division,img)
                    
                    if ((hight *img.shape[0])*(width*img.shape[1])) >110:
                        yolo_format.append([y_pos, x_pos,hight,width , count_y,count_x])


    yolo_format_dataframe = pd.DataFrame(yolo_format,columns=['y_pos', 'x_pos','hight','width' , 'count_y','count_x'])

    yolo_format_dataframe_remained = yolo_format_dataframe.__deepcopy__()

    dataframe_update_list =[]

    dataframe_del_list =[]



    for y in range(0,division):
        for x in range(0,division):
            condition1 = (yolo_format_dataframe['count_x'] == x )
            condition2 = (yolo_format_dataframe['count_y'] == y )

            condition_x = (yolo_format_dataframe['count_x'] == x+1 )
            condition_y = (yolo_format_dataframe['count_y'] == y+1 )

            now_values = yolo_format_dataframe.loc[condition1&condition2]
            if len(now_values) > 0:
                x_plus_val = yolo_format_dataframe.loc[condition_x & condition2]
                y_plus_val = yolo_format_dataframe.loc[condition1 & condition_y]

                for num1, now in now_values.iterrows():
                    for num2, x_plus in x_plus_val.iterrows():

                        del_x,del_y,x1,y1,x2,y2= cal_IOU_x(now,x_plus)

                        if del_x >0 and del_y >0:
                            x1 = x1 *img.shape[1]
                            x2 = x2 *img.shape[1]
                            y1 = y1 *img.shape[0]
                            y2 = y2 *img.shape[0]
                            y_pos, x_pos, hight, width = Change_big2Yolo(x1,x2,y1,y2,img)
                            dataframe_del_list.append(now.tolist())
                            dataframe_del_list.append(x_plus.tolist())
                            dataframe_update_list.append([y_pos, x_pos,hight,width])



                    for num2, y_plus in y_plus_val.iterrows():

                        del_x,del_y,x1,y1,x2,y2= cal_IOU_y(now,y_plus)

                        if del_x >0 and del_y >0:
                            x1 = x1 *img.shape[1]
                            x2 = x2 *img.shape[1]
                            y1 = y1 *img.shape[0]
                            y2 = y2 *img.shape[0]
                            y_pos, x_pos,hight,width = Change_big2Yolo(x1,x2,y1,y2,img)
                            dataframe_del_list.append(now.tolist())
                            dataframe_del_list.append(y_plus.tolist())
                            dataframe_update_list.append([y_pos, x_pos,hight,width])


    yolo_format_dataframe_update = pd.DataFrame(dataframe_update_list,columns=['y_pos', 'x_pos','hight','width'])


    for num,data_fram in yolo_format_dataframe_remained.iterrows():
        for data_eq in dataframe_del_list:
            if data_fram.eq(data_eq).all() == True:
                try :
                    yolo_format_dataframe_remained = yolo_format_dataframe_remained.drop([num],axis=0)  
                except:
                    pass

    result_text = ""

    for num1 , update in yolo_format_dataframe_remained.iterrows():

        result_text = result_text + str(update['x_pos']) + " " + str(update['y_pos']) + " " + str(update['width']) + " " + str(update['hight']) + "\n"
        x1 = update['x_pos'] - update['width']*0.5
        x2 = update['x_pos'] + update['width']*0.5
        y1 = update['y_pos'] - update['hight']*0.5
        y2 = update['y_pos'] + update['hight']*0.5

        x1= int(x1*img.shape[1])
        x2= int(x2*img.shape[1])
        y1= int(y1*img.shape[0])
        y2= int(y2*img.shape[0])

        cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,0),2)

    for num1 , update in yolo_format_dataframe_update.iterrows():

        result_text = result_text + str(update['x_pos']) + " " + str(update['y_pos']) + " " + str(update['width']) + " " + str(update['hight']) + "\n"
        x1 = update['x_pos'] - update['width']*0.5
        x2 = update['x_pos'] + update['width']*0.5
        y1 = update['y_pos'] - update['hight']*0.5
        y2 = update['y_pos'] + update['hight']*0.5

        x1= int(x1*img.shape[1])
        x2= int(x2*img.shape[1])
        y1= int(y1*img.shape[0])
        y2= int(y2*img.shape[0])

        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)

    with open(save_dir + "\\" + file_name + "_modified.txt", "w") as file:
        file.write(result_text)
        file.close()

    resize1 = cv2.resize(img,(0,0),fx=0.6,fy=0.5)
    # cv2.imshow('123',resize1)
    # cv2.waitKey(0)
    cv2.imwrite(save_dir + "\\" + file_name + "_modified.png", img)

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
file_list = os.listdir(select_dir)
for i, name in enumerate(file_list):
    print(name)
    detect_dir = select_dir + "\\" + name + "\detects"
    file_name = name
    save_dir = select_dir + "\\" + name + "\modified"
    createFolder(save_dir)
    merge_selected_img(png_dir, detect_dir, file_name, save_dir, 8)




'''
yolo_format_dataframe_del= pd.DataFrame(dataframe_del_list,columns=['y_pos', 'x_pos','hight','width'])

yolo_format_dataframe = yolo_format_dataframe[['y_pos', 'x_pos','hight','width']]
'''