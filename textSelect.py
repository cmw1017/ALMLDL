
from os import read
import cv2
from matplotlib import pyplot as plt
import numpy as np
from pandas.core.indexes.base import Index
import pytesseract
from difflib import SequenceMatcher
import pandas as pd
import os

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"


# def textSelect(text_result,data_eq_ori,data_PNID_ori) :
#     text_result1_split = text_result.split('\n')
#     text_result1_split.sort(key=len)
    
#     data_list_eq=[]
#     data_list_PNID=[]
#     ranking_eq =[0]
#     ranking_PNID =[0]

#     for i, data in enumerate(text_result1_split):

#         data = data.replace(" ","")
#         print(data + 'data')

#         if data ==' ' or data =='\x0c':
#             pass
    
#         for j, data2 in enumerate(data_eq_ori):
#             "print(str(SequenceMatcher(None,data,data2).ratio()) + ' : ' + data2)"

#             if SequenceMatcher(None,data,data2).ratio() > ranking_eq[0]:
#                 data_list_eq = data2
#                 "print(data_list_eq + ' :  sub_eq')"
#                 ranking_eq[0] = SequenceMatcher(None,data,data2).ratio() 
   

#         for j, data2 in enumerate(data_PNID_ori):

#             "print(str(SequenceMatcher(None,data,data2).ratio()) + ' : ' + data2)"
#             if SequenceMatcher(None,data,data2).ratio() == 0:
#                 break
            
#             elif SequenceMatcher(None,data,data2).ratio() > ranking_PNID[0]:
                
#                 data_list_PNID = data2
                
#                 ranking_PNID[0] = SequenceMatcher(None,data,data2).ratio()
    
#     return data_list_eq,data_list_PNID
    
def GetText(image):
    dsize =(0,0) 
    disp_resize = cv2.resize(image,dsize,fx=3,fy=3)
    image_gauss = cv2.GaussianBlur(disp_resize,(5,5),0)

    sharpening_1 = np.array([[-1, -1, -1, -1, -1],
                            [-1, 2, 2, 2, -1],
                            [-1, 2, 9, 2, -1],
                            [-1, 2, 2, 2, -1],
                            [-1, -1, -1, -1, -1]])/9

    image_sharp = cv2.filter2D(image_gauss, -1, sharpening_1)

    image_gauss = cv2.GaussianBlur(image_sharp,(5,5),0)

    sharpening_2 = np.array([[-1, -1, -1, -1, -1],
                            [-1, 2, 2, 2, -1],
                            [-1, 2, 9, 2, -1],
                            [-1, 2, 2, 2, -1],
                            [-1, -1, -1, -1, -1]])/16

    image_sharp = cv2.filter2D(image_gauss, -1, sharpening_2)

    text_result = pytesseract.image_to_string(image_sharp)

    return text_result, image_sharp

def Data_to_csv(data_ori, path = os.getcwd()):
    for name in data_ori:
        data_ori[name].to_csv(path + "/" + name + ".csv", mode="w",index=False)

def text_modifing(input_text):
    replace_text = input_text.strip()
    replace_text = replace_text.replace("|", "")
    replace_text = replace_text.replace("—", "-")
    replace_text = replace_text.replace("--", "-")
    replace_text = replace_text.replace("\n", " ")
    return replace_text



# crop된 이미지가 저장되는 위치 
result_path = "data\\PNID\\PDFs\\results"
result_file_list = os.listdir(result_path)
result_pd = pd.DataFrame(columns=["PNID","items"])
num = 0
for i, name in enumerate(result_file_list):
    print(name)
    image_dir = os.getcwd() + "\\" + result_path + "\\" + name + "\\crops"
    file_list = os.listdir(image_dir)

    for i, crop_name in enumerate(file_list):
        # print(crop_name)
        img = cv2.imread(image_dir + "\\" + crop_name, 0)
        text_result1, image_sharp = GetText(img)
        if text_result1.strip() != "":
            replace_text = text_modifing(text_result1)
            print(replace_text)
            result_pd.loc[num] = [name, replace_text]
            num = num + 1

print(result_pd)
result_pd.to_csv(os.getcwd() + "\\" + result_path + "\\" + "result.csv", mode="w",index=False)
