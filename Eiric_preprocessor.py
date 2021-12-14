# -*- coding: utf-8 -*-

import cv2
import os
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import pandas as pd
import csv


# 폴더 생성
def dir_create(dir_name):
    try:
        if not (os.path.isdir(dir_name)):
            os.makedirs(os.path.join(dir_name))
    except OSError as e:
        print("Failed to create directory!!!!!")
        raise


# 이미지 형식 변환
def reform_image(path1, path2, file_name):
    for name2 in enumerate(file_name):
        # path1이 origin_data의 path
        input_path3 = []
        input_path3.append(path1)
        input_path3.append("/{}".format(name2[1]))
        input_path3 = ''.join(input_path3)
        # path2가 trans_data의 path
        output_path3 = []
        output_path3.append(path2)
        output_path3.append(("/{}".format(name2[1]))[:-4])
        output_path3.append(".jpeg")
        output_path3 = ''.join(output_path3)
        print(input_path3)
        print(output_path3)
        img = cv2.imread(input_path3, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("test Image", img)
        # cv2.waitKey(0)
        # 이미지 윈도우 삭제
        # cv2.destroyAllWindows()
        # 이미지 다른 파일로 저장
        cv2.imwrite(output_path3, img)


# # original_data 폴더 리스트 출력(예제용!!)
# input_path1 = "./data/Eiric/original_data/"
# output_path1 = "./data/Eiric/train_data/"
# file_list = os.listdir(input_path1)


# # origin_data에서 파일을 가져와서 JPEG 파일로 변환하여 trans_data에 넣고 trans_data에서의 JPEG 파일을 경량화 하여 train_data에 넣기 위한 과정들
# # 1. 폴더 생성 로직
# for name in enumerate(file_list):
#     print("file list : %s" % name[1])
#     dir_name = "./data/Eiric/trans_data/{}".format(name[1])
#     print(dir_name)
#     dir_create(dir_name)

# # 2. 파일 변환 및 저장 로직
# error_num = 0
# for name in enumerate(file_list):
#     input_path2 = []
#     input_path2.append(input_path1)
#     input_path2.append("{}".format(name[1]))
#     input_path2 = ''.join(input_path2)
#     output_path2 = []
#     output_path2.append(output_path1)
#     output_path2.append("{}".format(name[1]))
#     output_path2 = ''.join(output_path2)
#     print(input_path2)
#     print(output_path2)
#     input_file_list = os.listdir(input_path2)
#     # print(input_file_list)
#     try:
#         reform_image(input_path2, output_path2, input_file_list)
#     except:
#         error_num += 1
# print(error_num)

# # 3. 파일 resize 및 저장(4와 같이 실행 불가)
# TrainType1은 128x128 사이즈이고 TrainType2는 32x32 사이즈
# file_list = os.listdir("./data/Eiric/TrainType2/trans_data")
# # 3-1  train_data 폴더 및 하위 폴더 생성
# dir_name = "./data/Eiric/TrainType2/train_data/"
# dir_create(dir_name)
# for name in enumerate(file_list):
#     print("file list : %s" % name[1])
#     dir_name = "./data/Eiric/TrainType2/train_data/{}".format(name[1])
#     print(dir_name)
#     dir_create(dir_name)
# # 3-2. 데이터 레이블 저장
# dataframe = pd.DataFrame(file_list)
# dataframe.to_csv("./data/Eiric/TrainType2/index.csv", header=False, index=True)
# trans = transforms.Compose([
#     transforms.Resize((32, 32))
# ])
# train_data = torchvision.datasets.ImageFolder(root='./data/Eiric/TrainType2/trans_data', transform=trans)
# for num, value in enumerate(train_data):
#     data, label = value
#     print(num, data, file_list[label])
#     data.save('./data/Eiric/TrainType2/train_data/%s/%d_%d.jpeg'%(file_list[label], num, label))


# 4. 필터 리스트에 있는 파일만 가져와서 resize 및 저장(3과 같이 실행 불가)
# TrainType3은 128x128 사이즈이고 TrainType4는 32x32 사이즈
file_list = os.listdir("./data/Eiric/TrainType4/trans_data")
# 4-1 파일 리스트의 파일 이름 가져오기
filter_file = open('./data/Eiric/TrainType4/filter.csv', 'r')
filter_csv = csv.reader(filter_file)
filter_list = []
for list in filter_csv:
    print("add to list (", list[0], ")")
    filter_list.append(list[0])
# 4-2-1 train_data 폴더 및 하위 폴더 생성(필터 리스트에 없으면 폴더를 생성하지 않음)
dir_name = "./data/Eiric/TrainType4/train_data/"
dir_create(dir_name)
for name in enumerate(file_list):
    if name[1] in filter_list:
        print("file list : %s" % name[1])
    else:
        continue
    dir_name = "./data/Eiric/TrainType4/train_data/{}".format(name[1])
    print(dir_name)
    dir_create(dir_name)
# 4-2-2 test_data 폴더 및 하위 폴더 생성(필터 리스트에 없으면 폴더를 생성하지 않음)
dir_name = "./data/Eiric/TrainType4/test_data/"
dir_create(dir_name)
for name in enumerate(file_list):
    if name[1] in filter_list:
        print("file list : %s" % name[1])
    else:
        continue
    dir_name = "./data/Eiric/TrainType4/test_data/{}".format(name[1])
    print(dir_name)
    dir_create(dir_name)
# 4-3. 데이터 저장(필터에 없으면 변환하지 않고 넘김)
trans = transforms.Compose([
    transforms.Resize((32, 32))
])
train_data = torchvision.datasets.ImageFolder(root='./data/Eiric/TrainType4/trans_data', transform=trans)
for num, value in enumerate(train_data):
    data, label = value
    if file_list[label] in filter_list:
        print("file list : %s" % file_list[label])
    else:
        continue
    if num % 4 == 0:
        print(num, data, file_list[label])
        data.save('./data/Eiric/TrainType4/test_data/%s/%d_%d.jpeg'%(file_list[label], num, label))
    else:
        print(num, data, file_list[label])
        data.save('./data/Eiric/TrainType4/train_data/%s/%d_%d.jpeg'%(file_list[label], num, label))
# 4-4. 인덱스 저장
file_list = os.listdir("./data/Eiric/TrainType4/train_data")
dataframe = pd.DataFrame(file_list)
dataframe.to_csv("./data/Eiric/TrainType4/index.csv", header=False, index=True)



# 유니코드 데이터 셋 출력
# out_df = pd.DataFrame(columns=['label', 'value'])
# out_df = out_df.set_index('label')
# for i in range(0xac00, 0xd7a4):
#     print(hex(i),'|', chr(i))
#     out_df.loc[hex(i)] = [chr(i)]
# print(out_df)
# out_df.to_csv("./data/Eiric/tot_index.csv", header=False, index=True)


# out_df = pd.DataFrame(columns=['label', 'value'])
# out_df = out_df.set_index('label')
# str = "한칸씩 분리할 텍스트 입력"
# str = list(str)
# for i in range(len(str)):
#     print(str[i])
#     out_df.loc[str[i]] = [","]
# out_df.to_csv("./data/Eiric/temp.csv", header=False, index=True)
