# CRAFT를 이용한 텍스트 영역 추출
# github : https://github.com/fcakyon/craft-text-detector
# pip install craft-text-detector

from craft_text_detector import (
    read_image,
    load_craftnet_model,
    load_refinenet_model,
    get_prediction,
    export_detected_regions,
    export_extra_results,
    empty_cuda_cache
)
import cv2
import os

def detect_text_from_image(image_path, output_dir, save_file_name):
    # read image
    image = read_image(image_path)

    # load models
    refine_net = load_refinenet_model(cuda=True)
    craft_net = load_craftnet_model(cuda=True)

    # perform prediction
    prediction_result = get_prediction(
        image=image,
        craft_net=craft_net,
        refine_net=refine_net,
        text_threshold=0.7,
        link_threshold=0.4,
        low_text=0.4,
        cuda=True,
        long_size=1280
    )

    # # export detected text regions
    # exported_file_paths = export_detected_regions(
    #     image=image,
    #     regions=prediction_result["boxes"],
    #     output_dir=output_dir,
    #     rectify=True
    # )

    # export heatmap, detection points, box visualization
    export_extra_results(
        image=image,
        regions=prediction_result["boxes"],
        heatmaps=prediction_result["heatmaps"],
        file_name=save_file_name,
        output_dir=output_dir
    )

    # unload models from gpu
    empty_cuda_cache()

def save_img(image, posH_ori, posH, posW_ori, posW, output_dir, save_file_name):
    dsize =(0,0) 
    crop_image = image[posH_ori : posH, posW_ori:posW]
    disp_resize1 = cv2.resize(crop_image,dsize,fx=1,fy=1)
    # 저장파일 이름
    cv2.imwrite(output_dir + save_file_name + ".png", disp_resize1)

def draw_box(image, text_file_dir, text_file_name, H, W):
    position = open(text_file_dir + text_file_name,"r")
    position = position.read().split('\n')
    for i in range(len(position)):
        if len(position[i]) != 0:
            rect = position[i].split(',')
            posW_1 = rect[0]
            posH_1 = rect[1]
            posW_2 = rect[4]
            posH_2 = rect[5]
            image = cv2.rectangle(image, (int(posW_1) + W, int(posH_1) + H), (int(posW_2) + W, int(posH_2)+ H), (255,0,0), 2)
    return image

def image_save_detection(image_path, output_dir, save_file_name, division):
    image = cv2.imread(image_path)
    image_h, image_w, _ = image.shape
    posH = 0
    posH_ori = 0
    posW = 0
    posW_ori = 0
    for i in range(1, division+1):
        posW = 0
        posW_ori = 0
        if i == division:
            posH_ori = posH + 1
            posH = image_h
        elif i != 1:
            posH_ori = posH + 1
            posH += int(image_h / division)
        else:
            posH_ori = posH
            posH += int(image_h / division)
        for j in range(1, division+1):
            if j == division:
                posW_ori = posW + 1
                posW = image_w
            elif j != 1:
                posW_ori = posW + 1
                posW += int(image_w / division)
            else:
                posW_ori = posW
                posW += int(image_w / division)

            # print(posH_ori, posH, posW_ori, posW)
            # save crop image
            crop_output_dir = output_dir + "crops/"
            if not (os.path.isdir(crop_output_dir)):
                os.makedirs(os.path.join(crop_output_dir))
            mod_save_file_name = save_file_name + "(" + str(i) + "," + str(j) + ")"
            save_img(image, posH_ori, posH, posW_ori, posW, crop_output_dir, mod_save_file_name)

            # crop image detection
            detect_image_path = crop_output_dir + mod_save_file_name + ".png"
            detect_output_dir = output_dir + "detects/"
            if not (os.path.isdir(detect_output_dir)):
                os.makedirs(os.path.join(detect_output_dir))
            detect_text_from_image(detect_image_path, detect_output_dir, mod_save_file_name)

            # merge image detection
            print("(" + str(i) + "," + str(j) + ")", "H : " + str(posH), "/ W : " + str(posW))
            image = draw_box(image, detect_output_dir, mod_save_file_name + "_text_detection.txt", posH_ori, posW_ori)

    result_output_dir = output_dir + "result/"
    if not (os.path.isdir(result_output_dir)):
        os.makedirs(os.path.join(result_output_dir))
    cv2.imwrite(result_output_dir + save_file_name + "_division" + str(division) + ".png", image)


# 실행예시
image_path = 'data/PNID/input_datas/TSL_1_pdf12.png' # can be filepath, PIL image or numpy array
output_dir = 'data/PNID/result_datas/'
# detect_text_from_image(image_path, output_dir, "outputs")
image_save_detection(image_path, output_dir, "TSL_1_pdf12", 8)

