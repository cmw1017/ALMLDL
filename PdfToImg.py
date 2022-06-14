import os
from pdf2image import convert_from_path 

# 여러개의 장으로 이루어진 PDF 파일을 모두 쪼개서 png 파일로 다시 저장
# PDF 파일이 있는 위치
PDF_dir = os.getcwd() + '/data/PNID/PDFs/origin'
file_list = os.listdir(PDF_dir)

# png 파일이 저장되는 위치
save_dir = os.getcwd() + '/data/PNID/PDFs/divided/'

# 메인 프로세스
for i, name in enumerate(file_list):
    print(name)
    pages = convert_from_path(PDF_dir+ '/' + name) 

    for j, page in enumerate(pages):
        name1 = name.split('.')[0]
        page.save(save_dir + name1 +'_' +str(j)+'.png', "PNG")


