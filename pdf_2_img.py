import os
from pdf2image import convert_from_path 


class pdf_to_image():

    def __init__(self):
        self.__pdf_dir = []
        self.__save_dir = []
        self.process = []

    def Getpaths(self,image_dir,save_dir):
        
        self.__pdf_dir = image_dir
        self.__save_dir = save_dir

    def make_image(self):
        file_list = self.__pdf_dir
        save_path = self.__save_dir
        try :
            for i, data in enumerate(file_list):
                print(data)
                name = data.split('/')[-1]
                name = name.split('.')[0]
                pages = convert_from_path(data)

                for j, page in enumerate(pages):
                    page.save(save_path + '/' + name + '_' + str(j) + '.png', "PNG")
            return 1
        except:
            return 0


pdf_2_image = pdf_to_image()
pdf_2_image.Getpaths(os.getcwd() + '/data/PNID/PDFs/origin', os.getcwd() + '/data/PNID/PDFs/divided')
pdf_2_image.make_image()


            
        



