'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-05-04 08:48:36
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-07-11 16:38:38
FilePath: \tfservingconvert\tif_data\images_spiltter.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import cv2
import numpy as np
import json
from json import  dumps
import base64
# from labelme import utils
import PIL.Image
# from labelme.logger import logger
import io
import os.path as osp
import xml.etree.ElementTree as ET
import glob
from xml.dom.minidom import Document

# from labelme import PY2
# from labelme import QT4
"""
tif拆图
"""

# def load_image_file(filename):
#     try:
#         image_pil = PIL.Image.open(filename)
#     except IOError:
#         logger.error("Failed opening image file: {}".format(filename))
#         return

#     # apply orientation to image according to exif
#     image_pil = utils.apply_exif_orientation(image_pil)

#     with io.BytesIO() as f:
#         ext = osp.splitext(filename)[1].lower()
#         if PY2 and QT4:
#             format = "PNG"
#         elif ext in [".jpg", ".jpeg"]:
#             format = "JPEG"
#         else:
#             format = "PNG"
#         image_pil.save(f, format=format)
#         f.seek(0)
#         return f.read()

class CreateAnno:
    def __init__(self, ):
        self.doc = Document()  # 创建DOM文档对象
        self.anno = self.doc.createElement('annotation')  # 创建根元素
        self.doc.appendChild(self.anno)

        self.add_folder()
        self.add_path()
        self.add_source()
        self.add_segmented()

        # self.add_filename()
        # self.add_pic_size(width_text_str=str(width), height_text_str=str(height), depth_text_str=str(depth))

    def add_folder(self, floder_text_str='JPEGImages'):
        floder = self.doc.createElement('floder')  ##建立自己的开头
        floder_text = self.doc.createTextNode(floder_text_str)  ##建立自己的文本信息
        floder.appendChild(floder_text)  ##自己的内容
        self.anno.appendChild(floder)

    def add_filename(self, filename_text_str='00000.jpg'):
        filename = self.doc.createElement('filename')
        filename_text = self.doc.createTextNode(filename_text_str)
        filename.appendChild(filename_text)
        self.anno.appendChild(filename)

    def add_path(self, path_text_str="None"):
        path = self.doc.createElement('path')
        path_text = self.doc.createTextNode(path_text_str)
        path.appendChild(path_text)
        self.anno.appendChild(path)

    def add_source(self, database_text_str="Unknow"):
        source = self.doc.createElement('source')
        database = self.doc.createElement('database')
        database_text = self.doc.createTextNode(database_text_str)  # 元素内容写入
        database.appendChild(database_text)
        source.appendChild(database)
        self.anno.appendChild(source)

    def add_pic_size(self, width_text_str="0", height_text_str="0", depth_text_str="3"):
        size = self.doc.createElement('size')
        width = self.doc.createElement('width')
        width_text = self.doc.createTextNode(width_text_str)  # 元素内容写入
        width.appendChild(width_text)
        size.appendChild(width)

        height = self.doc.createElement('height')
        height_text = self.doc.createTextNode(height_text_str)
        height.appendChild(height_text)
        size.appendChild(height)

        depth = self.doc.createElement('depth')
        depth_text = self.doc.createTextNode(depth_text_str)
        depth.appendChild(depth_text)
        size.appendChild(depth)

        self.anno.appendChild(size)

    def add_segmented(self, segmented_text_str="0"):
        segmented = self.doc.createElement('segmented')
        segmented_text = self.doc.createTextNode(segmented_text_str)
        segmented.appendChild(segmented_text)
        self.anno.appendChild(segmented)

    def add_object(self,
                   name_text_str="None",
                   xmin_text_str="0",
                   ymin_text_str="0",
                   xmax_text_str="0",
                   ymax_text_str="0",
                   pose_text_str="Unspecified",
                   truncated_text_str="0",
                   difficult_text_str="0"):
        object = self.doc.createElement('object')
        name = self.doc.createElement('name')
        name_text = self.doc.createTextNode(name_text_str)
        name.appendChild(name_text)
        object.appendChild(name)

        pose = self.doc.createElement('pose')
        pose_text = self.doc.createTextNode(pose_text_str)
        pose.appendChild(pose_text)
        object.appendChild(pose)

        truncated = self.doc.createElement('truncated')
        truncated_text = self.doc.createTextNode(truncated_text_str)
        truncated.appendChild(truncated_text)
        object.appendChild(truncated)

        difficult = self.doc.createElement('Difficult')
        difficult_text = self.doc.createTextNode(difficult_text_str)
        difficult.appendChild(difficult_text)
        object.appendChild(difficult)

        bndbox = self.doc.createElement('bndbox')
        xmin = self.doc.createElement('xmin')
        xmin_text = self.doc.createTextNode(xmin_text_str)
        xmin.appendChild(xmin_text)
        bndbox.appendChild(xmin)

        ymin = self.doc.createElement('ymin')
        ymin_text = self.doc.createTextNode(ymin_text_str)
        ymin.appendChild(ymin_text)
        bndbox.appendChild(ymin)

        xmax = self.doc.createElement('xmax')
        xmax_text = self.doc.createTextNode(xmax_text_str)
        xmax.appendChild(xmax_text)
        bndbox.appendChild(xmax)

        ymax = self.doc.createElement('ymax')
        ymax_text = self.doc.createTextNode(ymax_text_str)
        ymax.appendChild(ymax_text)
        bndbox.appendChild(ymax)
        object.appendChild(bndbox)

        self.anno.appendChild(object)

    def get_anno(self):
        return self.anno

    def get_doc(self):
        return self.doc

    def save_doc(self, save_path):
        with open(save_path, "w") as f:
            self.doc.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')



def get_json_shapes(path_json):
    with open(path_json,'r',encoding="utf_8") as path_json:
        jsonx=json.load(path_json)
        shape_polygons = jsonx['shapes']
        print(shape_polygons)
        print(len(shape_polygons))
    return shape_polygons
        

def splitAndScale(img_path,imageScale,subimageScale,xml_path,targetdir,resize800=False):
    subimageWidth = 640 / subimageScale
    subimageHeight = 640 / subimageScale
      
    im = cv2.imread(img_path)
    new_w = im.shape[1] / imageScale
    new_h = im.shape[0] / imageScale
    im_ = im.copy()
    
    # shape_polygons = get_json_shapes(labelmejsondir)
    tree = ET.parse(xml_path)
    imgPath = tree.findall("path")[0].text
    objects = tree.findall("object")
    

    for y in range(int(subimageHeight /2) ,int(im.shape[0] - subimageHeight / 2),int(subimageHeight)):
        for x in range(int(subimageWidth /2) ,int(im.shape[1] - subimageWidth / 2),int(subimageWidth)):
            res = cv2.getRectSubPix(im_,(int(subimageWidth),int(subimageHeight)),(x,y))
           
            im_file = f"{targetdir}/{os.path.basename(img_path).split('.')[0]}_yc{int(y-subimageHeight/2)}_xc{int(x-subimageWidth/2)}.jpg"
            x1,y1,x2,y2 = x-subimageWidth/2,y-subimageHeight/2,x+subimageWidth/2,y+subimageHeight/2
            print(f"imfile:{im_file}")
            bbox = []
            for i in range(len(objects)):
                # bndbox_ = objects[i].find("bndbox")[0].text
                box_ = [int(objects[i].find("bndbox")[x].text) for x in range(4)]
                box_flag = np.array(box_)
                
                box_.append(objects[i].find("name").text)
            # for shape in shape_polygons:
            #     xy=np.array(shape['points'])
            #     a,b = np.max(xy[:,0]),np.max(xy[:,1])
                
                if box_flag[2]>x1 and box_flag[2]<x2:
                    if box_flag[3]>y1 and box_flag[3]<y2:
                        
                        box_[0] = box_[0]-int(x1)
                        box_[1] = box_[1]-int(y1)
                        box_[2] = box_[2]-int(x1)
                        box_[3] = box_[3]-int(y1)
                        
                        # if box_[0] <=0:
                        #     box_[0] = 1
                        # if box_[1] <=0:
                        #     box_[1] = 1
                        
                        # if box_[1] <=0 or box_[1] <=0:
                        #     continue
                        # if resize800:
                        #     res = cv2.resize(res,(int(subimageWidth/2),int(subimageHeight/2)))
                        #     if len(box_flag[box_flag<0])==0:
                        #         bbox.append(box_/2)
                        # else:
                        print(i," : ",box_)
                        flag = np.array([box_[0],box_[1],box_[2],box_[3]])
                        if len(flag[flag<=0])==0:
                            bbox.append(box_)
                            
            print("============================================================")
            if len(bbox)>1:
                cv2.imwrite(im_file,res)
                
                xml_anno = CreateAnno()
                xml_anno.add_filename(os.path.basename(im_file))
                xml_anno.add_pic_size(width_text_str=str(res.shape[1]), height_text_str=str(res.shape[0]), depth_text_str=str(3))
                for data in bbox:
                    # if ((xmax - xmin) < (width * 2 / 3)):
                        # xml_anno.add_object(name_text_str=str("text"),
                    xml_anno.add_object(name_text_str=str(data[4]),
                                        xmin_text_str=str(int(data[0])),
                                        ymin_text_str=str(int(data[1])),
                                        xmax_text_str=str(int(data[2])),
                                        ymax_text_str=str(int(data[3])))

                xml_anno.save_doc(im_file.replace(".jpg", ".xml"))
            #     s_list = []
            #     for p in bbox:
                    
            #         dict_p = {"label": "light_group",
            #                     "points": p.tolist(),
            #                     "group_id": None,
            #                     "description": "",
            #                     "shape_type": "polygon",
            #                     "flags": {}}
                    
            #         s_list.append(dict_p)
                
            #     base64_str = image_to_base64(im_file)

            #     json_label = {
            #                 "version": "5.0.1",
            #                 "flags": {},
            #                 "shapes": s_list,
            #                 "imagePath": os.path.basename(im_file),
            #                 "imageData": base64_str,
            #                 "imageHeight": res.shape[0],
            #                 "imageWidth": res.shape[1],
            #                 }

            #     # 将字典变成json格式，缩进为2个空格
            #     # save_to = open(im_file.replace(".jpg", ".json"), 'w')
            #     # json.dump(json_label, save_to, indent=4)

            #     json_data = dumps(json_label, indent=2)
            #     with open(im_file.replace(".jpg", ".json"), "w") as fp:
            #         fp.write(json_data)
            
                    

            
def image_to_base64(image_path):
    with open(image_path, 'rb') as jpg_file:
        byte_content = jpg_file.read()
        
    base64_bytes = base64.b64encode(byte_content)
    base64_string = base64_bytes.decode('utf-8')

    return base64_string



def parseXmlFiles(xml_path):
    tree = ET.parse(xml_path)
    imgPath = tree.findall("path")[0].text
    objects = tree.findall("object")
    for i in range(len(objects)):
        bndbox_ = objects[i].find("bndbox")[0].text
        [int(objects[3].find("bndbox")[i].text) for i in range(4)]

        a = 1

if __name__ == "__main__":

    # folder = r"F:\data\xinjin_rx"
    # print(os.listdir(folder))
    # for i in os.listdir(folder):
    #     print(i.split("_")[1])
    #     if i.split("_")[1] != "lg":continue
    imageScale = 1
    subimageScale = 1
    # targetdir = r"D:\A_data_Puan\puan_light\dirt" 
    targetdir = r"C:\Test_Image\bzz_gx_data\bzz_gx_hot01_1600"
    # labelme_json_dir = r"F:\data\xinjin_rx/xinjin_light.json"
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)
    
    # splitAndScale(f"{folder}/xinjin_light.tif", 1, 1, labelme_json_dir, targetdir, resize800=True)
    # splitAndScale(f"{folder}/xinjin_light.tif", 1, 1, labelme_json_dir, targetdir, resize800=False)
    # xmls = r"D:\A_data_Puan\puan_light\tif_data"
    xmls = r"C:\Test_Image\bzz_gx_data"
    for xml_path in glob.glob(xmls+"/*.xml"):
        # parseXmlFiles(xml_path)
        img_path = xml_path.replace(".xml", ".tif")
        splitAndScale(img_path,imageScale,subimageScale,xml_path,targetdir,resize800=False)
    

    
    