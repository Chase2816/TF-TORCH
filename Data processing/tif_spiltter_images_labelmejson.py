'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-05-04 08:48:36
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-05-05 17:17:19
FilePath: \tfservingconvert\tif_data\images_spiltter.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import cv2
import numpy as np
import json
from json import  dumps
import base64
from labelme import utils
import PIL.Image
from labelme.logger import logger
import io
import os.path as osp
from labelme import PY2
from labelme import QT4


def load_image_file(filename):
    try:
        image_pil = PIL.Image.open(filename)
    except IOError:
        logger.error("Failed opening image file: {}".format(filename))
        return

    # apply orientation to image according to exif
    image_pil = utils.apply_exif_orientation(image_pil)

    with io.BytesIO() as f:
        ext = osp.splitext(filename)[1].lower()
        if PY2 and QT4:
            format = "PNG"
        elif ext in [".jpg", ".jpeg"]:
            format = "JPEG"
        else:
            format = "PNG"
        image_pil.save(f, format=format)
        f.seek(0)
        return f.read()
    
def get_json_shapes(path_json):
    with open(path_json,'r',encoding="utf_8") as path_json:
        jsonx=json.load(path_json)
        shape_polygons = jsonx['shapes']
        print(shape_polygons)
        print(len(shape_polygons))
    return shape_polygons
        

def splitAndScale(file,imageScale,subimageScale,labelmejsondir,targetdir,resize800=False):
    subimageWidth = 1600 / subimageScale
    subimageHeight = 1600 / subimageScale
      
    im = cv2.imread(file)
    new_w = im.shape[1] / imageScale
    new_h = im.shape[0] / imageScale
    im_ = im.copy()
    
    shape_polygons = get_json_shapes(labelmejsondir)
    
    for y in range(int(subimageHeight /2) ,int(im.shape[0] - subimageHeight / 2),int(subimageHeight/3)):
        for x in range(int(subimageWidth /2) ,int(im.shape[1] - subimageWidth / 2),int(subimageWidth/6)):
            res = cv2.getRectSubPix(im_,(int(subimageWidth),int(subimageHeight)),(x,y))
           
            im_file = f"{targetdir}/{os.path.basename(file).split('.')[0]}_yc{int(y-subimageHeight/2)}_xc{int(x-subimageWidth/2)}.jpg"
            x1,y1,x2,y2 = x-subimageWidth/2,y-subimageHeight/2,x+subimageWidth/2,y+subimageHeight/2
            
            bbox = []
            for shape in shape_polygons:
                xy=np.array(shape['points'])
                a,b = np.max(xy[:,0]),np.max(xy[:,1])
                
                if a>x1 and a<x2:
                    if b>y1 and b<y2:
                        
                        xy[:,0] = xy[:,0]-float(x1)
                        xy[:,1] = xy[:,1]-float(y1)

                        if resize800:
                            res = cv2.resize(res,(int(subimageWidth/2),int(subimageHeight/2)))
                            if len(xy[xy<0])==0:
                                bbox.append(xy/2)
                        else:
                            if len(xy[xy<0])==0:
                                bbox.append(xy)
                            
            print("============================================================")
            if len(bbox)>1:
                cv2.imwrite(im_file,res)
                
                s_list = []
                for p in bbox:
                    
                    dict_p = {"label": "light_group",
                                "points": p.tolist(),
                                "group_id": None,
                                "description": "",
                                "shape_type": "polygon",
                                "flags": {}}
                    
                    s_list.append(dict_p)
                
                base64_str = image_to_base64(im_file)

                json_label = {
                            "version": "5.0.1",
                            "flags": {},
                            "shapes": s_list,
                            "imagePath": os.path.basename(im_file),
                            "imageData": base64_str,
                            "imageHeight": res.shape[0],
                            "imageWidth": res.shape[1],
                            }

                # 将字典变成json格式，缩进为2个空格
                # save_to = open(im_file.replace(".jpg", ".json"), 'w')
                # json.dump(json_label, save_to, indent=4)

                json_data = dumps(json_label, indent=2)
                with open(im_file.replace(".jpg", ".json"), "w") as fp:
                    fp.write(json_data)
            
                    

            
def image_to_base64(image_path):
    with open(image_path, 'rb') as jpg_file:
        byte_content = jpg_file.read()
        
    base64_bytes = base64.b64encode(byte_content)
    base64_string = base64_bytes.decode('utf-8')

    return base64_string



if __name__ == "__main__":

    folder = "E:/data/bzz_data"
    print(os.listdir(folder))
    # for i in os.listdir(folder):
    #     print(i.split("_")[1])
    #     if i.split("_")[1] != "lg":continue
    targetdir = f"E:/data/bzz_data/bzz_lg_01_800"
    labelme_json_dir = r"E:\data\bzz_lg_01.json"
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)
    
    splitAndScale(f"{folder}/bzz_lg_01.tif", 1, 1, labelme_json_dir, targetdir, resize800=True)
    
    

    
    