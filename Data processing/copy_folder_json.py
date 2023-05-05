'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2021-07-30 14:01:47
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-05-04 10:54:17
FilePath: \tfservingconvert\maskrcnn_data_deal\copy_folder_json.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import shutil
import glob
import json
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

def process_json(in_json_file,out_json_file, new_imagePath):
    file_in = open(in_json_file, "r")
    file_out = open(out_json_file, "w")
    # load数据到变量json_data
    json_data = json.load(file_in)

    imagePath = json_data["imagePath"]
    print("imagePath修改前：",imagePath)
    #截取路径中图像的文件名
    # new_imagePath = imagePath.split("\\")[-1]
    # 修改json中图像路径

    # 修改imagedata
    imageData = load_image_file(new_imagePath)
    mageData = base64.b64encode(imageData).decode("utf-8")
    json_data["imageData"] = mageData
    # print(imageData)
    # print("-----------------")
    # print(mageData)

    json_data["imagePath"] = os.path.basename(new_imagePath)
    imagePath2 = json_data["imagePath"]
    print("imagePath修改后：",imagePath2)
    print(json_data)
    # 将修改后的数据写回文件

    file_out.write(json.dumps(json_data,indent=2, ensure_ascii=False))
    file_in.close()
    file_out.close()

def copy_json(imagePath):
    json_file = glob.glob(imagePath+"/*.json")[0]
    print(json_file)
    im_files = glob.glob(imagePath+"/*.jpg")

    for i in im_files:
        if i == json_file.replace("json","jpg"):
            continue
        dst_dir = i.replace(".jpg",".json")
        print(dst_dir)

        # shutil.copyfile(json_file,dst_dir)

        new_imagePath = os.path.basename(i)
        print(new_imagePath)

        process_json(json_file,dst_dir, i)



# imagePath = "F:\data\mountain0727\88082\split-light_group\yc0"
#
# copy_json(imagePath)

folder = r"F:\data\light_group_total_no\test"
for i in os.listdir(folder):
    if i == "yc0": continue
    print(osp.join(folder,i))
    copy_json(osp.join(folder,i))



