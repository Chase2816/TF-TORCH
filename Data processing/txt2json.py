import json
import base64
from PIL import Image
import io
import os
import cv2
import numpy as np


def generate_json(file_dir, file_name,class_name="light_group"):
    str_json = {}
    shapes = []
    # 读取坐标
    fr = open(os.path.join(file_dir, file_name))
    for line in fr.readlines():  # 逐行读取，滤除空格等
        lineArr = line.strip().split(',')
        print(type(lineArr),type(lineArr[0]))
        points = []
        points.append([float(lineArr[0]), float(lineArr[1])])
        points.append([float(lineArr[2]), float(lineArr[3])])
        points.append([float(lineArr[4]), float(lineArr[5])])
        points.append([float(lineArr[6]), float(lineArr[7])])
        print(points)
        shape = {}
        shape["label"] = class_name
        shape["points"] = points
        shape["line_color"] = []
        shape["fill_color"] = []
        shape["flags"] = {}
        shapes.append(shape)
    str_json["version"] = "4.5.6"
    str_json["flags"] = {}
    str_json["shapes"] = shapes
    str_json["lineColor"] = [0, 255, 0, 128]
    str_json["fillColor"] = [255, 0, 0, 128]
    picture_basename = file_name.replace('.txt', '.jpg')
    str_json["imagePath"] = picture_basename
    img = cv2.imread(os.path.join(file_dir, picture_basename))
    str_json["imageHeight"] = img.shape[0]
    str_json["imageWidth"] = img.shape[1]
    str_json["imageData"] = base64encode_img(os.path.join(file_dir, picture_basename))
    return str_json


def base64encode_img(image_path):
    src_image = Image.open(image_path)
    output_buffer = io.BytesIO()
    src_image.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return base64_str


# file_dir = r"E:\pycharm_project\tfservingconvert\maskrcnn_data_deal\test_demo\data"
file_dir = r"E:\data\total_Plant_inspection\test"
file_name_list = [file_name for file_name in os.listdir(file_dir) \
                  if file_name.lower().endswith('txt')]

for file_name in file_name_list:
    str_json = generate_json(file_dir, file_name)
    json_data = json.dumps(str_json,indent=4)
    jsonfile_name = file_name.replace(".txt", ".json")
    f = open(os.path.join(file_dir, jsonfile_name), 'w')
    f.write(json_data)
    f.close()
