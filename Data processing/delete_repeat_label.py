# coding:utf-8
import json
import base64
from PIL import Image
import io
import os
import json
import numpy as np
import glob
import cv2


def json2txt(path_json, path_txt):
    with open(path_json, 'r', encoding='utf-8') as path_json:
        jsonx = json.load(path_json)
        a = []
        with open(path_txt, 'w+') as ftxt:
            for shape in jsonx['shapes']:
                xy = np.array(shape['points'])
                label = str(shape['label'])
                strxy = ''
                a.append(str(xy.tolist()))
                for m, n in xy:
                    strxy += str(m) + ',' + str(n) + ','
                strxy += label
                ftxt.writelines(strxy + "\n")


def base64encode_img(image_path):
    src_image = Image.open(image_path)
    output_buffer = io.BytesIO()
    src_image.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return base64_str


def generate_json(file_dir, file_name, class_name="light_group"):
    str_json = {}
    shapes = []
    with open(
            r"E:\pycharm_project\tfservingconvert\maskrcnn_data_deal\test_demo\test20211118\ChaoYang1_hot_yc533xc2926.txt") as f:
        strs = f.readlines()
        print(len(strs), strs)
        print(len(set(strs)))
        print(set(strs))
        for line in list(set(strs)):
            lineArr = line.strip().split(',')
            print(type(lineArr), type(lineArr[0]))
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


dir_json = r'E:\pycharm_project\tfservingconvert\maskrcnn_data_deal\test_demo\test20211118/'
dir_txt = r'E:\pycharm_project\tfservingconvert\maskrcnn_data_deal\test_demo\test20211118/'
if not os.path.exists(dir_txt):
    os.makedirs(dir_txt)
list_json = os.listdir(dir_json)
# for cnt,json_name in enumerate(list_json):
for cnt, json_name in enumerate(glob.glob(dir_json + "/*.json")):
    print('cnt=%d,name=%s' % (cnt, json_name))
    path_json = json_name
    path_txt = json_name.replace('.json', '.txt')
    # print(path_json, path_txt)
    json2txt(path_json, path_txt)

for cnt, json_name in enumerate(glob.glob(dir_json + "/*.txt")):
    str_json = generate_json(dir_json, os.path.basename(json_name))
    json_data = json.dumps(str_json,indent=4)
    jsonfile_name = os.path.basename(json_name).replace(".txt", "_new.json")
    f = open(os.path.join(dir_json, jsonfile_name), 'w')
    f.write(json_data)
    f.close()
