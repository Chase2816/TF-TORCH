import json
import os
import glob
import shutil
from labelme import utils
from PIL import Image
from labelme.logger import logger
import io
import os.path as osp
from labelme import PY2
from labelme import QT4
import base64
import collections
import cv2
import numpy as np
import time


def base64encode_img(image_path):
    src_image = Image.open(image_path)
    output_buffer = io.BytesIO()
    src_image.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return base64_str


def generate_json(new_dict, imfile_dir, imfile_name):
    print("imdir:",imfile_dir)
    str_json = {}
    shapes = []
    # 读取坐标
    for x in new_dict.items():
        print(x)
        print(x[0])
        # print(x[1])
        for line in x[1]:
            print(line)
            print(len(line))
            print(type(line))
            # if len(line) < 4:
            #     os.remove(imfile_dir)
            #     break
            shape = {}
            shape["label"] = x[0]
            shape["points"] = line
            shape["line_color"] = []
            shape["fill_color"] = []
            shape["flags"] = {}
            shapes.append(shape)
    print(shapes)
    print(len(shapes))

    str_json["version"] = "4.5.6"
    str_json["flags"] = {}
    str_json["shapes"] = shapes
    str_json["lineColor"] = [0, 255, 0, 128]
    str_json["fillColor"] = [255, 0, 0, 128]
    # picture_basename = file_name.replace('.txt', '.jpg')
    str_json["imagePath"] = imfile_name
    img = cv2.imread(imfile_dir)
    str_json["imageHeight"] = img.shape[0]
    str_json["imageWidth"] = img.shape[1]
    str_json["imageData"] = base64encode_img(imfile_dir)
    return str_json


def export_json_shapes_im1(json_file, imfile_dir, imfile_name):
    # j = open(
    #     r"E:\pycharm_project\tfservingconvert\maskrcnn_data_deal\labelme2mask\chaitu\data\yc2132xc2128.json").read()  # json文件读入成字符串格式

    j = open(json_file).read()
    jj = json.loads(j)  # 载入字符串，json格式转python格式
    print(len(jj["shapes"]))  # 获取标签的个数，shapes包含所有的标签
    print(jj["shapes"][0])  # 输出第一个标签信息
    print(jj["shapes"])

    classes_name = ['Hot', 'Stain', 'Diode', 'Shadow', 'Reflect']
    new_dict = {}
    Hot_points = []
    Stain_points = []
    Diode_points = []
    Shadow_points = []
    Reflect_points = []
    for i, data in enumerate(jj["shapes"]):
        print(i, data)
        label_name = data["label"]
        points_list = data["points"]
        print(label_name)
        print(points_list)
        points_list_np = np.array(points_list)
        print(points_list_np)
        print(max(points_list_np[:, 0]))
        print(max(points_list_np[:, 1]))
        x_min = min(points_list_np[:, 0])
        x_max = max(points_list_np[:, 0])
        y_max = max(points_list_np[:, 1])
        # list_max = max(max(row) for row in points_list)
        # if list_max <= 900 - 5:
        if x_max <= 900 - 2 and y_max <= 900 - 2:
            if label_name == classes_name[0]:
                Hot_points.append(points_list_np.tolist())
            if label_name == classes_name[1]:
                Stain_points.append(points_list_np.tolist())
            if label_name == classes_name[2]:
                Diode_points.append(points_list_np.tolist())
            if label_name == classes_name[3]:
                Shadow_points.append(points_list_np.tolist())
            if label_name == classes_name[4]:
                Reflect_points.append(points_list_np.tolist())

        new_dict[classes_name[0]] = Hot_points
        new_dict[classes_name[1]] = Stain_points
        new_dict[classes_name[2]] = Diode_points
        new_dict[classes_name[3]] = Shadow_points
        new_dict[classes_name[4]] = Reflect_points
    print(new_dict)
    print(len(new_dict[classes_name[0]]))
    print(len(new_dict[classes_name[1]]))
    print(len(new_dict[classes_name[2]]))
    print(len(new_dict[classes_name[3]]))
    print(len(new_dict[classes_name[4]]))

    str_json = generate_json(new_dict, imfile_dir, imfile_name)
    print(len(str_json))

    json_data = json.dumps(str_json, indent=4)
    jsonfile_name = imfile_dir.replace(".jpg", ".json")
    f = open(os.path.join(jsonfile_name), 'w')
    f.write(json_data)
    f.close()
    # return new_dict


def export_json_shapes_im2(json_file, imfile_dir, imfile_name):
    # j = open(
    #     r"E:\pycharm_project\tfservingconvert\maskrcnn_data_deal\labelme2mask\chaitu\data\yc2132xc2128.json").read()  # json文件读入成字符串格式

    j = open(json_file).read()
    jj = json.loads(j)  # 载入字符串，json格式转python格式
    print(len(jj["shapes"]))  # 获取标签的个数，shapes包含所有的标签
    print(jj["shapes"][0])  # 输出第一个标签信息
    print(jj["shapes"])

    classes_name = ['Hot', 'Stain', 'Diode', 'Shadow', 'Reflect']
    new_dict = {}
    Hot_points = []
    Stain_points = []
    Diode_points = []
    Shadow_points = []
    Reflect_points = []
    for i, data in enumerate(jj["shapes"]):
        print(i, data)
        label_name = data["label"]
        points_list = data["points"]
        print(label_name)
        print(points_list)
        points_list_np = np.array(points_list)
        print(points_list_np)
        print(max(points_list_np[:, 0]))
        print(max(points_list_np[:, 1]))
        x_min = min(points_list_np[:, 0])
        y_max = max(points_list_np[:, 1])
        print(points_list_np[:, 0] - 100)
        print(points_list_np[:, 1] - 200)
        # list_max = max(max(row) for row in points_list)
        # if list_max <= 900 - 5:
        if x_min >= 700 + 2 and y_max <= 900 - 2:
            print(points_list_np)
            points_list_np[:, 0] = points_list_np[:, 0] - 700
            print(points_list_np.tolist())
            if label_name == classes_name[0]:
                Hot_points.append(points_list_np.tolist())
            if label_name == classes_name[1]:
                Stain_points.append(points_list_np.tolist())
            if label_name == classes_name[2]:
                Diode_points.append(points_list_np.tolist())
            if label_name == classes_name[3]:
                Shadow_points.append(points_list_np.tolist())
            if label_name == classes_name[4]:
                Reflect_points.append(points_list_np.tolist())

        new_dict[classes_name[0]] = Hot_points
        new_dict[classes_name[1]] = Stain_points
        new_dict[classes_name[2]] = Diode_points
        new_dict[classes_name[3]] = Shadow_points
        new_dict[classes_name[4]] = Reflect_points
    print(new_dict)
    print(len(new_dict[classes_name[0]]))
    print(len(new_dict[classes_name[1]]))
    print(len(new_dict[classes_name[2]]))
    print(len(new_dict[classes_name[3]]))
    print(len(new_dict[classes_name[4]]))
    # return new_dict

    str_json = generate_json(new_dict, imfile_dir, imfile_name)
    print(len(str_json))

    json_data = json.dumps(str_json, indent=4)
    jsonfile_name = imfile_dir.replace(".jpg", ".json")
    f = open(os.path.join(jsonfile_name), 'w')
    f.write(json_data)
    f.close()


def export_json_shapes_im3(json_file, imfile_dir, imfile_name):
    # j = open(
    #     r"E:\pycharm_project\tfservingconvert\maskrcnn_data_deal\labelme2mask\chaitu\data\yc2132xc2128.json").read()  # json文件读入成字符串格式

    j = open(json_file).read()
    jj = json.loads(j)  # 载入字符串，json格式转python格式
    print(len(jj["shapes"]))  # 获取标签的个数，shapes包含所有的标签
    print(jj["shapes"][0])  # 输出第一个标签信息
    print(jj["shapes"])

    classes_name = ['Hot', 'Stain', 'Diode', 'Shadow', 'Reflect']
    new_dict = {}
    Hot_points = []
    Stain_points = []
    Diode_points = []
    Shadow_points = []
    Reflect_points = []
    for i, data in enumerate(jj["shapes"]):
        print(i, data)
        label_name = data["label"]
        points_list = data["points"]
        print(label_name)
        print(points_list)
        points_list_np = np.array(points_list)
        print(points_list_np)
        print(max(points_list_np[:, 0]))
        print(max(points_list_np[:, 1]))
        x_min = min(points_list_np[:, 0])
        x_max = max(points_list_np[:, 0])
        y_min = min(points_list_np[:, 1])
        y_max = max(points_list_np[:, 1])
        print(points_list_np[:, 0] - 100)
        print(points_list_np[:, 1] - 200)
        # list_max = max(max(row) for row in points_list)
        # if list_max <= 900 - 5:
        if x_max <= 900 - 2 and y_min >= 700 + 2:
            print(points_list_np)
            points_list_np[:, 1] = points_list_np[:, 1] - 700
            print(points_list_np.tolist())
            if label_name == classes_name[0]:
                Hot_points.append(points_list_np.tolist())
            if label_name == classes_name[1]:
                Stain_points.append(points_list_np.tolist())
            if label_name == classes_name[2]:
                Diode_points.append(points_list_np.tolist())
            if label_name == classes_name[3]:
                Shadow_points.append(points_list_np.tolist())
            if label_name == classes_name[4]:
                Reflect_points.append(points_list_np.tolist())

        new_dict[classes_name[0]] = Hot_points
        new_dict[classes_name[1]] = Stain_points
        new_dict[classes_name[2]] = Diode_points
        new_dict[classes_name[3]] = Shadow_points
        new_dict[classes_name[4]] = Reflect_points
    print(new_dict)
    print(len(new_dict[classes_name[0]]))
    print(len(new_dict[classes_name[1]]))
    print(len(new_dict[classes_name[2]]))
    print(len(new_dict[classes_name[3]]))
    print(len(new_dict[classes_name[4]]))

    str_json = generate_json(new_dict, imfile_dir, imfile_name)

    json_data = json.dumps(str_json, indent=4)
    jsonfile_name = imfile_dir.replace(".jpg", ".json")
    f = open(os.path.join(jsonfile_name), 'w')
    f.write(json_data)
    f.close()
    # return new_dict


def export_json_shapes_im4(json_file, imfile_dir, imfile_name):
    print("--------------")
    # j = open(
    #     r"E:\pycharm_project\tfservingconvert\maskrcnn_data_deal\labelme2mask\chaitu\data\yc2132xc2128.json").read()  # json文件读入成字符串格式

    j = open(json_file).read()
    jj = json.loads(j)  # 载入字符串，json格式转python格式
    print(len(jj["shapes"]))  # 获取标签的个数，shapes包含所有的标签
    print(jj["shapes"][0])  # 输出第一个标签信息
    print(jj["shapes"])

    classes_name = ['Hot', 'Stain', 'Diode', 'Shadow', 'Reflect']
    new_dict = {}
    Hot_points = []
    Stain_points = []
    Diode_points = []
    Shadow_points = []
    Reflect_points = []
    for i, data in enumerate(jj["shapes"]):
        print(i, data)
        label_name = data["label"]
        points_list = data["points"]
        print(label_name)
        print(points_list)
        points_list_np = np.array(points_list)
        print(points_list_np)
        print(max(points_list_np[:, 0]))
        print(max(points_list_np[:, 1]))
        x_min = min(points_list_np[:, 0])
        x_max = max(points_list_np[:, 0])
        y_min = min(points_list_np[:, 1])
        y_max = max(points_list_np[:, 1])
        print(points_list_np[:, 0] - 100)
        print(points_list_np[:, 1] - 200)
        # list_max = max(max(row) for row in points_list)
        # if list_max <= 900 - 5:
        if x_min >= 700 + 2 and y_min >= 700 + 2:
            print(points_list_np)
            points_list_np[:, 0] = points_list_np[:, 0] - 700
            points_list_np[:, 1] = points_list_np[:, 1] - 700
            print(points_list_np.tolist())
            if label_name == classes_name[0]:
                Hot_points.append(points_list_np.tolist())
            if label_name == classes_name[1]:
                Stain_points.append(points_list_np.tolist())
            if label_name == classes_name[2]:
                Diode_points.append(points_list_np.tolist())
            if label_name == classes_name[3]:
                Shadow_points.append(points_list_np.tolist())
            if label_name == classes_name[4]:
                Reflect_points.append(points_list_np.tolist())

        new_dict[classes_name[0]] = Hot_points
        new_dict[classes_name[1]] = Stain_points
        new_dict[classes_name[2]] = Diode_points
        new_dict[classes_name[3]] = Shadow_points
        new_dict[classes_name[4]] = Reflect_points
    print(new_dict)
    print("------------fff--------")

    print(len(new_dict[classes_name[0]]))
    print(len(new_dict[classes_name[1]]))
    print(len(new_dict[classes_name[2]]))
    print(len(new_dict[classes_name[3]]))
    print(len(new_dict[classes_name[4]]))

    str_json = generate_json(new_dict, imfile_dir, imfile_name)

    json_data = json.dumps(str_json, indent=4)
    jsonfile_name = imfile_dir.replace(".jpg", ".json")
    f = open(os.path.join(jsonfile_name), 'w')
    f.write(json_data)
    f.close()
    # return new_dict


# 相对路径
# imfile_dir = r"E:\pycharm_project\tfservingconvert\maskrcnn_data_deal\labelme2mask\chaitu\cat_ROI_4.jpg"
# imfile_name = "cat_ROI_4.jpg"
# json_file = r"E:\pycharm_project\tfservingconvert\maskrcnn_data_deal\labelme2mask\chaitu\data\yc2132xc2128.json"

folder = r"D:\CC\Hot_5classes\data"
save_dir = r"D:\CC\CF_Hot5\new_three_classes"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for image_path in glob.glob(folder + "/*.jpg"):
    json_file = image_path.replace(".jpg", ".json")
    srcImg = cv2.imread(image_path)
    img_roi1 = srcImg[0:900, 0:900]
    img_roi2 = srcImg[0:900, 700:1600]
    img_roi3 = srcImg[700:1600, 0:900]
    img_roi4 = srcImg[700:1600, 700:1600]

    new_name = os.path.basename(image_path).split(".")
    print(new_name)
    cv2.imwrite(f"{save_dir}/{new_name[0]}_1.jpg", img_roi1)
    cv2.imwrite(f"{save_dir}/{new_name[0]}_2.jpg", img_roi2)
    cv2.imwrite(f"{save_dir}/{new_name[0]}_3.jpg", img_roi3)
    cv2.imwrite(f"{save_dir}/{new_name[0]}_4.jpg", img_roi4)

    export_json_shapes_im1(json_file, f"{save_dir}/{new_name[0]}_1.jpg", f"{new_name[0]}_1.jpg")
    export_json_shapes_im2(json_file, f"{save_dir}/{new_name[0]}_2.jpg", f"{new_name[0]}_2.jpg")
    export_json_shapes_im3(json_file, f"{save_dir}/{new_name[0]}_3.jpg", f"{new_name[0]}_3.jpg")
    export_json_shapes_im4(json_file, f"{save_dir}/{new_name[0]}_4.jpg", f"{new_name[0]}_4.jpg")
