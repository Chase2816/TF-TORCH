import shutil
import numpy as np
import base64
from PIL import Image
import io
import os
import json
import glob
import cv2

def base64encode_img(image_path):
    src_image = Image.open(image_path)
    output_buffer = io.BytesIO()
    src_image.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return base64_str

def generate_json(file_dir, file_name, data,class_name="light_group"):
    str_json = {}
    shapes = []

    for line in data:
        # lineArr = line.strip().split(',')
        # print(type(lineArr), type(lineArr[0]))
        # points = []
        # points.append([float(lineArr[0]), float(lineArr[1])])
        # points.append([float(lineArr[2]), float(lineArr[3])])
        # points.append([float(lineArr[4]), float(lineArr[5])])
        # points.append([float(lineArr[6]), float(lineArr[7])])
        # print(points)
        points = line.tolist()
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
    picture_basename = file_name.replace('.json', '.jpg')
    str_json["imagePath"] = picture_basename
    print(os.path.join(file_dir, picture_basename))
    img = cv2.imread(os.path.join(file_dir, picture_basename))
    str_json["imageHeight"] = img.shape[0]
    str_json["imageWidth"] = img.shape[1]
    str_json["imageData"] = base64encode_img(os.path.join(file_dir, picture_basename))
    return str_json

json_files = r"D:\CC\total_data"
save_json = r"D:\CC\20211118"
if not os.path.exists(save_json):
    os.mkdir(save_json)

for json_in in glob.glob(json_files+"/*.json"):
    print("开始处理：",json_in)
    if os.path.isfile(os.path.join(save_json,os.path.basename(json_in))):continue
    dst_json = os.path.join(save_json,os.path.basename(json_in))

    j = open(json_in).read()  # json文件读入成字符串格式

    jj = json.loads(j)  # 载入字符串，json格式转python格式
    print(len(jj["shapes"]))  # 获取标签的个数，shapes包含所有的标签
    # print(jj["shapes"][0])  # 输出第一个标签信息
    # if jj["shapes"][0] == "hot_group":
    #     continue

    data = []
    for i in range(len(jj['shapes'])):
        # jj["shapes"][i]["label"] = 'light_group'  # 把所有label的值都改成‘10’
        # print(jj["shapes"][i]["label"])
        a = jj["shapes"][i]["points"]
        # print(a)
        # print(type(a))
        data.append(a)
    print(len(data),data)
    print(np.array(data).shape)
    if len(data) == 0:
        continue
    # 删除重复行
    uniques = np.unique(np.array(data),axis = 0)
    # print(uniques)
    print(uniques.shape)
    print(len(uniques))

    file_name = os.path.basename(json_in)
    str_json = generate_json(json_files,file_name,uniques)
    json_data = json.dumps(str_json, indent=4)
    jsonfile_name = file_name.replace(".json", ".jpg")
    shutil.copyfile(json_in.replace(".json",".jpg"),os.path.join(save_json,jsonfile_name))
    f = open(os.path.join(save_json, file_name), 'w')
    f.write(json_data)
    f.close()

    # for box in uniques:
    #     print(box)
    #     print(type(box))
    #     jj["shapes"][i]["points"] = box.tolist()


    # 删除重复列
    # uniques = np.unique(data,axis = 1)

    # 把修改后的python格式的json文件，另存为新的json文件
    # with open(dst_json, 'w') as f:
    #     json.dump(jj, f, indent=4)  # indent=4缩进保存json文件

