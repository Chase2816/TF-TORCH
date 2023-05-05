import cv2
import numpy as np
import time
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

json_files = r"D:\CC\CF_Hot5\new_three_classes"
save_dir = r"D:\CC\CF_Hot5\new_three_classes_error"
if not os.path.exists(save_dir):os.makedirs(save_dir)

for json_in in glob.glob(json_files + "/*.json"):

    print("开始处理：", json_in)
    # dst_json = os.path.join(save_json, os.path.basename(json_in))

    j = open(json_in).read()  # json文件读入成字符串格式
    # j = open(os.path.join(json_files,'ChaoYang1_light_yc0xc4256_1.json')).read()  # json文件读入成字符串格式

    jj = json.loads(j)  # 载入字符串，json格式转python格式
    print(len(jj["shapes"]))  # 获取标签的个数，shapes包含所有的标签
    # print(jj["shapes"][0])  # 输出第一个标签信息
    # if jj["shapes"][0] == "hot_group":
    #     continue
    print(jj['shapes'])
    if len(jj['shapes']) == 0:
        shutil.move(json_in,os.path.join(save_dir,os.path.basename(json_in)))
        shutil.move(json_in.replace(".json",".jpg"),os.path.join(save_dir,os.path.basename(json_in.replace(".json",".jpg"))))
    # exit()
    # print(jj["shapes"])
    # for i in range(len(jj['shapes'])):