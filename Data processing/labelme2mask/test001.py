import cv2
import numpy as np
import os
import glob
import shutil


folder = r'E:\data\light_group_hot\kjgdt-sm'
save_dir = r'E:\data\light_group_hot\data'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i,j,k in os.walk(folder):
    print(f"处理文件路径：{i}")
    for idx,data in enumerate(k):
        im_path = os.path.join(i,data)

        if im_path.split(".")[-1] == 'json':
            print(f"c处理json：{im_path}")

            json_src = im_path
            json_dst = os.path.join(save_dir,im_path.split("dt-sm\\")[-1].replace("\\","_"))

            im_src = im_path.replace(".json",".jpg")
            im_dst = os.path.join(save_dir,os.path.basename(json_dst).replace("json","jpg"))

            print(json_src)
            print(json_dst)
            print(im_src)
            print(im_dst)

            shutil.copyfile(im_src,im_dst)
            shutil.copyfile(json_src,json_dst)

