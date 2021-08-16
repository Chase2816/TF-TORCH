import cv2
import os
import os.path as osp
import shutil

imagePath = "F:\data\mountain0727\88083\split-material-0"
save_path = "F:\data\mountain0727\88083\split-hot_group"

def split_images(imagePath,save_path):
    files = os.listdir(imagePath)
    yc = set([a.split("xc")[0] for a in files])

    for i in yc:
        out_dir = save_path + "/" + i
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for j in files:
            if j.split("xc")[0] == i:
                src_dir = imagePath + "/" + j
                dst_dir = out_dir + "/" + j
                print(dst_dir)
                print(src_dir)
                shutil.copyfile(src_dir, dst_dir)

split_images(imagePath,save_path)