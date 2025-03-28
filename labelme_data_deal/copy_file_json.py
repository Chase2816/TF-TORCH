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

# copy_json(imagePath)

# imagePath = r"D:\渔光物联\split-hot_group\yc1599\yc1599xc1596.jpg"
# in_json = r"D:\渔光物联\split-hot_group\yc1066\yc1066xc1596.json"
# out_json = r"D:\渔光物联\split-hot_group\yc1599\yc1599xc1596.json"
# process_json(in_json ,out_json, imagePath)
#
imagePath2 = r"F:\data\mountain0727\88083\split-hot_group\data\yc7462xc1862.jpg"
in_json2 = r"F:\data\mountain0727\88083\split-hot_group\data\yc6929xc1330.json"

out_json2 = r"F:\data\mountain0727\88083\split-hot_group\data\yc7462xc1862.json"
process_json(in_json2 ,out_json2, imagePath2)

# in_json = r"F:\data\mountain0727\88082\split-light_group\yc2132\yc2132xc1064.json"
# imagePath = r"F:\data\mountain0727\88082\split-light_group\yc2132"
#
# json_files = glob.glob(imagePath+"/*.json")
# print(json_files[0].split(".")[0])
# a = [a.split(".")[0] for a in json_files]
# print(a)
# for i in glob.glob(imagePath+"/*.jpg"):
#     if i.split(".")[0] in a:continue
#     print(i)
#     out_json = i.split(".")[0]+".json"
#     print(out_json)
#
#     process_json(in_json,out_json,i)