import json
import os
import glob
import shutil
from labelme import utils
import PIL.Image
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


def reset_label(json_folder,save_json):
    """
    修改lebleme得json文件标签label
    @param json_folder: json文件夹路径
    @param save_json: 保存文件夹路径
    """
    if not os.path.exists(save_json):
        os.mkdir(save_json)

    for json_in in glob.glob(json_folder + "/*.json"):
        print("开始处理：", json_in)
        dst_json = os.path.join(save_json, os.path.basename(json_in))

        j = open(json_in).read()  # json文件读入成字符串格式

        jj = json.loads(j)  # 载入字符串，json格式转python格式
        print(len(jj["shapes"]))  # 获取标签的个数，shapes包含所有的标签
        print(jj["shapes"][0])  # 输出第一个标签信息
        if jj["shapes"][0] == "hot_group":
            continue

        for i in range(len(jj['shapes'])):
            jj["shapes"][i]["label"] = 'hot_group'  # 把所有label的值都改成‘10’
            print(jj["shapes"][i]["label"])

        # 把修改后的python格式的json文件，另存为新的json文件
        with open(dst_json, 'w') as f:
            json.dump(jj, f, indent=2)  # indent=4缩进保存json文件

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

def process_json(in_json_file, out_json_file, new_imagePath):
    file_in = open(in_json_file, "r")
    file_out = open(out_json_file, "w")
    # load数据到变量json_data
    json_data = json.load(file_in)

    imagePath = json_data["imagePath"]
    print("imagePath修改前：", imagePath)
    # 截取路径中图像的文件名
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
    print("imagePath修改后：", imagePath2)
    print(json_data)
    # 将修改后的数据写回文件

    file_out.write(json.dumps(json_data, indent=2, ensure_ascii=False))
    file_in.close()
    file_out.close()

def reset_image_json_name(json_folder,out_folder,date=time.strftime("_%m%d_", time.localtime())):
    """
    重命名image和json文件，更新json文件得imagedata和imagepath参数
    @param json_folder: 包含image和json得文件夹
    @param out_folder: 保存得文件夹
    """
    if not osp.exists(out_folder):
        os.mkdir(out_folder)
    json_files = glob.glob(json_folder + "/*.json")
    print(json_files[0].split(".")[0])

    for x, i in enumerate(glob.glob(json_folder + "/*.jpg")):
        flag = i.split(".")[0]
        src_json = i.split(".")[0] + ".json"
        print("----------------")
        print(i)
        print(src_json)

        rename = osp.basename(i).split(".")[0]

        dst_img = osp.join(out_folder, rename + date + str(x) + ".jpg")
        dst_json = osp.join(out_folder, rename + date + str(x) + ".json")
        print(dst_img)
        print(dst_json)
        shutil.copyfile(i, dst_img)
        process_json(src_json, dst_json, dst_img)
        print(f"处理结束 num：{x + 1}")

def reset_labelme_json(src_dir,dst_dir,resize=800):
    """
    labelme 修改1600*1600图片尺寸为800*800  json标注点point同比例缩放
    @param src_dir: 包含image和json得文件夹路径
    @param dst_dir: 修改后存放路径
    """

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    # 先收集一下文件夹中图片的格式列表，例如 ['.jpg', '.JPG']
    exts = dict()
    filesnames = os.listdir(src_dir)
    for filename in filesnames:
        name, ext = filename.split('.')
        if ext != 'json':
            if exts.__contains__(ext):
                exts[ext] += 1
            else:
                exts[ext] = 1

    anno = collections.OrderedDict()  # 这个可以保证保存的字典顺序和读取出来的是一样的，直接使用dict()的话顺序会很乱（小细节哦）
    for key in exts.keys():
        for img_file in glob.glob(os.path.join(src_dir, '*.' + key)):
            file_name = os.path.basename(img_file)
            print(f"Processing {file_name}")
            img = cv2.imread(img_file)
            (h, w, c) = img.shape
            # 都等比例地将宽resize为1344(这里可以自己修改)
            w_new = resize
            h_new = int(h / w * w_new)  # 高度等比例缩放
            ratio = w_new / w  # 标注文件里的坐标乘以这个比例便可以得到新的坐标值
            img_resize = cv2.resize(img, (w_new, h_new))  # resize中的目标尺寸参数为(width, height)
            cv2.imwrite(os.path.join(dst_dir, file_name), img_resize)

            # 接下来处理标注文件json中的标注点的resize
            json_file = os.path.join(src_dir, file_name.split('.')[0] + '.json')
            save_to = open(os.path.join(dst_dir, file_name.split('.')[0] + '.json'), 'w')
            print(json_file)
            with open(json_file, 'rb') as f:
                anno = json.load(f)
                for shape in anno["shapes"]:
                    points = shape["points"]
                    points = (np.array(points) * ratio).astype(int).tolist()
                    shape["points"] = points

                # 注意下面的img_resize编码加密之前要记得将通道顺序由BGR变回RGB
                anno['imageData'] = str(utils.img_arr_to_b64(img_resize[..., (2, 1, 0)]), encoding='utf-8')

                anno["imageHeight"] = img_resize.shape[0]
                anno["imageWidth"] = img_resize.shape[1]

                json.dump(anno, save_to, indent=4)
    print("Done")

if __name__ == '__main__':
    # labelme 修改1600*1600图片尺寸为800*800  json标注点point同比例缩放
    src_dir = r'E:\data\total_Plant_inspection\GuangDongTaiShan\pv_light'
    dst_dir = r'E:\data\total_Plant_inspection\GuangDongTaiShan\wide_800'
    reset_labelme_json(src_dir, dst_dir,resize=1600)

    # 重命名image和json文件
    # json_folder = r"F:\data\mountain0727\light_group\test_data\da_yjb-lg"
    # out_folder = r"F:\data\mountain0727\light_group\test_data\new_json"
    # reset_image_json_name(json_folder,out_folder)

    # 修改lebleme得json文件标签labelF:\data\mountain0727\light_group\test_data
    # json_files = r"F:\data\mountain0727\light_group\test_data\new_json"
    # save_json = r"F:\data\mountain0727\light_group\test_data\new_json2"
    # reset_label(json_files,save_json)

