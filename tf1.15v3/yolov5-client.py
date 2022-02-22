import requests
import numpy as np
import cv2
import sys, os, shutil
import json
import tensorflow as tf

sys.path.append("../")
from PIL import Image, ImageDraw


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


names = {}
with open(r"E:\pycharm_project\tfservingconvert\tf2.3v4\data\classes\coco.names", 'r') as data:
    for ID, name in enumerate(data):
        names[ID] = name.strip("\n")

# input_img = r"E:\data\starin-diode\diode-0605\images\20200605_5.jpg"
# input_img = "diode.jpg"
# input_imgs = r"E:\data\diode-opt\imgs/"
input_imgs = r"D:\ArcGIS 10.5\yolov5\data\images/"
# input_imgs = r"E:\pycharm_project\Data-proessing\break-group-yolo\test01\result/"
files = os.listdir(input_imgs)
save_path = r"out_imgs3"
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)

for file in files:
    # input_img = r"E:\data\diode-opt\imgs\20200611_84.jpg"
    input_img = input_imgs + file
    img = Image.open(input_img)
    # img = Image.open(r"D:\ArcGIS 10.5\yolov5\data\images\bus.jpg")
    img1 = img.resize((640, 640))
    # img1 = img.resize((224, 224))
    image_np = np.array(img1)
    # image_np = image_np / 255.

    # image_np = image_np.transpose([2,0,1])  # yolov4 NHWC注释  YOLOV3 NCHW
    # print(image_np.shape)
    # print(image_np[np.newaxis,:].shape)
    # img_data = image_np[np.newaxis, :].tolist()
    img_data = image_np[np.newaxis, :]
    img_data = img_data / 255.
    # print(img_data)
    img_data = img_data.tolist()
    # print(image_np[np.newaxis, :].shape)
    data = {"instances": img_data}
    # exit()
    # data = {"inputs": img_data}
    # data = {"data": img_data}

    # http://172.20.112.102:8701/v1/models/model-fish/metadata
    # preds1 = requests.post("http://172.20.112.102:9901/v1/models/yolov4:predict", json=data)
    # # preds1 = requests.post("http://172.20.112.102:4000/predictions/resnet34", json=data)
    # print(preds1.json())
    # predictions1 = json.loads(preds1.content.decode('utf-8'))
    # print(predictions1)
    # exit()

    # preds = requests.post("http://172.20.112.102:9101/v1/models/model-diode:predict", json=data)
    preds = requests.post("http://172.20.112.102:9911/v1/models/yolov5:predict", json=data)

    predictions = json.loads(preds.content.decode('utf-8'))["predictions"]
    # predictions = json.loads(preds.content.decode('utf-8'))["outputs"]

    pred = np.array(predictions)

    pred[..., :4] *= [640, 640, 640, 640]
    a = pred[..., 0]


    nc = pred.shape[2] - 5
    xc = pred[..., 4] > 0.25

    draw = ImageDraw.Draw(img1)

    for xi, x in enumerate(pred):
        x[((x[..., 2:4] < 2) | (x[..., 2:4] > 7680)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]
        x[:, 5:] *= x[:, 4:5]

        box = xywh2xyxy(x[:, :5])

        for j in box:
            print(j)
            if j[4] > 0.25:
                draw.rectangle((j[0], j[1], j[2], j[3]), outline="red")
        img1.show()
