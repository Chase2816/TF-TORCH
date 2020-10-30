import requests
import numpy as np
import cv2
import sys, os, shutil
import json
import tensorflow as tf

sys.path.append("../")
from PIL import Image


names = {}
with open("coco.names",'r') as data:
    for ID,name in enumerate(data):
        names[ID] = name.strip("\n")

# input_img = r"E:\data\starin-diode\diode-0605\images\20200605_5.jpg"
# input_img = "diode.jpg"
# input_imgs = r"E:\data\diode-opt\imgs/"
input_imgs = r"E:\pycharm_project\Data-proessing\server_test\imgs/"
# input_imgs = r"E:\pycharm_project\Data-proessing\break-group-yolo\test01\result/"
files = os.listdir(input_imgs)
save_path = r"out_imgs"
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)
for file in files:
    # input_img = r"E:\data\diode-opt\imgs\20200611_84.jpg"
    input_img = input_imgs + file
    img = Image.open(input_img)
    print(np.array(img))
    img1 = img.resize((416, 416))
    # img1 = img.resize((224, 224))
    image_np = np.array(img1)
    image_np = image_np / 255.
    print(image_np.shape)
    # image_np = image_np.transpose([2,0,1])  # yolov4 NHWC注释  YOLOV3 NCHW
    # print(image_np.shape)
    # print(image_np[np.newaxis,:].shape)
    img_data = image_np[np.newaxis, :].tolist()
    print(image_np[np.newaxis,:].shape)
    data = {"instances": img_data}
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
    preds = requests.post("http://172.20.112.102:9901/v1/models/yolov4:predict", json=data)

    print(preds)
    predictions = json.loads(preds.content.decode('utf-8'))["predictions"][0]
    print(predictions)
    print(np.array(predictions)[:,4].max())
    pred = np.array(predictions)
    print(pred.shape)
    # pred = pred[None]
    # print(pred.shape)
    # print(pred)

    boxes = pred[:,0:4]
    pred_conf = pred[:,4:]
    print(boxes.shape)
    print(pred_conf.shape)
    # boxes = boxes.reshape(boxes.shape[0],1,-1)
    # scores = pred_conf.reshape()
    print(boxes)
    # print(pred_conf[33,0].argmax())

    image = cv2.imread(input_img)
    image_h, image_w, _ = image.shape

    for i,coor in enumerate(boxes):
        label = pred_conf[i].argmax()
        scores = pred_conf[i][label]
        if scores<0.25:
            continue
        coor[0] = int(coor[0] * image_h)
        coor[2] = int(coor[2] * image_h)
        coor[1] = int(coor[1] * image_w)
        coor[3] = int(coor[3] * image_w)
        c1, c2 = (int(coor[1]), int(coor[0])), (int(coor[3]), int(coor[2]))
        print(c1,c2,scores,label)


        cv2.rectangle(image, c1, c2, (0,255,0))
        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(image,names[label]+"-"+str(scores),(c1),font,1.2,(255,255,255))
    cv2.imshow("ss",image)
    cv2.waitKey(0)
