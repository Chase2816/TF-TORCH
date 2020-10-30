import requests
import numpy as np
import cv2
import sys, os, shutil
import json

sys.path.append("../")
from PIL import Image

# input_img = r"E:\data\starin-diode\diode-0605\images\20200605_5.jpg"
# input_img = "diode.jpg"
input_imgs = r"E:\data\diode-opt\imgs/"
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
    print(image_np.shape)
    image_np = image_np.transpose([2,0,1])
    print(image_np.shape)
    # print(image_np[np.newaxis,:].shape)
    img_data = image_np[np.newaxis, :].tolist()
    print(image_np[np.newaxis,:].shape)
    data = {"instances": img_data}
    # data = {"inputs": img_data}
    # data = {"data": img_data}

    # http://172.20.112.102:8701/v1/models/model-fish/metadata
    preds1 = requests.post("http://172.20.112.102:9101/v1/models/model-diode:predict", json=data)
    # preds1 = requests.post("http://172.20.112.102:4000/predictions/resnet34", json=data)
    print(preds1.json())
    predictions1 = json.loads(preds1.content.decode('utf-8'))
    print(predictions1)
    exit()

    preds = requests.post("http://172.20.112.102:9101/v1/models/model-diode:predict", json=data)
    print(preds)
    predictions = json.loads(preds.content.decode('utf-8'))["predictions"][0]
    print(predictions)
    print(np.array(predictions)[:,4].max())
    pred = np.array(predictions)
    print(pred.shape)
    # exit()
    a = pred[:, 4] > 0.25
    print(pred[a])
    print(len(pred[a]))
    # exit()
    im = cv2.imread(input_img)
    # print(im.shape) # hwc
    h_s = im.shape[0] / 416
    w_s = im.shape[1] / 416

    box = []
    for i in range(len(pred[a])):
        # print(pred[a][i])
        x1 = pred[a][i][0]
        y1 = pred[a][i][1]
        x2 = pred[a][i][2]
        y2 = pred[a][i][3]
        xx1 = (x1 - x2 / 2)
        yy1 = (y1 - y2 / 2)
        xx2 = (x1 + x2 / 2)
        yy2 = (y1 + y2 / 2)
        box.append([xx1, yy1, xx2, yy2, pred[a][i][4], pred[a][i][5:]])

        cv2.rectangle(im, (int(xx1 * w_s), int(yy1 * h_s)), (int(xx2 * w_s), int(yy2 * h_s)), (255, 0, 0))
    # print(box)
    # cv2.imshow('ss1', im)
    cv2.imwrite(save_path + '/' + file, im)
    # cv2.waitKey(0)

# filtered_boxes = np.array([[6.64219894e+01,3.38672394e+02,1.29818487e+01,1.43017712e+01]])
# draw_boxes(filtered_boxes, img, "classes", (416, 416), True)
# img.show()


# im = cv2.imread(input_img)
# # print(im.shape) # hwc
# x_ = im.shape[0]/416
# y_ = im.shape[1]/416
# # a = min(x_,y_)
# # p = 0.5*(im.shape[:2] - a*np.array([416,416]))
# # s = (np.array([66,338])-p)/a
# # print(s)
# # s1 = (np.array([12,14])-p)/a
# # print(s1)
# # w,h,cx,cy
# img = cv2.resize(im,(416,416))
# # cv2.rectangle(img,(79,182),(9,15),(0,0,255))
# # cv2.rectangle(img,(int((79-9/2)),int((182-15/2))),(int((79+9/2)),int((182+15/2))),(0,0,255))
# # cv2.rectangle(im,(int((338/2-12)*y_),int((66/2-14)*x_)),(int((12+338/2)*y_),int((14+66/2)*x_)),(0,0,255))
# h_s = im.shape[0]/416
# w_s = im.shape[1]/416
# x1 = 79
# y1 = 182
# x2 = 9
# y2 = 15
# xx1 = (x1 - x2/2)
# yy1 = (y1 - y2/2)
# xx2 = (x1 + x2/2)
# yy2 = (y1 + y2/2)

# cv2.rectangle(img,(xx1,yy1),(xx2,yy2),(0,0,255))
# cv2.rectangle(im,(int(xx1*w_s),int(yy1*h_s)),(int(xx2*w_s),int(yy2*h_s)),(255,0,0))
# x1,y1,x2,y2  x1- x2/2    y1- y2/2    x1+x2/2  y1 + y2/2
# cv2.imshow("ss",img)
# cv2.imshow('ss1',im)
# cv2.waitKey(0)
