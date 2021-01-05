from socket import *
import numpy as np
import cv2
import base64
import os
import datetime
import json
from torchvision import transforms
import torch
from PIL import Image, ImageDraw
import shutil
from torch import nn
import torchvision
import time
import io
import threading


class Ensemble(nn.ModuleList):
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.stack(y).mean(0)
        return y, None


def attempt_load(weights, map_location=None):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        model.append(torch.load(w, map_location=map_location)['model'].float().fuse().eval())

    if len(model) == 1:
        return model[-1]
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model


def xywh2xyxy(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def box_iou(box1, box2):
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    if prediction.dtype is torch.float16:
        prediction = prediction.float()

    nc = prediction[0].shape[1] - 5
    xc = prediction[..., 4] > conf_thres

    min_wh, max_wh = 2, 4096
    max_det = 300
    time_limit = 10.0
    redundant = True
    multi_label = nc > 1

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if not x.shape[0]:
            continue

        x[:, 5:] *= x[:, 4:5]

        box = xywh2xyxy(x[:, :4])

        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        n = x.shape[0]
        if not n:
            continue

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        if merge and (1 < n < 3E3):
            try:
                iou = box_iou(boxes[i], boxes) > iou_thres
                weights = iou * scores[None]
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
                if redundant:
                    i = i[iou.sum(1) > 1]
            except:
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break

    return output


def box_detect(img):
    tf = transforms.Compose([
        transforms.Resize((512, 640)),
        transforms.ToTensor(),
    ])
    model = attempt_load(r"D:\GoogleEarthProPortable\yolov5\boat1215\last.pt")
    # model = attempt_load(r"D:\GoogleEarthProPortable\yolov5\runs\exp13\weights\last.pt")

    model.eval()

    img_tensor = tf(img)
    pred = model(img_tensor[None])[0]
    # pred = non_max_suppression(pred, 0.4, 0.5)
    pred = non_max_suppression(pred, 0.25, 0.45)
    # print(pred)

    return pred

def main():
    HOST = '127.0.0.1'
    PORT = 9999
    BUFSIZ = 1024*20
    ADDR = (HOST, PORT)
    tcpSerSock = socket(AF_INET, SOCK_STREAM)
    tcpSerSock.bind(ADDR)
    tcpSerSock.listen(5)
    while True:
        rec_d = bytes([])
        print('waiting for connection...')
        tcpCliSock, addr = tcpSerSock.accept()
        print('...connected from:', addr)
        while True:
            data = tcpCliSock.recv(BUFSIZ)
            if not data or len(data) == 0:
                break
            else:
                rec_d = rec_d + data
        rec_d = base64.b64decode(rec_d)
        # np_arr = np.fromstring(rec_d, np.uint8)
        # image = cv2.imdecode(np_arr, 1)


        image = Image.open(io.BytesIO(rec_d))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        pred = box_detect(image)
        print(pred)
        a = pred[0].cpu().detach().numpy()
        draw = ImageDraw.Draw(image)
        for i in a:
            draw.rectangle((i[0],i[1]*480/512,i[2],i[3]*480/512))
            tcpCliSock.sendall(f"{i[0]},{i[1]},{i[2]},{i[3]}\n".encode("utf8"))
            image.show()

        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        tcpCliSock.send("   --|    0001".encode())
        tcpCliSock.send("返回值".encode("utf8"))
        tcpCliSock.close()
    tcpSerSock.close()

if __name__ == "__main__":
    main()