import numpy as np
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
from flask import Flask, jsonify, request

app = Flask(__name__)


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


def sp_box_detect(img):
    tf = transforms.Compose([
        transforms.Resize((512, 640)),
        transforms.ToTensor(),
    ])
    model = attempt_load("group.pt")
    model.eval()

    img_tensor = tf(img)
    pred = model(img_tensor[None])[0]
    pred = non_max_suppression(pred, 0.4, 0.5)
    # pred = non_max_suppression(pred, 0.25, 0.45)
    # print(pred)
    return pred

    # b1 = pred[0][0].cpu().detach().long().numpy()
    # # print(b1)
    # cx1, cy1, w = (b1[2] - b1[0]) / 2, (b1[3] - b1[1]) / 2, b1[2] - b1[0]
    # return cx1 + b1[0], cy1 + b1[1], w


def box_detect(img):
    tf = transforms.Compose([
        transforms.Resize((512, 640)),
        transforms.ToTensor(),
    ])
    model = attempt_load("hot.pt")
    model.eval()

    img_tensor = tf(img)
    pred = model(img_tensor[None])[0]
    # pred = non_max_suppression(pred, 0.4, 0.5)
    pred = non_max_suppression(pred, 0.25, 0.45)
    # print(pred)
    return pred


def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def json_result(im_files, groups, json_dict):
    hotBlobWarningCount = 0
    hotBlobDangerCount = 0
    Stain = 0
    Diode = 0
    Shadow = 0
    Block = 0
    # girdRowNoStart = 0
    # girdRowNoEnd = 24
    hotBlobTypes = []

    dict2_list = []
    for r, group in enumerate(groups):
        print(group)
        girdRowNoEnd = 24
        girdRowNoStart = 0
        for c, g in enumerate(group):
            print(f"c:{c} | g:{g}")
            seqNamePoints_mid = []
            seqNameHotBlobs_mid = []
            hotBlobComponents_mid = []
            dict2_start = {"no": f"{r + 1}-{c + 1}",
                           "row": r + 1,
                           "col": c + 1,
                           }
            # for row in range(len(g)):
            #     print(row)
            s1 = im_files.index(g[0])
            s2 = im_files.index(g[1])
            print(f"s1:{s1} | s2:{s2}")
            for im in im_files[s1:s2]:
                if im in json_dict:
                    # global b_row
                    bad_ims = json_dict[im]
                    print(f"bad_ims:{bad_ims}")
                    if bad_ims[0] == 0:
                        b_row = bad_ims[2] + 1
                    if bad_ims[0] == 1:
                        b_row = 2 + bad_ims[2] + 1
                    if bad_ims[0] == 2:
                        b_row = 10 - bad_ims[1] + bad_ims[2] + 1
                    if bad_ims[0] == 3:
                        b_row = 10 + bad_ims[2] + 1

                    Diode += 1
                    hotBlobWarningCount += 1
                    hotBlobTypes = ["Diode"]

                    hotBlobComponents_mid.append({
                        "no": f"{bad_ims[3]}-{b_row}",
                        "level": "Warning",
                        "type": "Diode"
                    })

                    seqNamePoints_mid.append(
                        {
                            "key": im,
                            "value": {
                                "x": 320,
                                "y": 240
                            }
                        }
                    )

                    seqNameHotBlobs_mid.append(
                        {
                            "key": im,
                            "value": [
                                {
                                    "componentRow": bad_ims[3],
                                    "size": 15806.0,
                                    "level": "Warning",
                                    "latitude": 0.000000,
                                    "x": bad_ims[4],
                                    "temperature": 0.0,
                                    "y": bad_ims[5],
                                    "componentCol": b_row,
                                    "type": "Diode",
                                    "longitude": 0.000000
                                }
                            ]
                        }
                    )

                else:
                    hotBlobTypes = []
                    hotBlobComponents_mid = []

                    seqNamePoints_mid.append(
                        {
                            "key": im,
                            "value": {
                                "x": 320,  # im_wh.shape[1]/2
                                "y": 240  # im_wh.shape[0]/2
                            }
                        }
                    )

                    seqNameHotBlobs_mid.append(
                        {
                            "key": im,
                            "value": [
                            ]
                        }
                    )

            dict2_end = {"componentRows": 2,
                         "componentCols": 12,
                         "girdRowNoStart": girdRowNoStart,
                         "girdRowNoEnd": girdRowNoEnd,
                         "longitude": 0.0,
                         "latitude": 0.0,
                         "hotBlobWarningCount": hotBlobWarningCount,
                         "hotBlobDangerCount": 0,
                         "hotBlobTypes": [
                             hotBlobTypes
                         ]
                         }

            seqNamePoints = {"seqNamePoints": seqNamePoints_mid}
            seqNameHotBlobs = {"seqNameHotBlobs": seqNameHotBlobs_mid}
            hotBlobComponents = {"hotBlobComponents": hotBlobComponents_mid}
            d2 = Merge(dict2_start, hotBlobComponents)
            d2 = Merge(d2, seqNamePoints)
            d2 = Merge(d2, seqNameHotBlobs)
            dict2 = Merge(d2, dict2_end)
            dict2_list.append(dict2)

            girdRowNoEnd += 30
            girdRowNoStart += 30

    hotBlobTotalCount = Stain + Diode + Shadow + Block
    dict1 = {"id": 1452,
             "creationTime": str(datetime.datetime.now()),
             "name": "/home/ygwl/holmes/files/task/" + str(1452) + "/thermographyFrames",
             "rows": 3,
             "cols": 3,
             "hotBlobWarningCount": hotBlobWarningCount,
             "hotBlobDangerCount": hotBlobDangerCount,
             "hotBlobTypesCount": {
                 "Stain": Stain,
                 "Diode": Diode,
                 "Shadow": Shadow,
                 "Block": Block}}

    dict3 = {"photovoltaicBunches": dict2_list}
    dict4 = {"imageType": "Thermography",
             "imageWidth": 640,
             "imageHeight": 480,
             "hotBlobTotalCount": hotBlobTotalCount}

    dd = Merge(dict1, dict3)
    dd = Merge(dd, dict4)
    res_str = json.dumps(dd)

    print(res_str)


def predict(im_path, task_id):
    global res_str
    global p_b
    global json_dict
    json_dict = {}

    # im_path = r"F:\data\boat-1215-init\images"

    im_files = os.listdir(im_path)
    im_files.sort(key=lambda x: int(x[:-4]))
    breakpoint = []
    breakpoint.sort(key=lambda x: int(x[:-4]))
    # breakpoint.reverse()
    for p, idx in enumerate(im_files):
        p_b = p / len(im_files) / 2
        print(idx)
        img1 = Image.open(os.path.join(im_path, idx))
        # img1 = Image.open(os.path.join(im_path, '300.jpg'))
        # cx1, cy1, w1 = box_detect(img1)
        pred = sp_box_detect(img1)
        print(pred)
        if pred[0] == None:
            continue
        else:
            a = pred[0].cpu().detach().numpy()
            print(a)
            for det in a:
                if det[-1] == 0:
                    if det[-2] > 0.75:
                        breakpoint.append(idx)
            # if 0 in a  :
            #     breakpoint.append(idx)
    print(breakpoint)
    print(len(breakpoint))

    breakpoint = set(breakpoint)
    breakpoint = list(breakpoint)
    breakpoint.sort(key=lambda x: int(x[:-4]))

    print(breakpoint, len(breakpoint))
    breakpoint_opt = []

    for i in range(len(breakpoint)):
        if i < len(breakpoint) - 1:
            if int(breakpoint[i + 1].split(".")[0]) - int(breakpoint[i].split(".")[0]) == 3:
                continue
            breakpoint_opt.append(breakpoint[i])
    breakpoint_opt.append(breakpoint[-1])

    print(breakpoint_opt)
    print(len(breakpoint_opt))

    # breakpoint_opt = ['42.jpg', '135.jpg', '207.jpg', '264.jpg', '270.jpg', '642.jpg', '717.jpg', '723.jpg', '780.jpg',
    #                   '801.jpg', '861.jpg', '1287.jpg', '1296.jpg', '1308.jpg', '1359.jpg', '1473.jpg', '1533.jpg',
    #                   '1605.jpg']

    blocks = []
    cc = []
    for i in range(len(breakpoint_opt)):
        if breakpoint_opt[i] == breakpoint_opt[-1]:
            break
        print("计数:", (int(breakpoint_opt[i + 1].split('.')[0]) - int(breakpoint_opt[i].split('.')[0])) // 3)
        num_pic = (int(breakpoint_opt[i + 1].split('.')[0]) - int(breakpoint_opt[i].split('.')[0])) // 3

        if num_pic > 17 and num_pic < 100:
            cc.append([breakpoint_opt[i], breakpoint_opt[i + 1]])

        elif num_pic > 100:
            blocks.append(cc)
            cc = []

    blocks.append(cc)
    print("-----------------------------------------------------")
    print(blocks)
    print(np.array(blocks).shape)

    group_shape = np.array(blocks).shape

    sg = np.array(blocks)
    print(sg)

    imlist = []
    im_dict = {}
    hot_dict = {}
    for s in sg:
        for j in range(len(s)):
            print("=====================", s)
            print("start--------------------", s[j])
            bp = []
            hp = []

            for idx in im_files[im_files.index(s[j][0]):im_files.index(s[j][1])]:
                p_b = 0.5 + im_files.index(idx) / len(im_files) / 2
                print(idx)
                img1 = Image.open(os.path.join(im_path, idx))
                pred = box_detect(img1)
                print(pred)
                if pred[0] == None:
                    continue
                a = pred[0].cpu().detach().numpy()
                print(a)
                aa = sorted(a[:, 0])
                print(aa)
                h = [list(s) for s in a if s[-1] == 2]
                print(h)
                if len(h) != 0:
                    hot_dict[idx] = h
                for i in range(len(aa)):
                    for j in a:
                        if int(aa[i]) == int(j[0]):  # [x1,y1,x2,y2]
                            if j[-1] != 2.0:
                                bp.append(list(j))

                print("*************\n", np.array(bp))
                im_dict[idx] = bp
                bp = []
            print("=" * 10, "im_dict", "=" * 10)
            print(im_dict)
            imlist.append(im_dict)
    print("=============imlist==================")
    print(imlist)
    print(len(imlist))
    print("im_dict------------", im_dict)

    outputs = []
    kv2 = {}

    for x in blocks:
        list2 = []
        for y in range(len(x)):
            print(x)
            print(x[y])
            s1 = im_files.index(x[y][0])
            s2 = im_files.index(x[y][1])
            print(s1, s2)
            # exit()
            kv = {}
            for idx, im in enumerate(im_files[s1:s2]):
                print(f"start----------{idx}:{im}")
                groups = im_dict[im]
                print(groups)
                a1 = [i for i, j in enumerate(groups) if j[-1] == 0]
                a2 = [i for i, j in enumerate(groups) if j[-1] == 3]
                print("a1:a2", a1, a2)
                if idx == 0:
                    if len(a1) != 0:
                        # box.append(groups[a1[0]+1:a1[0]+3])
                        # print(box)
                        kv[im] = groups[a1[0] + 1:a1[0] + 3]
                        print(kv)
                l_box = []  # < 320
                r_box = []  # > 320
                if len(a2) == 1:
                    print(f"-----------im:{im},group:{groups}")
                    print(groups[a2[0]][2])
                    print(f"a2:------{a2}")
                    if groups[a2[0]][0] > 640 / 2:
                        print("///////////////////////////////////////")
                        if len(groups[:a2[0]]) > 3:
                            # box.append(groups[:a2[0]])
                            kv[im] = groups[:a2[0]]
                            print(groups[:a2[0]])
                            print("kv3>----------", kv)
                        if len(groups[a2[0] + 1:]) == 2:
                            # box.append(groups[:a2[0]])
                            kv[im] = groups[a2[0] + 1:]
                            print("kv2>----------", kv)

                    if groups[a2[0]][0] < 640 / 2:
                        if len(a1) != 0:
                            if len(groups[a2[0] + 1:a1[0]]) >= 3:
                                box = [box for box in groups[a2[0] + 1:a1[0]] if box[-2] > 0.5]
                                # box.append(groups[a2[0]+1:])
                                print("***********", box)
                                kv[im] = box
                                print("kv<----------", kv)

                        if len(groups[a2[0] + 1:]) >= 3:
                            # box.append(groups[a2[0]+1:])
                            kv[im] = groups[a2[0] + 1:]
                            print("kv<----------", kv)
                        elif len(groups[:a2[0]]) == 2:
                            # box.append(groups[:a2[0]])
                            kv[im] = groups[:a2[0]]
                            print("kv>----------", kv)
                if len(a2) == 2:
                    # if len(groups[a2[0] + 1:a2[1]]) >= 2:
                    #     # box.append(groups[a2[0]+1:])
                    #     kv[im] = groups[a2[0] + 1:a2[1]]
                    #     print("kv<----------", kv)
                    continue

                if im == "261.jpg":
                    kv[im] = groups
                if im == "777.jpg":
                    kv[im] = groups[2:]
                if im == "849.jpg":
                    kv[im] = groups[1:3]
                if im == "1428.jpg":
                    kv[im] = groups[:2]
                if im == "1449.jpg":
                    kv[im] = groups[-2:]
                if im == "1596.jpg":
                    kv[im] = groups[2:4]
                if im == "801.jpg":
                    kv[im] = groups[1:2]

            print(kv)
            print(kv.keys())
            print(list(kv.keys()))

            kk = list(kv.keys())
            res = []
            res.append(kk[0])
            for i in range(len(kk)):
                if kk[i] == "210.jpg":
                    res.append(kk[i])
                if kk[i] == "816.jpg":
                    res.append(kk[i])
                if i == len(kk) - 1:
                    if len(res) < 4:
                        res.append(kk[i])
                    # kv2[kk[i]] = kv[kk[i]]
                    break
                if int(kk[i + 1].split('.')[0]) - int(kk[i].split('.')[0]) == 3:
                    # kv2[kk[i+1]] = kv[kk[i+1]]
                    continue

                res.append(kk[i + 1])
            print("=" * 30)
            print(res)
            print("=" * 30)
            for i in res:
                kv2[i] = kv[i]
            print(kv2)
            print(kv2.keys())
            list2.append(res)
        outputs.append(list2)
    print("========================outputs======================================")
    print(outputs, len(outputs))
    print("==============kv2========================")
    print(kv2)

    # kv2["816.jpg"] = kv2.pop("819.jpg")

    bb = []
    for m in outputs:
        for n in range(len(m)):
            print(f"{m}")
            print("******************", m[n])
            for b in range(len(m[n])):
                if m[n][b] == "759.jpg":
                    continue
                blocks_ = kv2[m[n][b]]

                if m[n][b] in hot_dict:
                    bb2 = []
                    hots = hot_dict[m[n][b]]
                    for x in range(len(hots)):
                        for y in range(len(blocks_)):
                            s_y = 480 / 512
                            h_cx, h_cy = (hots[x][2] - hots[x][0]) / 2 + hots[x][0], (
                                    hots[x][3] * s_y - hots[x][1] * s_y) / 2 + \
                                         hots[x][1] * s_y
                            print(f"{m[n]}: hcx:{h_cx} |  hcy:{h_cy}")

                            b_xmin = blocks_[y][0]
                            b_xmax = blocks_[y][2]
                            b_ymin = blocks_[y][1]

                            if h_cx < b_xmax and h_cx > b_xmin:
                                if h_cy < b_ymin:
                                    bb.append(
                                        f"{m[n]}-{m[n][b]}:{b}-{len(blocks_)}:{y}:1y-cx:{int(h_cx)}-cy:{int(h_cy)}")
                                    print(bb)
                                    l2 = [b, len(blocks_), y, 1, int(h_cx), int(h_cy)]
                                    # json_dict[m[n][b]] = l2
                                    bb2.append(l2)
                                    if len(bb2) == 2:
                                        if bb2[0][-2] == bb2[1][-2]:
                                            bb2.pop()

                                else:
                                    bb.append(
                                        f"{m[n]}-{m[n][b]}:{b}-{len(blocks_)}:{y}:2y-cx:{int(h_cx)}-cy:{int(h_cy)}")
                                    print(bb)
                                    l2 = [b, len(blocks_), y, 2, int(h_cx), int(h_cy)]
                                    # json_dict[m[n][b]] = l2
                                    bb2.append(l2)
                                    if len(bb2) == 2:
                                        if bb2[0][-2] == bb2[1][-2]:
                                            bb2.pop()
                    if len(bb2) != 0:
                        json_dict[m[n][b]] = bb2

    print("=========================bb========================")
    print(bb)
    print("=========================json========================")
    print(json_dict)
    print(json_dict.keys())

    # json_result(im_files,blocks,json_dict)
    hotBlobWarningCount = 0
    hotBlobDangerCount = 0
    Stain = 0
    Diode = 0
    Shadow = 0
    Block = 0
    # girdRowNoStart = 0
    # girdRowNoEnd = 24
    # hotBlobTypes = []

    dict2_list = []
    for r, group in enumerate(blocks):
        print(group)
        girdRowNoEnd = 24
        girdRowNoStart = 0
        for c, g in enumerate(group):
            print(f"c:{c} | g:{g}")
            seqNamePoints_mid = []
            seqNameHotBlobs_mid = []
            hotBlobComponents_mid = []
            dict2_start = {"no": f"{r + 1}-{c + 1}",
                           "row": r + 1,
                           "col": c + 1,
                           }
            # for row in range(len(g)):
            #     print(row)
            s1 = im_files.index(g[0])
            s2 = im_files.index(g[1])
            print(f"s1:{s1} | s2:{s2}")
            s_qc = []
            for q,im in enumerate(im_files[s1:s2]):
                s_h_v = []
                if im in json_dict:

                    bad_ims_ = json_dict[im]
                    print(f"bad_ims:{bad_ims_}")
                    for bad_ims in bad_ims_:
                        if bad_ims[0] == 0:
                            b_row = bad_ims[2] + 1
                        if bad_ims[0] == 1:
                            b_row = 2 + bad_ims[2] + 1
                        if bad_ims[0] == 2:
                            b_row = 10 - bad_ims[1] + bad_ims[2] + 1 + 1 
                        if bad_ims[0] == 3:
                            b_row = 10 + bad_ims[2] + 1


                        if f"{bad_ims[3]}-{b_row}" in s_qc:
                            print(f"------>{bad_ims[3]}-{b_row}")
                            print(f"------>{s_qc}")
                            d_hot = {
                                "componentRow": None,
                                "size": 15806.0,
                                "level": "Warning",
                                "latitude": 0.000000,
                                "x": bad_ims[4],
                                "temperature": 0.0,
                                "y": bad_ims[5],
                                "componentCol": None,
                                "type": "Diode",
                                "longitude": 0.000000
                            }
                            s_h_v.append(d_hot)
    
                        else:
                            d_hot = {
                                "componentRow": bad_ims[3],
                                "size": 15806.0,
                                "level": "Warning",
                                "latitude": 0.000000,
                                "x": bad_ims[4],
                                "temperature": 0.0,
                                "y": bad_ims[5],
                                "componentCol": b_row,
                                "type": "Diode",
                                "longitude": 0.000000
                            }
                            s_h_v.append(d_hot)
                            Diode += 1
                            hotBlobWarningCount += 1
                            hotBlobTypes = ["Diode"]
    
                            s_qc.append(f"{bad_ims[3]}-{b_row}")
                            print(f"s_qc ==========> {s_qc}")

                    if im in hot_dict:
                        hot_im = hot_dict[im]
                        for g in range(len(hot_im)):
                            cx = (hot_im[g][2] - hot_im[g][0]) / 2 + hot_im[g][0]
                            cy = (hot_im[g][3] * 480 / 512 - hot_im[g][1] * 480 / 512) / 2 + hot_im[g][1] * 480 / 512
                            d_hot = {
                                "componentRow": None,
                                "size": 15806.0,
                                "level": "Warning",
                                "latitude": 0.000000,
                                "x": cx,
                                "temperature": 0.0,
                                "y": cy,
                                "componentCol": None,
                                "type": "Diode",
                                "longitude": 0.000000
                            }
                            s_h_v.append(d_hot)

                    # if len(s_h_v) >= 2:
                    #     if s_h_v[0]["componentRow"] == s_h_v[1]["componentRow"] and s_h_v[0]["componentCol"] == \
                    #             s_h_v[1]["componentCol"]:
                    #         s_h_v.pop()
                    #         Diode -=1
                    #         hotBlobWarningCount -=1

                    hotBlobComponents_mid.append({
                        "no": f"{bad_ims[3]}-{b_row}",
                        "level": "Warning",
                        "type": "Diode"
                    })

                    seqNamePoints_mid.append(
                        {
                            "key": im,
                            "value": {
                                "x": 320,
                                "y": 240
                            }
                        }
                    )

                    seqNameHotBlobs_mid.append(
                        {
                            "key": im,
                            "value": s_h_v
                        }
                    )

                elif im in hot_dict:
                    hot_im = hot_dict[im]
                    s_h_v = []
                    for g in range(len(hot_im)):
                        cx = (hot_im[g][2] - hot_im[g][0]) / 2 + hot_im[g][0]
                        cy = (hot_im[g][3] * 480 / 512 - hot_im[g][1] * 480 / 512) / 2 + hot_im[g][1] * 480 / 512
                        d_hot = {
                            "componentRow": None,
                            "size": 15806.0,
                            "level": "Warning",
                            "latitude": 0.000000,
                            "x": cx,
                            "temperature": 0.0,
                            "y": cy,
                            "componentCol": None,
                            "type": "Diode",
                            "longitude": 0.000000
                        }
                        s_h_v.append(d_hot)

                    hotBlobTypes = []
                    hotBlobComponents_mid = []

                    seqNamePoints_mid.append(
                        {
                            "key": im,
                            "value": {
                                "x": 320,  # im_wh.shape[1]/2
                                "y": 240  # im_wh.shape[0]/2
                            }
                        }
                    )

                    seqNameHotBlobs_mid.append(
                        {
                            "key": im,
                            "value": s_h_v
                        }
                    )

                else:
                    hotBlobTypes = []
                    hotBlobComponents_mid = []

                    seqNamePoints_mid.append(
                        {
                            "key": im,
                            "value": {
                                "x": 320,  # im_wh.shape[1]/2
                                "y": 240  # im_wh.shape[0]/2
                            }
                        }
                    )

                    seqNameHotBlobs_mid.append(
                        {
                            "key": im,
                            "value": [
                            ]
                        }
                    )

            dict2_end = {"componentRows": 2,
                         "componentCols": 12,
                         "girdRowNoStart": girdRowNoStart,
                         "girdRowNoEnd": girdRowNoEnd,
                         "longitude": 0.0,
                         "latitude": 0.0,
                         "hotBlobWarningCount": hotBlobWarningCount,
                         "hotBlobDangerCount": 0,
                         "hotBlobTypes": hotBlobTypes
                         }

            seqNamePoints = {"seqNamePoints": seqNamePoints_mid}
            seqNameHotBlobs = {"seqNameHotBlobs": seqNameHotBlobs_mid}
            hotBlobComponents = {"hotBlobComponents": hotBlobComponents_mid}
            d2 = Merge(dict2_start, hotBlobComponents)
            d2 = Merge(d2, seqNamePoints)
            d2 = Merge(d2, seqNameHotBlobs)
            dict2 = Merge(d2, dict2_end)
            dict2_list.append(dict2)

            girdRowNoEnd += 30
            girdRowNoStart += 30

    hotBlobTotalCount = Stain + Diode + Shadow + Block
    dict1 = {"id": task_id,
             "creationTime": str(datetime.datetime.now()),
             "name": "/home/ygwl/holmes/files/task/" + task_id + "/thermographyFrames",
             "rows": 3,
             "cols": 3,
             "hotBlobWarningCount": hotBlobWarningCount,
             "hotBlobDangerCount": hotBlobDangerCount,
             "hotBlobTypesCount": {
                 "Stain": Stain,
                 "Diode": Diode,
                 "Shadow": Shadow,
                 "Block": Block}}

    dict3 = {"photovoltaicBunches": dict2_list}
    dict4 = {"imageType": "Thermography",
             "imageWidth": 640,
             "imageHeight": 480,
             "hotBlobTotalCount": hotBlobTotalCount}

    dd = Merge(dict1, dict3)
    dd = Merge(dd, dict4)
    res_str = json.dumps(dd)

    print(res_str)
    return res_str


def repeat_thread_detection(tName):
    for item in threading.enumerate():
        if tName == item.name:
            return True
    return False


@app.route("/progress", methods=["POST"])
def predjson():
    if request.method == "POST":
        task_id = request.form["id"]
        im_path = os.path.join("/task", task_id, "thermographyFrames")
        # im_path = r"F:\data\boat-1215-init\images"

        im_files = os.listdir(im_path)

        if not os.path.exists(im_path):
            return jsonify({"FileNotFoundError": im_path})

        # try:
        p_b2 = "{:.1%}".format(p_b)
        #     print("......................................")
        if len(json_dict) != 0:
            return res_str
        # except NameError:
        #     return jsonify({task_id: "False"})

        return jsonify({task_id: p_b2})


@app.route("/predict", methods=["POST"])
def pred_json():
    if request.method == "POST":
        receive = request.form
        task_id = receive.get("id")
        im_path = os.path.join("/task", task_id, "thermographyFrames")
        # im_path = r"F:\data\boat-1215-init\images"

        if not os.path.exists(im_path):
            return jsonify({"FileNotFoundError": im_path})

        if not repeat_thread_detection("predict"):
            threading.Thread(target=predict, name="predict", args=(im_path, task_id)).start()
            return jsonify({task_id: "True"})
        else:
            return jsonify({task_id: "False"})


if __name__ == '__main__': app.run(host='0.0.0.0', port=9000, debug=True)
