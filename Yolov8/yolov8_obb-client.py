import requests
import numpy as np
import cv2
import sys, os, shutil
import json
from PIL import Image, ImageDraw, ImageFont
import torch
import time
import torchvision
# import ops

from ultralytics.utils import LOGGER, is_colab, is_kaggle, ops

def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(box1, box2):
    area1 = box_area(box1)  # N
    area2 = box_area(box2)  # M
    # broadcasting, 两个数组各维度大小 从后往前对比一致， 或者 有一维度值为1；
    lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
    rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])
    wh = rb - lt
    wh = np.maximum(0, wh) # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, np.newaxis] + area2 - inter)
    return iou  # NxM

def numpy_nms(boxes, scores, iou_threshold):
    idxs = scores.argsort()  # 按分数 降序排列的索引 [N]
    keep = []
    while idxs.size > 0:  # 统计数组中元素的个数
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]
        keep.append(max_score_index)
        if idxs.size == 1:
            break
        idxs = idxs[:-1]  # 将得分最大框 从索引中删除； 剩余索引对应的框 和 得分最大框 计算IoU；
        other_boxes = boxes[idxs]  # [?, 4]
        ious = box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
        idxs = idxs[ious[0] <= iou_threshold]
    keep = np.array(keep)  
    return keep

def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner.

    Args:
        x (np.ndarray) or (torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
    Returns:
        y (np.ndarray) or (torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    """
    Rescales bounding boxes (in the format of xyxy) from the shape of the image they were originally specified in
    (img1_shape) to the shape of a different image (img0_shape).

    Args:
      img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
      boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
      img0_shape (tuple): the shape of the target image, in the format of (height, width).
      ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
                         calculated based on the size difference between the two images.

    Returns:
      boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes

def clip_boxes(boxes, shape):
    """
    It takes a list of bounding boxes and a shape (height, width) and clips the bounding boxes to the
    shape

    Args:
      boxes (torch.Tensor): the bounding boxes to clip
      shape (tuple): the shape of the image
    """
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def nms(dets, iou_thred, cfd_thred):
    if len(dets) == 0: return []
    bboxes = np.array(dets)
    ## 对整个bboxes排序
    bboxes = bboxes[np.argsort(bboxes[:, 4])]
    pick_bboxes = []
    #     print(bboxes)
    while bboxes.shape[0] and bboxes[-1, 4] >= cfd_thred:
        # while bboxes.shape[0] and bboxes[-1, -1] >= cfd_thred:
        bbox = bboxes[-1]
        x1 = np.maximum(bbox[0], bboxes[:-1, 0])
        y1 = np.maximum(bbox[1], bboxes[:-1, 1])
        x2 = np.minimum(bbox[2], bboxes[:-1, 2])
        y2 = np.minimum(bbox[3], bboxes[:-1, 3])
        inters = np.maximum(x2 - x1 + 1, 0) * np.maximum(y2 - y1 + 1, 0)
        unions = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1) + (bboxes[:-1, 2] - bboxes[:-1, 0] + 1) * (
                bboxes[:-1, 3] - bboxes[:-1, 1] + 1) - inters
        ious = inters / unions
        keep_indices = np.where(ious < iou_thred)
        bboxes = bboxes[keep_indices]  ## indices一定不包括自己
        pick_bboxes.append(bbox)
    return np.asarray(pick_bboxes)

def sigmoid(x): # sigmoid 函数实现
    return 1 / (1 + np.exp(-x))

def process_mask(protos, masks_in, bboxes, shape, upsample=False):
    """
    Apply masks to bounding boxes using the output of the mask head.

    Args:
        protos (torch.Tensor): A tensor of shape [mask_dim, mask_h, mask_w].
        masks_in (torch.Tensor): A tensor of shape [n, mask_dim], where n is the number of masks after NMS.
        bboxes (torch.Tensor): A tensor of shape [n, 4], where n is the number of masks after NMS.
        shape (tuple): A tuple of integers representing the size of the input image in the format (h, w).
        upsample (bool): A flag to indicate whether to upsample the mask to the original image size. Default is False.

    Returns:
        (torch.Tensor): A binary mask tensor of shape [n, h, w], where n is the number of masks after NMS, and h and w
            are the height and width of the input image. The mask is applied to the bounding boxes.
    """

    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    # masks_in = [9,32] protos
    masks = sigmoid((masks_in @ protos.reshape(c, -1))).reshape(-1, mh, mw)  # CHW

    downsampled_bboxes = bboxes.copy()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
    return masks.gt_(0.5)

MASK_COLORS = np.array([(255, 56, 56), (255, 157, 151), (255, 112, 31),
                        (255, 178, 29), (207, 210, 49), (72, 249, 10),
                        (146, 204, 23), (61, 219, 134), (26, 147, 52),
                        (0, 212, 187), (44, 153, 168), (0, 194, 255),
                        (52, 69, 147), (100, 115, 255), (0, 24, 236),
                        (132, 56, 255), (82, 0, 133), (203, 56, 255),
                        (255, 149, 200), (255, 55, 199)],
                       dtype=np.float32)

# names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#          'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#          'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#          'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
#          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#          'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#          'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
#          'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
#          'scissors',
#          'teddy bear', 'hair drier', 'toothbrush']

names = ['light_group']
# input_imgs = r"E:\pycharm_project\tfservingconvert\yolov8\model-yolo_lg\images/"
input_imgs = "/home/ygwl/Project_Image/test_data/"
files = os.listdir(input_imgs)
# save_path = r"E:\pycharm_project\tfservingconvert\water_flowers\outputs\out_imgs2"
save_path = "/home/ygwl/Project_Image/test_data/tfserver_out/"
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)
for file in files:
    # input_img = r"E:\data\diode-opt\imgs\20200611_84.jpg"
    # input_img = r"E:\pycharm_project\tfservingconvert\tf1.15v3\yc960xc1484.jpg"
    # if file.split('.')[0] != "yc0xc6118": continue
    input_img = input_imgs + file
    print(input_img)
    # img = Image.open(r"E:\data\bzz_data\split-material-0\yc4264xc266.jpg")

    img = Image.open(input_img)
    img1 = img.resize((640, 640))
    # img1 = img.resize((224, 224))
    image_np = np.array(img1)
    image_np = image_np / 255
    img_data = image_np[np.newaxis, :].tolist()
    # data = {"instances": img_data}
    data = {"inputs":img_data}
    # preds = requests.post("http://172.20.112.102:9911/v1/models/yolov8:predict", json=data)
    preds = requests.post("http://172.20.112.102:9991/v1/models/model-tm_obb:predict", json=data,verify=False)

    # preds = requests.post("http://localhost:8101/v1/models/model-lg:predict", json=data)
    # preds = requests.post("http://localhost:8201/v1/models/model-yolo_lg:predict", json=data)
    print(preds)
    #predictions = json.loads(preds.content.decode('utf-8'))["predictions"][0]
    predictions = json.loads(preds.content.decode('utf-8'))["outputs"][0]
    print(predictions)
    #print(json.loads(preds.content.decode('utf-8'))["predictions"])
    print(len(predictions))
    pred_det = np.array(predictions)
    print(pred_det.shape)
    pred_det = pred_det[np.newaxis,:]
    print("=============",pred_det.shape)
    p = ops.non_max_suppression(torch.tensor(torch.from_numpy(pred_det)),
                                    0.25,
                                    0.7,
                                    agnostic=False,
                                    max_det=300,
                                    nc=len({0:"light_group"}),
                                    classes=None,
                                    rotated=True)
    results = []
    pred = p[0]
    print("ppppp",p[0],image_np.shape)
    pred[:, :4] = ops.scale_boxes(image_np.shape[:2], pred[:, :4], (1600,1600,3),xywh=True)
    obb = torch.cat([pred[:, :4], pred[:, -1:], pred[:, 4:6]], dim=-1)
    # for d in reversed(obb):
    xywhr = obb[:,:5]
    xyxyxyxy = ops.xywhr2xyxyxyxy(xywhr)
    box = xyxyxyxy.reshape(-1, 4, 2).squeeze()
    box = box.tolist()
    print("======box:",box)
    cv_im = cv2.imread(input_img)
    # cv_im = cv2.imread(r"D:\data\Temperature\test\yolo_obb\images\train\yc0xc6118.jpg")
    for d in box:
        lw = max(round(sum(cv_im.shape) / 2 * 0.003), 2) # line width
        tf = max(lw - 1, 1) # font thickness
        sf = lw / 3 # font scale
        p1 = [int(b) for b in d[0]]
        cv2.polylines(cv_im, [np.asarray(d, dtype=int)], True, (255, 255, 0), lw)
        w, h = cv2.getTextSize("light_group", 0, fontScale=sf, thickness=tf)[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(cv_im, p1, p2, (0,255,0), -1, cv2.LINE_AA)  # filled
        cv2.putText(
                    cv_im,
                    "light_group",
                    (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    sf,
                    (255, 255, 255),
                    thickness=tf,
                    lineType=cv2.LINE_AA,
                )
    cv2.imwrite(save_path + file, cv_im)
    cv2.imshow("cv_im", cv2.resize(cv_im,(800,800)))
    cv2.waitKey(0)
