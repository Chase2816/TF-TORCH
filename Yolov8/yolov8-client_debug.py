import requests
import numpy as np
import cv2
import sys, os, shutil
import json
from PIL import Image, ImageDraw, ImageFont,ImageColor
import torch
import time
import torchvision
import ops
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import batched_nms
from numpy import ndarray
import random

random.seed(0)

def upsample_forward_numpy(input, scales=(1,1), method="nearest_neighbor", align_corners=False):
    """
    param input: (N, C, H, W)
    param scales: (scale_h, scale_w)
    param method: [nearest_neighbor, bilinear]
    param align_corners: [True, False]
    """
    N, C, H, W = input.shape

    out_h = scales[0] * H
    out_w = scales[1] * W

    output = np.zeros((N, C, out_h, out_w), dtype=np.float32)

    if method == "nearest_neighbor":
        for n in np.arange(N):
            for c in np.arange(C):
                for oh in np.arange(out_h):
                    for ow in np.arange(out_w):
                        ih = oh // scales[0]
                        iw = ow // scales[1]
                        output[n, c, oh, ow] = input[n, c, ih, iw]
    elif method == "bilinear":
        if align_corners == False:
            """中心对齐，投影目标图的横轴和纵轴到原图上"""
            hs_p = np.array([(i + 0.5) / out_h * H - 0.5 for i in range(out_h)], dtype=np.float32)
            ws_p = np.array([(i + 0.5) / out_w * W - 0.5 for i in range(out_w)], dtype=np.float32)
        else:
            stride_h = (H - 1) / (out_h - 1)
            stride_w = (W - 1) / (out_w - 1)
            hs_p = np.array([i * stride_h for i in range(out_h)], dtype=np.float32)
            ws_p = np.array([i * stride_w for i in range(out_w)], dtype=np.float32)
        hs_p = np.clip(hs_p, 0, H - 1)
        ws_p = np.clip(ws_p, 0, W - 1)

        """找出每个投影点在原图纵轴方向的近邻点坐标对"""
        # ih_0的取值范围是0 ~(H - 2), 因为ih_1 = ih_0 + 1
        hs_0 = np.clip(np.floor(hs_p), 0, H - 2).astype(np.int)
        """找出每个投影点在原图横轴方向的近邻点坐标对"""
        # iw_0的取值范围是0 ~(W - 2), 因为iw_1 = iw_0 + 1
        ws_0 = np.clip(np.floor(ws_p), 0, W - 2).astype(np.int)

        """
        计算目标图各个点的像素值
        """
        us = hs_p - hs_0
        vs = ws_p - ws_0
        _1_us = 1 - us
        _1_vs = 1 - vs
        for n in np.arange(N):
            for c in np.arange(C):
                for oh in np.arange(out_h):
                    ih_0, ih_1 = hs_0[oh], hs_0[oh] + 1 # 原图的坐标
                    for ow in np.arange(out_w):
                        iw_0, iw_1 = ws_0[ow], ws_0[ow] + 1 # 原图的坐标
                        output[n, c, oh, ow] = input[n, c, ih_0, iw_0] * _1_us[oh] * _1_vs[ow] + input[n, c, ih_0, iw_1] * _1_us[oh] * vs[ow] + input[n, c, ih_1, iw_0] * us[oh] * _1_vs[ow] + input[n, c, ih_1, iw_1] * us[oh] * vs[ow]
    return  output

def fromarray(im):
        # Update self.im from a numpy array
        im = im if isinstance(im, Image.Image) else Image.fromarray(im)
        draw = ImageDraw.Draw(im)
        
def scale_image(im1_shape, masks, im0_shape, ratio_pad=None):
    """
    img1_shape: model input shape, [h, w]
    img0_shape: origin pic shape, [h, w, 3]
    masks: [h, w, num]
    """
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    # masks = masks.permute(2, 0, 1).contiguous()
    # masks = F.interpolate(masks[None], im0_shape[:2], mode='bilinear', align_corners=False)[0]
    # masks = masks.permute(1, 2, 0).contiguous()
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))

    if len(masks.shape) == 2:
        masks = masks[:, :, None]
    return masks

def masks_show(im,masks, colors, im_gpu, alpha=0.5, pil=False,retina_masks=False):
        """Plot masks at once.
        Args:
            masks (tensor): predicted masks on cuda, shape: [n, h, w]
            colors (List[List[Int]]): colors for predicted masks, [[r, g, b] * n]
            im_gpu (tensor): img is in cuda, shape: [3, h, w], range: [0, 1]
            alpha (float): mask transparency: 0.0 fully transparent, 1.0 opaque
        """
        if pil:
            # convert to numpy first
            im = np.asarray(im).copy()
        if len(masks) == 0:
            im[:] = im_gpu.permute(1, 2, 0).contiguous().cpu().numpy() * 255
        colors = torch.tensor(colors, dtype=torch.float32) / 255.0
        colors = colors[:, None, None]  # shape(n,1,1,3)
        # masks = masks.unsqueeze(3)  # shape(n,h,w,1)
        masks_color = masks * (colors * alpha)  # shape(n,h,w,3)

        inv_alph_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)
        mcs = (masks_color * inv_alph_masks).sum(0) * 2  # mask color summand shape(n,h,w,3)

        im_gpu = im_gpu.flip(dims=[0])  # flip channel
        im_gpu = im_gpu.permute(1, 2, 0).contiguous()  # shape(h,w,3)
        im_gpu = im_gpu * inv_alph_masks[-1] + mcs
        im_mask = (im_gpu * 255).byte().cpu().numpy()
        im[:] = im_mask if retina_masks else scale_image(im_gpu.shape, im_mask, im.shape)
        if pil:
            # convert im back to PIL and update draw
            fromarray(im)

def letterbox(im: ndarray,
              new_shape: Union[Tuple, List] = (640, 640),
              color: Union[Tuple, List] = (114, 114, 114)) \
        -> Tuple[ndarray, float, Tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # new_shape: [width, height]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    # Compute padding [width, height]
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[
        1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=color)  # add border
    return im, r, (dw, dh)

def blob(im: ndarray, return_seg: bool = False) -> Union[ndarray, Tuple]:
    if return_seg:
        seg = im.astype(np.float32) / 255
    im = im.transpose([2, 0, 1])
    im = im[np.newaxis, ...]
    im = np.ascontiguousarray(im).astype(np.float32) / 255
    if return_seg:
        return im, seg
    else:
        return im


def seg_postprocess(
        data: Tuple[Tensor],
        shape: Union[Tuple, List],
        conf_thres: float = 0.25,
        iou_thres: float = 0.65) \
        -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    assert len(data) == 2
    h, w = shape[0] // 4, shape[1] // 4  # 4x downsampling
    outputs, proto = (i[0] for i in data)
    bboxes, scores, labels, maskconf = outputs.split([4, 1, 1, 32], 1)
    scores, labels = scores.squeeze(), labels.squeeze()
    idx = scores > conf_thres
    bboxes, scores, labels, maskconf = \
        bboxes[idx], scores[idx], labels[idx], maskconf[idx]
    idx = batched_nms(bboxes, scores, labels, iou_thres)
    bboxes, scores, labels, maskconf = \
        bboxes[idx], scores[idx], labels[idx].int(), maskconf[idx]
    masks = (maskconf @ proto).sigmoid().view(-1, h, w)
    masks = crop_mask(masks, bboxes / 4.)
    masks = F.interpolate(masks[None],
                          shape,
                          mode='bilinear',
                          align_corners=False)[0]
    masks = masks.gt_(0.5)[..., None]
    return bboxes, scores, labels, masks


def det_postprocess(data: Tuple[Tensor, Tensor, Tensor, Tensor]):
    assert len(data) == 4
    num_dets, bboxes, scores, labels = (i[0] for i in data)
    nums = num_dets.item()
    bboxes = bboxes[:nums]
    scores = scores[:nums]
    labels = labels[:nums]
    return bboxes, scores, labels


def crop_mask(masks: Tensor, bboxes: Tensor) -> Tensor:
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(bboxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = torch.arange(w, device=masks.device,
                     dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = torch.arange(h, device=masks.device,
                     dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

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

def crop_mask_np(masks: Tensor, bboxes: Tensor) -> Tensor:
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(bboxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    r = torch.arange(w, device=masks.device,
                     dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = torch.arange(h, device=masks.device,
                     dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def sigmoid(x): # sigmoid 函数实现
    return 1.0 / (1 + np.exp(-x))

def process_mask_np(protos, masks_in, bboxes, shape, upsample=False):
    """
    Crop before upsample.
    proto_out: [mask_dim, mask_h, mask_w]
    out_masks: [n, mask_dim], n is number of masks after nms
    bboxes: [n, 4], n is number of masks after nms
    shape:input_image_size, (h, w)

    return: h, w, n
    """

    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    masks = sigmoid((masks_in @ protos.float().reshape(c, -1))).reshape(-1, mh, mw)  # CHW

    downsampled_bboxes = bboxes.copy()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih

    masks = crop_mask_np(masks, downsampled_bboxes)  # CHW
    if upsample:
        # masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
        masks = cv2.resize(masks, shape, interpolation = cv2.INTER_LINEAR)  # 缩小到0.3 

    return masks.gt_(0.5)

def inter(img):
    '''
    Task01 图像缩放
    :param img:
    :return:
    '''
    scale_percent = 30       # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # 双线性插值缩小
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)  # 缩小到0.3 

    fx = 1.2  
    fy = 1.2
    # 方法1：最近邻插值放大
    nearest_resized = cv2.resize(img, dsize=None, fx=fx, fy=fy, interpolation = cv2.INTER_NEAREST)
    # 方法2：双线性插值放大
    linear_resized = cv2.resize(img, dsize=None, fx=fx, fy=fy, interpolation = cv2.INTER_LINEAR)
    print('Resized Dimensions : ',resized.shape)

    cv2.imshow("Resized image", resized)
    cv2.imshow("INTER_NEAREST image", nearest_resized)
    cv2.imshow("INTER_LINEAR image", linear_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

# detection model classes
CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
           'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
           'scissors', 'teddy bear', 'hair drier', 'toothbrush')

# colors for per classes
COLORS = {
    cls: [random.randint(0, 255) for _ in range(3)]
    for i, cls in enumerate(CLASSES)
}

# colors for segment masks
MASK_COLORS = np.array([(255, 56, 56), (255, 157, 151), (255, 112, 31),
                        (255, 178, 29), (207, 210, 49), (72, 249, 10),
                        (146, 204, 23), (61, 219, 134), (26, 147, 52),
                        (0, 212, 187), (44, 153, 168), (0, 194, 255),
                        (52, 69, 147), (100, 115, 255), (0, 24, 236),
                        (132, 56, 255), (82, 0, 133), (203, 56, 255),
                        (255, 149, 200), (255, 55, 199)],
                       dtype=np.float32) / 255.

# alpha for segment masks
ALPHA = 0.5
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
input_imgs = r"E:\pycharm_project\tfservingconvert\yolov8\model-yolo_lg\images/"
files = os.listdir(input_imgs)
# save_path = r"E:\pycharm_project\tfservingconvert\water_flowers\outputs\out_imgs2"
save_path = r"E:\pycharm_project\tfservingconvert\yolov8\model-yolo_lg\tfserver_out"
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)


for file in files:
    # input_img = r"E:\data\diode-opt\imgs\20200611_84.jpg"
    # input_img = r"E:\pycharm_project\tfservingconvert\tf1.15v3\yc960xc1484.jpg"
    if file.split('.')[1] == "xml": continue
    input_img = input_imgs + file
    img = Image.open(input_img)
    img1 = img.resize((640, 640))
    # img1 = img.resize((224, 224))
    image_np = np.array(img1)
    image_np = image_np / 255.
    img_data = image_np[np.newaxis, :].tolist()
    
    print()
    data = {"instances": img_data}

    frame = cv2.imread(input_img)
    # frame = cv2.imread(r"E:\data\bzz_data\split-material-0\yc4264xc266.jpg")
    fh, fw, fc = frame.shape
    im, r, (dw, dh)= letterbox(frame) # Resize to new shape by letterbox
    blob = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    blob = np.ascontiguousarray(blob)  # contiguous
    blob = np.float32(blob) / 255.0    # 0 - 255 to 0.0 - 1.0
    blob = blob[None]  # expand for batch dim
    
    # data = {"instances": blob.transpose(0,2,3,1).tolist()}

    
    preds = requests.post("http://172.20.112.102:9911/v1/models/yolov8:predict", json=data)
    # preds = requests.post("http://10.8.111.172:8301/v1/models/model-yolo_lg:predict", json=data)

    # preds = requests.post("http://localhost:8101/v1/models/model-lg:predict", json=data)
    # preds = requests.post("http://localhost:8201/v1/models/model-yolo_lg:predict", json=data)
    print(preds)
    predictions = json.loads(preds.content.decode('utf-8'))["predictions"][0]
    pred_det = np.array(predictions['output0']) # 37,8400
    
    pred_mask = np.array(predictions['output1'])    
          
    # np操作
    # xc = pred_det[4:5] > 0.25  # [1,8400] cfd_thred
    # bboxes = pred_det.transpose(1,0)[xc[0]]
    # box, cls, mask = bboxes[:,0:4],bboxes[:,4:5],bboxes[:,5:] # [91,4] [91,1] [91,32]
    # box = xywh2xyxy(box)  # center_x, center
    # conf, j = cls[cls.argmax(1)],cls.argmax(1)[:,np.newaxis]
    # x = np.concatenate((box, conf, j, mask), 1)[conf.reshape(-1) > 0.25] # [91,38] cfd_thred
    # # x = x[x[:, 4].argsort(descending=True)[:max_nms]]
    # c = x[:, 5:6] * 7680  # classes max_wh=7680  [91,1]
    
    # boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
    
    # cv_im = cv2.imread(input_img)
    # pred = nms(x, 0.45,0.25)
    
    # pred[:, :4] = scale_boxes(frame.shape[:2], pred[:, :4], (800,800,3))
    # mw,mh = pred_mask.shape[:2]
    # # ih,iw = frame.shape[:2]
    # ih,iw = frame.shape[:2]
    # masks = sigmoid((pred[:,6:] @ np.float32(pred_mask).transpose(2,0,1).reshape(32,-1))).reshape(-1, 160, 160)  # CHW
    
    # masks_roi = []
    # for obj_mask,bbox in zip(masks, pred[:, :4]):
    #     mx1 = max(0,np.int32((bbox[0] * 0.25)))
    #     my1 = max(0,np.int32((bbox[1] * 0.25)))
    #     mx2 = max(0,np.int32((bbox[2] * 0.25)))
    #     my2 = max(0,np.int32((bbox[3] * 0.25)))
    #     masks_roi.append(obj_mask[my1:my2,mx1:mx2])
        
    
    # color_mask = np.zeros((ih,iw,3),dtype=np.uint8)
    # black_mask = np.zeros((ih,iw),dtype=np.float32)
    # mv = cv2.split(color_mask)
    
    # for i, det in enumerate(pred):        

    #     # solid_color = np.expand_dims(np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
    #     # masks_ = cv2.resize(masks[i][None], image_np.shape[1:], interpolation = cv2.INTER_LINEAR)  # 缩小到0.3 
        
        
    #     score = det[4]
    #     boxes = det[:4]        
    #     label = int(det[5])
    #     x1,y1,x2,y2 = int(boxes[0]), int(boxes[1]), int(boxes[2]), int(boxes[3])    

    #     color = randomcolor()
    #     rgb = ImageColor.getrgb(color)
    #     result_mask = cv2.resize(masks_roi[i],(int(boxes[2]) - int(boxes[0]),int(boxes[3])-int(boxes[1])))
    #     result_mask[result_mask > 0.5] = 1.0
    #     result_mask[result_mask <= 0.5] = 0.0
    #     rh, rw = result_mask.shape
    #     if (y1+rh) >= ih:
    #         rh = ih - y1
    #     if (x1+rw) >= iw:
    #         rw = iw - x1
    #     black_mask[y1:y1+rh, x1:x1+rw] = result_mask[0:rh, 0:rw]
    #     mv[2][black_mask == 1], mv[1][black_mask == 1], mv[0][black_mask == 1] = \
    #             [np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256)]
            
        
    #     # pil_solid_color = Image.fromarray(np.uint8(masks[0][i])).convert("RGBA")
    #     # mask_pil = Image.fromarray(np.uint8(255.0 * 0.5 * masks[0][i])).convert("L")
    #     # img_pil = Image.composite(pil_solid_color, img1, mask_pil)
    #     # print(img_pil)
    #     # m1 = np.uint8(masks[0][i] * 255.)
    #     cv2.imshow("q",black_mask)
    #     cv2.imshow("q1",result_mask)
    #     cv2.imshow("q2",color_mask)
    #     # cv2.waitKey(0)
    #     # img_pil.show()
    #     print(f"det: {i}/{pred.shape[0]}- boxes: {boxes} - score: {score} - label: {names[label]}")
    #     cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])),
    #                     (int(boxes[2]), int(boxes[3])), (0, 255, 0))
    #     cv2.putText(frame, f"{names[label]}-{score:.2f}",
    #                     (int(boxes[0]+3), int(boxes[1]-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
    #                     (255, 255, 0))
    
    # color_mask = cv2.merge(mv)
    # dst = cv2.addWeighted(frame,0.5,color_mask,0.5,0)
    # cv2.imshow("dst", dst)
    # cv2.imshow("cv_im", frame)
    # cv2.waitKey(0)
    

        
    
    
    
    pred_det = pred_det[np.newaxis,:]
    pred_mask = pred_mask[np.newaxis,:]
    pred_mask = np.transpose(pred_mask,(0,3,1,2))

    p = ops.non_max_suppression(torch.tensor(torch.from_numpy(pred_det)),
                                    0.25,
                                    0.7,
                                    agnostic=False,
                                    max_det=300,
                                    nc=len({0:"light_group"}),
                                    classes=None)
    results = []
 
    proto = torch.from_numpy(pred_mask) # second output is len 3 if pt, but only 1 if exported
    
    for i, pred in enumerate(p):
        # orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
        # path = self.batch[0]
        # img_path = path[i] if isinstance(path, list) else path
        print(type(pred[:,6:].float))
        print(image_np.shape[:2])
        masks = ops.process_mask(proto[i], pred[:, 6:], pred[:, :4], image_np.shape[:2], upsample=True)  # HWC

        # masks_show(bgr,masks[0],colors=[0],im_gpu=bgr) 
        color = randomcolor()
        rgb = ImageColor.getrgb(color)
        # solid_color = np.expand_dims(np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
        pil_solid_color = Image.fromarray(np.uint8(masks[0])).convert("RGBA")
        mask_pil = Image.fromarray(np.uint8(255.0 * 0.5 * masks[0])).convert("L")
        img_pil = Image.composite(pil_solid_color, img1, mask_pil)
        print(img_pil)
        
        img_pil.show()
        m1 = np.uint8(masks[0].numpy() * 255.)
        cv2.imshow("q",m1)
        cv2.waitKey(0)
        
        pred[:, :4] = ops.scale_boxes(image_np.shape[:2], pred[:, :4], (800,800,3))
        boxes = pred[:,:6]
        print(boxes)
        boxes = boxes.numpy()
        cv_im = cv2.imread(input_img)
        
        seg_img = torch.asarray(seg_img[dh:H - dh, dw:W - dw, [2, 1, 0]])
        
        masks = masks[:, dh:H - dh, dw:W - dw]
        indices = (labels % len(MASK_COLORS)).long()
        mask_colors = torch.asarray(MASK_COLORS)[indices]
        mask_colors = mask_colors.view(-1, 1, 1, 3) * ALPHA
        mask_colors = masks @ mask_colors
        inv_alph_masks = (1 - masks * 0.5).cumprod(0)
        mcs = (mask_colors * inv_alph_masks).sum(0) * 2
        seg_img = (seg_img * inv_alph_masks[-1] + mcs) * 255
        draw = cv2.resize(seg_img.cpu().numpy().astype(np.uint8),
                            draw.shape[:2][::-1])
        cv2.imshow("d",draw)
        cv2.waitKey(0)
        
        for i, det in enumerate(boxes):
            
        
            score = det[4]
            boxes = det[:4]
            conf = det[5:]
            label = det[5:].argmax()
            
            cv2.rectangle(cv_im, (int(boxes[0]), int(boxes[1])),
                        (int(boxes[2]), int(boxes[3])), (0, 255, 0))
            cv2.putText(cv_im, names[0] + "-" + str(score),
                        (int(boxes[0]+3), int(boxes[1]-3)), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                        (255, 255, 0))
        cv2.imshow("cv_im", cv_im)
        cv2.waitKey(0)

