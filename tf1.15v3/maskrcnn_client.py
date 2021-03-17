import requests,os
import json
import PIL
import numpy as np
from PIL import Image, ImageDraw, ImageColor, ImageFont
import cv2
from matplotlib import pyplot as plt
import random


def resize_predictions_detection_masks(image_shape=('height', 'width', 'channels'), mask_threshold=0.):
    # http://172.20.112.102:8501/v1/models/model-group/metadata
    # preds = requests.post("http://172.20.112.102:8501/v1/models/model-group:predict", json=data)
    preds = requests.post("http://172.20.112.102:8511/v1/models/model-light_group:predict", json=data)
    predictions = json.loads(preds.content.decode('utf-8'))["predictions"][0]

    detection_masks = []
    detection_boxes = []

    for idx, detection_mask in enumerate(predictions['detection_masks']):
        if predictions['detection_scores'][idx] < 0.1:  # 0.5
            continue

        height, width, _ = image_shape
        mask = np.zeros((height, width))
        ymin, xmin, ymax, xmax = predictions['detection_boxes'][idx]
        ymin, xmin, ymax, xmax = int(ymin * height), int(xmin * width), int(ymax * height), int(xmax * width)
        bbox_height, bbox_width = ymax - ymin, xmax - xmin
        print(bbox_height, bbox_width)

        bbox_mask = np.array(
            PIL.Image.fromarray(np.uint8(np.array(detection_mask) * 255), mode='L').resize(
                size=(bbox_width, bbox_height), resample=Image.NEAREST)
        )

        mask[ymin:ymax, xmin:xmax] = bbox_mask
        mask_threshold = mask_threshold  # 0.3 0.5
        mask = np.where(mask > (mask_threshold * 255), 1,
                        mask)
        if mask_threshold > 0: mask = np.where(mask != 1, 0, mask)

        print('Index (Example): ', idx)
        print('detection_mask shape():', np.array(detection_mask).shape)
        print('detection_boxes:', predictions['detection_boxes'][idx])
        print('detection_boxes@image.size (ymin, xmin, ymax, xmax) :', [ymin, xmin, ymax, xmax])
        print('detection_boxes (height, width):', (bbox_height, bbox_width))
        draw = ImageDraw.Draw(img)
        draw.rectangle((xmin, ymin, xmax, ymax))
        # img.show()

        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
        detection_boxes.append([xmin, ymin, xmax, ymax])

        detection_masks.append(mask.astype(np.uint8))

    return detection_masks, detection_boxes


def get_classes_name():
    # Load names of classes
    classesFile = "mscoco_labels.names"
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    return classes

# img_path = r"F:\java_demo\maskrcnn-demo\src\main\resources\images\202005130001.jpg"
# img_path = r"E:\pycharm_project\Data-proessing\server_test\2.jpg"
img_path = r"F:\无人机数据\h20t红外\可见光组串训练样本\result3\yc1599xc1596sr1.jpg"
img = Image.open(img_path)
print(img.size)
image_np = np.array(img)
print(image_np[np.newaxis, :].shape)
img_data = image_np[np.newaxis, :].tolist()
data = {"instances": img_data}

detection_masks, detection_boxes = resize_predictions_detection_masks(
    image_shape=image_np.shape
)
print(detection_masks)
print(detection_masks[1])
print("-----------")
print(len(detection_masks[1]))
print(detection_boxes)

img_pil = Image.open(img_path)
alpha = 0.5
for mask in detection_masks:
    print(mask)
    # exit()
    color = "red"
    rgb = ImageColor.getrgb(color)
    solid_color = np.expand_dims(np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
    pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert("RGBA")
    mask_pil = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert("L")
    img_pil = Image.composite(pil_solid_color, img_pil, mask_pil)
    print(img_pil)

setFont = ImageFont.truetype(r'E:\pycharm_project\tfservingconvert\tf1.15v3\arial.ttf', 50)
colors = ["red","yellow","blue","white","black","red","yellow","blue","white","black","red","yellow","blue","white","black","red","yellow","blue","white","black"]

for i, boxes in enumerate(detection_boxes):
    print(detection_masks)
    draw = ImageDraw.Draw(img_pil)
    draw.rectangle((boxes[0], boxes[1], boxes[2], boxes[3]), outline=colors[i],width=3)
    draw.text((boxes[0], boxes[1]), str(i), font=setFont, fill=colors[i+1])
img_pil.show()
img_pil.save("outputs/"+os.path.basename(img_path))

#
# plt.figure(figsize=(10, 8))
# plt.imshow(img_pil)
# plt.title("TensorFlow Mask RCNN-Inception_v2_coco")
# plt.axis("off")
# plt.show()
