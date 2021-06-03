import requests
import json
import PIL
import numpy as np
from PIL import Image, ImageDraw
import os

from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util


def resize_predictions_detection_masks(image_shape=('height', 'width', 'channels'), mask_threshold=0.):
    # http://172.20.112.102:8501/v1/models/model-group/metadata
    preds = requests.post("http://172.20.112.102:8501/v1/models/model-group:predict", json=data)
    predictions = json.loads(preds.content.decode('utf-8'))["predictions"][0]

    detection_masks = []

    for idx, detection_mask in enumerate(predictions['detection_masks']):
        if predictions['detection_scores'][idx] < 0.5:
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
        # draw = ImageDraw.Draw(img)
        # draw.rectangle((xmin, ymin, xmax, ymax))
        # img.show()
        detection_masks.append(mask.astype(np.uint8))

    return detection_masks, predictions


NUM_CLASSES = 1

PATH_TO_LABELS = os.path.join(r'F:\A_I\models-master\research\object_detection\my_maskrcnn\data',
                              'label_map.pbtxt')
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)
# img_path = r"F:\java_demo\maskrcnn-demo\src\main\resources\images\202005130001.jpg"
img_path = r"E:\data\maskrcnn-data\data0528\test_img\1908.jpg"
img = Image.open(img_path)
print(img.size)
image_np = np.array(img)
print(image_np[np.newaxis, :].shape)
img_data = image_np[np.newaxis, :].tolist()
data = {"instances": img_data}

detection_masks, predictions = resize_predictions_detection_masks(
    image_shape=image_np.shape
)

print(detection_masks)
print(np.array(predictions["detection_boxes"]).shape)
print(np.array(predictions["detection_classes"]).shape)
print(np.array(predictions["detection_scores"]).shape)
print(np.array(predictions["detection_scores"]).shape)
print(np.array(predictions["detection_classes"],dtype=np.int32))


f=1
vis_util.visualize_boxes_and_labels_on_image_array(img_path,f,
    image_np,
    np.array(predictions["detection_boxes"]),
    np.array(predictions["detection_classes"],dtype=np.int32),
    np.array(predictions["detection_scores"]),
    category_index,
    instance_masks=detection_masks,
    use_normalized_coordinates=True,
    line_thickness=4,
    skip_scores=False,
    skip_labels=False,
    min_score_thresh=0.3
)

from matplotlib import pyplot as plt

plt.figure(figsize=(12,8))
plt.imshow(image_np)
arr = img_path.split('\\')
arr = arr[-1]
plt.show()
plt.savefig(r"F:\A_I\models-master\research\object_detection\my_maskrcnn\maskrcnn-data-deal" + '\\' + arr.split('.')[0] + '_labeled.jpg')
