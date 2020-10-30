import requests
import json
import PIL
import numpy as np
from PIL import Image,ImageDraw


def resize_predictions_detection_masks(image_shape=('height', 'width', 'channels'), mask_threshold=0.):
    #http://172.20.112.102:8501/v1/models/model-group/metadata
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
        print(bbox_height,bbox_width)

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
        draw.rectangle((xmin,ymin,xmax,ymax))
        img.show()
        detection_masks.append(mask.astype(np.uint8))

    return detection_masks

# img_path = r"F:\java_demo\maskrcnn-demo\src\main\resources\images\202005130001.jpg"
img_path = r"E:\pycharm_project\Data-proessing\server_test\2.jpg"
img = Image.open(img_path)
print(img.size)
image_np = np.array(img)
print(image_np[np.newaxis,:].shape)
img_data = image_np[np.newaxis,:].tolist()
data = {"instances": img_data}

detection_masks = resize_predictions_detection_masks(
    image_shape=image_np.shape
)

print(detection_masks)
