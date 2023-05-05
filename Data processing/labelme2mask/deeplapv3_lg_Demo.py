import requests,os
import json
import PIL
import numpy as np
from PIL import Image, ImageDraw, ImageColor, ImageFont
import cv2
from matplotlib import pyplot as plt
import random
# import request

# http://172.20.112.102:8501/v1/models/model-group/metadata

def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

img_path = r"E:\data\image_segmentation_test\eval\Images\ChaoYang1_hot_yc2132xc2926.jpg"
img = Image.open(img_path)
print(img.size)
resizeimg = img.resize((513,513))
print(resizeimg)
image_np = np.array(resizeimg)
print(image_np[np.newaxis, :].shape)
img_data = image_np[np.newaxis, :].tolist()
data = {"signature_def": img_data}
print(image_np.shape)

# request.model_spec.signature_name = 'serving_default'
data = json.dumps({"signature_name": "predict_object", "instances": img_data})

# preds = requests.post("http://172.20.112.102:8791/v1/models/model-deeplapv3_lg:predict", json=data)
preds = requests.post("http://172.20.112.102:8791/v1/models/model-deeplapv3_lg:predict", data=data)
predictions = json.loads(preds.content.decode('utf-8'))["predictions"][0]
print(predictions)
print(np.array(predictions).shape)

mask = np.array(predictions).astype(np.uint8)
img_pil = Image.open(img_path)
img_pil = img_pil.resize((513,513))

alpha = 0.5

color = randomcolor()
rgb = ImageColor.getrgb(color)
solid_color = np.expand_dims(np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert("RGBA")
mask_pil = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert("L")
img_pil = Image.composite(pil_solid_color, img_pil, mask_pil)
print(img_pil)

img_pil.show()
# img_pil.save("outputs/"+os.path.basename(img_path))

