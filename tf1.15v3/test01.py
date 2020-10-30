import requests
import json
import cv2
import numpy as np
import tensorflow as tf
from tfv3 import core as utils
from PIL import Image


names = {}
with open("data/classes/coco.names",'r') as data:
    for ID,name in enumerate(data):
        names[ID] = name.strip("\n")

im = cv2.imread("data/kite.jpg")
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
img = cv2.resize(im,(416,416))
img = img/255.

data = {"instances":img[None].tolist()}
preds = requests.post("http://172.20.112.102:9901/v1/models/yolov4:predict",json=data)
# print(preds.json()['predictions'])
predictions = json.loads(preds.content.decode('utf-8'))["predictions"][0]
print(predictions)
print(np.array(predictions)[:,4].max())
pred = np.array(predictions,dtype=np.float32)
print(pred.shape)

boxes = pred[:,0:4]
pred_conf = pred[:,4:]

# (1, 45, 4) (1, 45, 80)
boxes = boxes[None]
pred_conf = pred_conf[None]
boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes[None], (tf.shape(boxes[None])[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.25
    )
pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
image = utils.draw_bbox(im, pred_bbox, classes='')
# image = utils.draw_bbox(image_data*255, pred_bbox)
image = Image.fromarray(image.astype(np.uint8))
image.show()
image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
cv2.waitKey(0)



# image_h, image_w, _ = im.shape
# for i,coor in enumerate(boxes):
#     label = pred_conf[i].argmax()
#     scores = pred_conf[i][label]
#     if scores<0.25:
#         continue
#     coor[0] = int(coor[0] * image_h)
#     coor[2] = int(coor[2] * image_h)
#     coor[1] = int(coor[1] * image_w)
#     coor[3] = int(coor[3] * image_w)
#     c1, c2 = (int(coor[1]), int(coor[0])), (int(coor[3]), int(coor[2]))
#     print(c1,c2,scores,label)
#
#
#     cv2.rectangle(im, c1, c2, (0,255,0))
#     font = cv2.FONT_HERSHEY_SIMPLEX
#
#     cv2.putText(im,names[label]+"-"+str(scores),(c1),font,1.2,(255,255,255))
# cv2.imshow("ss",im)
# cv2.waitKey(0)
