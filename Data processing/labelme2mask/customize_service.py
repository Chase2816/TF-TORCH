import cv2
import numpy as np
import json


category_types = ["Background", "Tom", "Jerry"]

img = cv2.imread("images/image.png")
h, w = img.shape[:2]
mask = np.zeros([h, w, 1], np.uint8)    # 创建一个大小和原图相同的空白图像

with open("images/image.json", "r") as f:
    label = json.load(f)

shapes = label["shapes"]
for shape in shapes:
    category = shape["label"]
    points = shape["points"]
    # 填充
    points_array = np.array(points, dtype=np.int32)
    mask = cv2.fillPoly(mask, [points_array], category_types.index(category))

cv2.imwrite("masks/image.png", mask)

#
# for shape in shapes:
#     category = shape["label"]
#     points = shape["points"]
#     points_array = np.array(points, dtype=np.int32)
#     if category == "Tom":
#     	# 调试时将Tom的填充颜色改为255，便于查看
#         mask = cv2.fillPoly(mask, [points_array], 255)
#     else:
#         mask = cv2.fillPoly(mask, [points_array], category_types.index(category))
#
# cv2.imshow("mask", mask)
# cv2.waitKey(0)
