import cv2  # [1]导入OpenCv开源库
import numpy as np

image_path = r"E:\pycharm_project\tfservingconvert\maskrcnn_data_deal\labelme2mask\chaitu\data\yc2132xc2128.jpg"
srcImg = cv2.imread(image_path)  # [2]将图片加载到内存

# cv2.namedWindow("[srcImg]", cv2.WINDOW_AUTOSIZE)  # [3]创建显示窗口
# cv2.imshow("[srcImg]", srcImg)  # [4]在刚才创建的显示窗口中显示刚在加载的图片
# cv2.waitKey(0)

# ========================================================================================================
# 模块说明:
#    由于OpenCv中,imread()函数读进来的图片,其本质上就是一个三维的数组,这个NumPy中的三维数组是一致的,所以设置图片的
#  ROI区域的问题,就转换成数组的切片问题,在Python中,数组就是一个列表序列,所以使用列表的切片就可以完成ROI区域的设置
# ========================================================================================================

img_roi1 = srcImg[0:900,0:900]
img_roi2 = srcImg[0:900,700:1600]
img_roi3 = srcImg[700:1600,0:900]
img_roi4 = srcImg[700:1600,700:1600]
# image_save_path = "%s%d%s" % (image_save_path_head, seq, image_save_path_tail)  ##将整数和字符串连接在一起
# cv2.imshow("im1",img_roi1)
# print(img_roi1.shape)
# cv2.waitKey(0)
cv2.imwrite("cat_ROI_1.jpg", img_roi1)
cv2.imwrite("cat_ROI_2.jpg", img_roi2)
cv2.imwrite("cat_ROI_3.jpg", img_roi3)
cv2.imwrite("cat_ROI_4.jpg", img_roi4)
