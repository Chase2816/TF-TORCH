import cv2
from ultralytics import YOLO
import glob
import os


import onnx
from onnx_tf.backend import prepare

model_path = r"D:\data\Temperature\my_work\runs\obb\train4\weights\best.onnx"
onnx_model = onnx.load(model_path)

tf_rep = prepare(onnx_model)
tf_rep.export_graph(r"D:\data\Temperature\my_work\runs\obb\train4\weights\best_obb_saved_model")

exit()
# model = YOLO(r'F:\data\hot_group\workspace\runs\segment\train7\weights\best.pt') 

model = YOLO(r'F:\data\hot_group\pose\train\weights\best.pt') 
# model.export(format='onnx')
# model.export(format='saved_model')
# yolo export model=best.pt format=saved_model -osd
# """
# onnx2tf -i yolov8n.onnx -o saved_model -osd
# onnx2tf -i F:\data\light_group_rx_total\segment\train5\weights\best.onnx -o F:\data\light_group_rx_total\segment\train5\weights\saved_model -osd
# onnx2tf -i E:\pycharm_project\tfservingconvert\ultralytics\my_work\runs\segment\train5\weights\best.onnx -o E:\pycharm_project\tfservingconvert\ultralytics\my_work\runs\segment\train5\weights\best_saved_model -osd
# """

# for im_path in glob.glob(r"F:\data\hot_group\total_seg\all_labelme_hg\*.jpg"):
for im_path in glob.glob(r"F:\A_I\models-master\research\object_detection\my_maskrcnn\deal_data\hot_group\test\*.jpg"):
    
    res = model(im_path)
    cv2.imshow("result", res[0].plot())
    cv2.waitKey(0)
    # savedir = r"F:\data\light_group_rx_total\images\rotateverify\yolo_labels\images\output/" + os.path.basename(im_path)
    # cv2.imwrite(savedir,res[0].plot())
