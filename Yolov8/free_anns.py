
from ultralytics.yolo.data.annotator import auto_annotate

auto_annotate(data=r"E:\pycharm_project\tfservingconvert\maskrcnn_data_deal\mmdet_datadeal\data\images\test2019", det_model="yolov8x.pt", sam_model='sam_b.pt')
