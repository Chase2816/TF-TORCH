docker run --name model-yolo_lg -p 8201:8501 -p 8200:8500 \
-v E:/ygwl/bzz_data/yolov8/runs/segment/train/weights/model-yolo_lg:/models/model-yolo_lg \
-e MODEL_NAME=model-yolo_lg -t tensorflow/serving
