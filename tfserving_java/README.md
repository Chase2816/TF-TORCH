yolov3: https://github.com/pjreddie/darknet.git --> https://github.com/mystic123/tensorflow-yolo-v3.git
tf1: yolov3.weights --> frozen-model.pb --> save_model.pb

yolov4: https://github.com/AlexeyAB/darknet.git --> https://github.com/hunglc007/tensorflow-yolov4-tflite.git
tf2: yolov4.weights --> save_model.pb

maskrcnn: https://github.com/tensorflow/models.git
model.ckpt --> https://github.com/tensorflow/models/blob/master/research/object_detection/export_inference_graph.py --> save_model.pb
