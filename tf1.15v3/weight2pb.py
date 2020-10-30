import tensorflow as tf
import sys
sys.path.append("../")
from tfv3 import yolo_v3
from tfv3.utils import load_weights, load_coco_names, detections_boxes, freeze_graph

"""
python weight2pb.py --class_names fish.names --data_format NHWC --weights_file D:\Projects\darknet\build\darknet\x64\myfile\fish_weights\yolov3_8000.weights --output_graph 1027_fish_model.pb
"""

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'class_names', 'coco.names', 'File with class names')
tf.app.flags.DEFINE_string(
    'weights_file', 'yolov3.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NCHW', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_string(
    'output_graph', 'frozen_darknet_yolov3_model.pb', 'Frozen tensorflow protobuf model output path')

tf.app.flags.DEFINE_bool(
    'tiny', False, 'Use tiny version of YOLOv3')
tf.app.flags.DEFINE_bool(
    'spp', False, 'Use SPP version of YOLOv3')
tf.app.flags.DEFINE_integer(
    'size', 416, 'Image size')



def main(argv=None):
    if FLAGS.tiny:
        model = yolo_v3_tiny.yolo_v3_tiny
    elif FLAGS.spp:
        model = yolo_v3.yolo_v3_spp
    else:
        model = yolo_v3.yolo_v3

    classes = load_coco_names(FLAGS.class_names)

    # placeholder for detector inputs
    inputs = tf.placeholder(tf.float32, [None, FLAGS.size, FLAGS.size, 3], "inputs")

    with tf.variable_scope('detector'):
        detections = model(inputs, len(classes), data_format=FLAGS.data_format)
        load_ops = load_weights(tf.global_variables(scope='detector'), FLAGS.weights_file)

    # Sets the output nodes in the current session
    boxes = detections_boxes(detections)

    with tf.Session() as sess:
        sess.run(load_ops)
        freeze_graph(sess, FLAGS.output_graph)

if __name__ == '__main__':
    tf.app.run()