from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf
import sys
sys.path.append("../")
#  faster rcnn pb 转换 tensorflowservice savemodel pb
# export_dir = r'E:\cvode\models\research\object_detection\mycode\logs/frcnn_res_model/000001'
# graph_pb = r'E:\cvode\models\research\object_detection\mycode\data\VOC6_augment50000_graph\frozen_inference_graph.pb'
# builder = tf.frcnn_res_model.builder.SavedModelBuilder(export_dir)
#
# with tf.gfile.GFile(graph_pb, "rb") as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#
#
# sigs = {}
#
#
# with tf.Session(graph=tf.Graph()) as sess:# name="" is important to ensure we don't get spurious prefixing
#     tf.import_graph_def(graph_def, name="")
#     g = tf.get_default_graph()
#
#     image_tensor = g.get_tensor_by_name('image_tensor:0')
#     detection_boxes = g.get_tensor_by_name('detection_boxes:0')
#     detection_scores = g.get_tensor_by_name('detection_scores:0')
#     detection_classes = g.get_tensor_by_name('detection_classes:0')
#     num_detections = g.get_tensor_by_name('num_detections:0')
#     print(type(detection_boxes))
#     out = {'boxes':detection_boxes,'scores':detection_scores,'classes':detection_classes,'num':num_detections}
#     print(type(out))
#     sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
#         tf.frcnn_res_model.signature_def_utils.predict_signature_def(
#             {"in": image_tensor},
#             {'boxes':detection_boxes,'scores':detection_scores,'classes':detection_classes,'num':num_detections},)
#             # {"scores":detection_scores},
#             # {"classes":detection_classes},
#             # {"nums":num_detections})
#
#     builder.add_meta_graph_and_variables(sess,
#                                          [tag_constants.SERVING],
#                                          signature_def_map = sigs)
# builder.save()


# docker run -p 8501:8501 --mount type=bind,source=C:\tmp\tfserving\serving\tensorflow_serving\servables\tensorflow\testdata\voc6_frcnn_model,target=/models/my_model -e MODEL_NAME=my_model -t tensorflow/serving '&'
#
# curl -XPOST http://localhost:8501/v1/models/my_model:predict -d "{\"instances\":'http://img0.imgtn.bdimg.com/it/u=58173181,2870617913&fm=26&gp=0.jpg'}"

# saved_model_cli show --dir C:\tmp\tfserving\serving\tensorflow_serving\servables\tensorflow\testdata\voc6_frcnn_model\000001/ --all



# yolo3 pb转换tensorflowservice savemodel pb
export_dir = 'model-best-0727/1'
graph_pb = '0727_best_model.pb'
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

with tf.gfile.GFile(graph_pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

sigs = {}

with tf.Session(graph=tf.Graph()) as sess:   # name="" is important to ensure we don't get spurious prefixing
    tf.import_graph_def(graph_def, name="")
    g = tf.get_default_graph()

    image_tensor = g.get_tensor_by_name("inputs:0")
    detection_sbbox = g.get_tensor_by_name("detector/yolo-v3/detections:0")
    # detection_mbbox = g.get_tensor_by_name("pred_mbbox/concat_2:0")
    # detection_lbbox = g.get_tensor_by_name("pred_lbbox/concat_2:0")
    print(type(detection_sbbox))
    # out = {'sbbox':detection_sbbox,'mbbox':detection_mbbox,'lbbox':detection_lbbox}
    # out = {''}
    # print(type(out))
    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        tf.saved_model.signature_def_utils.predict_signature_def(
            {"in": image_tensor},
            {"bboxs":detection_sbbox})

            # {'sbbox':detection_sbbox,'mbbox':detection_mbbox,'lbbox':detection_lbbox},)
            # {"scores":detection_scores},
            # {"classes":detection_classes},
            # {"nums":num_detections})

    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         signature_def_map = sigs)
builder.save()














# #coding:utf-8
# import sys, os, io
# import tensorflow as tf
#
# def restore_and_save(input_checkpoint, export_path_base):
#     checkpoint_file = tf.train.latest_checkpoint(input_checkpoint)
#     graph = tf.Graph()
#
#     with graph.as_default():
#         session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
#         sess = tf.Session(config=session_conf)
#
#         with sess.as_default():
#             # 载入保存好的meta graph，恢复图中变量，通过SavedModelBuilder保存可部署的模型
#             print(checkpoint_file)
#             # exit()
#             saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
#             saver.restore(sess, checkpoint_file)
#             print (graph.get_name_scope())
#
#             export_path_base = export_path_base
#             export_path = os.path.join(
#                 tf.compat.as_bytes(export_path_base),
#                 tf.compat.as_bytes(str(count)))
#             print('Exporting trained model to', export_path)
#             builder = tf.frcnn_res_model.builder.SavedModelBuilder(export_path)
#
#             # 建立签名映射，需要包括计算图中的placeholder（ChatInputs, SegInputs, Dropout）和我们需要的结果（project/logits,crf_loss/transitions）
#             """
#             build_tensor_info：建立一个基于提供的参数构造的TensorInfo protocol buffer，
#             输入：tensorflow graph中的tensor；
#             输出：基于提供的参数（tensor）构建的包含TensorInfo的protocol buffer
#                         get_operation_by_name：通过name获取checkpoint中保存的变量，能够进行这一步的前提是在模型保存的时候给对应的变量赋予name
#             """
#
#             image_tensor =tf.frcnn_res_model.utils.build_tensor_info(graph.get_operation_by_name("image_tensor").outputs[0])
#             detection_boxes =tf.frcnn_res_model.utils.build_tensor_info(graph.get_operation_by_name("detection_boxes:0").outputs[0])
#             detection_scores =tf.frcnn_res_model.utils.build_tensor_info(graph.get_operation_by_name("detection_scores:0").outputs[0])
#             detection_classes =tf.frcnn_res_model.utils.build_tensor_info(graph.get_operation_by_name("detection_classes:0").outputs[0])
#
#             num_detections =tf.frcnn_res_model.utils.build_tensor_info(graph.get_operation_by_name("num_detections:0").outputs[0])
#
#             """
#             signature_constants：SavedModel保存和恢复操作的签名常量。
#             在序列标注的任务中，这里的method_name是"tensorflow/serving/predict"
#             """
#             # 定义模型的输入输出，建立调用接口与tensor签名之间的映射
#             labeling_signature = (
#                 tf.frcnn_res_model.signature_def_utils.build_signature_def(
#                     inputs={
#                         "image_tensor":
#                             image_tensor,
#
#                     },
#                     outputs={
#                         "detection_boxes":
#                             detection_boxes,
#                         "detection_scores":
#                             detection_scores,
#                         "detection_classes":
#                             detection_classes,
#                         "num_detections":
#                             num_detections
#                     },
#                     method_name="tensorflow/serving/predict"))
#
#             """
#             tf.group : 创建一个将多个操作分组的操作，返回一个可以执行所有输入的操作
#             """
#             legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
#
#             """
#             add_meta_graph_and_variables：建立一个Saver来保存session中的变量，
#                                           输出对应的原图的定义，这个函数假设保存的变量已经被初始化；
#                                           对于一个SavedModelBuilder，这个API必须被调用一次来保存meta graph；
#                                           对于后面添加的图结构，可以使用函数 add_meta_graph()来进行添加
#             """
#             # 建立模型名称与模型签名之间的映射
#             builder.add_meta_graph_and_variables(
#                 sess, [tf.frcnn_res_model.tag_constants.SERVING],
#                 # 保存模型的方法名，与客户端的request.model_spec.signature_name对应
#                 signature_def_map={
#                     tf.frcnn_res_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
#                        labeling_signature},
#                 legacy_init_op=legacy_init_op)
#
#             builder.save()
#             print("Build Done")
#
# ### 测试模型转换
# tf.flags.DEFINE_string("ckpt_path",     r"E:\cvode\models\research\object_detection\mycode\checkpoint_six\model.ckpt-50000.meta",             "path of source checkpoints")
# tf.flags.DEFINE_string("pb_path",       r"E:\cvode\models\research\object_detection\mycode\checkpoint_six\model",             "path of servable models")
# tf.flags.DEFINE_integer("version",      1,              "the number of model version")
# tf.flags.DEFINE_string("classes",       'LOC',          "multi-models to be converted")
# FLAGS = tf.flags.FLAGS
#
# classes = FLAGS.classes
# input_checkpoint = r'E:\cvode\models\research\object_detection\mycode\checkpoint_six'
# model_path = FLAGS.pb_path
#
# # 版本号控制
# count = FLAGS.version
# modify = False
# if not os.path.exists(model_path):
#     os.mkdir(model_path)
# # else:
# #     for v in os.listdir(model_path):
# #         print(type(v), v)
# #         if int(v) >= count:
# #             count = int(v)
# #             modify = True
# #     if modify:
# #         count += 1
#
# # 模型格式转换
# restore_and_save(input_checkpoint, model_path)