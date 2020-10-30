import cv2 as cv
import numpy as np
import os

def read_class(file):
    with open(file, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    return classes


class cv_yolov3(object):
    def __init__(self, class_path, net_width, net_height, confThreshold, nmsThreshold):
        '''
        Initialize the parameters
        :param class_path:
        :param net_width: default 416, Width of network's input image
        :param net_height: default 416,Height of network's input image
        :param confThreshold: default 0.5, Confidence threshold
        :param nmsThreshold: default 0.5,Non-maximum suppression threshold
        '''
        self.classes = read_class(class_path)
        self.net_width = net_width
        self.net_height = net_height
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold

    def cv_dnn_init(self, modelConfiguration, modelWeights):
        '''
        Give the configuration and weight files for the model and load the network using them.
        eg:
        modelConfiguration = "checkpoint-bk/yolov3.cfg";
        modelWeights = "checkpoint-bk/yolov3.weights";
        :param modelConfiguration:
        :param modelWeights:
        :return:
        '''
        self.net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    def getOutputsNames(self, net):
        '''
        Get the names of the output layers
        :param net:
        :return:
        '''
        layersNames = net.getLayerNames()
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def drawPred(self, frame, classes, classId, conf, left, top, right, bottom):
        '''
        Draw the predicted bounding box
        :param frame:
        :param classes:
        :param classId:
        :param conf:
        :param left:
        :param top:
        :param right:
        :param bottom:
        :return:
        '''
        cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 1)

        # label = '%.2f' % conf
        #
        # if classes:
        #     assert (classId < len(classes))
        #     label = '%s:%s' % (classes[classId], label)
        #
        # # Display the label at the top of the bounding box
        # labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        # top = max(top, labelSize[1])
        # cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
        #              (255, 255, 255), cv.FILLED)
        # cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)

    def postprocess(self, frame, classes, outs):
        '''
        Remove the bounding boxes with low confidence using non-maxima suppression
        :param frame:
        :param classes:
        :return: outs:[507*85 =(13*13*3)*(5+80),
                 2028*85=(26*26*3)*(5+80),
                 8112*85=(52*52*3)*(5+80)]
        outs中每一行是一个预测值：[x,y,w,h,confs,class_probs_0,class_probs_1,..,class_probs_78,class_probs_79]
        :return:
        '''
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])


        indices = cv.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.drawPred(frame, classes, classIds[i], confidences[i], left, top, left + width, top + height)

    def cv_dnn_forward(self, frame):
        '''
        :param frame:
        :return: outs:[507*85 =13*13*3*(5+80),
                       2028*85=26*26*3*(5+80),
                       8112*85=52*52*3*(5+80)]
        '''

        blob = cv.dnn.blobFromImage(frame, 1 / 255, (self.net_width, self.net_height), [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.getOutputsNames(self.net))
        print(outs[0].shape)

        runtime, _ = self.net.getPerfProfile()
        # print(runtime)
        # exit()
        return outs, runtime

    def yolov3_predict(self, image_path):
        '''
        :param image_path:
        :return:
        '''
        winName = 'im'
        # cv.namedWindow(winName, cv.WINDOW_NORMAL)

        frame = cv.imread(image_path)
        outs, runtime = self.cv_dnn_forward(frame)
        self.postprocess(frame, self.classes, outs)

        label = 'Inference time: %.2f ms' % (runtime * 1000.0 / cv.getTickFrequency())
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        cv.imshow(winName, frame)
        cv.waitKey(0)

    def yolov3_save_predict(self, image_path,save_path=None):
        '''
        :param image_path:
        :return:
        '''
        winName = 'im'
        # cv.namedWindow(winName, cv.WINDOW_NORMAL)

        frame = cv.imread(image_path)
        outs, runtime = self.cv_dnn_forward(frame)
        self.postprocess(frame, self.classes, outs)

        label = 'Inference time: %.2f ms' % (runtime * 1000.0 / cv.getTickFrequency())
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # cv.imshow(winName, frame)
        # cv.waitKey(0)
        file_name = os.path.basename(image_path)
        cv.imwrite(os.path.join(save_path,file_name),frame)


if __name__ == "__main__":

    confThreshold = 0.25  #0.3
    nmsThreshold = 0.45   #0.45  0.5
    net_input_width = 416
    net_input_height = 416

    classesFile = r'E:\pycharm_project\Data-proessing\break-group-yolo\hot.names'
    modelConfiguration = r"E:\pycharm_project\Data-proessing\break-group-yolo\hot-yolov3.cfg"
    modelWeights = r"E:\pycharm_project\Data-proessing\break-group-yolo\hot-yolov3_5000.weights"
    cv_model = cv_yolov3(classesFile, net_input_width, net_input_height, confThreshold, nmsThreshold)
    cv_model.cv_dnn_init(modelConfiguration, modelWeights)
    # cv_model.yolov3_predict(image_path)
    images_path = r'E:\pycharm_project\Data-proessing\break-group-yolo\image'
    save_path = r'E:\pycharm_project\Data-proessing\break-group-yolo\result3'
    img_files = os.listdir(images_path)
    print(f"检测图片数量:{len(img_files)}")
    for i,file in enumerate(img_files):
        print(f"{i}/{len(img_files)} | {file}")
        image_path = os.path.join(images_path , file)
        cv_model.yolov3_save_predict(image_path,save_path)