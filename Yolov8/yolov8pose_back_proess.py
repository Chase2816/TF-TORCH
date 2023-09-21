'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-09-20 14:43:54
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-09-20 15:22:50
FilePath: \tfservingconvert\yolov8\onnx_proess\yolov8_back_proess.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import numpy as np
import onnxruntime

def preDeal(img:np.ndarray,H:int,W:int):
    right = (32 - W % 32) % 32
    down = (32 - H % 32) % 32
    x = cv2.copyMaskBorder(img,0,down,0,right,cv2.BORDER_CONSTANT,value=[114,114,114])
    print(x.shape) # (608,448,3)
    x = x/255
    x = np.transpose(x, (2,0,1)) 
    x = np.expand_dims(x, 0)
    x = x.astype(np.float32)
    return x


def postDeal_v8_pose(output:np.ndarray,conf_threshold:float,iou_threshold:float):
    # print(output.shape)  # (56,5584)
    output = np.transpose(output,(1,0))
    
    # [x,y,w,h,conf,点1的x，点1的y，点1的v，点2的x，点2的y，点2的v，....]
    confs = output[:,4]
    conf_index = np.where(confs > conf_threshold)[0]
    output = output[conf_index]
    print(output.shape) # (10,56)
    
    xls = output[:,0] - output[:,2] / 2
    yls = output[: , 1] - output[: , 3] / 2
    output[:,0] = xls
    output[:,1] = yls
    boxes = output[:,:4]  # [x1,y1,w,h]
    
    confs = output[:,4]
    indexes = cv2.dnn.NMSBoxes(boxes, confs, conf_threshold, iou_threshold)
    output = output[indexes]
    print(output.shape) # (1,56)
    
    boxes = output[:,:4]
    confs = output[:,4]
    allTargets_pointXs = output[: , 5::3]
    print(allTargets_pointXs.shape) # (2,17) 两个框，每个框17个点
    allTargets_pointYs = output[: , 6::3]
    
    return boxes, confs, allTargets_pointXs, allTargets_pointYs, len(indexes)


if __name__ == '__main__':
    onnx_path = r"yolov8l-pose.onnx"
    test_path = r"a.jpg"
    conf_threshold = 0.5
    iou_threshold = 0.1
    
    net = onnxruntime.InferenceSession(onnx_path, providers=[('CUDAExecutionProvider',{"device_id":0}),'CPUExecutionProvider'])
    net.set_providers(['CUDAExecutionProvider',[{'device_id':0}]])
    
    img = np.fromfile(test_path,np.uint8)
    img = cv2.imdecode(img,1)
    H,W = img.shape[:2]
    
    x = preDeal(img,H,W)
    output = net.run([net.get_outputs()[0].name], {net.get_inputs()[0].name:x})[0][0]
    boxes,confs,allTargets_pointXs,allTargets_pointYs,length = postDeal_v8_pose(output, conf_threshold, iou_threshold)
    
    for i in range(length):
        box = boxes[i]
        x1,y1,w,h = box
        x2 = x1 + w
        y2 = y1 + h
        
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        if x2 > W:
            x2 = W
        if y2 < H:
            y2 = H
        
        img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
        
        xs = allTargets_pointXs[i]
        ys = allTargets_pointYs[i]
        for (x,y) in zip(xs,ys):
            x = int(x)
            y = int(y)
            img = cv2.circle(img,(x,y),2,(0,255,0),5)
            
    cv2.imencode(".png",img)[1].tof.istitle("b.png")
    