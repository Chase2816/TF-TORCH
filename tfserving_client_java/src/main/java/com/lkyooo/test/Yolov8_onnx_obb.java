package com.lkyooo.test;

import ai.onnxruntime.*;
import com.alibaba.fastjson.JSONObject;
import org.apache.commons.math3.fitting.leastsquares.EvaluationRmsChecker;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.indexer.Indexer;
import org.bytedeco.javacpp.indexer.IntIndexer;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
import org.bytedeco.opencv.opencv_core.*;
import org.opencv.core.MatOfFloat;

import java.nio.FloatBuffer;
import java.text.DecimalFormat;
import java.util.*;
import java.util.Arrays;

import java.util.ArrayList;
import java.util.List;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.javacpp.IntPointer;

import org.bytedeco.artoolkitplus.*;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_imgproc.*;
import org.opencv.core.MatOfInt;

import static org.bytedeco.artoolkitplus.global.ARToolKitPlus.*;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_highgui.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;


/**
 * onnx学习笔记  GTianyu
 */
public class Yolov8_onnx_obb {
    public static OrtEnvironment env;
    public static OrtSession session;
    public static JSONObject names;
    public static long count;
    public static long channels;
    public static long netHeight;
    public static long netWidth;
    public static  float srcw;
    public static  float srch;
    public static float confThreshold = 0.25f;
    public static float nmsThreshold = 0.5f;
    public static Mat src;

    public static void load(String path) {
        String weight = path;
        try{
            env = OrtEnvironment.getEnvironment();
            session = env.createSession(weight, new OrtSession.SessionOptions());
            OnnxModelMetadata metadata = session.getMetadata();
            Map<String, NodeInfo> infoMap = session.getInputInfo();
            TensorInfo nodeInfo = (TensorInfo)infoMap.get("images").getInfo();
//            String nameClass = metadata.getCustomMetadata().get("names");
            System.out.println("getProducerName="+metadata.getProducerName());
            System.out.println("getGraphName="+metadata.getGraphName());
            System.out.println("getDescription="+metadata.getDescription());
            System.out.println("getDomain="+metadata.getDomain());
            System.out.println("getVersion="+metadata.getVersion());
            System.out.println("getCustomMetadata="+metadata.getCustomMetadata());
            System.out.println("getInputInfo="+infoMap);
            System.out.println("nodeInfo="+nodeInfo);
//            System.out.println(nameClass);
//            names = JSONObject.parseObject(nameClass.replace("\"","\"\""));
            count = nodeInfo.getShape()[0];//1 模型每次处理一张图片
            channels = nodeInfo.getShape()[1];//3 模型通道数
            netHeight = nodeInfo.getShape()[2];//640 模型高
            netWidth = nodeInfo.getShape()[3];//640 模型宽
//            System.out.println(names.get(0));
        }
        catch (Exception e){
            e.printStackTrace();
            System.exit(0);
        }
    }

    public static Map<Object, Object> predict(String imgPath) throws Exception {
        src = imread(imgPath);


        return predictor();
    }

    public static Map<Object, Object> predict(Mat mat) throws Exception {
        src=mat;
        return predictor();
    }

    public static OnnxTensor transferTensor(Mat dst){
        cvtColor(dst, dst, COLOR_BGR2RGB);
        dst.convertTo(dst, CV_32FC1, 1f / 255f,0);
        float[] whc = new float[ Long.valueOf(channels).intValue() * Long.valueOf(netWidth).intValue() * Long.valueOf(netHeight).intValue() ];
        FloatIndexer dstindexer = dst.createIndexer();
        dstindexer.get(0, 0, whc);
        float[] chw = whc2cwh(whc);

        OnnxTensor tensor = null;
        try {
            tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), new long[]{count,channels,netWidth,netHeight});
        }
        catch (Exception e){
            e.printStackTrace();
            System.exit(0);
        }
        return tensor;
    }


    //宽 高 类型 to 类 宽 高
    public static float[] whc2cwh(float[] src) {
        float[] chw = new float[src.length];
        int j = 0;
        for (int ch = 0; ch < 3; ++ch) {
            for (int i = ch; i < src.length; i += 3) {
                chw[j] = src[i];
                j++;
            }
        }
        return chw;
    }

    public static Map<Object, Object> predictor() throws Exception{
        srcw = src.cols();
        srch = src.rows();
        System.out.println("width:"+srcw+" hight:"+srch);
        System.out.println("resize: \n width:"+netWidth+" hight:"+netHeight);
        float scaleW=srcw/netWidth;
        float scaleH=srch/netHeight;
        // resize
        Mat dst=new Mat(netWidth, netHeight);
        resize(src, dst, new Size((int) netWidth, (int) netHeight));
        // 转换成Tensor数据格式
        OnnxTensor tensor = transferTensor(dst);
        OrtSession.Result result = session.run(Collections.singletonMap("images", tensor));
        System.out.println("res Data: "+result.get(0));
        OnnxTensor res = (OnnxTensor)result.get(0);
        float[][][] dataRes = (float[][][])res.getValue();
        float[][] data = dataRes[0];

        // 将矩阵转置
        // 先将xywh部分转置
        float rawData[][]=new float[data[0].length][8];
        System.out.println(data.length-1);
        for(int i=0;i<6;i++){
            for(int j=0;j<data[0].length;j++){
                rawData[j][i]=data[i][j];
            }
        }
        // 保存每个检查框置信值最高的类型置信值和该类型下标
//        for(int i=0;i<data[0].length;i++){
//            for(int j=4;j<data.length;j++){
//                if(rawData[i][4]<data[j][i]){
//                    rawData[i][4]=data[j][i];           //置信值
//                    rawData[i][6]=j-4;                  //类型编号
////                    rawData[i][6]=data[5][i];                  //类型编号
//                }
//            }
//        }
        List<ArrayList<Float>> boxes=new LinkedList<ArrayList<Float>>();
        ArrayList<Float> box=null;
        // 置信值过滤,xywh转xyxy
        for(float[] d:rawData){
            // 置信值过滤
            if(d[4]>confThreshold){
                System.out.println("dd[4]===="+d[4]);
                // xywh(xy为中心点)转xyxy
//                d[0]=d[0]-d[2]/2;
//                d[1]=d[1]-d[3]/2;
//                d[2]=d[0]+d[2];
//                d[3]=d[1]+d[3];
                System.out.println("d------"+d);

                // 置信值符合的进行插入法排序保存
                box = new ArrayList<Float>();
                for(float num:d) {
                    box.add(num);
                }
                System.out.println("box------"+box);

                if(boxes.size()==0){
                    boxes.add(box);
                }else {
                    int i;
                    for(i=0;i<boxes.size();i++){
                        if(box.get(4)>boxes.get(i).get(4)){
                            boxes.add(i,box);
                            System.out.println(box.get(4));
                            System.out.println(boxes.get(i).get(4));
                            break;
                        }
                    }
                    // 插入到最后
                    if(i==boxes.size()){
                        boxes.add(box);
                    }
                }
            }
        }

        // 每个框分别有x1、x1、x2、y2、conf、class
        System.out.println(boxes);
        System.out.println(22222);


        // 非极大值抑制
        int[] indexs=new int[boxes.size()];
        Arrays.fill(indexs,1);                       //用于标记1保留，0删除
        for(int cur=0;cur<boxes.size();cur++){
            if(indexs[cur]==0){
                continue;
            }
            ArrayList<Float> curMaxConf=boxes.get(cur);   //当前框代表该类置信值最大的框
            for(int i=cur+1;i<boxes.size();i++){
                if(indexs[i]==0){
                    continue;
                }
                float classIndex=boxes.get(i).get(5);
                // 两个检测框都检测到同一类数据，通过iou来判断是否检测到同一目标，这就是非极大值抑制
//                if(classIndex==curMaxConf.get(5)){
                    double cos_value = Math.cos(curMaxConf.get(5));
                    double sin_value = Math.sin(curMaxConf.get(5));
                    double a1 = curMaxConf.get(2) * Math.sqrt( cos_value) + curMaxConf.get(3) * Math.sqrt(sin_value);
                    double b1 = curMaxConf.get(2) * Math.sqrt(sin_value) + curMaxConf.get(3) * Math.sqrt(cos_value);
                    double c1 = curMaxConf.get(2) * cos_value * sin_value - curMaxConf.get(3) * sin_value * cos_value;

                    double a2 = boxes.get(i).get(2) * Math.sqrt(cos_value) + boxes.get(i).get(3) * Math.sqrt(sin_value);
                    double b2 = boxes.get(i).get(2) * Math.sqrt(sin_value) + boxes.get(i).get(3) * Math.sqrt(cos_value);
                    double c2 = boxes.get(i).get(2) * cos_value * sin_value - boxes.get(i).get(3) * sin_value * cos_value;

                    float x1=curMaxConf.get(0);
                    float y1=curMaxConf.get(1);
//                    float a1=curMaxConf.get(2);
//                    float b1=curMaxConf.get(3);
//                    float c1=curMaxConf.get(6);
                    float x2=boxes.get(i).get(0);
                    float y2=boxes.get(i).get(1);
//                    float a2=boxes.get(i).get(2);
//                    float b2=boxes.get(i).get(3);
//                    float c2=boxes.get(i).get(6);

                    double t1 = (((a1 + a2) * (Math.pow(y1 - y2, 2)) + (b1 + b2) * (Math.pow(x1 - x2, 2)))
                            / ((a1 + a2) * (b1 + b2) - (Math.pow(c1 + c2, 2)) + 1e-7)) * 0.25;
                    double t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / ((a1 + a2) * (b1 + b2) - (Math.pow(c1 + c2, 2)) + 1e-7)) * 0.5;
                    double t3 = (Math.log(((a1 + a2) * (b1 + b2) - (Math.pow(c1 + c2, 2)))
                            / (4 * Math.sqrt(Math.max((a1 * b1 - Math.pow(c1, 2)),0) * Math.max((a2 * b2 - Math.pow(c2, 2)),0)) + 1e-7) + 1e-7) * 0.5);

                    double bd = Math.max(1e-7, Math.min(t1 + t2 + t3, 100.0));
                    double hd = Math.sqrt(1.0 - Math.exp(-bd) + 1e-7);
                    double iou = 1 - hd;
                    // 计算IoU
//                    float iou = intersectionArea / unionArea;
                    // 对交并比超过阈值的标记
                    indexs[i]=(float) iou>nmsThreshold?0:1;
//                    System.out.println(cur+" "+i+" class"+curMaxConf.get(5)+" "+classIndex+"  u:"+unionArea+" i:"+intersectionArea+"  iou:"+ iou);
//                }
            }
        }


        List<ArrayList<Float>> resBoxes=new LinkedList<ArrayList<Float>>();
        for(int index=0;index<indexs.length;index++){
            if(indexs[index]==1) {
                resBoxes.add(boxes.get(index));
            }
        }
        boxes=resBoxes;

        System.out.println("boxes.size : "+boxes.size());
        for(ArrayList<Float> box1:boxes){
            double cos_value = Math.cos(box1.get(5));
            double sin_value = Math.sin(box1.get(5));
            double vec1x = box1.get(2) / 2 * cos_value;
            double vec1y = box1.get(2) / 2 * sin_value;
            double vec2x = -box1.get(3) / 2 * sin_value;
            double vec2y = box1.get(3) / 2 * cos_value;
            double x1 = box1.get(0) + vec1x + vec2x;
            double y1 = box1.get(1) + vec1y + vec2y;
            double x2 = box1.get(0) + vec1x - vec2x;
            double y2 = box1.get(1) + vec1y - vec2y;
            double x3 = box1.get(0) - vec1x - vec2x;
            double y3 = box1.get(1) - vec1y - vec2y;
            double x4 = box1.get(0) - vec1x + vec2x;
            double y4 = box1.get(1) - vec1y + vec2y;
            box1.set(0,(float) x1*scaleW);
            box1.set(1,(float) y1*scaleH);
            box1.set(2,(float) x2*scaleW);
            box1.set(3,(float) y2*scaleH);
            box1.set(4,(float) x3*scaleW);
            box1.set(5,(float) y3*scaleH);
            box1.set(6,(float) x4*scaleW);
            box1.set(7,(float) y4*scaleH);
        }
        System.out.println("boxes: "+boxes);
        //detect(boxes);
        Map<Object,Object> map=new HashMap<Object,Object>();
        map.put("boxes",boxes);
//        map.put("classNames",names);
        return map;
    }




    public static Mat showDetect(Map<Object,Object> map){
        List<ArrayList<Float>> boxes=(List<ArrayList<Float>>)map.get("boxes");
//        JSONObject names=(JSONObject) map.get("classNames");
        resize(src,src,new Size((int)srcw,(int)srch));


        for(ArrayList<Float> box:boxes){
            float x1=box.get(0);
            float y1=box.get(1);
            float x2=box.get(2);
            float y2=box.get(3);
            float x3=box.get(4);
            float y3=box.get(5);
            float x4=box.get(6);
            float y4=box.get(7);
//            float config=box.get(4);
//            String className=(String)names.get(String.valueOf(box.get(5).intValue()));
//            System.out.println(className);
            System.out.println(x1);
            System.out.println(y1);
            System.out.println(x2);
            System.out.println(y2);

            Point point1 = new Point((int)x1,(int)y1);
            Point point2 = new Point((int)x2,(int)y2);
            Point point3 = new Point((int)x3,(int)y3);
            Point point4 = new Point((int)x4,(int)y4);

            circle(src, point1, 3, Scalar.BLUE, -1, 8, 0);
            circle(src, point2, 3, Scalar.RED, -1, 8, 0);
            circle(src, point3, 3, Scalar.YELLOW, -1, 8, 0);
            circle(src, point4, 3, Scalar.BLACK, -1, 8, 0);

            MatVector mat = new MatVector(4);
            Mat pointMat = new Mat(4, 2, CV_32SC1);
            IntIndexer indexer = pointMat.createIndexer();
            for (int x = 0; x < indexer.rows(); x++) {
                for (int y = 0; y < indexer.cols(); y++) {
                    float p =box.get(2*x+y);
                    indexer.put(x, y, (int)p);
                }
            }
            mat.push_back(pointMat);
            polylines(src,mat,true,Scalar.YELLOW);
        }

        imwrite("demo.png", src);
        imshow("YOLO", src);
        waitKey();

        return src;
    }

    public static void main(String[] args) throws Exception {
        String modelPath="D:\\data\\Temperature\\my_work\\runs\\obb\\train4\\weights\\best.onnx";
        String path="D:\\data\\Temperature\\test\\yolo_obb\\images\\train\\yc0xc6118.jpg";
        Yolov8_onnx_obb.load(modelPath);
        Map<Object,Object> map= Yolov8_onnx_obb.predict(path);
        showDetect(map);
    }
}

