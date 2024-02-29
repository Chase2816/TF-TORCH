package com.lkyooo.test;

import ai.onnxruntime.*;
import com.alibaba.fastjson.JSONObject;

import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.javacv.*;
import org.bytedeco.javacpp.indexer.*;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_highgui.*;

import java.nio.FloatBuffer;
import java.text.DecimalFormat;
import java.util.*;
import java.util.List;
import java.util.Arrays;


/**
 * onnx学习笔记  GTianyu
 */
public class Yolov8_onnx {
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
            String nameClass = metadata.getCustomMetadata().get("names");
            System.out.println("getProducerName="+metadata.getProducerName());
            System.out.println("getGraphName="+metadata.getGraphName());
            System.out.println("getDescription="+metadata.getDescription());
            System.out.println("getDomain="+metadata.getDomain());
            System.out.println("getVersion="+metadata.getVersion());
            System.out.println("getCustomMetadata="+metadata.getCustomMetadata());
            System.out.println("getInputInfo="+infoMap);
            System.out.println("nodeInfo="+nodeInfo);
            System.out.println(nameClass);
            names = JSONObject.parseObject(nameClass.replace("\"","\"\""));
            count = nodeInfo.getShape()[0];//1 模型每次处理一张图片
            channels = nodeInfo.getShape()[1];//3 模型通道数
            netHeight = nodeInfo.getShape()[2];//640 模型高
            netWidth = nodeInfo.getShape()[3];//640 模型宽
            System.out.println(names.get(0));
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
        float rawData[][]=new float[data[0].length][6];
        System.out.println(data.length-1);
        for(int i=0;i<4;i++){
            for(int j=0;j<data[0].length;j++){
                rawData[j][i]=data[i][j];
            }
        }
        // 保存每个检查框置信值最高的类型置信值和该类型下标
        for(int i=0;i<data[0].length;i++){
            for(int j=4;j<data.length;j++){
                if(rawData[i][4]<data[j][i]){
                    rawData[i][4]=data[j][i];           //置信值
                    rawData[i][5]=j-4;                  //类型编号
                }
            }
        }
        List<ArrayList<Float>> boxes=new LinkedList<ArrayList<Float>>();
        ArrayList<Float> box=null;
        // 置信值过滤,xywh转xyxy
        for(float[] d:rawData){
            // 置信值过滤
            if(d[4]>confThreshold){
                // xywh(xy为中心点)转xyxy
                d[0]=d[0]-d[2]/2;
                d[1]=d[1]-d[3]/2;
                d[2]=d[0]+d[2];
                d[3]=d[1]+d[3];
                // 置信值符合的进行插入法排序保存
                box=new ArrayList<Float>();
                for(float num:d) {
                    box.add(num);
                }
                if(boxes.size()==0){
                    boxes.add(box);
                }else {
                    int i;
                    for(i=0;i<boxes.size();i++){
                        if(box.get(4)>boxes.get(i).get(4)){
                            boxes.add(i,box);
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
        //System.out.println(boxes);

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
                if(classIndex==curMaxConf.get(5)){
                    float x1=curMaxConf.get(0);
                    float y1=curMaxConf.get(1);
                    float x2=curMaxConf.get(2);
                    float y2=curMaxConf.get(3);
                    float x3=boxes.get(i).get(0);
                    float y3=boxes.get(i).get(1);
                    float x4=boxes.get(i).get(2);
                    float y4=boxes.get(i).get(3);
                    //将几种不相交的情况排除。提示:x1y1、x2y2、x3y3、x4y4对应两框的左上角和右下角
                    if(x1>x4||x2<x3||y1>y4||y2<y3){
                        continue;
                    }
                    // 两个矩形的交集面积
                    float intersectionWidth =Math.max(x1, x3) - Math.min(x2, x4);
                    float intersectionHeight=Math.max(y1, y3) - Math.min(y2, y4);
                    float intersectionArea =Math.max(0,intersectionWidth * intersectionHeight);
                    // 两个矩形的并集面积
                    float unionArea = (x2-x1)*(y2-y1)+(x4-x3)*(y4-y3)-intersectionArea;
                    // 计算IoU
                    float iou = intersectionArea / unionArea;
                    // 对交并比超过阈值的标记
                    indexs[i]=iou>nmsThreshold?0:1;
                    //System.out.println(cur+" "+i+" class"+curMaxConf.get(5)+" "+classIndex+"  u:"+unionArea+" i:"+intersectionArea+"  iou:"+ iou);
                }
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
            box1.set(0,box1.get(0)*scaleW);
            box1.set(1,box1.get(1)*scaleH);
            box1.set(2,box1.get(2)*scaleW);
            box1.set(3,box1.get(3)*scaleH);
        }
        System.out.println("boxes: "+boxes);
        //detect(boxes);
        Map<Object,Object> map=new HashMap<Object,Object>();
        map.put("boxes",boxes);
        map.put("classNames",names);
        return map;
    }


    public static Mat showDetect(Map<Object,Object> map){
        List<ArrayList<Float>> boxes=(List<ArrayList<Float>>)map.get("boxes");
        JSONObject names=(JSONObject) map.get("classNames");
        resize(src,src,new Size((int)srcw,(int)srch));

        for(ArrayList<Float> box:boxes){
            float x1=box.get(0);
            float y1=box.get(1);
            float x2=box.get(2);
            float y2=box.get(3);
            float config=box.get(4);
            String className=(String)names.get(String.valueOf(box.get(5).intValue()));
            System.out.println(className);
            System.out.println(x1);
            System.out.println(y1);
            System.out.println(x2);
            System.out.println(y2);
            System.out.println(config);

            Point point1=new Point((int)x1,(int)y1);
            Point point2=new Point((int)x2,(int)y2);
            rectangle(src,point1,point2,Scalar.RED,2, CV_AA, 0);
            String conf=new DecimalFormat("#.###").format(config);
            putText(src,className+" "+conf,new Point((int)x1,(int)y1-5),0,0.5,Scalar.BLUE);
        }
        imshow("YOLO", src);
        waitKey();

        return src;
    }

//    private static void imshow(String txt, Mat img) {
//        CanvasFrame canvasFrame = new CanvasFrame(txt);
//        canvasFrame.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
//        canvasFrame.setCanvasSize(img.cols(), img.rows());
//        canvasFrame.showImage(new OpenCVFrameConverter.ToMat().convert(img));
//    }

    public static void main(String[] args) throws Exception {
        String modelPath="E:\\pycharm_project\\tfservingconvert\\yolov8\\runs\\detect\\train\\weights\\best.onnx";
        String path="E:\\pycharm_project\\tfservingconvert\\yolov8\\model-yolo_lg\\bus.jpg";

//        String modelPath="D:\\data\\Temperature\\my_work\\runs\\obb\\train4\\weights\\best.onnx";
//        String path="D:\\data\\Temperature\\test\\yolo_obb\\images\\train\\yc0xc6118.jpg";
        Yolov8_onnx.load(modelPath);
        Map<Object,Object> map=Yolov8_onnx.predict(path);
        showDetect(map);
    }
}

