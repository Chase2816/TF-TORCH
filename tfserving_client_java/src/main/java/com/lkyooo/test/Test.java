package com.lkyooo.test;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Size;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;
import tensorflow.serving.Model;
import tensorflow.serving.Predict;
import tensorflow.serving.PredictionServiceGrpc;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.text.DecimalFormat;
import java.util.*;
import java.util.List;

import com.alibaba.fastjson.JSONObject;
import com.alibaba.fastjson.JSONArray;

import static org.bytedeco.opencv.global.opencv_core.CV_8UC3;

public class Test {
    public static JSONObject names;
    public static long count;
    public static long channels;
    public static long netHeight = 640;
    public static long netWidth = 640;
    public static float confThreshold = 0.25f;
    public static float nmsThreshold = 0.45f;

    public static void main(String[] args) throws Exception {
        find("E:\\pycharm_project\\tfservingconvert\\yolov8\\model-yolo_lg\\images/bzz_lg_01_yc4264_xc1330.jpg");
    }

    static List<Map<String, Object>> find(String filename) {
        Mat imageMat = opencv_imgcodecs.imread(filename);
        int width = imageMat.cols();
        int height = imageMat.rows();
        int dimSize = 640;
//        int tfsPort = 8300;
//        String tfsServer = "10.8.111.172";
//        String tfsModelName = "model-yolo_lg";
        int tfsPort = 9910;
        String tfsServer = "172.20.112.102";
        String tfsModelName = "yolov8";
        String tfsSignatureName = "serving_default";

//        ManagedChannel channel = ManagedChannelBuilder.forAddress(tfsServer, tfsPort).usePlaintext(true).build();
        ManagedChannel channel = ManagedChannelBuilder.forAddress(tfsServer, tfsPort).usePlaintext(true).maxInboundMessageSize(256 * 1024 * 1024).build();
//        ManagedChannel channel = NettyChannelBuilder.forAddress(tfsServer, tfsPort).usePlaintext(true).build();

        PredictionServiceGrpc.PredictionServiceBlockingStub stub = PredictionServiceGrpc.newBlockingStub(channel);
        Predict.PredictRequest.Builder predictRequestBuilder = Predict.PredictRequest.newBuilder();
        Model.ModelSpec.Builder modelSpecBuilder = Model.ModelSpec.newBuilder();
        modelSpecBuilder.setName(tfsModelName);
        modelSpecBuilder.setSignatureName(tfsSignatureName);
        predictRequestBuilder.setModelSpec(modelSpecBuilder);
        TensorProto.Builder tensorProtoBuilder = TensorProto.newBuilder();
        tensorProtoBuilder.setDtype(DataType.DT_FLOAT);
        TensorShapeProto.Builder tensorShapeBuilder = TensorShapeProto.newBuilder();
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(1));
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(dimSize));
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(dimSize));
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(3));
        tensorProtoBuilder.setTensorShape(tensorShapeBuilder.build());
        Mat inputMat = new Mat(dimSize, dimSize, imageMat.type());
        opencv_imgproc.resize(imageMat, inputMat, new Size(dimSize, dimSize));
        UByteRawIndexer inputMatIndexer = inputMat.createIndexer();

        Mat shapeMat = new Mat(imageMat.size(), opencv_core.CV_8UC1);
        UByteRawIndexer shapeMatIndexer = shapeMat.createIndexer();
        Mat detailImageMat = imageMat.clone();
        UByteRawIndexer detailImageMatIndexer = detailImageMat.createIndexer();
        UByteRawIndexer imageMatIndexer = imageMat.createIndexer();

        for (int y = 0; y < inputMat.rows(); y++) {
            for (int x = 0; x < inputMat.cols(); x++) {
                int baseX = x * 3;
                int blue = inputMatIndexer.get(y, baseX);
                int green = inputMatIndexer.get(y, baseX + 1);
                int red = inputMatIndexer.get(y, baseX + 2);
                tensorProtoBuilder.addFloatVal(red / 255.0f);
                tensorProtoBuilder.addFloatVal(green / 255.0f);
                tensorProtoBuilder.addFloatVal(blue / 255.0f);
            }
        }
        predictRequestBuilder.putInputs("images", tensorProtoBuilder.build());
        Predict.PredictResponse predictResponse = stub.predict(predictRequestBuilder.build());
        channel.shutdown();
        Map<String, TensorProto> outputsMap = predictResponse.getOutputsMap();
//        System.out.println(outputsMap);
        TensorProto output0 = predictResponse.getOutputsOrThrow("output0"); // 1,37,8400
        TensorProto output1 = predictResponse.getOutputsOrThrow("output1"); // 1,160,160,32
        System.out.println(output1);
        System.out.println(output1.getTensorShape().getDim(2));


//        System.exit(0);
        List<Float> boxes = predictResponse.getOutputsOrThrow("output0").getFloatValList(); // 310800
        List<Float> masks = predictResponse.getOutputsOrThrow("output1").getFloatValList(); // 819200

        float[][] proto = new float[32][160 * 160];
        for (int i = 0; i < proto.length; i++) {
            for (int j = 0; j < proto[i].length; j++) {
                System.out.println(masks.get(j));
                int f = i+j*32;
                proto[i][j] = masks.get(f);
            }
        }

        float[][] detect = new float[37][8400];
        for (int x = 0; x < detect.length; x++) {
            for (int y = 0; y < detect[x].length; y++) {
                System.out.println(boxes.get(y));
                detect[x][y] = boxes.get(y);
            }
        }

        generateMaskInfo(boxes,proto,800,800);

//        JSONArray srcRec = filterRec1(detect);
//        JSONArray srcRec2 = filterRec2(srcRec);
//        JSONArray dstRec = transferSrc2Dst(srcRec2,800,800);
//
//        dstRec.stream().forEach(n->{
//            JSONObject obj = JSONObject.parseObject(n.toString());
//            String name = obj.getString("name");
//            float percentage = obj.getFloat("percentage");
//            float xmin = obj.getFloat("xmin");
//            float ymin = obj.getFloat("ymin");
//            float xmax = obj.getFloat("xmax");
//            float ymax = obj.getFloat("ymax");
//            float w = xmax - xmin;
//            float h = ymax - ymin;
//        });







        List<Map<String, Object>> blobs = new ArrayList<>();
        return blobs;
    }


    // 参考 general.py 中的 process_mask 函数
    public static void generateMaskInfo(List<Float> boxes,float[][] proto,int width,int height){
        // 32 * 160 * 160 这个是mask原型 c h w
//        float[][][] maskSrc = proto[0];
        // 转为二维矩阵也就是 32 * 25600,也就是 32 行 25600 列,相当于把 160*160展平
//        float[][] flattenedData = floatArray2floatArray(maskSrc);
        // 再转为矩阵
//        RealMatrix m1 = MatrixUtils.createRealMatrix(floatArray2doubleArray(proto));
        // 每个目标框
        List<Float> list = new ArrayList<>();
        for (int j = 0; j < boxes.size() / 37; j++) {
            for (int k = 0; k < boxes.size(); k++) {
                if (k % 8400 == 0){
                    int f = j + k;
                    list.add(boxes.get(f));
                }
            }
        }
        System.out.println(list);

        int dimSize = 640;
        float limit = 0.25f;

        List<Float> boxPoints_ = boxes.subList(4 * 8400, 4 * 8400 + 8400);
        for (int i = 0; i < boxPoints_.size(); i++) {
            if (boxPoints_.get(i) > limit) {
                System.out.println("============================"+i+"============================");
                int baseIndex_ = i * 37;
                List<Float> boxPoints = list.subList(baseIndex_, baseIndex_ + 37);
                System.out.println("=============: " + i);
                System.out.println(baseIndex_ + "====" + (baseIndex_ + 37));

                int boxCenterX = (int) (boxPoints.get(0) * width / dimSize);
                int boxCenterY = (int) (boxPoints.get(1) * height / dimSize);
                int boxWidth = (int) (boxPoints.get(2) * width / dimSize);
                int boxHeight = (int) (boxPoints.get(3) * height / dimSize);
                int left = boxCenterX - boxWidth / 2;
                int top = boxCenterY - boxHeight / 2;
                int right = boxCenterX + boxWidth / 2;
                int down = boxCenterY + boxHeight / 2;

                float[] maskWeight = new float[32];
                for (int j = 0; j < maskWeight.length; j++) {
                    maskWeight[j] = list.get(j+6);
                }

                RealMatrix m1 = MatrixUtils.createRealMatrix(floatArray2doubleArray(proto));
                RealMatrix m2 = MatrixUtils.createRowRealMatrix(floatArray2doubleArray(maskWeight));
                RealMatrix m3 = m2.multiply(m1);
                RealMatrix m4 = transfer_25600_To_160_160(m3);
                RealMatrix m5 = getSigmod(m4);
                RealMatrix m6 = resizeRealMatrix(m5,height,width);
                showMatrixWithBox(m6,left,top,right,down);



            }
        }


//        detections.stream().forEach(detection -> {
//            // 32 这个是mask 掩膜系数,也就是权重,转为矩阵
//            float[] maskWeight = detection.getMaskWeight();
//            // 作为一个行向量存储在m1中,也就是 1 行 32 列
//            RealMatrix m2 = MatrixUtils.createRowRealMatrix(floatArray2doubleArray(maskWeight));
//            // 矩阵乘法 1*32 乘 32*25600 得到 1*25600
//            RealMatrix m3 = m2.multiply(m1);
//            // 再将 1*25600 转回 160*160 也就是一个缩小的掩膜图
//            RealMatrix m4 = transfer_25600_To_160_160(m3);
//            // 对每个元素求sigmod限制到0~1,后续根据阈值进行二值化
//            RealMatrix m5 = getSigmod(m4);
//            // 将160*160上采样到图片原始尺寸
//            RealMatrix m6 = resizeRealMatrix(m5,height,width);
//            // 目标在原始图片上的xyxy
//            showMatrixWithBox(m6,detection.getSrcXYXY()[0],detection.getSrcXYXY()[1],detection.getSrcXYXY()[2],detection.getSrcXYXY()[3]);
//        });
    }


    public static float[][] floatArray2floatArray(float[][][] data){
        float[][] flattenedData = new float[data.length][data[0].length * data[0][0].length];
        for (int i = 0; i < data.length; i++) {
            float[][] slice = data[i];
            for (int j = 0; j < slice.length; j++) {
                System.arraycopy(slice[j], 0, flattenedData[i], j * slice[j].length, slice[j].length);
            }
        }
        return flattenedData;
    }
    public static double[] floatArray2doubleArray(float[] data){
        double[] maskDouble = new double[data.length];
        for (int j = 0; j < data.length; j++) {
            maskDouble[j] = (double) data[j];
        }
        return maskDouble;
    }
    public static double[][] floatArray2doubleArray(float[][] data){
        double[][] maskDouble = new double[data.length][data[0].length];
        for (int i = 0; i < data.length; i++) {
            for(int j=0; j<data[0].length;j++){
                maskDouble[i][j] = data[i][j];
            }
        }
        return maskDouble;
    }
    // 再将 1*25600 转回 160*160
    public static RealMatrix transfer_25600_To_160_160(RealMatrix data){
        RealMatrix res = new Array2DRowRealMatrix(160, 160);
        for (int i = 0; i < 160; i++) {
            for (int j = 0; j < 160; j++) {
                int index = i * 160 + j;
                double value = data.getEntry(0, index);
                res.setEntry(i, j, value);
            }
        }
        return res;
    }

    public static RealMatrix getSigmod(RealMatrix data){
        RealMatrix res = new Array2DRowRealMatrix(160, 160);
        for (int i = 0; i < 160; i++) {
            for (int j = 0; j < 160; j++) {
                double x = data.getEntry(i, j);
                double value = sigmoidF(x);
                res.setEntry(i,j,value);
//                if (value >= 0.5) {
//                    res.setEntry(i, j, 1);
//                }else {
//                    res.setEntry(i, j, 0);
//                }
            }
        }
        return res;
    }

    private static double sigmoidF(double x) {
        return 1f / (1f + Math.pow(Math.E, -1 * x));
    }


    public static RealMatrix resizeRealMatrix(RealMatrix matrix, int newRows, int newCols) {
        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();
        RealMatrix resizedMatrix = MatrixUtils.createRealMatrix(newRows, newCols);
        for (int i = 0; i < newRows; i++) {
            for (int j = 0; j < newCols; j++) {
                int origI = (int) Math.floor(i * rows / newRows);
                int origJ = (int) Math.floor(j * cols / newCols);
                double d = matrix.getEntry(origI, origJ);
                if(d>=0.5f){
                    d = 1;
                }else{
                    d = 0;
                }
                resizedMatrix.setEntry(i, j, d);
            }
        }
        return resizedMatrix;
    }
    // 弹窗显示一个 showMatrix 并画框
    public static void showMatrixWithBox(RealMatrix matrix,float xmin,float ymin,float xmax,float ymax){
        // 转换 RealMatrix to BufferedImage
        int numRows = matrix.getRowDimension();
        int numCols = matrix.getColumnDimension();
        BufferedImage image = new BufferedImage(numCols, numRows, BufferedImage.TYPE_INT_RGB);
        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                double value = matrix.getEntry(i, j);
                int grayValue = (int) Math.round(value * 255.0);
                grayValue = Math.min(grayValue, 255);
                grayValue = Math.max(grayValue, 0);
                int pixelValue = (grayValue << 16) | (grayValue << 8) | grayValue;
                image.setRGB(j, i, pixelValue);
            }
        }
        // image 上画框
        Graphics2D graph = image.createGraphics();
        graph.setStroke(new BasicStroke(3));// 线粗细
        graph.setColor(Color.RED);
        // 画矩形
        graph.drawRect(
                Float.valueOf(xmin).intValue(),
                Float.valueOf(ymin).intValue(),
                Float.valueOf(xmax-xmin).intValue(),
                Float.valueOf(ymax-ymin).intValue());
        // 提交画框
        graph.dispose();
        // 弹窗显示
        JFrame frame = new JFrame("Image Dialog");
        frame.setSize(image.getWidth(), image.getHeight());
        JLabel label = new JLabel(new ImageIcon(image));
        frame.getContentPane().add(label);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

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

    public static int getMaxIndex(float[] array) {
        int maxIndex = 0;
        float maxVal = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > maxVal) {
                maxVal = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }


    public static float[] xywh2xyxy(float[] bbox) {
        float x = bbox[0];
        float y = bbox[1];
        float w = bbox[2];
        float h = bbox[3];
        float x1 = x - w * 0.5f;
        float y1 = y - h * 0.5f;
        float x2 = x + w * 0.5f;
        float y2 = y + h * 0.5f;
        return new float[]{
                x1 < 0 ? 0 : x1,
                y1 < 0 ? 0 : y1,
                x2 > netWidth ? netWidth:x2,
                y2 > netHeight? netHeight:y2};
    }

    public static JSONArray filterRec1(float[][] data){
        JSONArray recList = new JSONArray();
        for (float[] bbox : data){
            float[] xywh = new float[] {bbox[0],bbox[1],bbox[2],bbox[3]};
            float[] mask = new float[32];
            for (int i = 0; i < 32; i++) {
                mask[i] = bbox[i+6];
            }
            float[] xyxy = xywh2xyxy(xywh);
            float confidence = bbox[4];
            float[] classInfo = Arrays.copyOfRange(bbox, 5, 85);
            int maxIndex = getMaxIndex(classInfo);
            float maxValue = classInfo[maxIndex];
//            String maxClass = (String)names.get(Integer.valueOf(maxIndex));
            // 首先根据框图置信度粗选
            if(confidence>=confThreshold){
                JSONObject detect = new JSONObject();
//                detect.put("name",maxClass);// 类别
                detect.put("percentage",maxValue);// 概率
                detect.put("xmin",xyxy[0]);
                detect.put("ymin",xyxy[1]);
                detect.put("xmax",xyxy[2]);
                detect.put("ymax",xyxy[3]);
//                detect.put("mask",mask);
                recList.add(detect);
            }
        }
        return recList;
    }

    public static JSONArray filterRec2(JSONArray data){
        JSONArray res = new JSONArray();
        data.sort(Comparator.comparing(obj->((JSONObject)obj).getString("percentage")).reversed());
        while (!data.isEmpty()){
            JSONObject max = data.getJSONObject(0);
            res.add(max);
            Iterator<Object> it = data.iterator();
            while (it.hasNext()) {
                JSONObject obj = (JSONObject)it.next();
                double iou = calculateIoU(max, obj);
                if (iou > nmsThreshold) {
                    it.remove();
                }
            }
        }
        return res;
    }

    private static double calculateIoU(JSONObject box1, JSONObject box2) {
        double x1 = Math.max(box1.getDouble("xmin"), box2.getDouble("xmin"));
        double y1 = Math.max(box1.getDouble("ymin"), box2.getDouble("ymin"));
        double x2 = Math.min(box1.getDouble("xmax"), box2.getDouble("xmax"));
        double y2 = Math.min(box1.getDouble("ymax"), box2.getDouble("ymax"));
        double intersectionArea = Math.max(0, x2 - x1 + 1) * Math.max(0, y2 - y1 + 1);
        double box1Area = (box1.getDouble("xmax") - box1.getDouble("xmin") + 1) * (box1.getDouble("ymax") - box1.getDouble("ymin") + 1);
        double box2Area = (box2.getDouble("xmax") - box2.getDouble("xmin") + 1) * (box2.getDouble("ymax") - box2.getDouble("ymin") + 1);
        double unionArea = box1Area + box2Area - intersectionArea;
        return intersectionArea / unionArea;
    }

    public static JSONArray transferSrc2Dst(JSONArray data,int srcw,int srch){
        JSONArray res = new JSONArray();
        float gain = Math.min((float) netWidth / srcw, (float) netHeight / srch);
        float padW = (netWidth - srcw * gain) * 0.5f;
        float padH = (netHeight - srch * gain) * 0.5f;
        data.stream().forEach(n->{
            JSONObject obj = JSONObject.parseObject(n.toString());
            float xmin = obj.getFloat("xmin");
            float ymin = obj.getFloat("ymin");
            float xmax = obj.getFloat("xmax");
            float ymax = obj.getFloat("ymax");
            float xmin_ = Math.max(0, Math.min(srcw - 1, (xmin - padW) / gain));
            float ymin_ = Math.max(0, Math.min(srch - 1, (ymin - padH) / gain));
            float xmax_ = Math.max(0, Math.min(srcw - 1, (xmax - padW) / gain));
            float ymax_ = Math.max(0, Math.min(srch - 1, (ymax - padH) / gain));
//            float mask = obj.getFloat("mask");
            obj.put("xmin",xmin_);
            obj.put("ymin",ymin_);
            obj.put("xmax",xmax_);
            obj.put("ymax",ymax_);
//            obj.put("mask",mask);
            res.add(obj);
        });
        return res;
    }
    public static void pointBox(String pic,JSONArray box){
        if(box.size()==0){
            System.out.println("暂无识别目标");
            return;
        }
        try {
            File imageFile = new File(pic);
            BufferedImage img = ImageIO.read(imageFile);
            Graphics2D graph = img.createGraphics();
            graph.setStroke(new BasicStroke(2));
            graph.setFont(new Font("Serif", Font.BOLD, 20));
            graph.setColor(Color.RED);
            box.stream().forEach(n->{
                JSONObject obj = JSONObject.parseObject(n.toString());
                String name = obj.getString("name");
                float percentage = obj.getFloat("percentage");
                float xmin = obj.getFloat("xmin");
                float ymin = obj.getFloat("ymin");
                float xmax = obj.getFloat("xmax");
                float ymax = obj.getFloat("ymax");
                float w = xmax - xmin;
                float h = ymax - ymin;
                graph.drawRect(
                        Float.valueOf(xmin).intValue(),
                        Float.valueOf(ymin).intValue(),
                        Float.valueOf(w).intValue(),
                        Float.valueOf(h).intValue());
                DecimalFormat decimalFormat = new DecimalFormat("#.##");
                String percentString = decimalFormat.format(percentage);
                graph.drawString(name+" "+percentString, xmin-1, ymin-5);
            });
            graph.dispose();
            JFrame frame = new JFrame("Image Dialog");
            frame.setSize(img.getWidth(), img.getHeight());
            JLabel label = new JLabel(new ImageIcon(img));
            frame.getContentPane().add(label);
            frame.setVisible(true);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        }
        catch (Exception e){
            e.printStackTrace();
            System.exit(0);
        }
    }
//————————————————
//    版权声明：本文为CSDN博主「0x13」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
//    原文链接：https://blog.csdn.net/qq_34448345/article/details/129692031
}
