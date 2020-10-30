package org.example;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

import java.awt.image.BufferedImage;
import java.io.File;
import java.util.*;
import java.util.Arrays;

import javaxt.io.Image;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;

import tensorflow.serving.Model;
import tensorflow.serving.Predict;
import tensorflow.serving.PredictionServiceGrpc;

import org.bytedeco.javacpp.*;
import org.bytedeco.opencv.global.opencv_dnn;

import org.bytedeco.opencv.opencv_core.Point;
import org.bytedeco.opencv.opencv_dnn.*;
import org.bytedeco.opencv.opencv_core.*;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_dnn.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

import org.bytedeco.opencv.opencv_text.FloatVector;
import org.bytedeco.opencv.opencv_text.IntVector;

public class SFishDemo {
    public static void main(String[] args) {
        String modelName = "model-sfish";
        String signatureName = "serving_default";

        try {

            String file = "F:\\tfm\\src\\main\\223666666.jpg";

            BufferedImage image = new javaxt.io.Image(new File(file)).getBufferedImage();
            System.out.println(image);

            List<Integer> intList = new ArrayList<>();
            int pixels[] = image.getRGB(0, 0, image.getWidth(), image.getHeight(), null, 0, image.getWidth());
            // RGB转BGR格式
            for (int i = 0, j = 0; i < pixels.length; ++i, j += 3) {
                intList.add(pixels[i] & 0xff);
                intList.add((pixels[i] >> 8) & 0xff);
                intList.add((pixels[i] >> 16) & 0xff);
            }

            //记个时
            long t = System.currentTimeMillis();
            //创建连接，注意usePlaintext设置为true表示用非SSL连接
            ManagedChannel channel = ManagedChannelBuilder.forAddress("172.20.112.102", 8710).usePlaintext(true).build();

            //这里还是先用block模式
            PredictionServiceGrpc.PredictionServiceBlockingStub stub = PredictionServiceGrpc.newBlockingStub(channel);
            //创建请求
            Predict.PredictRequest.Builder predictRequestBuilder = Predict.PredictRequest.newBuilder();

            //模型名称和模型方法名预设
            Model.ModelSpec.Builder modelSpecBuilder = Model.ModelSpec.newBuilder();
            modelSpecBuilder.setName(modelName);
            modelSpecBuilder.setSignatureName(signatureName);
            predictRequestBuilder.setModelSpec(modelSpecBuilder);


            //设置入参,访问默认是最新版本，如果需要特定版本可以使用tensorProtoBuilder.setVersionNumber方法
            TensorProto.Builder tensorProtoBuilder = TensorProto.newBuilder();
            tensorProtoBuilder.setDtype(DataType.DT_FLOAT);

            TensorShapeProto.Builder tensorShapeBuilder = TensorShapeProto.newBuilder();
            tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(1));
            tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(416));
            tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(416));
            tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(3));

            tensorProtoBuilder.setTensorShape(tensorShapeBuilder.build());
            tensorProtoBuilder.addAllIntVal(intList);
//            tensorProtoBuilder.addAllFloatVal(intList);
            System.out.println(intList.subList(0,12));
            System.out.println("-============================================");

            predictRequestBuilder.putInputs("in", tensorProtoBuilder.build());
//            System.out.println(tensorProtoBuilder.build());
//            System.exit(1);

            //访问并获取结果
            Predict.PredictResponse predictResponse = stub.predict(predictRequestBuilder.build());

            // http://172.20.112.102:8600/v1/models/model-hot/metadata
            List<Float> boxes = predictResponse.getOutputsOrThrow("bboxs").getFloatValList();
//            List<Float> scores = predictResponse.getOutputsOrThrow("detection_scores").getFloatValList();
//            List<Float> classes = predictResponse.getOutputsOrThrow("detection_classes").getFloatValList();
            System.out.println(boxes);
            System.out.println(boxes.size() / 8);
//            List<List<Float>> bbox = getSplitList(8, boxes);
            List<List<Float>> bbox = getSplitList(6, boxes);
            System.out.println(bbox);
            System.out.println(bbox.size());

            List<List<Float>> bb = new ArrayList<>();
            for (int i = 0; i < bbox.size(); i++) {
//                System.out.println(bbox.get(i));
//                System.out.println(bbox.get(i).size());
//                System.out.println(bbox.get(i).get(4));
                if (bbox.get(i).get(4) < 0.1){
                    continue;
                }
                bb.add(bbox.get(i));
            }
            System.out.println(bb);
            System.exit(1);
            System.out.println("cost time: " + (System.currentTimeMillis() - t));


        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static List<List<Float>> getSplitList(int splitNum, List<Float> list) {
        List<List<Float>> splitList = new ArrayList<>();
        int groupFlag = list.size() % splitNum == 0 ? (list.size() / splitNum) : (list.size() / splitNum + 1);
        for (int j = 0; j < groupFlag; j++) {
            if ((j * splitNum + splitNum) <= list.size()) {
                splitList.add(list.subList(j * splitNum, j * splitNum + splitNum));
            } else if ((j * splitNum + splitNum) > list.size()) {
                splitList.add(list.subList(j * splitNum, list.size()));
            } else if (list.size() < splitNum) {
                splitList.add(list.subList(0, list.size()));
            }
        }
        return splitList;
    }
}
