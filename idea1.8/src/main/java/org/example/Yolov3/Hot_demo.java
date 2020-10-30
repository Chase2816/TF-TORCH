package org.example.Yolov3;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import javaxt.io.Image;
import net.coobird.thumbnailator.Thumbnails;
import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.*;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;
import tensorflow.serving.Model;
import tensorflow.serving.Predict;
import tensorflow.serving.PredictionServiceGrpc;

import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class Hot_demo {
    public static void main(String[] args) throws Exception {
        String modelName = "model-group";
        String signatureName = "serving_default";
        String filename = "F:\\tfm\\src\\main\\21.jpg";

//        BufferedImage im = Thumbnails.of(filename).forceSize(416, 416).outputFormat("bmp").asBufferedImage();
        BufferedImage im = Thumbnails.of(filename).scale(1).asBufferedImage();
        Raster raster = im.getRaster();
        List<Float> floatList = new ArrayList<>();
        float[] tmp = new float[raster.getWidth() * raster.getHeight() * raster.getNumBands()];
        float[] pixels = raster.getPixels(0, 0, raster.getWidth(), raster.getHeight(), tmp);
        for (float pixel : pixels) {
            floatList.add(pixel);
        }

        long t = System.currentTimeMillis();
        //创建连接，注意usePlaintext设置为true表示用非SSL连接
        ManagedChannel channel = ManagedChannelBuilder.forAddress("172.20.112.102", 8500).usePlaintext(true).build();
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
        tensorProtoBuilder.setDtype(DataType.DT_UINT8);
        TensorShapeProto.Builder tensorShapeBuilder = TensorShapeProto.newBuilder();
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(1));
        // 1 * w * h * c
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(im.getWidth()));
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(im.getHeight()));
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(3));

        tensorProtoBuilder.setTensorShape(tensorShapeBuilder.build());
        tensorProtoBuilder.addAllFloatVal(floatList);
        predictRequestBuilder.putInputs("inputs", tensorProtoBuilder.build());
        //访问并获取结果
        Predict.PredictResponse predictResponse = stub.predict(predictRequestBuilder.build());
        List<Float> boxes = predictResponse.getOutputsOrThrow("detection_boxes").getFloatValList();
        List<Float> scores = predictResponse.getOutputsOrThrow("detection_scores").getFloatValList();
        List<Float> classes = predictResponse.getOutputsOrThrow("detection_classes").getFloatValList();
        List<Float> masks = predictResponse.getOutputsOrThrow("detection_masks").getFloatValList();
        System.out.println(boxes.size()/4);
        System.out.println(scores);
        System.out.println(classes);
        System.out.println(masks);
        System.exit(1);

        List<List<Float>> bbox = getSplitList(8, boxes);

        List<List<Float>> bb = new ArrayList<>();
        for (int i = 0; i < bbox.size(); i++) {
//                System.out.println(bbox.get(i));
//                System.out.println(bbox.get(i).size());
//                System.out.println(bbox.get(i).get(4));
            if (bbox.get(i).get(4) < 0.9) {
                continue;
            }
            bb.add(bbox.get(i));
        }
        System.out.println("=====================================");
        System.out.println(bb);
        System.exit(1);
        System.out.println("cost time: " + (System.currentTimeMillis() - t));
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
