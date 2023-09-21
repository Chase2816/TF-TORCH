package com.lkyooo.test;

import io.grpc.ClientInterceptor;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
//import io.grpc.netty.NettyChannelBuilder;
import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.AbstractScalar;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;
import org.bytedeco.opencv.opencv_core.Size;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;
import tensorflow.serving.Model;
import tensorflow.serving.Predict;
import tensorflow.serving.PredictionServiceGrpc;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Yolov8Det {

    static List<Map<String, Object>> find(String filename) {
        Mat imageMat = opencv_imgcodecs.imread(filename);
        int width = imageMat.cols();
        int height = imageMat.rows();
        int dimSize = 640;
        int tfsPort = 8300;
        String tfsServer = "10.8.111.172";
        String tfsModelName = "model-yolo_lg";
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
        System.out.println(outputsMap);
//        TensorProto output0 = predictResponse.getOutputsOrThrow("output0"); // 1,37,8400
//        TensorProto output1 = predictResponse.getOutputsOrThrow("output1"); // 1,160,160,32
//        System.out.println(output1);
//        System.out.println(output1.getTensorShape().getDim(2));

        List<Float> boxes = predictResponse.getOutputsOrThrow("output0").getFloatValList(); // 310800
        List<Float> masks = predictResponse.getOutputsOrThrow("output1").getFloatValList(); // 819200

        int maskDimSize = 160;

        float limit = 0.25f;
        Mat out1 = new Mat(37, 8400, opencv_core.CV_32F);
        List<Map<String, Object>> blobs = new ArrayList<>();
        Mat detailImageMat = imageMat.clone();

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

        List<Float> boxPoints_ = boxes.subList(4 * 8400, 4 * 8400 + 8400);
        for (int i = 0; i < boxPoints_.size(); i++) {
            if (boxPoints_.get(i) > limit) {
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

                System.out.println(boxPoints);
                Rect rect = new Rect(left, top, boxWidth, boxHeight);

                Map<String, Object> blob = new HashMap<>();
                blob.put("x", rect.x() + rect.width() / 2.0f);
                blob.put("y", rect.y() + rect.height() / 2.0f);
                blob.put("size", rect.area() / 2.0f);
                if (boxPoints.get(5) > boxPoints.get(6) && boxPoints.get(5) > boxPoints.get(7)) {
                    blob.put("type", "Diode");
                } else if (boxPoints.get(6) > boxPoints.get(7)) {
                    blob.put("type", "Shadow");
                } else {
                    blob.put("type", "Blob");
                }
                blobs.add(blob);
                opencv_imgproc.rectangle(detailImageMat, rect, AbstractScalar.YELLOW);

            }

        }

        System.out.println(blobs.size());

        opencv_imgcodecs.imwrite(new File("target/tfs/blob." + System.currentTimeMillis() + ".jpg").getAbsolutePath(), detailImageMat);
        detailImageMat.deallocate();
        inputMat.deallocate();
        inputMatIndexer.release();
        System.out.println(blobs);
        return blobs;
    }

    public static float[][] transposeMatrix(float[][] m) {
        float[][] temp = new float[m[0].length][m.length];
        for (int i = 0; i < m.length; i++)
            for (int j = 0; j < m[0].length; j++)
                temp[j][i] = m[i][j];
        return temp;
    }


    public static void main(String[] args) throws Exception {
        find("E:\\pycharm_project\\tfservingconvert\\yolov8\\model-yolo_lg\\images/bzz_lg_01_yc4264_xc1330.jpg");
    }

}

