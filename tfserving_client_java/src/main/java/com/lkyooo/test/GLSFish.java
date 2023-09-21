package com.lkyooo.test;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import org.bytedeco.javacpp.indexer.UByteRawIndexer;
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

public class GLSFish {

    public static void main(String[] args) throws Exception {
        find("/Users/Administrator/Documents/工作文档/通威渔光物联/养殖智能运维/Poseidon/poseidon-video/和县/IMG_0601-frame/181400000.jpg");
    }

    static List<Map<String, Object>> find(String filename) {
        Mat imageMat = opencv_imgcodecs.imread(filename);
        int width = imageMat.cols();
        int height = imageMat.rows();
        int dimSize = 416;
        int tfsPort = 8710;
        String tfsServer = "172.20.112.102";
        String tfsModelName = "model-sfish";
        String tfsSignatureName = "serving_default";
        ManagedChannel channel = ManagedChannelBuilder.forAddress(tfsServer, tfsPort).usePlaintext(true).build();
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
                tensorProtoBuilder.addFloatVal(red);
                tensorProtoBuilder.addFloatVal(green);
                tensorProtoBuilder.addFloatVal(blue);
            }
        }
        predictRequestBuilder.putInputs("in", tensorProtoBuilder.build());
        Predict.PredictResponse predictResponse = stub.predict(predictRequestBuilder.build());
        channel.shutdown();
        List<Float> boxes = predictResponse.getOutputsOrThrow("bboxs").getFloatValList();
        float limit = 0.25f;
        List<Map<String, Object>> fishes = new ArrayList<>();
        Mat detailImageMat = imageMat.clone();
        for (int i = 0; i < boxes.size() / 6; i++) {
            int baseIndex = i * 6;
            List<Float> boxPoints = boxes.subList(baseIndex, baseIndex + 6);
            if (boxPoints.get(4) > limit) {
                int boxCenterX = (int) (boxPoints.get(0) * width / dimSize);
                int boxCenterY = (int) (boxPoints.get(1) * height / dimSize);
                int boxWidth = (int) (boxPoints.get(2) * width / dimSize);
                int boxHeight = (int) (boxPoints.get(3) * height / dimSize);
                int left = boxCenterX - boxWidth / 2;
                int top = boxCenterY - boxHeight / 2;
                Rect rect = new Rect(left, top, boxWidth, boxHeight);
                Map<String, Object> fish = new HashMap<>();
                fish.put("x", rect.x() + rect.width() / 2.0f);
                fish.put("y", rect.y() + rect.height() / 2.0f);
                fish.put("size", rect.area() / 2.0f);
                fishes.add(fish);
                opencv_imgproc.rectangle(detailImageMat, rect, AbstractScalar.YELLOW);
            }
        }
        opencv_imgcodecs.imwrite(new File("target/tfs/fish." + System.currentTimeMillis() + ".jpg").getAbsolutePath(), detailImageMat);
        detailImageMat.deallocate();
        inputMat.deallocate();
        inputMatIndexer.release();
        System.out.println(fishes);
        return fishes;
    }
}
