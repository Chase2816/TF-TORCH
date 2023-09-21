package com.lkyooo.test;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import org.bytedeco.javacpp.indexer.UByteRawIndexer;
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

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class EngineSelect {

    public static void main(String[] args) throws IOException {
        File root = new File("/Users/Administrator/Dev/Projects/tongwei-projects/tw-ygwl-ai-holmes/files/task/752/thermographyFrames");
//        File root = new File("/Users/Administrator/Dev/Projects/tongwei-projects/tw-ygwl-ai-holmes/files/task/902/thermographyFrames");
        //                File root = new File("/Users/Administrator/Dev/Projects/tongwei-projects/tw-ygwl-ai-holmes/files/task/952/thermographyFrames");
        //                File root = new File("/Users/Administrator/Dev/Projects/tongwei-projects/tw-ygwl-ai-holmes/files/task/1752/thermographyFrames");
        //        File root = new File("/Users/Administrator/Dev/Projects/tongwei-projects/tw-ygwl-ai-holmes/files/task/2002/thermographyFrames");
        File[] files = root.listFiles();
        String tfsServer = "172.20.112.102";
        int tfsPort = 8800;
        int dimSize = 224;
        int groundCount = 0;//地面
        int longCount = 0;//超规格
        int waterCount = 0;//水面
        int waterNoiseCount = 0;//背景复杂的水面
        int backCount = 0;//背景复杂的水面
        for (int k = 0; k < files.length; k++) {
            if (k % 5 == 0) {
                System.out.println("inspect image:" + files[k]);
                ManagedChannel channel = ManagedChannelBuilder.forTarget(tfsServer + ":" + tfsPort).usePlaintext(true).build();
                PredictionServiceGrpc.PredictionServiceBlockingStub stub = PredictionServiceGrpc.newBlockingStub(channel);
                Predict.PredictRequest.Builder predictRequestBuilder = Predict.PredictRequest.newBuilder();
                Model.ModelSpec.Builder modelSpecBuilder = Model.ModelSpec.newBuilder();
                modelSpecBuilder.setName("auto-group");
                modelSpecBuilder.setSignatureName("serving_default");
                predictRequestBuilder.setModelSpec(modelSpecBuilder);
                TensorProto.Builder tensorProtoBuilder = TensorProto.newBuilder();
                tensorProtoBuilder.setDtype(DataType.DT_FLOAT);
                TensorShapeProto.Builder tensorShapeBuilder = TensorShapeProto.newBuilder();
                tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(1));
                tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(dimSize));
                tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(dimSize));
                tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(3));
                tensorProtoBuilder.setTensorShape(tensorShapeBuilder.build());
                Mat imageMat = opencv_imgcodecs.imread(files[k].getAbsolutePath());
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
                predictRequestBuilder.putInputs("image", tensorProtoBuilder.build());
                Predict.PredictResponse predictResponse = stub.predict(predictRequestBuilder.build());
                List<Float> result = new ArrayList<>();
                for (int j = 0; j < 5; j++) {
                    result.add(predictResponse.getOutputsOrThrow("prediction").getFloatVal(j));
                }
                int idx = result.indexOf(Collections.max(result));
                if (idx == 0) {
                    backCount++;
                } else if (idx == 1) {
                    groundCount++;
                } else if (idx == 2) {
                    longCount++;
                } else if (idx == 3) {
                    waterCount++;
                } else {
                    waterNoiseCount++;
                }
                System.out.println("BackSpec: " + backCount + ",LongSpec: " + longCount + ", WaterNoise: " + waterNoiseCount + ", Ground: " + groundCount + ", Water: " + waterCount);
            }
        }
        String result;
        if (backCount > 0) {
            result = "BackSpec";
        } else if (longCount > 0) {
            result = "LongSpec";
        } else if (waterNoiseCount > 0) {
            result = "WaterNoise";
        } else if (groundCount > 0) {
            result = "Ground";
        } else {
            result = "Water";
        }
        System.out.println(result);
    }
}
