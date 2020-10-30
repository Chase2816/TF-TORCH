package org.example.Classifier;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import net.coobird.thumbnailator.Thumbnails;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;
import tensorflow.serving.Model;
import tensorflow.serving.Predict;
import tensorflow.serving.PredictionServiceGrpc;

import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

//import net.dongliu.requests.RawResponse;
//import net.dongliu.requests.Requests;

public class AutoGroup4Demo {
    public static void main(String[] args) throws IOException {

        String filename = "F:\\tfm\\src\\main\\21.jpg";

        BufferedImage im = Thumbnails.of(filename).forceSize(224, 224).outputFormat("bmp").asBufferedImage();
        Raster raster = im.getData();

        List<Float> floatList = new ArrayList<>();

        float[] temp = new float[raster.getWidth() * raster.getHeight() * raster.getNumBands()];
        float[] pixels = raster.getPixels(0, 0, raster.getWidth(), raster.getHeight(), temp);

        for (float pixel : pixels) {
            floatList.add(pixel / 255);
        }

        ManagedChannel channel = ManagedChannelBuilder.forAddress("172.20.112.102", 8800).usePlaintext(true).build();
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
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(224));
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(224));
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(3));
        tensorProtoBuilder.setTensorShape(tensorShapeBuilder.build());
        tensorProtoBuilder.addAllFloatVal(floatList);

        predictRequestBuilder.putInputs("image", tensorProtoBuilder.build());

        Predict.PredictResponse predictResponse = stub.predict(predictRequestBuilder.build());

        List<Float> result = new ArrayList<>();
        for (int j = 0; j < 4; j++) {
            result.add(predictResponse.getOutputsOrThrow("prediction").getFloatVal(j));
        }

        System.out.println(result);
        System.out.println(result.indexOf(Collections.max(result)));

        String[] class_names = new String[]{"ground", "group", "water", "water noise"};

        System.out.println("im: " + filename + " pred_result: " + class_names[result.indexOf(Collections.max(result))]);
//            System.out.println(class_names[result.indexOf(Collections.max(result))]);
    }
}
