package com.lkyooo.test;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
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

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class Bunch1600 {

    static List<Mat> find(String filepath) {
        Mat imageMat = opencv_imgcodecs.imread(filepath);
        int width = imageMat.cols();
        int height = imageMat.rows();
        int tfsPort = 8510;
        String tfsServer = "172.20.112.102";
        String tfsModelName = "model-light_group";
        String tfsSignatureName = "serving_default";
        ManagedChannel channel = ManagedChannelBuilder.forAddress(tfsServer, tfsPort).usePlaintext(true).build();
        PredictionServiceGrpc.PredictionServiceBlockingStub stub = PredictionServiceGrpc.newBlockingStub(channel);
        Predict.PredictRequest.Builder predictRequestBuilder = Predict.PredictRequest.newBuilder();
        Model.ModelSpec.Builder modelSpecBuilder = Model.ModelSpec.newBuilder();
        modelSpecBuilder.setName(tfsModelName);
        modelSpecBuilder.setSignatureName(tfsSignatureName);
        predictRequestBuilder.setModelSpec(modelSpecBuilder);
        TensorProto.Builder tensorProtoBuilder = TensorProto.newBuilder();
        tensorProtoBuilder.setDtype(DataType.DT_UINT8);
        TensorShapeProto.Builder tensorShapeBuilder = TensorShapeProto.newBuilder();
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(1));
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(height));
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(width));
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(3));
        tensorProtoBuilder.setTensorShape(tensorShapeBuilder.build());
        Mat shapeMat = new Mat(imageMat.size(), opencv_core.CV_8UC1);
        UByteRawIndexer shapeMatIndexer = shapeMat.createIndexer();
        Mat detailImageMat = imageMat.clone();
        UByteRawIndexer detailImageMatIndexer = detailImageMat.createIndexer();
        for (int y = 0; y < shapeMat.rows(); y++) {
            for (int x = 0; x < shapeMat.cols(); x++) {
                shapeMatIndexer.put(y, x, 0);
                int baseX = x * 3;
                int blue = detailImageMatIndexer.get(y, baseX);
                int green = detailImageMatIndexer.get(y, baseX + 1);
                int red = detailImageMatIndexer.get(y, baseX + 2);
                tensorProtoBuilder.addIntVal(red);
                tensorProtoBuilder.addIntVal(green);
                tensorProtoBuilder.addIntVal(blue);
            }
        }
        predictRequestBuilder.putInputs("inputs", tensorProtoBuilder.build());
        Predict.PredictResponse predictResponse = stub.predict(predictRequestBuilder.build());
        channel.shutdown();
        List<Float> boxes = predictResponse.getOutputsOrThrow("detection_boxes").getFloatValList();
        List<Float> scores = predictResponse.getOutputsOrThrow("detection_scores").getFloatValList();
        List<Float> masks = predictResponse.getOutputsOrThrow("detection_masks").getFloatValList();
        int maskDimSize = 33;
        for (int i = 0; i < scores.size(); i++) {
            if (scores.get(i) > 0.2) {
                int baseIndex = i * 4;
                List<Float> boxPoints = boxes.subList(baseIndex, baseIndex + 4);
                int boxImageTopLeftY = Math.round(boxPoints.get(0) * height);
                int boxImageTopLeftX = Math.round(boxPoints.get(1) * width);
                int boxImageBottomRightY = Math.round(boxPoints.get(2) * height);
                int boxImageBottomRightX = Math.round(boxPoints.get(3) * width);
                int boxWidth = boxImageBottomRightX - boxImageTopLeftX;
                int boxHeight = boxImageBottomRightY - boxImageTopLeftY;
                Rect rect = new Rect(new Point(boxImageTopLeftX, boxImageTopLeftY), new Point(boxImageBottomRightX, boxImageBottomRightY));
                opencv_imgproc.rectangle(detailImageMat, rect, AbstractScalar.YELLOW);
                baseIndex = i * maskDimSize * maskDimSize;
                List<Float> maskPoints = masks.subList(baseIndex, baseIndex + maskDimSize * maskDimSize);
                Mat maskNumMat = new Mat(maskDimSize, maskDimSize, opencv_core.CV_32F);
                FloatRawIndexer maskNumMatIndexer = maskNumMat.createIndexer();
                for (int y = 0; y < maskDimSize; y++) {
                    for (int x = 0; x < maskDimSize; x++) {
                        maskNumMatIndexer.put(y, x, maskPoints.get(y * maskDimSize + x));
                    }
                }
                Mat maskMat = new Mat(boxHeight, boxWidth, maskNumMat.type());
                opencv_imgproc.resize(maskNumMat, maskMat, new Size(boxWidth, boxHeight));
                FloatRawIndexer maskFloatRawIndexer = maskMat.createIndexer();
                for (int y = 0; y < boxHeight; y++) {
                    for (int x = 0; x < boxWidth; x++) {
                        int maskImageY = boxImageTopLeftY + y;
                        int maskImageX = boxImageTopLeftX + x;
                        if (maskFloatRawIndexer.get(y, x) > 0.016) {
                            detailImageMatIndexer.put(maskImageY, maskImageX * 3, 0);
                            detailImageMatIndexer.put(maskImageY, maskImageX * 3 + 1, 0);
                            detailImageMatIndexer.put(maskImageY, maskImageX * 3 + 2, 0);
                            shapeMatIndexer.put(maskImageY, maskImageX, 254);
                        }
                    }
                }
                maskNumMat.deallocate();
                maskMat.deallocate();
                maskNumMatIndexer.release();
                maskFloatRawIndexer.release();
            }
        }
        List<Mat> contoursMatList = new ArrayList<>();
        MatVector contoursMatVector = new MatVector();
        opencv_imgproc.findContours(shapeMat, contoursMatVector, opencv_imgproc.RETR_LIST, opencv_imgproc.CHAIN_APPROX_SIMPLE);
        for (int i = 0; i < contoursMatVector.size(); i++) {
            Mat contourMat = contoursMatVector.get(i);
            contoursMatList.add(contourMat.clone());
        }
        new File("target/tfs").mkdirs();
        long t = System.currentTimeMillis();
        opencv_imgcodecs.imwrite(new File("target/tfs/" + t + ".image.jpg").getAbsolutePath(), imageMat);
        opencv_imgcodecs.imwrite(new File("target/tfs/" + t + ".bunch.jpg").getAbsolutePath(), detailImageMat);
        opencv_imgcodecs.imwrite(new File("target/tfs/" + t + ".shape.jpg").getAbsolutePath(), shapeMat);
        detailImageMat.deallocate();
        shapeMat.deallocate();
        shapeMatIndexer.release();
        detailImageMatIndexer.release();
        System.out.println(contoursMatList);
        return contoursMatList;
    }

    public static void main(String[] args) throws Exception {
//        find("/Users/Administrator/Dev/Projects/tongwei-projects/tw-ygwl-ai-minerva/minerva-server/upload/tasks/252/split-material-0/yc3198xc2128.jpg");
        find("/Users/Administrator/Dev/Projects/tongwei-projects/tw-ygwl-ai-minerva/minerva-server/upload/tasks/402/split-material-0/yc3198xc4522.jpg");;
    }
}
