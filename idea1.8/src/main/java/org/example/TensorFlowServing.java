//package org.example;
//
//import io.grpc.ManagedChannel;
//import io.grpc.ManagedChannelBuilder;
//import org.apache.commons.lang3.tuple.Triple;
//import org.bytedeco.javacpp.indexer.FloatRawIndexer;
//import org.bytedeco.javacpp.indexer.UByteRawIndexer;
//import org.bytedeco.opencv.global.opencv_core;
//import org.bytedeco.opencv.global.opencv_imgcodecs;
//import org.bytedeco.opencv.global.opencv_imgproc;
//import org.bytedeco.opencv.opencv_core.*;
//import org.springframework.beans.factory.annotation.Value;
//import org.springframework.stereotype.Service;
//import org.springframework.util.FileSystemUtils;
//import org.tensorflow.framework.DataType;
//import org.tensorflow.framework.TensorProto;
//import org.tensorflow.framework.TensorShapeProto;
//import tensorflow.serving.Model;
//import tensorflow.serving.Predict;
//import tensorflow.serving.PredictionServiceGrpc;
//
//import java.io.File;
//import java.util.ArrayList;
//import java.util.List;
//
//@Service
//public class TensorFlowServing {
//
//    static boolean outputProcessingDetail = true;
//
//    static {
//        if (outputProcessingDetail) {
//            File temp = new File("target/tfs");
//            FileSystemUtils.deleteRecursively(temp);
//            temp.mkdirs();
//        }
//    }
//
//    @Value("${app.tfs.server}")
//    private String tfsServer;
//
//    public List<Mat> findBunch(Mat imageMat) {
//        int width = imageMat.cols();
//        int height = imageMat.rows();
//        int tfsPort = 8500;
//        String tfsModelName = "model-group";
//        String tfsSignatureName = "serving_default";
//        ManagedChannel channel = ManagedChannelBuilder.forAddress(tfsServer, tfsPort).usePlaintext(true).build();
//        PredictionServiceGrpc.PredictionServiceBlockingStub stub = PredictionServiceGrpc.newBlockingStub(channel);
//        Predict.PredictRequest.Builder predictRequestBuilder = Predict.PredictRequest.newBuilder();
//        Model.ModelSpec.Builder modelSpecBuilder = Model.ModelSpec.newBuilder();
//        modelSpecBuilder.setName(tfsModelName);
//        modelSpecBuilder.setSignatureName(tfsSignatureName);
//        predictRequestBuilder.setModelSpec(modelSpecBuilder);
//        TensorProto.Builder tensorProtoBuilder = TensorProto.newBuilder();
//        tensorProtoBuilder.setDtype(DataType.DT_UINT8);
//        TensorShapeProto.Builder tensorShapeBuilder = TensorShapeProto.newBuilder();
//        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(1));
//        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(height));
//        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(width));
//        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(3));
//        tensorProtoBuilder.setTensorShape(tensorShapeBuilder.build());
//        List<Integer> colorList = new ArrayList<>();
//        Mat detailImageMat = imageMat.clone();
//        UByteRawIndexer detailImageMatIndexer = detailImageMat.createIndexer();
//        Mat shapeMat = new Mat(imageMat.size(), opencv_core.CV_8UC1);
//        UByteRawIndexer shapeMatIndexer = shapeMat.createIndexer();
//        for (int y = 0; y < shapeMat.rows(); y++) {
//            for (int x = 0; x < shapeMat.cols(); x++) {
//                shapeMatIndexer.put(y, x, 0);
//                int baseX = x * 3;
//                int blue = detailImageMatIndexer.get(y, baseX);
//                int green = detailImageMatIndexer.get(y, baseX + 1);
//                int red = detailImageMatIndexer.get(y, baseX + 2);
//                colorList.add(blue);
//                colorList.add(green);
//                colorList.add(red);
//            }
//        }
//        tensorProtoBuilder.addAllIntVal(colorList);
//        predictRequestBuilder.putInputs("inputs", tensorProtoBuilder.build());
//        Predict.PredictResponse predictResponse = stub.predict(predictRequestBuilder.build());
//        List<Float> boxes = predictResponse.getOutputsOrThrow("detection_boxes").getFloatValList();
//        List<Float> scores = predictResponse.getOutputsOrThrow("detection_scores").getFloatValList();
//        List<Float> masks = predictResponse.getOutputsOrThrow("detection_masks").getFloatValList();
//        for (int i = 0; i < scores.size(); i++) {
//            if (scores.get(i) > 0.9) {
//                int baseIndex = i * 4;
//                List<Float> boxPoints = boxes.subList(baseIndex, baseIndex + 4);
//                int boxImageTopLeftY = Math.round(boxPoints.get(0) * height);
//                int boxImageTopLeftX = Math.round(boxPoints.get(1) * width);
//                int boxImageBottomRightY = Math.round(boxPoints.get(2) * height);
//                int boxImageBottomRightX = Math.round(boxPoints.get(3) * width);
//                int boxWidth = boxImageBottomRightX - boxImageTopLeftX;
//                int boxHeight = boxImageBottomRightY - boxImageTopLeftY;
//                Rect rect = new Rect(new Point(boxImageTopLeftX, boxImageTopLeftY), new Point(boxImageBottomRightX, boxImageBottomRightY));
//                if (outputProcessingDetail) {
//                    opencv_imgproc.rectangle(detailImageMat, rect, AbstractScalar.YELLOW);
//                }
//                baseIndex = i * 15 * 15;
//                List<Float> maskPoints = masks.subList(baseIndex, baseIndex + 15 * 15);
//                Mat maskNumMat = new Mat(15, 15, opencv_core.CV_32F);
//                FloatRawIndexer maskNumMatIndexer = maskNumMat.createIndexer();
//                for (int y = 0; y < 15; y++) {
//                    for (int x = 0; x < 15; x++) {
//                        maskNumMatIndexer.put(y, x, maskPoints.get(y * 15 + x));
//                    }
//                }
//                Mat maskMat = new Mat(boxHeight, boxWidth, maskNumMat.type());
//                opencv_imgproc.resize(maskNumMat, maskMat, new Size(boxWidth, boxHeight));
//                FloatRawIndexer maskFloatRawIndexer = maskMat.createIndexer();
//                for (int y = 0; y < boxHeight; y++) {
//                    for (int x = 0; x < boxWidth; x++) {
//                        int maskImageY = boxImageTopLeftY + y;
//                        int maskImageX = boxImageTopLeftX + x;
//                        if (maskFloatRawIndexer.get(y, x) > 0.08) {
//                            if (outputProcessingDetail) {
//                                detailImageMatIndexer.put(maskImageY, maskImageX * 3, 0);
//                                detailImageMatIndexer.put(maskImageY, maskImageX * 3 + 1, 0);
//                                detailImageMatIndexer.put(maskImageY, maskImageX * 3 + 2, 0);
//                            }
//                            shapeMatIndexer.put(maskImageY, maskImageX, 254);
//                        }
//                    }
//                }
//                maskNumMat.deallocate();
//                maskMat.deallocate();
//                maskNumMatIndexer.release();
//                maskFloatRawIndexer.release();
//            }
//        }
//        List<Mat> contoursMatList = new ArrayList<>();
//        MatVector contoursMatVector = new MatVector();
//        opencv_imgproc.findContours(shapeMat, contoursMatVector, opencv_imgproc.RETR_LIST, opencv_imgproc.CHAIN_APPROX_SIMPLE);
//        for (int i = 0; i < contoursMatVector.size(); i++) {
//            Mat contourMat = contoursMatVector.get(i);
//            contoursMatList.add(contourMat.clone());
//        }
//        if (outputProcessingDetail) {
//            opencv_imgcodecs.imwrite(new File("target/tfs/bunch." + System.currentTimeMillis() + ".jpg").getAbsolutePath(), detailImageMat);
//            opencv_imgcodecs.imwrite(new File("target/tfs/shape." + System.currentTimeMillis() + ".jpg").getAbsolutePath(), shapeMat);
//        }
//        detailImageMat.deallocate();
//        shapeMat.deallocate();
//        shapeMatIndexer.release();
//        detailImageMatIndexer.release();
//        return contoursMatList;
//    }
//
//    public Triple<List<Rect>, List<Rect>, List<Rect>> findBlob(Mat imageMat) {
//        int width = imageMat.cols();
//        int height = imageMat.rows();
//        int dimSize = 416;
//        Mat inputMat = new Mat(dimSize, dimSize, imageMat.type());
//        opencv_imgproc.resize(imageMat, inputMat, new Size(dimSize, dimSize));
//        int tfsPort = 8600;
//        String tfsModelName = "model-hot";
//        String tfsSignatureName = "serving_default";
//        ManagedChannel channel = ManagedChannelBuilder.forAddress(tfsServer, tfsPort).usePlaintext(true).build();
//        PredictionServiceGrpc.PredictionServiceBlockingStub stub = PredictionServiceGrpc.newBlockingStub(channel);
//        Predict.PredictRequest.Builder predictRequestBuilder = Predict.PredictRequest.newBuilder();
//        Model.ModelSpec.Builder modelSpecBuilder = Model.ModelSpec.newBuilder();
//        modelSpecBuilder.setName(tfsModelName);
//        modelSpecBuilder.setSignatureName(tfsSignatureName);
//        predictRequestBuilder.setModelSpec(modelSpecBuilder);
//        TensorProto.Builder tensorProtoBuilder = TensorProto.newBuilder();
//        tensorProtoBuilder.setDtype(DataType.DT_FLOAT);
//        TensorShapeProto.Builder tensorShapeBuilder = TensorShapeProto.newBuilder();
//        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(1));
//        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(dimSize));
//        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(dimSize));
//        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(3));
//        tensorProtoBuilder.setTensorShape(tensorShapeBuilder.build());
//        List<Integer> colorList = new ArrayList<>();
//        UByteRawIndexer inputMatIndexer = inputMat.createIndexer();
//        for (int y = 0; y < inputMat.rows(); y++) {
//            for (int x = 0; x < inputMat.cols(); x++) {
//                int baseX = x * 3;
//                int blue = inputMatIndexer.get(y, baseX);
//                int green = inputMatIndexer.get(y, baseX + 1);
//                int red = inputMatIndexer.get(y, baseX + 2);
//                colorList.add(blue);
//                colorList.add(green);
//                colorList.add(red);
//            }
//        }
//        tensorProtoBuilder.addAllIntVal(colorList);
//        predictRequestBuilder.putInputs("input", tensorProtoBuilder.build());
//        Predict.PredictResponse predictResponse = stub.predict(predictRequestBuilder.build());
//        List<Float> boxes = predictResponse.getOutputsOrThrow("detector/yolo-v3/detections").getFloatValList();
//        float limit = 0.25f;
//        List<Rect> diode = new ArrayList<>();
//        List<Rect> shadow = new ArrayList<>();
//        List<Rect> stain = new ArrayList<>();
//        Mat detailImageMat = imageMat.clone();
//        for (int i = 0; i < boxes.size() / 8; i++) {
//            int baseIndex = i * 8;
//            List<Float> boxPoints = boxes.subList(baseIndex, baseIndex + 8);
//            if (boxes.get(4) > limit) {
//                int boxCenterX = (int) (boxPoints.get(0) * imageMat.cols() / dimSize);
//                int boxCenterY = (int) (boxPoints.get(1) * imageMat.rows() / dimSize);
//                int boxWidth = (int) (boxPoints.get(2) * imageMat.cols() / dimSize);
//                int boxHeight = (int) (boxPoints.get(3) * imageMat.rows() / dimSize);
//                int left = boxCenterX - width / 2;
//                int top = boxCenterY - height / 2;
//                Rect rect = new Rect(left, top, boxWidth, boxHeight);
//                if (outputProcessingDetail) {
//                    opencv_imgproc.rectangle(detailImageMat, rect, AbstractScalar.YELLOW);
//                }
//            }
//        }
//        if (outputProcessingDetail) {
//            opencv_imgcodecs.imwrite(new File("target/tfs/blob." + System.currentTimeMillis() + ".jpg").getAbsolutePath(), detailImageMat);
//        }
//        detailImageMat.deallocate();
//        inputMat.deallocate();
//        inputMatIndexer.release();
//        return Triple.of(shadow, diode, stain);
//    }
//}
