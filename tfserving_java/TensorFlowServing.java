package com.tw.ygwl.ai.holmes.application;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import lombok.SneakyThrows;
import lombok.extern.java.Log;
import org.apache.commons.lang3.tuple.Triple;
import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.util.FileSystemUtils;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;
import tensorflow.serving.Model;
import tensorflow.serving.Predict;
import tensorflow.serving.PredictionServiceGrpc;

import java.io.File;
import java.io.FileFilter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@Service
@Log
public class TensorFlowServing {

    static boolean outputProcessingDetail = false;

    static {
        if (outputProcessingDetail) {
            File temp = new File("target/tfs");
            FileSystemUtils.deleteRecursively(temp);
            temp.mkdirs();
        }
    }

    @Value("${app.tfs.server}")
    private String tfsServer;

    public String selectEngine(File root) {
        FileFilter fileFilter = file -> (file.getName().endsWith(".jpg") || file.getName().endsWith(".JPG"));
        File[] files = root.listFiles(fileFilter);
        String tfsServer = "172.20.112.102";
        int tfsPort = 8800;
        int dimSize = 224;
        int groundCount = 0;//地面
        int waterLongCount = 0;//超规格
        int waterCount = 0;//水面
        int waterNoiseCount = 0;//背景复杂的水面
        int backCount = 0;//组串背侧
        for (int k = 0; k < files.length; k++) {
            if (k % 5 == 0) {
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
                    waterLongCount++;
                } else if (idx == 3) {
                    waterCount++;
                } else {
                    waterNoiseCount++;
                }
                channel.shutdown();
            }
        }
        String result;
        if (backCount > 0) {
            result = "Back";
        } else if (waterLongCount > 0) {
            result = "WaterLongSpec";
        } else if (waterNoiseCount > 0) {
            result = "WaterNoise";
        } else if (groundCount > 0) {
            result = "Ground";
        } else {
            result = "Water";
        }
        log.info("Result is " + result + ", Back: " + backCount + ", WaterLongSpec: " + waterLongCount + ", WaterNoise: " + waterNoiseCount + ", Ground: " + groundCount + ", Water: " + waterCount);
        return result;
    }

    public List<Mat> findBunch(Mat imageMat) {
        int width = imageMat.cols();
        int height = imageMat.rows();
        int tfsPort = 8500;
        String tfsModelName = "model-group";
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
            if (scores.get(i) > 0.9) {
                int baseIndex = i * 4;
                List<Float> boxPoints = boxes.subList(baseIndex, baseIndex + 4);
                int boxImageTopLeftY = Math.round(boxPoints.get(0) * height);
                int boxImageTopLeftX = Math.round(boxPoints.get(1) * width);
                int boxImageBottomRightY = Math.round(boxPoints.get(2) * height);
                int boxImageBottomRightX = Math.round(boxPoints.get(3) * width);
                int boxWidth = boxImageBottomRightX - boxImageTopLeftX;
                int boxHeight = boxImageBottomRightY - boxImageTopLeftY;
                Rect rect = new Rect(new Point(boxImageTopLeftX, boxImageTopLeftY), new Point(boxImageBottomRightX, boxImageBottomRightY));
                if (outputProcessingDetail) {
                    opencv_imgproc.rectangle(detailImageMat, rect, AbstractScalar.YELLOW);
                }
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
                        if (maskFloatRawIndexer.get(y, x) > 0.04) {
                            if (outputProcessingDetail) {
                                detailImageMatIndexer.put(maskImageY, maskImageX * 3, 0);
                                detailImageMatIndexer.put(maskImageY, maskImageX * 3 + 1, 0);
                                detailImageMatIndexer.put(maskImageY, maskImageX * 3 + 2, 0);
                            }
                            shapeMatIndexer.put(maskImageY, maskImageX, 254);
                        }
                    }
                }
                maskNumMat.releaseReference();
                maskMat.releaseReference();
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
        if (outputProcessingDetail) {
            opencv_imgcodecs.imwrite(new File("target/tfs/bunch." + System.currentTimeMillis() + ".jpg").getAbsolutePath(), detailImageMat);
            opencv_imgcodecs.imwrite(new File("target/tfs/shape." + System.currentTimeMillis() + ".jpg").getAbsolutePath(), shapeMat);
        }
        detailImageMat.releaseReference();
        shapeMat.releaseReference();
        shapeMatIndexer.release();
        detailImageMatIndexer.release();
        return contoursMatList;
    }

    public Triple<List<Rect>, List<Rect>, List<Rect>> findBlobDJ(Mat imageMat) {
        int width = imageMat.cols();
        int height = imageMat.rows();
        int dimSize = 416;
        String tfsModelName = "model-djhot";
        int tfsPort = 8610;
        //        int tfsPort = 9100; //测试
        //        String tfsModelName = "model-diode"; //测试
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
        float limit = 0.36f;
        List<Rect> diode = new ArrayList<>();
        List<Rect> shadow = new ArrayList<>();
        List<Rect> stain = new ArrayList<>();
        List<Rect> excludes = new ArrayList<>();
        Mat detailImageMat = imageMat.clone();
        out:
        for (int i = 0; i < boxes.size() / 8; i++) {
            int baseIndex = i * 8;
            List<Float> boxPoints = boxes.subList(baseIndex, baseIndex + 8);
            if (boxPoints.get(4) > limit) {
                int boxCenterX = (int) (boxPoints.get(0) * width / dimSize);
                int boxCenterY = (int) (boxPoints.get(1) * height / dimSize);
                int boxWidth = (int) (boxPoints.get(2) * width / dimSize);
                int boxHeight = (int) (boxPoints.get(3) * height / dimSize);
                int left = boxCenterX - boxWidth / 2;
                int top = boxCenterY - boxHeight / 2;
                Rect rect = new Rect(left, top, boxWidth, boxHeight);
                for (Rect exclude : new ArrayList<>(excludes)) {
                    int existBlobCenterX = exclude.x() + exclude.width() / 2;
                    int existBlobhCenterY = exclude.y() + exclude.height() / 2;
                    if (existBlobCenterX + 5 > boxCenterX && existBlobCenterX - 5 < boxCenterX
                            && existBlobhCenterY + 5 > boxCenterY && existBlobhCenterY - 5 < boxCenterY) {
                        if (exclude.area() > rect.area()) {
                            continue out;
                        } else {
                            diode.remove(exclude);
                            shadow.remove(exclude);
                            stain.remove(exclude);
                            excludes.remove(exclude);
                            break;
                        }
                    }
                }
                if (boxPoints.get(5) > boxPoints.get(6) && boxPoints.get(5) > boxPoints.get(7)) {
                    diode.add(rect);
                } else if (boxPoints.get(6) > boxPoints.get(7)) {
                    shadow.add(rect);
                } else {
                    stain.add(rect);
                }
                excludes.add(rect);
            }
        }
        if (outputProcessingDetail) {
            for (Rect rect : excludes) {
                opencv_imgproc.rectangle(detailImageMat, rect, AbstractScalar.YELLOW);
            }
        }
        if (outputProcessingDetail) {
            opencv_imgcodecs.imwrite(new File("target/tfs/blob." + System.currentTimeMillis() + ".jpg").getAbsolutePath(), detailImageMat);
        }
        detailImageMat.releaseReference();
        inputMat.releaseReference();
        inputMatIndexer.release();
        return Triple.of(shadow, diode, stain);
    }

    @SneakyThrows
    public Triple<List<Rect>, List<Rect>, List<Rect>> findBlob(Mat imageMat) {
        int width = imageMat.cols();
        int height = imageMat.rows();
        int dimSize = 416;
        String tfsModelName = "model-hot";
        int tfsPort = 8600;
        //        int tfsPort = 9100; //测试
        //        String tfsModelName = "model-diode"; //测试
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
        float limit = 0.36f;
        List<Rect> diode = new ArrayList<>();
        List<Rect> shadow = new ArrayList<>();
        List<Rect> stain = new ArrayList<>();
        List<Rect> excludes = new ArrayList<>();
        Mat detailImageMat = imageMat.clone();
        out:
        for (int i = 0; i < boxes.size() / 8; i++) {
            int baseIndex = i * 8;
            List<Float> boxPoints = boxes.subList(baseIndex, baseIndex + 8);
            if (boxPoints.get(4) > limit) {
                int boxCenterX = (int) (boxPoints.get(0) * width / dimSize);
                int boxCenterY = (int) (boxPoints.get(1) * height / dimSize);
                int boxWidth = (int) (boxPoints.get(2) * width / dimSize);
                int boxHeight = (int) (boxPoints.get(3) * height / dimSize);
                int left = boxCenterX - boxWidth / 2;
                int top = boxCenterY - boxHeight / 2;
                Rect rect = new Rect(left, top, boxWidth, boxHeight);
                for (Rect exclude : new ArrayList<>(excludes)) {
                    int existBlobCenterX = exclude.x() + exclude.width() / 2;
                    int existBlobCenterY = exclude.y() + exclude.height() / 2;
                    if (existBlobCenterX + 5 > boxCenterX && existBlobCenterX - 5 < boxCenterX
                            && existBlobCenterY + 5 > boxCenterY && existBlobCenterY - 5 < boxCenterY) {
                        if (exclude.area() > rect.area()) {
                            continue out;
                        } else {
                            diode.remove(exclude);
                            shadow.remove(exclude);
                            stain.remove(exclude);
                            excludes.remove(exclude);
                            break;
                        }
                    }
                }
                if (boxPoints.get(5) > boxPoints.get(6) && boxPoints.get(5) > boxPoints.get(7)) {
                    diode.add(rect);
                } else if (boxPoints.get(6) > boxPoints.get(7)) {
                    shadow.add(rect);
                } else {
                    stain.add(rect);
                }
                excludes.add(rect);
            }
        }
        if (outputProcessingDetail) {
            for (Rect rect : excludes) {
                opencv_imgproc.rectangle(detailImageMat, rect, AbstractScalar.YELLOW);
            }
        }
        if (outputProcessingDetail) {
            opencv_imgcodecs.imwrite(new File("target/tfs/blob." + System.currentTimeMillis() + ".jpg").getAbsolutePath(), detailImageMat);
        }
        detailImageMat.releaseReference();
        inputMat.releaseReference();
        inputMatIndexer.release();
        return Triple.of(shadow, diode, stain);
    }
}
