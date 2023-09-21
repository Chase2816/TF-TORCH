package com.lkyooo.test;

import io.grpc.ClientInterceptor;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
//import io.grpc.netty.NettyChannelBuilder;
import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.javacpp.indexer.IntRawIndexer;
import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;
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
import org.bytedeco.opencv.opencv_core.Range;
import org.bytedeco.opencv.opencv_dnn.SigmoidLayer;
import java.io.File;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;

import javax.xml.transform.Result;

public class Yolov8Seg {

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

        List<Float> boxes = predictResponse.getOutputsOrThrow("output0").getFloatValList(); // 310800
        List<Float> masks = predictResponse.getOutputsOrThrow("output1").getFloatValList(); // 819200

        int maskDimSize = 160;

        float limit = 0.25f;
        Mat out1 = new Mat(37, 8400, opencv_core.CV_32F);
        List<Map<String, Object>> blobs = new ArrayList<>();
//        Mat detailImageMat = imageMat.clone();

        List<Float> pred_det = new ArrayList<>();
        for (int j = 0; j < boxes.size() / 37; j++) {
            for (int k = 0; k < boxes.size(); k++) {
                if (k % 8400 == 0) {
                    int f = j + k;
                    System.out.println(j + "===========" + k);
                    pred_det.add(boxes.get(f));
                }
            }
        }

        System.out.println(pred_det);

        List<Float> pred_seg = new ArrayList<>();
        for (int x = 0; x < masks.size() / (160 * 160); x++) {
            for (int y = 0; y < masks.size(); y++) {
                if (y % 32 == 0) {
                    int f_ = x + y;
                    System.out.println(x + "=======" + y);
                    System.out.println(masks.get(f_));
                    pred_seg.add(masks.get(f_));
                }
            }
        }
        System.out.println(pred_seg);

        List<Float> boxPoints_ = boxes.subList(4 * 8400, 4 * 8400 + 8400);
        for (int i = 0; i < boxPoints_.size(); i++) {
            if (boxPoints_.get(i) > limit) {

                int baseIndex_ = i * 37;
                // mask
                List<Float> maskPoints = pred_det.subList(baseIndex_ + 5, baseIndex_ + 37); // 1 * 32

//                for (int j = 0; j < maskPoints.size(); j++) {
//                    for (int k = 0; k < pred_seg.size() / 32; k++) {
//                        float mask = 0;
//                        for (int l = 0; l < 32; l++) {
//                            int f = j * 32
//                            float mask_ = pred_det.get(l) * pred_seg.get(l);
//                            mask += mask_;
//                        }
////                        BigDecimal b = new BigDecimal(String.valueOf(mask));
////                        double d = b.doubleValue();
//
//                        pred_mask.add(mask);
//                    }
//                }
                double[][] arr1 = new double[1][32];
                double[][] arr2 = new double[32][160 * 160];
                for (int j = 0; j < arr1.length; j++) {
                    for (int k = 0; k < arr1[j].length; k++) {
                        System.out.println(maskPoints.get(k));
                        arr1[j][k] = maskPoints.get(k);
                    }
                }

                for (int j = 0; j < arr2.length; j++) {
                    for (int k = 0; k < arr2[j].length; k++) {
                        arr2[j][k] = pred_seg.get(k);
                    }
                }

                double[][] arr3 = new double[1][160 * 160];
                for (int j = 0; j < arr1.length; j++) {
                    for (int k = 0; k < arr1[j].length; k++) {
                        for (int l = 0; l < arr2[k].length; l++) {
                            arr3[j][l] += arr1[j][k] * arr2[k][l];
                        }
                    }
                }
                double[] pred_mask = out_sigmoid(arr3[0]);
                System.out.println(pred_mask);


                // box处理
                List<Float> boxPoints = pred_det.subList(baseIndex_, baseIndex_ + 37);
                System.out.println("=============: " + i);
                System.out.println(baseIndex_ + "====" + (baseIndex_ + 37));


                int boxCenterX = (int) (boxPoints.get(0) * width / dimSize);
                int boxCenterY = (int) (boxPoints.get(1) * height / dimSize);
                int boxWidth_ = (int) (boxPoints.get(2) * width / dimSize);
                int boxHeight_ = (int) (boxPoints.get(3) * height / dimSize);
                int left = boxCenterX - boxWidth_ / 2;
                int top = boxCenterY - boxHeight_ / 2;
                int right = boxCenterX + boxWidth_ / 2;
                int down = boxCenterY + boxHeight_ / 2;


                System.out.println(boxPoints);
                Rect rect = new Rect(left, top, boxWidth_, boxHeight_);

                int x1 = (int) ((boxPoints.get(0) - (0.5 * boxPoints.get(2))-320)/0.8);
                int y1 = (int) ((boxPoints.get(1) - (0.5 * boxPoints.get(3))-320)/0.8);
                int x2 = (int) ((boxPoints.get(0) - (0.5 * boxPoints.get(2))-320)/0.8);
                int y2 = (int) ((boxPoints.get(1) - (0.5 * boxPoints.get(3))-320)/0.8);

                int mx1 = Math.max(0, (int) ((x1*0.8+320) * 0.25f));
                int my1 = Math.max(0, (int) ((y1*0.8+320) * 0.25f));
                int mx2 = Math.max(0, (int) ((x2*0.8+320) * 0.25f));
                int my2 = Math.max(0, (int) ((y2*0.8+320) * 0.25f));

                Mat mask_160_160 = new Mat(maskDimSize,maskDimSize,opencv_core.CV_32F);
                Rect roi = new Rect(my1, my2, mx1, mx2);

                Mat cropped = new Mat(mask_160_160, roi);
                Mat rm = new Mat();
                opencv_imgproc.resize(cropped, rm, new Size((top - left), (down-right)));
                FloatRawIndexer rmIndexer = rm.createIndexer();
                for (int r = 0; r < rm.rows(); r++) {
                    for (int c = 0; c < rm.cols(); c++) {

                        float pv = (float) pred_mask[(int) (r * 160) + c];
                        if (pv > 0.5) {
                            rmIndexer.put(r,c,1f * 255f);
                        }
                        else {
                            rmIndexer.put(r,c,0f * 255f);
                        }
                    }
                }
                imshow("Final Result", rm);
                System.exit(0);


                Mat maskNumMat = new Mat(maskDimSize, maskDimSize, opencv_core.CV_32F);
                FloatRawIndexer maskNumMatIndexer = maskNumMat.createIndexer();
                for (int y = 0; y < maskDimSize; y++) {
                    for (int x = 0; x < maskDimSize; x++) {
                        if (pred_mask[y * maskDimSize + x] < 0.5) {
                            maskNumMatIndexer.put(y, x, 0f);
                        }
                        if (pred_mask[y * maskDimSize + x] >= 0.5) {
                            maskNumMatIndexer.put(y, x, (float) pred_mask[y * maskDimSize + x]);
                        }
                    }
                }
                imshow("Final Result", maskNumMat);

                int boxHeight = 640;
                int boxWidth = 640;
                Mat maskMat = new Mat(boxHeight, boxWidth, maskNumMat.type());
                opencv_imgproc.resize(maskNumMat, maskMat, new Size(boxWidth, boxHeight));
                FloatRawIndexer maskFloatRawIndexer = maskMat.createIndexer();
                for (int y = 0; y < boxHeight; y++) {
                    for (int x = 0; x < boxWidth; x++) {
                        int maskImageY = boxHeight_ + y;
                        int maskImageX = boxWidth_ + x;
                        if (maskFloatRawIndexer.get(y, x) >= 0.8) {
                            System.out.println("*******************************************************");
                            System.out.println(maskImageX * 3 + "==========" + maskImageY);
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

                opencv_imgproc.rectangle(detailImageMat, rect, AbstractScalar.YELLOW);

            }

        }

        List<Mat> contoursMatList = new ArrayList<>();
        MatVector contoursMatVector = new MatVector();
        opencv_imgproc.findContours(shapeMat, contoursMatVector, opencv_imgproc.RETR_LIST, opencv_imgproc.CHAIN_APPROX_SIMPLE);
//        opencv_imgcodecs.imwrite(new File("target/tfs_bzz/" + "51551.findContours.jpg").getAbsolutePath(), imageMat);
        for (int i = 0; i < contoursMatVector.size(); i++) {
            Mat contourMat = contoursMatVector.get(i);
            RotatedRect rotatedRect = opencv_imgproc.minAreaRect(contourMat);
            System.out.println(rotatedRect.size().area());
            contoursMatList.add(contourMat.clone());
            Mat rotatedRectBoxPointsMat = new Mat();
            opencv_imgproc.boxPoints(rotatedRect, rotatedRectBoxPointsMat);
            Mat rotatedRectContourMat = new Mat(rotatedRectBoxPointsMat.rows(), contourMat.cols(), contourMat.type());
            FloatRawIndexer rotatedRectBoxPointsMatIndexer = rotatedRectBoxPointsMat.createIndexer();
            IntRawIndexer rotatedRectContourMatIndexer = rotatedRectContourMat.createIndexer();
            for (int y = 0; y < rotatedRectBoxPointsMat.rows(); y++) {
                int px = (int) rotatedRectBoxPointsMatIndexer.get(y, 0);
                int py = (int) rotatedRectBoxPointsMatIndexer.get(y, 1);
                rotatedRectContourMatIndexer.put(y, 0, px);
                rotatedRectContourMatIndexer.put(y, 1, py);
            }
            MatVector contourVector = new MatVector();
            contourVector.put(rotatedRectContourMat);
            opencv_imgproc.drawContours(detailImageMat, contourVector, -1, Scalar.YELLOW);
        }
//        new File("target/tfs").mkdirs();
        new File("target/tfs_bzz").mkdirs();
        long t = System.currentTimeMillis();
        opencv_imgcodecs.imwrite(new File("target/tfs_bzz/" + t + ".image.jpg").getAbsolutePath(), imageMat);
        opencv_imgcodecs.imwrite(new File("target/tfs_bzz/" + t + ".bunch.jpg").getAbsolutePath(), detailImageMat);
        opencv_imgcodecs.imwrite(new File("target/tfs_bzz/" + t + ".shape.jpg").getAbsolutePath(), shapeMat);
        detailImageMat.deallocate();
        shapeMat.deallocate();
        shapeMatIndexer.release();
        detailImageMatIndexer.release();
        System.out.println(contoursMatList);

        System.out.println(blobs.size());

        opencv_imgcodecs.imwrite(new File("target/tfs/blob." + System.currentTimeMillis() + ".jpg").getAbsolutePath(), detailImageMat);
        detailImageMat.deallocate();
        inputMat.deallocate();
        inputMatIndexer.release();
        System.out.println(blobs);
        return blobs;
    }

    public static double[] out_sigmoid(double[] x) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = sigmoidF(x[i]);
        }
        return out;
    }

    private static double sigmoidF(double x) {
        return 1f / (1f + Math.pow(Math.E, -1 * x));
    }

    private static void imshow(String txt, Mat img) {
        CanvasFrame canvasFrame = new CanvasFrame(txt);
        canvasFrame.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
        canvasFrame.setCanvasSize(img.cols(), img.rows());
        canvasFrame.showImage(new OpenCVFrameConverter.ToMat().convert(img));
    }


    private static float sigmoid(double a) {
        float b = 1.0f / (1.0f + (float) Math.exp(-a));
        return b;
    }


    public static float[][] transposeMatrix(float[][] m) {
        float[][] temp = new float[m[0].length][m.length];
        for (int i = 0; i < m.length; i++)
            for (int j = 0; j < m[0].length; j++)
                temp[j][i] = m[i][j];
        return temp;
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
                x2 > 640 ? 640 : x2,
                y2 > 640 ? 640 : y2};
    }


    public static void main(String[] args) throws Exception {
        find("E:\\pycharm_project\\tfservingconvert\\yolov8\\model-yolo_lg\\images/bzz_lg_01_yc4264_xc1330.jpg");
    }


}

