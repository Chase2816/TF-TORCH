package org.example;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import javaxt.io.Image;
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
import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class MaskRcnnGroupDemo {

    public static void main(String[] args) throws Exception {
        String modelName = "model-group";
        String signatureName = "serving_default";
        try {
            String file = "F:\\新建文件夹\\tfm\\src\\main\\1908.jpg";
            BufferedImage image = new Image(new File(file)).getBufferedImage();
            List<Integer> intList = new ArrayList<>();
            int pixels[] = image.getRGB(0, 0, image.getWidth(), image.getHeight(), null, 0, image.getWidth());
            // RGB转BGR格式
            for (int i = 0, j = 0; i < pixels.length; ++i, j += 3) {
                intList.add(pixels[i] & 0xff);
                intList.add((pixels[i] >> 8) & 0xff);
                intList.add((pixels[i] >> 16) & 0xff);
            }
            long t = System.currentTimeMillis();
            // http://172.20.112.102:8501/v1/models/model-group:predict
            ManagedChannel channel = ManagedChannelBuilder.forAddress("172.20.112.102", 8500).usePlaintext(true).build();
            //            System.out.println(channel);
            PredictionServiceGrpc.PredictionServiceBlockingStub stub = PredictionServiceGrpc.newBlockingStub(channel);
            //            System.out.println(stub);
            //创建请求
            Predict.PredictRequest.Builder predictRequestBuilder = Predict.PredictRequest.newBuilder();
            Model.ModelSpec.Builder modelSpecBuilder = Model.ModelSpec.newBuilder();
            modelSpecBuilder.setName(modelName);
            modelSpecBuilder.setSignatureName(signatureName);
            predictRequestBuilder.setModelSpec(modelSpecBuilder);
            TensorProto.Builder tensorProtoBuilder = TensorProto.newBuilder();
            tensorProtoBuilder.setDtype(DataType.DT_UINT8);
            TensorShapeProto.Builder tensorShapeBuilder = TensorShapeProto.newBuilder();
            tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(1));
            tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(image.getHeight()));
            tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(image.getWidth()));
            tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(3));
            tensorProtoBuilder.setTensorShape(tensorShapeBuilder.build());
            tensorProtoBuilder.addAllIntVal(intList);
            //            System.out.println(tensorShapeBuilder);
            //            predictRequestBuilder.putInputs("image_tensor", tensorProtoBuilder.build());
            predictRequestBuilder.putInputs("inputs", tensorProtoBuilder.build());
            // 结果
            Predict.PredictResponse predictResponse = stub.predict(predictRequestBuilder.build());
            //            System.out.println(predictResponse);
            List<Float> boxes = predictResponse.getOutputsOrThrow("detection_boxes").getFloatValList();
            List<Float> scores = predictResponse.getOutputsOrThrow("detection_scores").getFloatValList();
            List<Float> classes = predictResponse.getOutputsOrThrow("detection_classes").getFloatValList();
            List<Float> masks = predictResponse.getOutputsOrThrow("detection_masks").getFloatValList();
            //                                    System.out.println(scores);
            //                                    System.out.println(boxes);
//                                    System.out.println(masks.get(0));
            //                                    System.out.println(scores.size());
            //                                    System.out.println(boxes.size());
            //                        System.out.println(classes.size());
            //            System.out.println(masks.size());
            Mat srcMat = opencv_imgcodecs.imread(file);
            UByteRawIndexer srcMatRawIndexer = srcMat.createIndexer();
            int width = 640;
            int height = 480;
            for (int i = 0; i < scores.size(); i++) {
                if (scores.get(i) > 0.9) {
                    System.out.println("\n****************************************************************************");
                    System.out.println("index " + i + " score " + scores.get(i));
                    //box
                    System.out.println("******** box ***********");
                    int baseIndex = i * 4;
                    System.out.println("base index " + baseIndex);
                    List<Float> boxPoints = boxes.subList(baseIndex, baseIndex + 4);
                    System.out.println(boxPoints);
                    int boxImageTopLeftY = Math.round(boxPoints.get(0) * height);
                    int boxImageTopLeftX = Math.round(boxPoints.get(1) * width);
                    int boxImageBottomRightY = Math.round(boxPoints.get(2) * height);
                    int boxImageBottomRightX = Math.round(boxPoints.get(3) * width);
                    int boxWidth = boxImageBottomRightX - boxImageTopLeftX;
                    int boxHeight = boxImageBottomRightY - boxImageTopLeftY;
                    Rect rect = new Rect(new Point(boxImageTopLeftX, boxImageTopLeftY), new Point(boxImageBottomRightX, boxImageBottomRightY));
                    //mask
                    System.out.println("******** mask ***********");
//                    baseIndex = i * 15 * 15;
                    baseIndex = i * 33 * 33;
                    System.out.println("base index " + baseIndex);
//                    List<Float> maskPoints = masks.subList(baseIndex, baseIndex + 15 * 15);
                    List<Float> maskPoints = masks.subList(baseIndex, baseIndex + 33 * 33);
                    System.out.println(maskPoints);
                    Mat maskNumMat = new Mat(33, 33, opencv_core.CV_32F);
                    FloatRawIndexer maskNumMatIndexer = maskNumMat.createIndexer();
                    for (int y = 0; y < 33; y++) {
                        for (int x = 0; x < 33; x++) {
                            maskNumMatIndexer.put(y, x, maskPoints.get(y * 33 + x));
                        }
                    }
                    Mat maskMat = new Mat(boxHeight, boxWidth, maskNumMat.type());
                    opencv_imgproc.resize(maskNumMat, maskMat, new Size(boxWidth, boxHeight));
                    FloatRawIndexer maskFloatRawIndexer = maskMat.createIndexer();
                    for (int y = 0; y < boxHeight; y++) {
                        for (int x = 0; x < boxWidth; x++) {
                            int maskImageY = boxImageTopLeftY + y;
                            int maskImageX = boxImageTopLeftX + x;
                            if (maskFloatRawIndexer.get(y, x) > 0.3) {
                                srcMatRawIndexer.put(maskImageY, maskImageX * 3, 0);
                                srcMatRawIndexer.put(maskImageY, maskImageX * 3 + 1, 0);
                                srcMatRawIndexer.put(maskImageY, maskImageX * 3 + 2, 0);
                            }
                        }
                    }
                    opencv_imgproc.rectangle(srcMat, rect, AbstractScalar.YELLOW);
                }
            }
            opencv_imgcodecs.imwrite(new File("target/maskrcnn.jpg").getAbsolutePath(), srcMat);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
