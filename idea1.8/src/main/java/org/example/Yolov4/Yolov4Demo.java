package org.example.Yolov4;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import net.coobird.thumbnailator.Thumbnails;
import org.bytedeco.opencv.opencv_core.CvPoint;
import org.bytedeco.opencv.opencv_core.CvScalar;
import org.bytedeco.opencv.opencv_core.IplImage;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;
import tensorflow.serving.Model;
import tensorflow.serving.Predict;
import tensorflow.serving.PredictionServiceGrpc;

import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.cvPoint;
import static org.bytedeco.opencv.global.opencv_core.cvScalar;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.helper.opencv_imgcodecs.cvLoadImage;
import static org.bytedeco.opencv.helper.opencv_imgcodecs.cvSaveImage;


public class Yolov4Demo {
    public static void main(String[] args) throws Exception {
        String modelName = "yolov4";
        String signatureName = "serving_default";
        String filename = "F:\\tfm\\src\\main\\resources\\kite.jpg";

        BufferedImage im = Thumbnails.of(filename).forceSize(416, 416).outputFormat("bmp").asBufferedImage();
        Raster raster = im.getRaster();
        List<Float> floatList = new ArrayList<>();
        float[] tmp = new float[raster.getWidth() * raster.getHeight() * raster.getNumBands()];
        float[] pixels = raster.getPixels(0, 0, raster.getWidth(), raster.getHeight(), tmp);
        for (float pixel : pixels) {
//            floatList.add(pixel);
            floatList.add(pixel / 255.0f);
        }

        long t = System.currentTimeMillis();
        //创建连接，注意usePlaintext设置为true表示用非SSL连接
        ManagedChannel channel = ManagedChannelBuilder.forAddress("172.20.112.102", 9900).usePlaintext(true).build();
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
        tensorProtoBuilder.setDtype(DataType.DT_FLOAT);
        TensorShapeProto.Builder tensorShapeBuilder = TensorShapeProto.newBuilder();
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(1));
        //150528 = 224 * 224 * 3
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(416));
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(416));
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(3));

        tensorProtoBuilder.setTensorShape(tensorShapeBuilder.build());
        tensorProtoBuilder.addAllFloatVal(floatList);
        predictRequestBuilder.putInputs("input_1", tensorProtoBuilder.build());
        //访问并获取结果
        Predict.PredictResponse predictResponse = stub.predict(predictRequestBuilder.build());
        List<Float> boxes = predictResponse.getOutputsOrThrow("tf_op_layer_concat_18").getFloatValList();
        System.out.println(boxes);


        List<List<Float>> bbox = getSplitList(84, boxes);
        System.out.println(boxes.size());
        System.out.println(bbox.size());

        List<List<Float>> bb = new ArrayList<>();

        IplImage rawImage = null;
        rawImage = cvLoadImage(filename);
        int height = rawImage.height();
        int width = rawImage.width();
        List<String> list = new ArrayList<String>();
        FileInputStream fis = new FileInputStream("F:\\tfm\\src\\main\\resources\\coco.names");
        InputStreamReader isr = new InputStreamReader(fis, "UTF-8");
        BufferedReader br = new BufferedReader(isr);
        String line;
        while ((line = br.readLine()) != null) {
            list.add(line);
        }
        String[] Labels = list.toArray(new String[list.size()]);
        //System.out.println("labels:"+Labels);
        br.close();
        isr.close();
        fis.close();

        for (int i = 0; i < bbox.size(); i++) {
            List<Float> xyxy = bbox.get(i).subList(0, 4);
            List<Float> conf = bbox.get(i).subList(5, bbox.get(i).size());
            System.out.println(conf);

            Float score = Collections.max(conf);
            int classes_id = conf.indexOf(score);
            System.out.println(score);
            System.out.println(classes_id);

//            if (score < 0.25) {
//                continue;
//            }

            float y1 = xyxy.get(0) * height;
            float y2 = xyxy.get(2) * height;
            float x1 = xyxy.get(1) * width;
            float x2 = xyxy.get(3) * width;

            CvPoint pt1 = cvPoint((int)x1,(int)y1);
            CvPoint pt2 = cvPoint((int)x2,(int)y2);
            CvScalar color = cvScalar(255,0,0,0);
            cvRectangle(rawImage,pt1,pt2,color,1,4,0);
            cvPutText(rawImage,Labels[classes_id],pt1,cvFont(1.5,1),CvScalar.BLUE);

            String result = String.format("x1：%s | y1：%s | x2：%s | y2：%s | score：%s | class_id：%s", x1, y1, x2, y2, score, classes_id);
            System.out.println(result);
        }

        String save_img = "result_dog.jpg" ;
        cvSaveImage(save_img, rawImage);

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
