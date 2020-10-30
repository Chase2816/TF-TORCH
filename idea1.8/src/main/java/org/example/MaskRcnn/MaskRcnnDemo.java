package org.example.MaskRcnn;

import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.opencv.global.opencv_dnn;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.Net;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_BACKEND_OPENCV;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_TARGET_CPU;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

/*
<dependency>
      <groupId>org.bytedeco</groupId>
      <artifactId>javacv-platform</artifactId>
      <version>1.5.3</version>
</dependency>
*/


public class MaskRcnnDemo {
    public static void main(String[] args) {
        String cfg = "F:\\tfm\\src\\main\\resources\\maskrcnndata\\mask_rcnn.pbtxt";
        String model = "F:\\tfm\\src\\main\\resources\\maskrcnndata\\mask_rcnn.pb";
        String img_path = "F:\\tfm\\src\\main\\resources\\maskrcnndata\\202005130001.jpg";

        Mat img = imread(img_path);

        Net net = opencv_dnn.readNetFromTensorflow(model, cfg);
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);

//        Mat blob = opencv_dnn.blobFromImage(img, 1.0, new Size(640, 480), new Scalar(0.0), true, false, CV_32F);
        Mat blob = opencv_dnn.blobFromImage(img, 1.0, img.size(), new Scalar(0.0), true, false, CV_32F);

        net.setInput(blob);

        StringVector out_names = new StringVector();
        out_names.push_back("detection_out_final");
        out_names.push_back("detection_masks");

//        List<StringVector> out_names = Arrays.asList(net.getUnconnectedOutLayersNames());

        MatVector outs = new MatVector(out_names.size());
        System.out.println(outs);

        net.forward(outs, out_names);

        System.out.println(out_names);
        System.out.println(outs);

        float threshold = 0.5f;      //0.5
        float maskThreshold = 0.3f;   //0.5

        Mat boxes = outs.get(0);
        Mat masks = outs.get(1);

        int numClasses = masks.size(1);
        int numDetection = boxes.size(2);

        boxes = boxes.reshape(1, (int) boxes.total() / 7);

        Size size = img.size();

        List<MatVector> group = new ArrayList<>();

        for (int i = 0; i < numDetection; i++) {
            FloatPointer data = new FloatPointer(boxes.row(i).data());
            float score = data.get(2);

            if (score < threshold) {
                continue;
            }

            int classId = (int) data.get(1);
            int left = Math.round(data.get(3) * size.width());
            int top = Math.round(data.get(4) * size.height());
            int right = Math.round(data.get(5) * size.width());
            int bottom = Math.round(data.get(6) * size.height());

            int width = right - left;
            int height = bottom - top;

            Mat objectMask = new Mat(masks.size(2), masks.size(3), CV_32F, masks.ptr(i, classId)); //cv32F  [15,15]
            resize(objectMask, objectMask, new Size(width, height));
//            objectMask.convertTo(objectMask, CV_8UC1, 255.0, 0.0);  //[524,66]
//            System.out.println(objectMask.rows()); // 66
//            System.out.println(objectMask.cols()); //524



//            Mat mas = new Mat(objectMask.size().width(), objectMask.size().height(), CV_8UC1);

            FloatRawIndexer objectMaskIndexer = objectMask.createIndexer();

            for (int j = 0; j < objectMask.rows(); j++) {
                for (int k = 0; k < objectMask.cols(); k++) {
//                    System.out.println("dsad");

                    System.out.println(objectMaskIndexer.get(j, k));
                    if (objectMaskIndexer.get(j, k) < maskThreshold) {
                        objectMaskIndexer.put(j, k, 0);
                    } else {
                        objectMaskIndexer.put(j, k, 1);
                    }
                }
            }
            objectMaskIndexer.release();
            objectMask.convertTo(objectMask, CV_8U, 255.0, 0.0);  //[524,66]

            MatVector contours = new MatVector();
            Mat hierarchy = new Mat();
            findContours(objectMask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            System.out.println(contours);

            CvScalar color = cvScalar(255,0,0,0);       // blue [green] [red]


            drawContours(objectMask, contours, 0, Scalar.RED, 2, 8, hierarchy, 1, new Point(0,0));
            imwrite("result1.jpg",img);


            group.add(contours);

        }
        System.out.println(group);
        System.out.println(group.size());
    }

}