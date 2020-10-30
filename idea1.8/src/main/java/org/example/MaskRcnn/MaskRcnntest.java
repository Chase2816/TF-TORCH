package org.example.MaskRcnn;

import com.sun.imageio.plugins.bmp.BMPConstants;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.opencv.global.opencv_dnn;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.Net;

import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.CV_32F;
import static org.bytedeco.opencv.global.opencv_core.CV_8UC1;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_BACKEND_OPENCV;
import static org.bytedeco.opencv.global.opencv_dnn.DNN_TARGET_CPU;
import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
import static org.bytedeco.opencv.global.opencv_imgproc.resize;


public class MaskRcnntest {
    public static void main(String[] args) {
        String cfg = "F:\\tfm\\src\\main\\resources\\maskrcnndata\\mask_rcnn.pbtxt";
        String model = "F:\\tfm\\src\\main\\resources\\maskrcnndata\\mask_rcnn.pb";
        String img_path = "F:\\tfm\\src\\main\\resources\\maskrcnndata\\202005130002.jpg";

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

        float threshold = 0.3f;      //0.5
        float maskThreshold = 0.3f;   //0.5

        Mat boxes = outs.get(0);
        Mat masks = outs.get(1);

        int numClasses = masks.size(1);
        int numDetection = boxes.size(2);

        boxes = boxes.reshape(1, (int) boxes.total() / 7);

        Size size = img.size();

        List<Mat> group = new ArrayList<>();

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

            Mat objectMask = new Mat(masks.size(2), masks.size(3), CV_32F, masks.ptr(i, classId)); //cv32F
            resize(objectMask, objectMask, new Size(width, height));
            objectMask.convertTo(objectMask, CV_8UC1, 255.0, 0.0);

            Mat mask = new Mat(objectMask.size().width(),objectMask.size().height(), BMPConstants.BI_RGB);
            System.out.println(mask);


//            PImage mask = new PImage(cvMask.size().width(), cvMask.size().height(), PConstants.RGB);
//            CvProcessingUtils.toPImage(detection.getMask(), mask);
//
//            blendMode(SCREEN);
//            tint(c, 200);
//            image(mask, detection.getX(), detection.getY());

//            List<MatVector> contours = new ArrayList<>();
//            org.opencv.core.Mat hierarchy = new org.opencv.core.Mat();
//            cvFindContours(objectMask, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

//            IplImage src = cvLoadImage(path);//hear path is actual path to image
//            IplImage grayImage    = IplImage.create(src.width(), src.height(), IPL_DEPTH_8U, 1);
//            cvCvtColor(src, grayImage, CV_RGB2GRAY);
//            cvThreshold(grayImage, grayImage, 127, 255, CV_THRESH_BINARY);
//            CvSeq cvSeq=new CvSeq();
//            CvMemStorage memory=CvMemStorage.create();
//            cvFindContours(objectMask, memory, cvSeq, Loader.sizeof(CvContour.class), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
//            MatVector contours = new MatVector();
//            Mat hierarchy=new Mat();
//            opencv_imgproc.findContours(objectMask, contours,hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
//            System.out.println(contours);
//
//            System.exit(1);
//

            group.add(objectMask);
//            System.out.println(objectMask);
//            System.out.println(classId);
//            System.out.println(data.get(2));
//            System.out.println(data.get(3));
//            System.out.println(data.get(4));
//            System.out.println(data.get(5));
//            System.out.println(data.get(6));

        }
        System.out.println(group);
    }

}