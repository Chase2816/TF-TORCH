package org.example.Yolov3;

import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.opencv.global.opencv_dnn;
import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_dnn.Net;
import org.bytedeco.opencv.opencv_text.FloatVector;
import org.bytedeco.opencv.opencv_text.IntVector;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_dnn.NMSBoxes;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;


public class Yolo {

    public static void main(String[] args) {

//        Thread.currentThread().getContextClassLoader();
//        URL l1 = Thread.currentThread().getContextClassLoader().getResource("20190520_132858.jpg");
//        System.out.println(l1.getPath());
//        String a1 = yolo.class.getClassLoader().getResource("20190520_132858.jpg").getPath();
//        System.out.println(yolo.class.getClassLoader().getResource("21.jpg").getPath());
//        String cfg = yolo.class.getClassLoader().getResource("hot-yolov3.cfg").getPath();
//        String model = yolo.class.getClassLoader().getResource("hot-yolov3.weights").getPath();
//        String name = yolo.class.getClassLoader().getResource("hot.name").getPath();

//        if (args.length!=1){
//            System.out.println("Input error!!!");
//            System.out.println("Please input image path.");
//            System.exit(0);
//        }
//        String img_path = args[0];

        String cfg = "F:\\tfm\\src\\main\\resources\\yolov3data\\hot-yolov3.cfg";
        String model = "F:\\tfm\\src\\main\\resources\\yolov3data\\hot-yolov3.weights";
        String img_path = "F:\\tfm\\src\\main\\resources\\yolov3data\\stain2diode.jpg";

        Net net = opencv_dnn.readNetFromDarknet(cfg, model);

        Mat img = imread(img_path);
        Mat blob = opencv_dnn.blobFromImage(img, 1 / 255.0, new Size(416, 416), new Scalar(0.0), true, false, CV_32F);
        //Mat blob = opencv_dnn.blobFromImage(img,1,new Size(416,416),new Scalar(0),true,false,CV_32F);
        //Mat blob = opencv_dnn.blobFromImage(img,0.00392,new Size(416,416),new Scalar(0),true,false,CV_32F);
        System.out.println(blob);

        net.setPreferableBackend(3); //0
        net.setPreferableTarget(0);

        net.setInput(blob);

        StringVector outNames = net.getUnconnectedOutLayersNames();
        //System.out.println(outNames);

        MatVector outs = new MatVector(outNames.size());

        net.forward(outs, outNames);
        //System.out.println(outs);

        float threshold = 0.25f;      //0.3
        float nmsThreshold = 0.45f;   //0.5

        Yolo yolo = new Yolo();
        yolo.GetResult(outs, img, threshold, nmsThreshold, true,img_path);
    }

    private void GetResult(MatVector output, Mat image, float threshold, float nmsThreshold, boolean nms,String img_path) {
        nms = false;
        IntVector classIds = new IntVector();
        FloatVector confidences = new FloatVector();
        RectVector boxes = new RectVector();
        try {
            for (int i = 0; i < output.size(); ++i) {
                Mat result = output.get(i);
                //System.out.println("result："+result.rows());

                for (int j = 0; j < result.rows(); j++) {
                    FloatPointer data = new FloatPointer(result.row(j).data());
                    Mat scores = result.row(j).colRange(5, result.cols());
                    //System.out.println("score:"+scores);

                    Point classIdPoint = new Point(1);
                    DoublePointer confidence = new DoublePointer(1);

                    minMaxLoc(scores, null, confidence, null, classIdPoint, null);

                    //System.out.println(confidence.get());

                    if (confidence.get() > threshold) {
                        //System.out.println("data:"+data.get(0));
                        //System.out.println(data.get(1));
                        //System.out.println(data.get(2));
                        //System.out.println(data.get(3));
                        int centerX = (int) (data.get(0) * image.cols());
                        int centerY = (int) (data.get(1) * image.rows());
                        int width = (int) (data.get(2) * image.cols());
                        int height = (int) (data.get(3) * image.rows());
                        int left = centerX - width / 2;
                        int top = centerY - height / 2;

                        classIds.push_back(classIdPoint.x());
                        confidences.push_back((float) confidence.get());
                        boxes.push_back(new Rect(left, top, width, height));
                    }
                }
            }

            //System.out.println("classIds:"+classIds);
            //System.out.println(confidences);
            //System.out.println(boxes);

            /*if (nms) {
                IntPointer indices = new IntPointer(confidences.size());
                for (int i = 0; i < confidences.size(); ++i) {
                    Rect box = boxes.get(i);
                    int classId = classIds.get(i);
                    String res = "idx="+classId+"conf="+confidences.get(i);
                    res += "box.x=" + box.x() +"box.y="+box.y()+"box.width"+box.width()+"box.height"+box.height();
                    System.out.println(res);
                }
            }*/

            IntPointer indices = new IntPointer(confidences.size());
            FloatPointer confidencesPointer = new FloatPointer(confidences.size());
            confidencesPointer.put(confidences.get());

            NMSBoxes(boxes, confidencesPointer, threshold, nmsThreshold, indices, 1.f, 0);
            //NMSBoxes(boxes, confidencesPointer, threshold, nmsThreshold, indices);

            List<String> list = new ArrayList<String>();
            FileInputStream fis = new FileInputStream("F:\\tfm\\src\\main\\resources\\yolov3data\\hot.names");
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

            List<Integer> diode = new ArrayList<>();
            List<Integer> shadow = new ArrayList<>();
            List<Integer> stain = new ArrayList<>();
            //System.out.println(indices.sizeof());
            //System.out.println(indices.limit());


            File tempFile =new File( img_path.trim());
            String fileName = tempFile.getName();
            //System.out.println("fileName:" + fileName);

            IplImage rawImage = null;
            rawImage = cvLoadImage(img_path);

            String detect = "detect_img："+fileName+"  |  object_num：" + indices.limit();
            System.out.println(detect);

            for (int m = 0; m < indices.limit(); ++m) {
                int i = indices.get(m);
                Rect box = boxes.get(i);
                //System.out.println(box);
                int classId = classIds.get(i);
                //System.out.println("name:"+Labels);
                //System.out.println("classid"+classId);
                //System.out.println(box.x());
                //System.out.println(getType(box.x()));
                //System.exit(0);
                if (classId == 0) {
                    CvPoint pt1 = cvPoint(box.x(),box.y());
                    CvPoint pt2 = cvPoint(box.width()+box.x(),box.height()+box.y());
                    CvScalar color = cvScalar(255,0,0,0);
                    cvRectangle(rawImage, pt1, pt2, color, 1, 4, 0);
                    cvPutText(rawImage,Labels[classId],pt1,cvFont(1.5,1),CvScalar.BLUE);

                    diode.add(box.x());
                    diode.add(box.y());
                    diode.add(box.width());
                    diode.add(box.height());
                }
                if (classId == 1) {
                    CvPoint pt1 = cvPoint(box.x(),box.y());
                    CvPoint pt2 = cvPoint(box.width()+box.x(),box.height()+box.y());
                    CvScalar color = cvScalar(0,255,0,0);
                    cvRectangle(rawImage, pt1, pt2, color, 1, 4, 0);
                    cvPutText(rawImage,Labels[classId],pt1,cvFont(1.5,1),CvScalar.BLUE);

                    shadow.add(box.x());
                    shadow.add(box.y());
                    shadow.add(box.width());
                    shadow.add(box.height());
                }
                if (classId == 2) {
                    CvPoint pt1 = cvPoint(box.x(),box.y());
                    CvPoint pt2 = cvPoint(box.width()+box.x(),box.height()+box.y());
                    CvScalar color = cvScalar(0,0,255,0);
                    cvRectangle(rawImage, pt1, pt2, color, 1, 4, 0);
                    cvPutText(rawImage,Labels[classId],pt1,cvFont(1.5,1),CvScalar.BLUE);

                    stain.add(box.x());
                    stain.add(box.y());
                    stain.add(box.width());
                    stain.add(box.height());
                }

                String res = "idx：" + (m+1) + "  |  label：" + Labels[classId];
                res += "  |  coord：[ box_x：" + box.x() + " , box_y：" + box.y() + " , box_width：" + box.width() + " , box_height：" + box.height()+" ]";
                System.out.println(res);

            }

            String save_img = "result_" + fileName;
            System.out.println("save_img path："+save_img);
            cvSaveImage(save_img, rawImage);

            String total_diode = "diode_num ：" + diode.size() / 4 + "  |  diode_list：" + diode;
            String total_shadow = "shadow_num：" + shadow.size() / 4 + "  |  shadow_list：" + shadow;
            String total_stain = "stain_num ：" + stain.size() / 4 + "  |  stain_list：" + stain;

            System.out.println(total_diode);
            System.out.println(total_shadow);
            System.out.println(total_stain);

        } catch (Exception e) {
            System.out.println("GetResult error:" + e.getMessage());
        }
    }
}
