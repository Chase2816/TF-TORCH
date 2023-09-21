//import org.bytedeco.javacpp.FloatPointer;
//import org.bytedeco.javacpp.IntPointer;
//import org.bytedeco.javacpp.indexer.FloatIndexer;
//import org.bytedeco.opencv.global.opencv_dnn;
//import org.bytedeco.opencv.opencv_core.*;
//import org.bytedeco.opencv.opencv_dnn.Net;
//import org.bytedeco.opencv.opencv_text.FloatVector;
//import org.bytedeco.opencv.opencv_text.IntVector;
//
//import java.io.IOException;
//import java.nio.file.Files;
//import java.nio.file.Path;
//import java.nio.file.Paths;
//import java.util.ArrayList;
//import java.util.List;
//
//import static org.bytedeco.opencv.global.opencv_core.CV_32F;
//import static org.bytedeco.opencv.global.opencv_core.getCudaEnabledDeviceCount;
//import static org.bytedeco.opencv.global.opencv_dnn.*;
//import static org.bytedeco.opencv.global.opencv_highgui.imshow;
//import static org.bytedeco.opencv.global.opencv_highgui.waitKey;
//import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;
//import static org.bytedeco.opencv.global.opencv_imgproc.LINE_8;
//import static org.bytedeco.opencv.global.opencv_imgproc.rectangle;
//
//public class YOLONet {
//
//    public static void main(String[] args) {
//        Mat image = imread("dog.jpg");
//
//        YOLONet yolo = new YOLONet(
//                "yolov4.cfg",
//                "yolov4.weights",
//                "coco.names",
//                608, 608);
//        yolo.setup();
//
//        List<ObjectDetectionResult> results = yolo.predict(image);
//
//        System.out.printf("Detected %d objects:\n", results.size());
//        for(ObjectDetectionResult result : results) {
//            System.out.printf("\t%s - %.2f%%\n", result.className, result.confidence * 100f);
//
//            // annotate on image
//            rectangle(image,
//                    new Point(result.x, result.y),
//                    new Point(result.x + result.width, result.y + result.height),
//                    Scalar.MAGENTA, 2, LINE_8, 0);
//        }
//
//        imshow("YOLO", image);
//        waitKey();
//    }
//
//    private Path configPath;
//    private Path weightsPath;
//    private Path namesPath;
//    private int width;
//    private int height;
//
//    private float confidenceThreshold = 0.5f;
//    private float nmsThreshold = 0.4f;
//
//    private Net net;
//    private StringVector outNames;
//
//    private List<String> names;
//
//    /**
//     * Creates a new YOLO network.
//     * @param configPath Path to the configuration file.
//     * @param weightsPath Path to the weights file.
//     * @param namesPath Path to the names file.
//     * @param width Width of the network as defined in the configuration.
//     * @param height Height of the network as defined in the configuration.
//     */
//    public YOLONet(String configPath, String weightsPath, String namesPath, int width, int height) {
//        this.configPath = Paths.get(configPath);
//        this.weightsPath = Paths.get(weightsPath);
//        this.namesPath = Paths.get(namesPath);
//        this.width = width;
//        this.height = height;
//    }
//
//    /**
//     * Initialises the network.
//     *
//     * @return True if the network initialisation was successful.
//     */
//    public boolean setup() {
//        net = readNetFromDarknet(
//                configPath.toAbsolutePath().toString(),
//                weightsPath.toAbsolutePath().toString());
//
//        // setup output layers
//        outNames = net.getUnconnectedOutLayersNames();
//
//        // enable cuda backend if available
////        if (getCudaEnabledDeviceCount() > 0) {
////            net.setPreferableBackend(opencv_dnn.DNN_BACKEND_CUDA);
////            net.setPreferableTarget(opencv_dnn.DNN_TARGET_CUDA);
////        }
//
//        // read names file
//        try {
//            names = Files.readAllLines(namesPath);
//        } catch (IOException e) {
//            System.err.println("Could not read names file!");
//            e.printStackTrace();
//        }
//
//        return !net.empty();
//    }
//
//    /**
//     * Runs the object detection on the frame.
//     *
//     * @param frame Input frame.
//     * @return List of objects that have been detected.
//     */
//    public List<ObjectDetectionResult> predict(Mat frame) {
//        Mat inputBlob = blobFromImage(frame,
//                1 / 255.0,
//                new Size(width, height),
//                new Scalar(0.0),
//                true, false, CV_32F);
//
//        net.setInput(inputBlob);
//
//        // run detection
//        MatVector outs = new MatVector(outNames.size());
//        net.forward(outs, outNames);
//
//        // evaluate result
//        List<ObjectDetectionResult> result = postprocess(frame, outs);
//
//        // cleanup
//        outs.releaseReference();
//        inputBlob.release();
//
//        return result;
//    }
//
//    /**
//     * Remove the bounding boxes with low confidence using non-maxima suppression
//     *
//     * @param frame Input frame
//     * @param outs  Network outputs
//     * @return List of objects
//     */
//    private List<ObjectDetectionResult> postprocess(Mat frame, MatVector outs) {
//        final IntVector classIds = new IntVector();
//        final FloatVector confidences = new FloatVector();
//        final RectVector boxes = new RectVector();
//
//        for (int i = 0; i < outs.size(); ++i) {
//            // extract the bounding boxes that have a high enough score
//            // and assign their highest confidence class prediction.
//            Mat result = outs.get(i);
//            FloatIndexer data = result.createIndexer();
//
//            for (int j = 0; j < result.rows(); j++) {
//                // minMaxLoc implemented in java because it is 1D
//                int maxIndex = -1;
//                float maxScore = Float.MIN_VALUE;
//                for (int k = 5; k < result.cols(); k++) {
//                    float score = data.get(j, k);
//                    if (score > maxScore) {
//                        maxScore = score;
//                        maxIndex = k - 5;
//                    }
//                }
//
//                if (maxScore > confidenceThreshold) {
//                    int centerX = (int) (data.get(j, 0) * frame.cols());
//                    int centerY = (int) (data.get(j, 1) * frame.rows());
//                    int width = (int) (data.get(j, 2) * frame.cols());
//                    int height = (int) (data.get(j, 3) * frame.rows());
//                    int left = centerX - width / 2;
//                    int top = centerY - height / 2;
//
//                    classIds.push_back(maxIndex);
//                    confidences.push_back(maxScore);
//
//                    boxes.push_back(new Rect(left, top, width, height));
//                }
//            }
//
//            data.release();
//            result.release();
//        }
//
//        // remove overlapping bounding boxes with NMS
//        IntPointer indices = new IntPointer(confidences.size());
//        FloatPointer confidencesPointer = new FloatPointer(confidences.size());
//        confidencesPointer.put(confidences.get());
//
//        NMSBoxes(boxes, confidencesPointer, confidenceThreshold, nmsThreshold, indices, 1.f, 0);
//
//        // create result list
//        List<ObjectDetectionResult> detections = new ArrayList<>();
//        for (int i = 0; i < indices.limit(); ++i) {
//            final int idx = indices.get(i);
//            final Rect box = boxes.get(idx);
//
//            final int clsId = classIds.get(idx);
//
//            detections.add(new ObjectDetectionResult() {{
//                classId = clsId;
//                className = names.get(clsId);
//                confidence = confidences.get(idx);
//                x = box.x();
//                y = box.y();
//                width = box.width();
//                height = box.height();
//            }});
//
//            box.releaseReference();
//        }
//
//        // cleanup
//        indices.releaseReference();
//        confidencesPointer.releaseReference();
//        classIds.releaseReference();
//        confidences.releaseReference();
//        boxes.releaseReference();
//
//        return detections;
//    }
//
//    /**
//     * Dataclass for object detection result.
//     */
//    public static class ObjectDetectionResult {
//        public int classId;
//        public String className;
//
//        public float confidence;
//
//        public int x;
//        public int y;
//        public int width;
//        public int height;
//    }
//}

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.bytedeco.javacpp.indexer.FloatIndexer;
import org.bytedeco.javacpp.indexer.IntIndexer;
import org.bytedeco.javacpp.indexer.UByteIndexer;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.OpenCVFrameConverter;

import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_core.*;
import static org.bytedeco.opencv.global.opencv_imgcodecs.*;
import static org.bytedeco.opencv.global.opencv_imgproc.*;

public class YOLONet {
    private static final int[] WHITE = {255, 255, 255};
    private static final int[] BLACK = {0, 0, 0};

    public static void main(String[] args) {
        // Load the image
        Mat src = imread(args[0]);
        // Check if everything was fine
        if (src.data().isNull())
            return;
        // Show source image
        imshow("Source Image", src);

        // Change the background from white to black, since that will help later to extract
        // better results during the use of Distance Transform
        UByteIndexer srcIndexer = src.createIndexer();
        for (int x = 0; x < srcIndexer.rows(); x++) {
            for (int y = 0; y < srcIndexer.cols(); y++) {
                int[] values = new int[3];
                srcIndexer.get(x, y, values);
                if (Arrays.equals(values, WHITE)) {
                    srcIndexer.put(x, y, BLACK);
                }
            }
        }
        // Show output image
        imshow("Black Background Image", src);

        // Create a kernel that we will use for accuting/sharpening our image
        Mat kernel = Mat.ones(3, 3, CV_32F).asMat();
        FloatIndexer kernelIndexer = kernel.createIndexer();
        kernelIndexer.put(1, 1, -8); // an approximation of second derivative, a quite strong kernel

        // do the laplacian filtering as it is
        // well, we need to convert everything in something more deeper then CV_8U
        // because the kernel has some negative values,
        // and we can expect in general to have a Laplacian image with negative values
        // BUT a 8bits unsigned int (the one we are working with) can contain values from 0 to 255
        // so the possible negative number will be truncated
        Mat imgLaplacian = new Mat();
        Mat sharp = src; // copy source image to another temporary one
        filter2D(sharp, imgLaplacian, CV_32F, kernel);
        src.convertTo(sharp, CV_32F);
        Mat imgResult = subtract(sharp, imgLaplacian).asMat();
        // convert back to 8bits gray scale
        imgResult.convertTo(imgResult, CV_8UC3);
        imgLaplacian.convertTo(imgLaplacian, CV_8UC3);
        // imshow( "Laplace Filtered Image", imgLaplacian );
        imshow("New Sharped Image", imgResult);

        src = imgResult; // copy back
        // Create binary image from source image
        Mat bw = new Mat();
        cvtColor(src, bw, CV_BGR2GRAY);
        threshold(bw, bw, 40, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
        imshow("Binary Image", bw);

        // Perform the distance transform algorithm
        Mat dist = new Mat();
        distanceTransform(bw, dist, CV_DIST_L2, 3);
        // Normalize the distance image for range = {0.0, 1.0}
        // so we can visualize and threshold it
        normalize(dist, dist, 0, 1., NORM_MINMAX, -1, null);
        imshow("Distance Transform Image", dist);

        // Threshold to obtain the peaks
        // This will be the markers for the foreground objects
        threshold(dist, dist, .4, 1., CV_THRESH_BINARY);
        // Dilate a bit the dist image
        Mat kernel1 = Mat.ones(3, 3, CV_8UC1).asMat();
        dilate(dist, dist, kernel1);
        imshow("Peaks", dist);
        // Create the CV_8U version of the distance image
        // It is needed for findContours()
        Mat dist_8u = new Mat();
        dist.convertTo(dist_8u, CV_8U);
        // Find total markers
        MatVector contours = new MatVector();
        findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        // Create the marker image for the watershed algorithm
        Mat markers = Mat.zeros(dist.size(), CV_32SC1).asMat();
        // Draw the foreground markers
        for (int i = 0; i < contours.size(); i++)
            drawContours(markers, contours, i, Scalar.all((i) + 1));
        // Draw the background marker
        circle(markers, new Point(5, 5), 3, RGB(255, 255, 255));
        imshow("Markers", multiply(markers, 10000).asMat());

        // Perform the watershed algorithm
        watershed(src, markers);
        Mat mark = Mat.zeros(markers.size(), CV_8UC1).asMat();
        markers.convertTo(mark, CV_8UC1);
        bitwise_not(mark, mark);
//            imshow("Markers_v2", mark); // uncomment this if you want to see how the mark
        // image looks like at that point
        // Generate random colors
        List<int[]> colors = new ArrayList<int[]>();
        for (int i = 0; i < contours.size(); i++) {
            int b = theRNG().uniform(0, 255);
            int g = theRNG().uniform(0, 255);
            int r = theRNG().uniform(0, 255);
            int[] color = { b, g, r };
            colors.add(color);
        }
        // Create the result image
        Mat dst = Mat.zeros(markers.size(), CV_8UC3).asMat();
        // Fill labeled objects with random colors
        IntIndexer markersIndexer = markers.createIndexer();
        UByteIndexer dstIndexer = dst.createIndexer();
        for (int i = 0; i < markersIndexer.rows(); i++) {
            for (int j = 0; j < markersIndexer.cols(); j++) {
                int index = markersIndexer.get(i, j);
                if (index > 0 && index <= contours.size())
                    dstIndexer.put(i, j, colors.get(index - 1));
                else
                    dstIndexer.put(i, j, BLACK);
            }
        }
        // Visualize the final image
        imshow("Final Result", dst);
    }

    //I wrote a custom imshow method for problems using the OpenCV original one
    private static void imshow(String txt, Mat img) {
        CanvasFrame canvasFrame = new CanvasFrame(txt);
        canvasFrame.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
        canvasFrame.setCanvasSize(img.cols(), img.rows());
        canvasFrame.showImage(new OpenCVFrameConverter.ToMat().convert(img));
    }

}