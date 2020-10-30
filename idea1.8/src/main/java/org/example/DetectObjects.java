//package org.example;/*
//G:\IdeaProjects\ai\src\main\resource/models/ssd_inception_v2_coco_2017_11_17/saved_model G:/IdeaProjects/ai/src/main/resource/labels/mscoco_label_map.pbtxt G:/IdeaProjects/ai/src/main/resource/images/test.jpg
//G:\IdeaProjects\ai\src\main\resource\models\VOC6_0801_graph\saved_model G:\IdeaProjects\ai\src\main\resource\labels\pascal_label6_map.pbtxt G:\IdeaProjects\ai\src\main\resource\images\1.jpg
//G:\IdeaProjects\ai\src\main\resource\models\VOC11_0802_graph\saved_model G:\IdeaProjects\ai\src\main\resource\labels\pascal_label11_map.pbtxt G:\IdeaProjects\ai\src\main\resource\images\1.jpg
//G:\IdeaProjects\ai\src\main\resource\models\frcnn_voc6\saved_model G:\IdeaProjects\ai\src\main\resource\labels\pascal_label6_map.pbtxt G:\IdeaProjects\ai\src\main\resource\images\1.jpg
//G:\IdeaProjects\ai\src\main\resource\models\VOC6_0801_graph\saved_model G:\IdeaProjects\ai\src\main\resource\labels\pascal_label6_map.pbtxt G:\IdeaProjects\test_image\IMG.jpg
//G:\IdeaProjects\ai\src\main\resource\models\VOC25_augment162659_graph\saved_model G:\IdeaProjects\ai\src\main\resource\labels\pascal_label25_map.pbtxt G:\IdeaProjects\test_image\IMG_001.jpg
//==============================================================================*/
//
//import static object_detection.protos.StringIntLabelMapOuterClass.StringIntLabelMap;
//import static object_detection.protos.StringIntLabelMapOuterClass.StringIntLabelMapItem;
//
//import com.google.protobuf.TextFormat;
//
//import java.awt.*;
//import java.awt.image.BufferedImage;
//import java.awt.image.DataBufferByte;
//import java.io.File;
//import java.io.IOException;
//import java.nio.ByteBuffer;
//import java.nio.charset.StandardCharsets;
//import java.nio.file.Files;
//import java.nio.file.Paths;
//import java.util.List;
//import java.util.Map;
//import javax.imageio.ImageIO;
//
//import org.tensorflow.SavedModelBundle;
//import org.tensorflow.Tensor;
//import org.tensorflow.framework.MetaGraphDef;
//import org.tensorflow.framework.SignatureDef;
//import org.tensorflow.framework.TensorInfo;
//import org.tensorflow.types.UInt8;
//
//import java.util.Arrays;
//import java.io.FileOutputStream;
//
///**
// * Java inference for the Object Detection API at:
// * https://github.com/tensorflow/models/blob/master/research/object_detection/
// */
//public class DetectObjects {
//    public static void main(String[] args) throws Exception {
//        if (args.length < 3) {
//            System.exit(1);
//        }
//        final String[] labels = loadLabels(args[1]);
//        try (SavedModelBundle model = SavedModelBundle.load(args[0], "serve")) {
////      printSignature(model);
//            for (int arg = 2; arg < args.length; arg++) {
//                String filename = args[arg];
//                List<Tensor<?>> outputs = null;
//                try (Tensor<UInt8> input = makeImageTensor(filename)) {
//                    outputs =
//                            model
//                                    .session()
//                                    .runner()
//                                    .feed("image_tensor", input)
//                                    .fetch("detection_scores")
//                                    .fetch("detection_classes")
//                                    .fetch("detection_boxes")
//                                    .run();
//                }
//                try (Tensor<Float> scoresT = outputs.get(0).expect(Float.class);
//                     Tensor<Float> classesT = outputs.get(1).expect(Float.class);
//                     Tensor<Float> boxesT = outputs.get(2).expect(Float.class)) {
//                    // All these tensors have:
//                    // - 1 as the first dimension
//                    // - maxObjects as the second dimension
//                    // While boxesT will have 4 as the third dimension (2 sets of (x, y) coordinates).
//                    // This can be verified by looking at scoresT.shape() etc.
//                    int maxObjects = (int) scoresT.shape()[1];
//                    float[] scores = scoresT.copyTo(new float[1][maxObjects])[0];
//                    float[] classes = classesT.copyTo(new float[1][maxObjects])[0];
//                    float[][] boxes = boxesT.copyTo(new float[1][maxObjects][4])[0];
//                    // Print all objects whose score is at least 0.5.
//                    System.out.printf("* %s\n", filename);
//                    boolean foundSomething = false;
//                    for (int i = 0; i < scores.length; ++i) {
////            System.out.printf("* %s\n", scores[i]);
//                        if (scores[i] < 0.5) {
////            if (scores[i] < 0.7) {
//                            continue;
//                        }
//                        foundSomething = true;
////            System.out.printf("\tFound: %-20s (score: %.4f) (box: %s)\n", labels[(int) classes[i]], scores[i],Arrays.toString(boxes[i]));
////            System.out.printf("\tFound %-20s (score: %.4f) (box:%s)\n", classes[i], scores[i],Arrays.toString(boxes[i]));
//
//                        // drawimage
//                        BufferedImage image = ImageIO.read(new File(filename));
////            System.out.println(image.getWidth());
////            System.out.println(image.getHeight());
//                        float imagew = image.getWidth();
//                        float imageh = image.getHeight();
//                        float ymin = imageh * boxes[i][0];
//                        float xmin = imagew * boxes[i][1];
//                        float ymax = imageh * boxes[i][2];
//                        float xmax = imagew * boxes[i][3];
//                        float w = xmax - xmin;
//                        float h = ymax - ymin;
//
//                        System.out.printf("\tFound: %-20s (score: %.4f) (box: %s) (image_w_h: %s)\n", labels[(int) classes[i]], scores[i], "[" + (int) xmin + ", " + (int) ymin + ", " + (int) xmax + ", " + (int) ymax + "]", "" + (int) imagew + "x" + (int) imageh);
//
//
//                        int iw = (int) w;
//                        int ih = (int) h;
//                        int iy = (int) ymin;
//                        int ix = (int) xmin;
//
//                        Graphics g = image.getGraphics();
//                        Graphics2D g2 = (Graphics2D) g;
//                        g2.setStroke(new BasicStroke(3.0f));
//                        g2.setColor(Color.GREEN);
//                        g2.drawRect(ix, iy, iw, ih);
//                        g2.drawString(labels[(int) classes[i]], ix, iy);
//
////            g.setColor(Color.RED);
////            g.drawRect(ix, iy, iw, ih);
////            g.drawString(labels[(int) classes[i]],ix,iy);
//
//                        //g.dispose();
////            String path = args[2].replace("IMG","result");
//                        String path = args[2].replace("IMG_024", "test024");
//                        FileOutputStream out = new FileOutputStream(path);
//                        filename = path;
//                        ImageIO.write(image, "jpeg", out);
//                    }
//
//                    if (!foundSomething) {
//                        System.out.println("No objects detected with a high enough score.");
//                    }
//                }
//            }
//        }
//    }
//
//    private static void printSignature(SavedModelBundle model) throws Exception {
//        MetaGraphDef m = MetaGraphDef.parseFrom(model.metaGraphDef());
//        SignatureDef sig = m.getSignatureDefOrThrow("serving_default");
//        int numInputs = sig.getInputsCount();
//        int i = 1;
//        System.out.println("MODEL SIGNATURE");
//        System.out.println("Inputs:");
//        for (Map.Entry<String, TensorInfo> entry : sig.getInputsMap().entrySet()) {
//            TensorInfo t = entry.getValue();
//            System.out.printf(
//                    "%d of %d: %-20s (Node name in graph: %-20s, type: %s)\n",
//                    i++, numInputs, entry.getKey(), t.getName(), t.getDtype());
//        }
//        int numOutputs = sig.getOutputsCount();
//        i = 1;
//        System.out.println("Outputs:");
//        for (Map.Entry<String, TensorInfo> entry : sig.getOutputsMap().entrySet()) {
//            TensorInfo t = entry.getValue();
//            System.out.printf(
//                    "%d of %d: %-20s (Node name in graph: %-20s, type: %s)\n",
//                    i++, numOutputs, entry.getKey(), t.getName(), t.getDtype());
//        }
//        System.out.println("-----------------------------------------------");
//    }
//
//    private static String[] loadLabels(String filename) throws Exception {
//        String text = new String(Files.readAllBytes(Paths.get(filename)), StandardCharsets.UTF_8);
//        StringIntLabelMap.Builder builder = StringIntLabelMap.newBuilder();
//        TextFormat.merge(text, builder);
//        StringIntLabelMap proto = builder.build();
//        int maxId = 0;
//        for (StringIntLabelMapItem item : proto.getItemList()) {
//            if (item.getId() > maxId) {
//                maxId = item.getId();
//            }
//        }
//        String[] ret = new String[maxId + 1];
//        for (StringIntLabelMapItem item : proto.getItemList()) {
////      ret[item.getId()] = item.getDisplayName();
//            ret[item.getId()] = item.getName();
//        }
//        return ret;
//    }
//
//    private static void bgr2rgb(byte[] data) {
//        for (int i = 0; i < data.length; i += 3) {
//            byte tmp = data[i];
//            data[i] = data[i + 2];
//            data[i + 2] = tmp;
//        }
//    }
//
//    private static Tensor<UInt8> makeImageTensor(String filename) throws IOException {
//        BufferedImage img = ImageIO.read(new File(filename));
//        if (img.getType() != BufferedImage.TYPE_3BYTE_BGR) {
//            throw new IOException(
//                    String.format(
//                            "Expected 3-byte BGR encoding in BufferedImage, found %d (file: %s). This code could be made more robust",
//                            img.getType(), filename));
//        }
//        byte[] data = ((DataBufferByte) img.getData().getDataBuffer()).getData();
//        // ImageIO.read seems to produce BGR-encoded images, but the model expects RGB.
//        bgr2rgb(data);
//        final long BATCH_SIZE = 1;
//        final long CHANNELS = 3;
//        long[] shape = new long[]{BATCH_SIZE, img.getHeight(), img.getWidth(), CHANNELS};
//        return Tensor.create(UInt8.class, shape, ByteBuffer.wrap(data));
//    }
//}
