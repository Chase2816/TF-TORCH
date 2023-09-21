package com.lkyooo.test;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point2f;
import org.bytedeco.opencv.opencv_core.RotatedRect;

import java.io.File;
import java.util.*;

public class ProcessBunchAndBlob {

    public static void main(String[] args) throws Exception {
        File[] imageFiles = new File("/Users/Administrator/Documents/工作文档/通威渔光物联/光伏智能运维/Holmes/holmes-video/image-split/ddq1").listFiles(file -> file.getName().endsWith(".jpg"));
        Arrays.sort(imageFiles, Comparator.comparingInt((File o) -> o.getName().length()).thenComparing(File::getName));
        Map<Mat, List<Map<String, Object>>> bunchesToBlobs = new LinkedHashMap<>();
        Map<Mat, Map<String, Integer>> bunchesOffset = new LinkedHashMap<>();
        int count = 0;
        for (File f : imageFiles) {
            count++;
            System.out.println(count + " " + f.getAbsolutePath());
            List<Mat> bunches = Bunch.find(f.getAbsolutePath());
            if (bunches.isEmpty()) {
                continue;
            }
            List<Map<String, Object>> blobs = Blob.find(f.getAbsolutePath());
            String filename = f.getName();
            int offsetY = Integer.parseInt(filename.substring(filename.indexOf("yc") + 2, filename.indexOf("xc")));
            int offsetX = Integer.parseInt(filename.substring(filename.indexOf("xc") + 2, filename.indexOf(".")));
            blobs:
            for (Map<String, Object> blob : blobs) {
                float blobX = (float) blob.get("x");
                float blobY = (float) blob.get("y");
                Mat bunch = null;
                for (int i = 0; i < bunches.size(); i++) {
                    bunch = bunches.get(i);
                    if (opencv_imgproc.pointPolygonTest(bunch, new Point2f(blobX, blobY), false) >= 0) {
                        break;
                    } else if (i == bunches.size() - 1) {
                        continue blobs;
                    }
                }
                blobX += offsetX;
                blobY += offsetY;
                List<Map<String, Object>> bunchBlobs = bunchesToBlobs.getOrDefault(bunch, new ArrayList<>());
                Map<String, Object> globalBlob = new LinkedHashMap<>();
                globalBlob.put("x", Math.round(blobX));
                globalBlob.put("y", Math.round(blobY));
                globalBlob.put("size", blob.get("size"));
                globalBlob.put("type", blob.get("type"));
                globalBlob.put("temperature", 35.12);
                bunchBlobs.add(globalBlob);
                bunchesToBlobs.put(bunch, bunchBlobs);
                bunchesOffset.put(bunch, new LinkedHashMap<String, Integer>() {

                    {
                        put("offsetX", offsetX);
                        put("offsetY", offsetY);
                    }
                });
            }
        }
        Map<Mat, Map<String, Object>> mergedBunchesToBlobs = new LinkedHashMap<>();
        bunchesToBlobs:
        for (Map.Entry<Mat, List<Map<String, Object>>> entry : bunchesToBlobs.entrySet()) {
            RotatedRect bunchRotatedRect = opencv_imgproc.minAreaRect(entry.getKey());
            Point2f bunchCenterPoint2f = bunchRotatedRect.center();
            float[] bunchRotatedRectBoxPoints = ImageMatUtils.boxPoints(bunchRotatedRect);
            bunchRotatedRectBoxPoints[0] = bunchesOffset.get(entry.getKey()).get("offsetX") + bunchRotatedRectBoxPoints[0];
            bunchRotatedRectBoxPoints[1] = bunchesOffset.get(entry.getKey()).get("offsetY") + bunchRotatedRectBoxPoints[1];
            bunchRotatedRectBoxPoints[2] = bunchesOffset.get(entry.getKey()).get("offsetX") + bunchRotatedRectBoxPoints[2];
            bunchRotatedRectBoxPoints[3] = bunchesOffset.get(entry.getKey()).get("offsetY") + bunchRotatedRectBoxPoints[3];
            bunchRotatedRectBoxPoints[4] = bunchesOffset.get(entry.getKey()).get("offsetX") + bunchRotatedRectBoxPoints[4];
            bunchRotatedRectBoxPoints[5] = bunchesOffset.get(entry.getKey()).get("offsetY") + bunchRotatedRectBoxPoints[5];
            bunchRotatedRectBoxPoints[6] = bunchesOffset.get(entry.getKey()).get("offsetX") + bunchRotatedRectBoxPoints[6];
            bunchRotatedRectBoxPoints[7] = bunchesOffset.get(entry.getKey()).get("offsetY") + bunchRotatedRectBoxPoints[7];
            for (Map.Entry<Mat, Map<String, Object>> mergedEntry : mergedBunchesToBlobs.entrySet()) {
                RotatedRect mergedBunchRotatedRect = opencv_imgproc.minAreaRect(mergedEntry.getKey());
                float[] mergedBunchRotatedRectBoxPoints = ImageMatUtils.boxPoints(mergedBunchRotatedRect);
                mergedBunchRotatedRectBoxPoints[0] = bunchesOffset.get(mergedEntry.getKey()).get("offsetX") + mergedBunchRotatedRectBoxPoints[0];
                mergedBunchRotatedRectBoxPoints[1] = bunchesOffset.get(mergedEntry.getKey()).get("offsetY") + mergedBunchRotatedRectBoxPoints[1];
                mergedBunchRotatedRectBoxPoints[2] = bunchesOffset.get(mergedEntry.getKey()).get("offsetX") + mergedBunchRotatedRectBoxPoints[2];
                mergedBunchRotatedRectBoxPoints[3] = bunchesOffset.get(mergedEntry.getKey()).get("offsetY") + mergedBunchRotatedRectBoxPoints[3];
                mergedBunchRotatedRectBoxPoints[4] = bunchesOffset.get(mergedEntry.getKey()).get("offsetX") + mergedBunchRotatedRectBoxPoints[4];
                mergedBunchRotatedRectBoxPoints[5] = bunchesOffset.get(mergedEntry.getKey()).get("offsetY") + mergedBunchRotatedRectBoxPoints[5];
                mergedBunchRotatedRectBoxPoints[6] = bunchesOffset.get(mergedEntry.getKey()).get("offsetX") + mergedBunchRotatedRectBoxPoints[6];
                mergedBunchRotatedRectBoxPoints[7] = bunchesOffset.get(mergedEntry.getKey()).get("offsetY") + mergedBunchRotatedRectBoxPoints[7];
                Point2f mergedBunchCenterPoint2f = mergedBunchRotatedRect.center();
                if (isInBox(bunchRotatedRectBoxPoints, bunchesOffset.get(mergedEntry.getKey()).get("offsetX") + mergedBunchCenterPoint2f.x(), bunchesOffset.get(mergedEntry.getKey()).get("offsetY") + mergedBunchCenterPoint2f.y())
                        || isInBox(mergedBunchRotatedRectBoxPoints, bunchesOffset.get(entry.getKey()).get("offsetX") + bunchCenterPoint2f.x(), bunchesOffset.get(entry.getKey()).get("offsetY") + bunchCenterPoint2f.y())) {
                    List<Map<String, Object>> mergedBunchBlobs = (List<Map<String, Object>>) mergedEntry.getValue().get("bunchBlobs");
                    bunchBlobs:
                    for (Map<String, Object> bunchBlob : entry.getValue()) {
                        int bunchBlobX = (int) bunchBlob.get("x");
                        int bunchBlobY = (int) bunchBlob.get("y");
                        for (Map<String, Object> mergedBunchBlob : mergedBunchBlobs) {
                            float mergedBunchBlobX = (int) mergedBunchBlob.get("x");
                            float mergedBunchBlobY = (int) mergedBunchBlob.get("y");
                            if (mergedBunchBlobX + 5 > bunchBlobX && mergedBunchBlobX - 5 < bunchBlobX
                                    && mergedBunchBlobY + 5 > bunchBlobY && mergedBunchBlobY - 5 < bunchBlobY) {
                                continue bunchBlobs;
                            }
                        }
                        mergedBunchBlobs.add(bunchBlob);
                    }
                    continue bunchesToBlobs;
                }
            }
            mergedBunchesToBlobs.put(entry.getKey(), new LinkedHashMap<String, Object>() {

                {
                    put("tl", new LinkedHashMap<String, Object>() {{
                        put("x", bunchRotatedRectBoxPoints[0]);
                        put("y", bunchRotatedRectBoxPoints[1]);
                    }});
                    put("tr", new LinkedHashMap<String, Object>() {{
                        put("x", bunchRotatedRectBoxPoints[2]);
                        put("y", bunchRotatedRectBoxPoints[3]);
                    }});
                    put("bl", new LinkedHashMap<String, Object>() {{
                        put("x", bunchRotatedRectBoxPoints[4]);
                        put("y", bunchRotatedRectBoxPoints[5]);
                    }});
                    put("br", new LinkedHashMap<String, Object>() {{
                        put("x", bunchRotatedRectBoxPoints[6]);
                        put("y", bunchRotatedRectBoxPoints[7]);
                    }});
                    put("bunchBlobs", new ArrayList<>(entry.getValue()));
                }
            });
        }
        System.out.println("*****************");
        System.out.println(new ObjectMapper().writeValueAsString(mergedBunchesToBlobs.values()));
    }

    private static boolean isInBox(float[] boxPoints, float x, float y) {
        return x >= boxPoints[0] && x <= boxPoints[6]
                && y >= boxPoints[1] && y <= boxPoints[7];
    }
}
