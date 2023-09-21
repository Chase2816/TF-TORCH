package com.lkyooo.test;

import javafx.util.Pair;
import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.opencv.global.opencv_core;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.*;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class ImageMatUtils {

    /**
     * lt rt lb rb
     *
     * @param src
     * @return
     */
    public static float[] orderPosition(float[] src) {
        List<Pair<Float, Float>> list = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            list.add(new Pair<>(src[i * 2], src[i * 2 + 1]));
        }
        list.sort(Comparator.comparing(Pair::getValue));
        List<Pair<Float, Float>> list1 = list.subList(0, 2);
        list1.sort(Comparator.comparing(Pair::getKey));
        List<Pair<Float, Float>> list2 = list.subList(2, 4);
        list2.sort(Comparator.comparing(Pair::getKey));
        return new float[] {
                list1.get(0).getKey(), list1.get(0).getValue(),
                list1.get(1).getKey(), list1.get(1).getValue(),
                list2.get(0).getKey(), list2.get(0).getValue(),
                list2.get(1).getKey(), list2.get(1).getValue() };
    }

    public static float[] boxPoints(RotatedRect rotatedRect) {
        Mat mat = new Mat();
        opencv_imgproc.boxPoints(rotatedRect, mat);
        FloatRawIndexer floatRawIndexer = mat.createIndexer();
        float[] src = new float[8];
        src[0] = floatRawIndexer.get(0, 0);
        src[1] = floatRawIndexer.get(0, 1);
        src[2] = floatRawIndexer.get(1, 0);
        src[3] = floatRawIndexer.get(1, 1);
        src[4] = floatRawIndexer.get(2, 0);
        src[5] = floatRawIndexer.get(2, 1);
        src[6] = floatRawIndexer.get(3, 0);
        src[7] = floatRawIndexer.get(3, 1);
        floatRawIndexer.release();
        return orderPosition(src);
    }

    public static CvBox2D toCvBox2D(RotatedRect rotatedRect) {
        CvBox2D cvBox2D = new CvBox2D();
        cvBox2D.angle(rotatedRect.angle());
        CvPoint2D32f center = new CvPoint2D32f();
        center.x(rotatedRect.center().x());
        center.y(rotatedRect.center().y());
        cvBox2D.center(center);
        CvSize2D32f size = new CvSize2D32f();
        size.width(rotatedRect.size().width());
        size.height(rotatedRect.size().height());
        cvBox2D.size(size);
        return cvBox2D;
    }

    public static Mat pryDownNoise(Mat srcMat, int pryTimes) {
        Mat pryMat;
        if (srcMat.type() == opencv_core.CV_8UC1) {
            pryMat = new Mat(srcMat.size(), opencv_core.CV_8UC3);
            opencv_imgproc.cvtColor(srcMat, pryMat, opencv_imgproc.COLOR_GRAY2RGB);
        } else {
            pryMat = srcMat.clone();
        }
        for (int i = 0; i < pryTimes; i++) {
            opencv_imgproc.pyrDown(pryMat, pryMat, new Size(pryMat.cols() / 2, pryMat.rows() / 2), opencv_core.BORDER_DEFAULT);
        }
        for (int i = 0; i < pryTimes; i++) {
            opencv_imgproc.pyrUp(pryMat, pryMat, new Size(pryMat.cols() * 2, pryMat.rows() * 2), opencv_core.BORDER_DEFAULT);
        }
        return pryMat;
    }

    public static Mat rotate(Mat srcMat, int rotationDegree) {
        Mat rotatedMat = new Mat(srcMat.size(), srcMat.type());
        Point2f rawCenter = new Point2f(srcMat.cols() / 2.0F, srcMat.rows() / 2.0F);
        Mat rotationMatrix = opencv_imgproc.getRotationMatrix2D(rawCenter, rotationDegree, 1.0);
        opencv_imgproc.warpAffine(srcMat, rotatedMat, rotationMatrix, srcMat.size());
        rawCenter.releaseReference();
        rotationMatrix.releaseReference();
        return rotatedMat;
    }

    public static Mat whiteBalance(Mat srcMat) {
        Mat whiteBalancedMat = new Mat(srcMat.size(), srcMat.type());
        MatVector rgbMatVector = new MatVector();
        opencv_core.split(srcMat, rgbMatVector);
        double red, green, blue;
        blue = opencv_core.mean(rgbMatVector.get(0)).get(0);
        green = opencv_core.mean(rgbMatVector.get(1)).get(0);
        red = opencv_core.mean(rgbMatVector.get(2)).get(0);
        opencv_core.multiplyPut(rgbMatVector.get(0), (red + green + blue) / (3 * blue));
        opencv_core.multiplyPut(rgbMatVector.get(1), (red + green + blue) / (3 * green));
        opencv_core.multiplyPut(rgbMatVector.get(2), (red + green + blue) / (3 * red));
        opencv_core.merge(rgbMatVector, whiteBalancedMat);
        rgbMatVector.releaseReference();
        return whiteBalancedMat;
    }

    public static Mat colorFilter(Mat srcMat, ColorFilter colorFilter) {
        Mat colorFilteredMat = new Mat(srcMat.size(), srcMat.type());
        UByteRawIndexer colorFilteredMatRawIndexer = colorFilteredMat.createIndexer();
        UByteRawIndexer srcScaleSizeMatRawIndexer = srcMat.createIndexer();
        for (int y = 0; y < srcMat.rows(); y++) {
            for (int x = 0; x < srcMat.cols(); x++) {
                int baseX = x * 3;
                int blue = srcScaleSizeMatRawIndexer.get(y, baseX);
                int green = srcScaleSizeMatRawIndexer.get(y, baseX + 1);
                int red = srcScaleSizeMatRawIndexer.get(y, baseX + 2);
                if (colorFilter.isRetain(blue, green, red)) {
                    colorFilteredMatRawIndexer.put(y, baseX, blue);
                    colorFilteredMatRawIndexer.put(y, baseX + 1, green);
                    colorFilteredMatRawIndexer.put(y, baseX + 2, red);
                } else {
                    colorFilteredMatRawIndexer.put(y, baseX, 0);
                    colorFilteredMatRawIndexer.put(y, baseX + 1, 0);
                    colorFilteredMatRawIndexer.put(y, baseX + 2, 0);
                }
            }
        }
        colorFilteredMatRawIndexer.release();
        srcScaleSizeMatRawIndexer.release();
        return colorFilteredMat;
    }

    public static Mat contrast(Mat srcMat, double alpha, int beta) {
        Mat contrastMat = new Mat(srcMat.size(), srcMat.type());
        UByteRawIndexer colorFilteredMatRawIndexer = contrastMat.createIndexer();
        UByteRawIndexer srcMatRawIndexer = srcMat.createIndexer();
        for (int y = 0; y < srcMat.rows(); y++) {
            for (int x = 0; x < srcMat.cols(); x++) {
                int baseX = x * 3;
                int blue = (int) (alpha * srcMatRawIndexer.get(y, baseX) + beta);
                int green = (int) (alpha * srcMatRawIndexer.get(y, baseX + 1) + beta);
                int red = (int) (alpha * srcMatRawIndexer.get(y, baseX + 2) + beta);
                colorFilteredMatRawIndexer.put(y, baseX, blue);
                colorFilteredMatRawIndexer.put(y, baseX + 1, green);
                colorFilteredMatRawIndexer.put(y, baseX + 2, red);
            }
        }
        colorFilteredMatRawIndexer.release();
        srcMatRawIndexer.release();
        return contrastMat;
    }

    public static Mat shape(Mat srcMat) {
        //        Mat kernel = new Mat(3, 3, opencv_core.CV_16SC1);
        //        kernel.put(0, 0, 0, -1, 0, -1, 5, -1, 0, -1, 0);
        return null;
    }

    public static Integer rotateToHorizontal(Mat thresholdMat) {
        int rotationDegree = 0;
        int whiteCountLimit = 0;
        for (int currentRotationDegree = -15; currentRotationDegree < 15; currentRotationDegree++) {
            Mat rotatedMat = ImageMatUtils.rotate(thresholdMat, currentRotationDegree);
            int whiteCountTotal = 0;
            UByteRawIndexer rotatedMatIndexer = rotatedMat.createIndexer();
            for (int y = 0; y < rotatedMat.rows(); y++) {
                int whiteCount = 0;
                for (int x = 0; x < rotatedMat.cols(); x++) {
                    if (rotatedMatIndexer.get(y, x) > 0) {
                        whiteCount++;
                    }
                }
                if (whiteCount > 50) {
                    whiteCountTotal += whiteCount;
                }
                if (whiteCountTotal > whiteCountLimit) {
                    whiteCountLimit = whiteCountTotal;
                    rotationDegree = currentRotationDegree;
                }
            }
            rotatedMat.releaseReference();
            rotatedMatIndexer.release();
        }
        return rotationDegree;
    }

    public static Pair<Mat, Integer> rotateToHorizontalErodeMat(Mat srcMat, int erodeSize) {
        int rotationDegree = 0;
        double limit = 1.01;
        for (int currentRotationDegree = -15; currentRotationDegree < 15; currentRotationDegree++) {
            Mat rotatedMat = ImageMatUtils.rotate(srcMat, currentRotationDegree);
            Mat tempMat = new Mat();
            opencv_imgproc.getRectSubPix(rotatedMat,
                                         new Size(srcMat.cols() - srcMat.cols() / 10, srcMat.rows() - srcMat.rows() / 10),
                                         new Point2f(srcMat.cols() / 2, srcMat.rows() / 2),
                                         tempMat);
            Mat rotatedErodeMat = new Mat();
            opencv_imgproc.erode(tempMat, rotatedErodeMat,
                                 opencv_imgproc.getStructuringElement(opencv_imgproc.MORPH_ERODE, new Size(erodeSize, 1)));
            UByteRawIndexer erodeMatIndexer = rotatedErodeMat.createIndexer();
            double blackCount = 0;
            for (int y = 0; y < rotatedErodeMat.rows(); y++) {
                for (int x = 0; x < rotatedErodeMat.cols(); x++) {
                    int baseX = x * 3;
                    int b = erodeMatIndexer.get(y, baseX);
                    int g = erodeMatIndexer.get(y, baseX + 1);
                    int r = erodeMatIndexer.get(y, baseX + 2);
                    if (b == 0 && g == 0 && r == 0) {
                        blackCount++;
                    }
                }
            }
            double rate = blackCount / rotatedErodeMat.size().area();
            if (rate < limit) {
                limit = rate;
                rotationDegree = currentRotationDegree;
            }
            erodeMatIndexer.release();
            tempMat.releaseReference();
            rotatedErodeMat.releaseReference();
        }
        Mat horizontalErodeMat = new Mat();
        opencv_imgproc.erode(ImageMatUtils.rotate(srcMat, rotationDegree), horizontalErodeMat,
                             opencv_imgproc.getStructuringElement(opencv_imgproc.MORPH_ERODE, new Size(erodeSize, 1)));
        return new Pair<>(horizontalErodeMat, rotationDegree);
    }

    public interface ColorFilter {

        boolean isRetain(int blue, int green, int red);
    }
}
