package com.lkyooo.test;

import org.apache.commons.io.FileUtils;
import org.bytedeco.opencv.global.opencv_imgcodecs;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Point2f;
import org.bytedeco.opencv.opencv_core.Size;

import java.io.File;

public class ImageSplitter {

    public static void main(String[] args) throws Exception {
                splitAndScale(new File("/Users/Administrator/Documents/工作文档/通威渔光物联/光伏智能运维/Holmes/holmes-video/山西泽州/山西泽州红外.geo.tif"), 1, 2);
//        splitAndScale("/Users/Administrator/Downloads/天津明致", 1, 1);
    }

    static void splitAndScale(File file, int imageScale, int subImageScale) throws Exception {
        int subImageWidth = 1600 / subImageScale;
        int subImageHeight = 1600 / subImageScale;
        System.out.println(file.getAbsoluteFile());
        File target1600Dir = new File(file.getParentFile().getAbsolutePath() + "/拆图1600/" + file.getName().substring(0, file.getName().length() - 4));
        File target800Dir = new File(file.getParentFile().getAbsolutePath() + "/拆图800/" + file.getName().substring(0, file.getName().length() - 4));
        if (target1600Dir.exists()) {
            FileUtils.deleteDirectory(target1600Dir);
        }
        target1600Dir.mkdirs();
        target800Dir.mkdirs();
        Mat imageRawMat = opencv_imgcodecs.imread(file.getAbsolutePath());
        Size size = new Size(imageRawMat.size().width() / imageScale, imageRawMat.size().height() / imageScale);
        Mat imageMat = new Mat(size, imageRawMat.type());
        opencv_imgproc.resize(imageRawMat, imageMat, size);
        Mat subImageMat;
        for (int subCenterY = subImageHeight / 2; subCenterY < imageMat.rows() - subImageHeight / 2; subCenterY += subImageHeight / 3) {
            for (int subCenterX = subImageWidth / 2; subCenterX < imageMat.cols() - subImageWidth / 2; subCenterX += subImageWidth / 6) {
                subImageMat = new Mat(subImageHeight, subImageWidth, imageMat.type());
                opencv_imgproc.getRectSubPix(imageMat,
                                             new Size(subImageWidth, subImageHeight),
                                             new Point2f(subCenterX, subCenterY),
                                             subImageMat);
                String filenameOutput = "yc" + (subCenterY - subImageHeight / 2) + "xc" + (subCenterX - subImageWidth / 2) + ".jpg";
                opencv_imgcodecs.imwrite(target1600Dir.getAbsolutePath() + "/" + filenameOutput, subImageMat);
                //
                Mat subImageZoomOutMat = new Mat(subImageHeight / 2, subImageWidth / 2, imageMat.type());
                Size subImageZoomOutSize = new Size(subImageWidth / 2, subImageHeight / 2);
                opencv_imgproc.resize(subImageMat, subImageZoomOutMat, subImageZoomOutSize);
                opencv_imgcodecs.imwrite(target800Dir.getAbsolutePath() + "/yc" + (subCenterY - subImageHeight / 2) + "xc" + (subCenterX - subImageWidth / 2) + ".jpg", subImageZoomOutMat);
                subImageMat.release();
            }
        }
    }

    static void splitAndScale(String filename, int imageScale, int subImageScale) throws Exception {
        for (File file : new File(filename).listFiles((dir, name) -> name.endsWith(".tif"))) {
            splitAndScale(file, imageScale, subImageScale);
        }
    }
}
