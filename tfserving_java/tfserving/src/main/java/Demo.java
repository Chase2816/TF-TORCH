import org.apache.commons.lang3.tuple.Triple;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.Rect;

import java.io.File;
import java.util.List;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imread;

public class Demo {
    public static void main(String[] args) {

        TensorFlowServing tensorFlowServing = new TensorFlowServing();

        File file = new File("F:\\data\\hot_1203_data\\images");

        // 判断光伏组组串背景
        String s = tensorFlowServing.selectEngine(file);
        System.out.println(s);

        // 光伏组串分组
        File[] files = file.listFiles();
        System.out.println(files[5].getAbsolutePath());
        Mat mat = imread("D:\\lib\\tw_ygwl_ai_java_deploy\\tfserving\\src\\main\\resources\\210425_yc533xc2128.jpg");
        if(mat==null){
            System.out.println("读取图像失败,mat为空mat="+mat);
            return;
        }

        List<Mat> bunch = tensorFlowServing.findBunch(mat);
        System.out.println(bunch);

        // 光伏组串热斑检测
        Triple<List<Rect>, List<Rect>, List<Rect>> blobDJ = tensorFlowServing.findBlobDJ(mat);
        System.out.println(blobDJ);
    }
}
