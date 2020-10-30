package org.example.BoatGroup;

import com.alibaba.fastjson.JSONObject;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.example.BoatGroup.Yolov5Sgd.formUpload;

public class SgdGroup {

    public static File[] sort(File[] s) {
        //中间值
        File temp = null;
        //外循环:我认为最小的数,从0~长度-1
        for (int j = 0; j < s.length - 1; j++) {
            //最小值:假设第一个数就是最小的
            String min = s[j].getName();
            //记录最小数的下标的
            int minIndex = j;
            //内循环:拿我认为的最小的数和后面的数一个个进行比较
            for (int k = j + 1; k < s.length; k++) {
                //找到最小值
                if (Integer.parseInt(min.substring(0, min.indexOf("."))) > Integer.parseInt(s[k].getName().substring(0, s[k].getName().indexOf(".")))) {
                    //修改最小
                    min = s[k].getName();
                    minIndex = k;
                }
            }
            //当退出内层循环就找到这次的最小值
            //交换位置
            temp = s[j];
            s[j] = s[minIndex];
            s[minIndex] = temp;
        }
        return s;
    }

    public static void main(String[] args) {
        String Aurl = "http://172.20.112.102:5001/predict";
        String Durl = "http://172.20.112.102:5001/detect";

        File file = new File("E:\\pycharm_project\\Data-proessing\\boat\\data");
        File[] fs = sort(file.listFiles());

        List<Integer> l = new ArrayList<>();
        for (File f : fs) {
            System.out.println(f.toString());

            Map<String, String> textMap = new HashMap<String, String>();
            //可以设置多个input的name，value
            //设置file的name，路径
            Map<String, String> fileMap = new HashMap<String, String>();
            fileMap.put("file", f.toString());
            String contentType = "";//image/png
            String Aret = formUpload(Aurl, textMap, fileMap, contentType);
            System.out.println(Aret);
//            String Dret = formUpload(Durl, textMap, fileMap,contentType);
//            System.out.println(Dret);

            JSONObject jsonObject = JSONObject.parseObject(Aret);
            Double s = jsonObject.getDouble("s");

            if (s <= 0.8) {
                System.out.println("ss");
                String s1 = f.getName().split(".jpg")[0];
                int a = Integer.parseInt(s1);
                l.add(a);
            }
        }
        System.out.println(l);
        System.out.println(l.size());


        List<Integer> l2 = new ArrayList<>();
        for (int i = 0; i < l.size(); i++) {
            if (i == 0) {
                l2.add(l.get(0));
            }
            if (i < l.size() - 1) {
                if (l.get(i + 1) - l.get(i) == 3) {
                    continue;
                }
                l2.add(l.get(i + 1));
            }
        }

        System.out.println(l2);
        System.out.println(l2.size());

        ArrayList arrayList = new ArrayList();

        for (int i = 1; i < l2.size(); i++) {
            if (i == 1) {
                List<String> a = new ArrayList<>();

                a.add("n");

                for (int j = 0; j < fs.length; j++) {
                    String s = fs[j].getName().split(".jpg")[0];
                    int aa = Integer.parseInt(s);

                    if (aa <= l2.get(i)) {
                        System.out.println(fs[j]);
                        Map<String, String> textMap = new HashMap<String, String>();
                        //可以设置多个input的name，value
                        //设置file的name，路径
                        Map<String, String> fileMap = new HashMap<String, String>();
                        fileMap.put("file", fs[j].toString());
                        String contentType = "";//image/png

                        String Dret = formUpload(Durl, textMap, fileMap, contentType);
                        System.out.println(Dret);

                        Map<String, String> fileMap2 = new HashMap<String, String>();
                        fileMap.put("file", fs[j + 1].toString());
                        String Dret2 = formUpload(Durl, textMap, fileMap, contentType);
                        System.out.println(Dret2);

                        JSONObject jsonObject = JSONObject.parseObject(Dret);
                        Double cx = jsonObject.getDouble("cx");
                        Double cy = jsonObject.getDouble("cy");
                        Double w = jsonObject.getDouble("w");
                        System.out.println(cx + "|" + cy + "|" + w);
                        JSONObject jsonObject2 = JSONObject.parseObject(Dret2);
                        Double cx2 = jsonObject2.getDouble("cx");
                        Double cy2 = jsonObject2.getDouble("cy");
                        Double w2 = jsonObject2.getDouble("w");
                        System.out.println(cx2 + "|" + cy2 + "|" + w2);

                        if (cx2 < cx & w2 / w > 0.95) {
                            System.out.println("ssssss");
                            a.add("n");
                        }
                    }

                }
                System.out.println(a);
                System.out.println(a.size());

            } else if (i < l2.size()) {
                List<String> b = new ArrayList<>();
                b.add("n");
                for (int j = 0; j < fs.length; j++) {
                    String s = fs[j].getName().split(".jpg")[0];
                    int aa = Integer.parseInt(s);
                    if (aa > l2.get(i - 1) + 3 & aa <= l2.get(i)) {

                    }
                }
            }
        }
    }
}
