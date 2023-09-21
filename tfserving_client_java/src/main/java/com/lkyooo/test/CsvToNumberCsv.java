package com.lkyooo.test;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class CsvToNumberCsv {

    static List<String> stringLabel = new ArrayList<>();

    public static void main(String[] args) throws Exception {
        {
            File f = new File("/Users/Administrator/Dev/Projects/research-projects/tfserving-maskrcnn/model/房价预测原始数据/train.csv");
            List<String> list = FileUtils.readLines(f, "UTF-8");
            StringBuffer sb = new StringBuffer(list.remove(0).substring(3));
            for (String s : list) {
                sb.append("\n");
                String[] ss = s.split(",");
                for (int i = 0; i < ss.length; i++) {
                    if (i == 1) {
                        sb.append(ss[i]);
                    } else if (i > 1) {
                        sb.append(",");
                        if (ss[i].matches("^[0-9]+$")) {
                            sb.append(ss[i]);
                        } else {
                            int num = stringLabel.indexOf(ss[i]);
                            if (-1 == num) {
                                stringLabel.add(ss[i]);
                                num = stringLabel.size() - 1;
                            }
                            sb.append(num);
                        }
                    }
                }
            }
            System.out.println(sb);
            FileUtils.write(new File("model/num." + f.getName().split("\\.")[0] + ".csv"), sb, "UTF-8");
        }
        {
            File f = new File("/Users/Administrator/Dev/Projects/research-projects/tfserving-maskrcnn/model/房价预测原始数据/test.csv");
            File f1 = new File("/Users/Administrator/Dev/Projects/research-projects/tfserving-maskrcnn/model/房价预测原始数据/sample_submission.csv");
            List<String> list = FileUtils.readLines(f, "UTF-8");
            List<String> list1 = FileUtils.readLines(f1, "UTF-8");
            list1.remove(0);
            StringBuffer sb = new StringBuffer(list.remove(0).substring(3) + ",SalePrice");
            for (String s : list) {
                sb.append("\n");
                String[] ss = s.split(",");
                for (int i = 0; i < ss.length; i++) {
                    if (i == 1) {
                        sb.append(ss[i]);
                    } else if (i > 1) {
                        sb.append(",");
                        if (ss[i].matches("^[0-9]+$")) {
                            sb.append(ss[i]);
                        } else {
                            int num = stringLabel.indexOf(ss[i]);
                            if (-1 == num) {
                                stringLabel.add(ss[i]);
                                num = stringLabel.size() - 1;
                            }
                            sb.append(num);
                        }
                    }
                }
                sb.append("," + list1.get(list.indexOf(s)).split(",")[1]);
            }
            System.out.println(sb);
            FileUtils.write(new File("model/num." + f.getName().split("\\.")[0] + ".csv"), sb, "UTF-8");
        }
    }
}
