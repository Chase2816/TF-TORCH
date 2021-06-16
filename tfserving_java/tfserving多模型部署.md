1.安装docker
win10安装借鉴：https://www.jianshu.com/p/0f927dc8b23d

2.拉取tensorflow-serving镜像

```bash
docker pull tensorflow/serving
```

3.测试tfserving的例子

```bash
docker run -p 8701:8501 -p 8700:8500 \
-v C:/tmp/tfserving/serving/tensorflow_serving/servables/tensorflow/testdata/saved_model_half_plus_two_cpu:/models/half_plus_two \
-e MODEL_NAME=half_plus_two -t tensorflow/serving
```

4.转换自己模型成save_model

```python
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf
import sys
sys.path.append("./")

export_dir = 'model-best-0619/1'
graph_pb = '0619_best_model.pb'
builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

with tf.gfile.GFile(graph_pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

sigs = {}

with tf.Session(graph=tf.Graph()) as sess:   # name="" is important to ensure we don't get spurious prefixing
    tf.import_graph_def(graph_def, name="")
    g = tf.get_default_graph()

    image_tensor = g.get_tensor_by_name("inputs:0")
    detection_sbbox = g.get_tensor_by_name("detector/yolo-v3/detections:0")
    # detection_mbbox = g.get_tensor_by_name("pred_mbbox/concat_2:0")
    # detection_lbbox = g.get_tensor_by_name("pred_lbbox/concat_2:0")
    print(type(detection_sbbox))
    # out = {'sbbox':detection_sbbox,'mbbox':detection_mbbox,'lbbox':detection_lbbox}
    # out = {''}
    # print(type(out))
    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        tf.saved_model.signature_def_utils.predict_signature_def(
            {"in": image_tensor},
            {"bboxs":detection_sbbox})

            # {'sbbox':detection_sbbox,'mbbox':detection_mbbox,'lbbox':detection_lbbox},)
            # {"scores":detection_scores},
            # {"classes":detection_classes},
            # {"nums":num_detections})

    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         signature_def_map = sigs)
builder.save()

```

5.部署
单模型：
```bash
docker run -p 8701:8501 -p 8700:8500 \
-v /home/ygwl/tensorflow-serving/model-fish:/models/model-fish \
-e MODEL_NAME=model-fish -t tensorflow/serving
```
多模型：

```bash
docker run -p 8901:8501 -p 8900:8500 -v G:/Gits/multimodels/:/models -e MODEL_NAME=models -t tensorflow/serving --model_config_file=/models/model.config
```

6.模型结构查看
http:///localhost:8701/v1/models/model-fish/metadata

7.REST和GRPC API客户端调用

```python
import requests
import numpy as np
import cv2
import sys, os, shutil
import json

sys.path.append("./")
from PIL import Image

# input_img = r"E:\data\starin-diode\diode-0605\images\20200605_5.jpg"
# input_img = "diode.jpg"
input_imgs = r"E:\data\diode-opt\imgs/"
files = os.listdir(input_imgs)
save_path = r"out_imgs"
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)
for file in files:
    # input_img = r"E:\data\diode-opt\imgs\20200611_84.jpg"
    input_img = input_imgs + file
    img = Image.open(input_img)
    print(np.array(img))
    img1 = img.resize((416, 416))
    image_np = np.array(img1)
    print(image_np)
    # print(image_np[np.newaxis,:].shape)
    img_data = image_np[np.newaxis, :].tolist()
    data = {"instances": img_data}

    # http://172.20.112.102:8701/v1/models/model-fish/metadata
    preds1 = requests.post("http://172.20.112.102:9201/v1/models/model-best-0619:predict", json=data)
    predictions1 = json.loads(preds1.content.decode('utf-8'))["predictions"][0]

    preds = requests.post("http://172.20.112.102:9101/v1/models/model-diode:predict", json=data)
    predictions = json.loads(preds.content.decode('utf-8'))["predictions"][0]
    print(predictions)
    # print(np.array(predictions)[:,4].max())
    pred = np.array(predictions)
    # print(pred.shape)
    # exit()
    a = pred[:, 4] > 0.25
    print(pred[a])
    print(len(pred[a]))
    # exit()
    im = cv2.imread(input_img)
    # print(im.shape) # hwc
    h_s = im.shape[0] / 416
    w_s = im.shape[1] / 416

    box = []
    for i in range(len(pred[a])):
        # print(pred[a][i])
        x1 = pred[a][i][0]
        y1 = pred[a][i][1]
        x2 = pred[a][i][2]
        y2 = pred[a][i][3]
        xx1 = (x1 - x2 / 2)
        yy1 = (y1 - y2 / 2)
        xx2 = (x1 + x2 / 2)
        yy2 = (y1 + y2 / 2)
        box.append([xx1, yy1, xx2, yy2, pred[a][i][4], pred[a][i][5:]])

        cv2.rectangle(im, (int(xx1 * w_s), int(yy1 * h_s)), (int(xx2 * w_s), int(yy2 * h_s)), (255, 0, 0))
    # print(box)
    # cv2.imshow('ss1', im)
    cv2.imwrite(save_path + '/' + file, im)
    # cv2.waitKey(0)

```

```java
package org.example;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

import java.awt.image.BufferedImage;
import java.awt.image.Raster;
import java.util.ArrayList;
import java.util.List;

import net.coobird.thumbnailator.Thumbnails;

import org.tensorflow.framework.DataType;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;

import tensorflow.serving.Model;
import tensorflow.serving.Predict;
import tensorflow.serving.PredictionServiceGrpc;

import javax.imageio.ImageIO;


public class Demo {
    public static void main(String[] args) throws Exception {
        String modelName = "model-hot";
        String signatureName = "serving_default";
        String filename = "F:\\tfm\\src\\main\\21.jpg";

        BufferedImage im = Thumbnails.of(filename).forceSize(416, 416).outputFormat("bmp").asBufferedImage();
        Raster raster = im.getRaster();
        List<Float> floatList = new ArrayList<>();
        float[] tmp = new float[raster.getWidth() * raster.getHeight() * raster.getNumBands()];
        float[] pixels = raster.getPixels(0, 0, raster.getWidth(), raster.getHeight(), tmp);
        for (float pixel : pixels) {
            floatList.add(pixel);
        }

        long t = System.currentTimeMillis();
        //创建连接，注意usePlaintext设置为true表示用非SSL连接
        ManagedChannel channel = ManagedChannelBuilder.forAddress("172.20.112.102", 8600).usePlaintext(true).build();
        //这里还是先用block模式
        PredictionServiceGrpc.PredictionServiceBlockingStub stub = PredictionServiceGrpc.newBlockingStub(channel);
        //创建请求
        Predict.PredictRequest.Builder predictRequestBuilder = Predict.PredictRequest.newBuilder();
        //模型名称和模型方法名预设
        Model.ModelSpec.Builder modelSpecBuilder = Model.ModelSpec.newBuilder();
        modelSpecBuilder.setName(modelName);
        modelSpecBuilder.setSignatureName(signatureName);
        predictRequestBuilder.setModelSpec(modelSpecBuilder);
        //设置入参,访问默认是最新版本，如果需要特定版本可以使用tensorProtoBuilder.setVersionNumber方法
        TensorProto.Builder tensorProtoBuilder = TensorProto.newBuilder();
        tensorProtoBuilder.setDtype(DataType.DT_FLOAT);
        TensorShapeProto.Builder tensorShapeBuilder = TensorShapeProto.newBuilder();
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(1));
        //150528 = 224 * 224 * 3
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(416));
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(416));
        tensorShapeBuilder.addDim(TensorShapeProto.Dim.newBuilder().setSize(3));

        tensorProtoBuilder.setTensorShape(tensorShapeBuilder.build());
        tensorProtoBuilder.addAllFloatVal(floatList);
        predictRequestBuilder.putInputs("in", tensorProtoBuilder.build());
        //访问并获取结果
        Predict.PredictResponse predictResponse = stub.predict(predictRequestBuilder.build());
        List<Float> boxes = predictResponse.getOutputsOrThrow("bboxs").getFloatValList();
        System.out.println(boxes);

        List<List<Float>> bbox = getSplitList(8, boxes);

        List<List<Float>> bb = new ArrayList<>();
        for (int i = 0; i < bbox.size(); i++) {
//                System.out.println(bbox.get(i));
//                System.out.println(bbox.get(i).size());
//                System.out.println(bbox.get(i).get(4));
            if (bbox.get(i).get(4) < 0.9) {
                continue;
            }
            bb.add(bbox.get(i));
        }
        System.out.println("=====================================");
        System.out.println(bb);
        System.exit(1);
        System.out.println("cost time: " + (System.currentTimeMillis() - t));
    }

    private static List<List<Float>> getSplitList(int splitNum, List<Float> list) {
        List<List<Float>> splitList = new ArrayList<>();
        int groupFlag = list.size() % splitNum == 0 ? (list.size() / splitNum) : (list.size() / splitNum + 1);
        for (int j = 0; j < groupFlag; j++) {
            if ((j * splitNum + splitNum) <= list.size()) {
                splitList.add(list.subList(j * splitNum, j * splitNum + splitNum));
            } else if ((j * splitNum + splitNum) > list.size()) {
                splitList.add(list.subList(j * splitNum, list.size()));
            } else if (list.size() < splitNum) {
                splitList.add(list.subList(0, list.size()));
            }
        }
        return splitList;
    }

}

```

