import xml.etree.ElementTree as ET
import os
import json
import cv2
from io import BytesIO
import base64
from PIL import Image


def makefile(path, content):
    with open(path, 'w') as f:  # 创建一个params.json文件
        f.write(content)  # 将json_str写到文件中

    # if os.path.exists(path):
    #     if os.path.isdir(path):
    #         #**
    #     else:
    #         print('please input the dir name')
    # else:
    #     print('the path is not exists')


def toJson(imagePath, imageHeight, imageWidth, shape_type, label, points, xml_path):
    print(imagePath)

    coco = dict()
    coco['version'] = "5.0.0"
    coco['flags'] = dict()
    coco['shapes'] = [1]
    coco['shapes'][0] = dict()
    coco['shapes'][0]['label'] = label
    coco['shapes'][0]['points'] = points
    coco['shapes'][0]['group_id'] = None
    coco['shapes'][0]['shape_type'] = "rectangle"
    coco['shapes'][0]['flags'] = dict()

    coco['imagePath'] = os.path.basename(imagePath)
    print("ssss",imagePath)
    img = cv2.imread(imagePath)
    pil_img = Image.fromarray(img)
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
    coco['imageData'] = new_image_string

    coco['imageHeight'] = imageHeight
    coco['imageWidth'] = imageWidth

    makefile(imagePath[:-4] + ".json", json.dumps(coco, indent=4))


def parseXmlFiles(xml_path):
    for f in os.listdir(xml_path):
        if not f.endswith('.xml'):
            continue

        size = dict()
        size['width'] = None
        size['height'] = None
        size['depth'] = None

        xml_file = os.path.join(xml_path, f)
        print(xml_file)

        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

        imagePath = ""
        imageHeight = 0
        imageWidth = 0
        shape_type = "rectangle"
        label = "normal"
        points = [[0, 0], [0, 0]]
        for elem in root:

            if elem.tag == 'folder' or elem.tag == 'path' or elem.tag == 'source' or elem.tag == 'segmented':
                continue
            elif elem.tag == 'filename':
                print(elem.text)
                imagePath = r"E:\data\total_Plant_inspection\pv_group_data\DongXin\mulyolov5\labelimg_xml/"+elem.text
            elif elem.tag == 'size':
                for subelem in elem:
                    if subelem.tag == 'width':
                        imageWidth = subelem.text
                    elif subelem.tag == 'height':
                        imageHeight = subelem.text
            elif elem.tag == 'object':
                for subelem in elem:
                    if subelem.tag == 'name':
                        label = subelem.text
                    if subelem.tag == 'bndbox':
                        for item in subelem:
                            if item.tag == 'xmin':
                                points[0][0] = int(item.text)
                            if item.tag == 'ymin':
                                points[0][1] = int(item.text)
                            if item.tag == 'xmax':
                                points[1][0] = int(item.text)
                            if item.tag == 'ymax':
                                points[1][1] = int(item.text)
        toJson(imagePath, imageHeight, imageWidth, shape_type, label, points, xml_path)


if __name__ == '__main__':
    xml_path = r'E:\data\total_Plant_inspection\pv_group_data\DongXin\mulyolov5\labelimg_xml/'  # 这是xml文件所在的地址
    parseXmlFiles(xml_path)