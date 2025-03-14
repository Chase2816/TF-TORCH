import cv2
import xml.etree.ElementTree as ET
import os
import sys
import lxml
import shutil


path = r"H:\ygwl\hbtm\tianmen_hot1_1600"
image_path = path + "/img/"
label_path = path + "/xml/"
min_size = 800


def search_jpg_xml(image_dir, label_dir):
    image_ext = '.jpg'
    img = [fn for fn in os.listdir(image_dir) if fn.endswith(image_ext)]
    label_ext = '.xml'
    label = [fn for fn in os.listdir(label_dir) if fn.endswith(label_ext)]
    return img, label


def copyfile():
    if "Annotations_temp" in os.listdir(path):
        shutil.rmtree(path + "/Annotations_temp")
    if "JPEGImages_temp" in os.listdir(path):
        shutil.rmtree(path + "/JPEGImages_temp")
    save_annotation_path = path + "/Annotations_temp/"
    save_jpg_path = path + "/JPEGImages_temp/"
    shutil.copytree(path + "/xml", save_annotation_path)
    shutil.copytree(path + "/img", save_jpg_path)
    return save_jpg_path, save_annotation_path


def write_xml_jpg(jpg_path, annotation_path):
    img, label = search_jpg_xml(jpg_path, annotation_path)
    sorted(img)
    sorted(label)
    print(img)
    print(label)
    if "Annotations_1" not in os.listdir(path):
        os.mkdir(path + "/Annotations_1")
    if "JPEGImages_1" not in os.listdir(path):
        os.mkdir(path + "/JPEGImages_1")
    new_image_path = path + "/JPEGImages_1/"
    new_annotation_path = path + "/Annotations_1/"
    for index, file in enumerate(label):
        cur_img = cv2.imread(jpg_path + img[index])
        width = cur_img.shape[1]
        height = cur_img.shape[0]
        if width < height:
            new_width = min_size
            new_height = int(min_size * height / width)
            w_ratio = new_width / width
            h_ratio = new_height / height
        elif width > height:
            new_width = int(min_size * width / height)
            new_height = min_size
            w_ratio = new_width / width
            h_ratio = new_height / height
        elif width == height:
            new_width = min_size
            new_height = min_size
            w_ratio = new_width / width
            h_ratio = new_height / height
        cur_img = cv2.resize(cur_img, (new_width, new_height))
        cv2.imwrite(new_image_path + img[index], cur_img)
        cur_xml = ET.parse(annotation_path + file)
        root = cur_xml.getroot()
        for node in root:
            if node.tag == 'size':
                node[0].text = str(new_width)
                node[1].text = str(new_height)
            elif node.tag == 'object':
                xmin = int(node[4][0].text)  # bbox position
                ymin = int(node[4][1].text)
                xmax = int(node[4][2].text)
                ymax = int(node[4][3].text)
                node[4][0].text = str(int(xmin * w_ratio))
                node[4][1].text = str(int(ymin * h_ratio))
                node[4][2].text = str(int(xmax * w_ratio))
                node[4][3].text = str(int(ymax * h_ratio))
        cur_xml.write(new_annotation_path + file)
    shutil.rmtree(path + "/JPEGImages_temp")
    shutil.rmtree(path + "/Annotations_temp")


if __name__ == "__main__":
    jpg_path, annotation_path = copyfile()
    write_xml_jpg(jpg_path, annotation_path)