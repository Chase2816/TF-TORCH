import xml.etree.ElementTree as ET
import pickle
import os
import shutil
import glob
import random


def convert(size, box):
 
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    x = x_center / size[0]
    y = y_center / size[1]
 
    w = (box[1] - box[0]) / size[0]
    h = (box[3] - box[2]) / size[1]
    
    return (x, y, w, h)


def convert_annotation(xml_files_path, save_txt_files_path, classes):
    if not os.path.exists(save_txt_files_path):os.makedirs(save_txt_files_path)
    # xml_files = os.listdir(xml_files_path)
    xml_files = glob.glob(xml_files_path+'/*.xml')
    print(xml_files)
    for xml_file in xml_files:
        xml_name = os.path.basename(xml_file)
        print(xml_name)
        # xml_file = os.path.join(xml_files_path, xml_name)
        out_txt_path = os.path.join(save_txt_files_path, xml_name.split('.')[0] + '.txt')
        out_txt_f = open(out_txt_path, 'w')
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
 
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            # b=(xmin, xmax, ymin, ymax)
            print(w, h, b)
            bb = convert((w, h), b)
            out_txt_f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def moveFile(imagesDir,labelsDir,tarDir,rate=0.1):
        pathDir = os.listdir(imagesDir)    #文件夹路径下的所有图片名以list方式存储
        filenumber=len(pathDir)
        #picknumber=100（想取图片数量）
        # 想以一定比率抽取图片则为
        # rate=0.2    #rate=想取图片数量/文件夹下面所有图片数量
        picknumber=int(filenumber*rate) #想取图片数量（整数）
        sample = random.sample(pathDir, picknumber)  #在图片名list中随机选取
        print (sample)
        
        train_imagedir = os.path.join(tarDir,'images','train') 
        valid_imagedir = os.path.join(tarDir,'images','val')
        train_labeldir = os.path.join(tarDir,'labels','train') 
        valid_labeldir = os.path.join(tarDir,'labels','val')
        for p in [train_labeldir, valid_labeldir,train_imagedir,valid_imagedir]:
            if not os.path.exists(p):
                os.makedirs(p)
                
        for name in sample:
            label_name = name.replace('.jpg', '.txt')
            # shutil.move(fileDir+name, tarDir+name)
            shutil.copyfile(os.path.join(imagesDir,name), os.path.join(train_imagedir,name))
            shutil.copyfile(os.path.join(labelsDir,label_name), os.path.join(train_labeldir,label_name))

        for i in pathDir:
            if i in sample:
                continue
            label_name = i.replace('.jpg', '.txt')
            shutil.copyfile(os.path.join(imagesDir,i), os.path.join(valid_imagedir,i))
            shutil.copyfile(os.path.join(labelsDir,label_name), os.path.join(valid_labeldir,label_name))


if __name__ == "__main__":
    # 把voc的xml标签文件转化为yolo的txt标签文件
    # 1、类别
    classes = ['head','trunk','tail','stomach']
    # 2、voc格式的xml标签文件路径
    xml_files = r'F:\data\shrimp\body\data'
    # 3、转化为yolo格式的txt标签文件存储路径
    save_txt_files = r'F:\data\shrimp\body\labels'
    convert_annotation(xml_files, save_txt_files, classes)
    moveFile(r'F:\data\shrimp\body\images', save_txt_files,tarDir=r'F:\data\shrimp\body\shrimp_data',rate=0.9)
