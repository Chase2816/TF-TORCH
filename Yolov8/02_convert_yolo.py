'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-07-14 15:38:45
LastEditors: ciao 18483666678@163.com
LastEditTime: 2023-10-18 11:05:03
FilePath: /tfservingconvert/ultralytics/my_work/convert_yolo.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from ultralytics.yolo.data.converter import convert_coco

# convert_coco(labels_dir='D:/data/Temperature/test/total_detect_coco/', use_segments=True)
convert_coco(labels_dir='D:/data/Temperature/test/total_detect_coco/', use_segments=False)

# from ultralytics.data.converter import convert_dota_to_yolo_obb
# convert_dota_to_yolo_obb('C:\myyolo\ultralytics-main\dataobb')
#关于dataobb文件下的目录下面会详细说明