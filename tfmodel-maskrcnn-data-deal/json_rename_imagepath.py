import json
import os, sys

json_path = "/root/AI/tfm/object_detection/my_maskrcnn/light_group/ann/"


def get_json_data(json_path):
    with open(json_path, 'rb')as f:
        params = json.load(f)

        # 加载json文件中的内容给params

        a = filename[:-5]

        params['imagePath'] = a + ".jpg"  # 这两行控制修改的内容 时间有限就写的很草率

        dict = params

        # 将修改后的内容保存在dict中

        f.close()

        # 关闭json读模式

        return dict


# 返回dict字典内容

def write_json_data(dict):
    # 写入json文件

    with open(json_path1, 'w')as r:
        # 定义为写模式，名称定义为r

        json.dump(dict, r, indent=2)  # indent控制间隔

        # 将dict写入名称为r的文件中

        r.close()


# 关闭json写模式

# 获取文件夹中的文件名称列表

filenames = os.listdir(json_path)

# 遍历文件名

for filename in filenames:
    filepath = json_path + '/' + filename

    # print(filepath)

    dict = {}

    the_revised_dict = get_json_data(filepath)

    json_path1 = "/root/AI/tfm/object_detection/my_maskrcnn/light_group/ann/" + filename  # 修改json文件后保存的路径

    write_json_data(the_revised_dict)