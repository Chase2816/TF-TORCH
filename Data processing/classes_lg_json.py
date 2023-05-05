import json
import os
import glob
import shutil

def angle(json_in):
    dst_json = json_in

    j = open(json_in).read()  # json文件读入成字符串格式

    jj = json.loads(j)  # 载入字符串，json格式转python格式
    print(len(jj["shapes"]))  # 获取标签的个数，shapes包含所有的标签
    print(jj["shapes"][0])  # 输出第一个标签信息
    # if jj["shapes"][0] == "hot_group":
    #     continue

    for i in range(len(jj['shapes'])):
        jj["shapes"][i]["label"] = 'lg_r3c28'  # 把所有label的值都改成‘10’
        print(jj["shapes"][i]["label"])

    # 把修改后的python格式的json文件，另存为新的json文件
    with open(dst_json, 'w') as f:
        json.dump(jj, f, indent=2)  # indent=4缩进保存json文件


# angle(r"F:\data\mountain0727\88082\light_group\Aug_lg_sd\classes_lg_total\w_lg_new_json\yc5330xc2128_127.json")
# exit()

json_files = r"F:\data\mountain0727\88082\light_group\Aug_lg_sd\classes_lg_total\lg_new_json"
save_json = r"F:\data\mountain0727\88082\light_group\Aug_lg_sd\classes_lg_total\lg_new_json"
if not os.path.exists(save_json):
    os.mkdir(save_json)

for json_in in glob.glob(json_files+"/*.json"):
    print("开始处理：",json_in)
    dst_json = os.path.join(save_json,os.path.basename(json_in))

    j = open(json_in).read()  # json文件读入成字符串格式

    jj = json.loads(j)  # 载入字符串，json格式转python格式
    print(len(jj["shapes"]))  # 获取标签的个数，shapes包含所有的标签
    print(jj["shapes"][0])  # 输出第一个标签信息
    # if jj["shapes"][0] == "hot_group":
    #     continue

    for i in range(len(jj['shapes'])):
        jj["shapes"][i]["label"] = 'lg_r3c58'  # 把所有label的值都改成‘10’
        print(jj["shapes"][i]["label"])

    # 把修改后的python格式的json文件，另存为新的json文件
    with open(dst_json, 'w') as f:
        json.dump(jj, f, indent=2)  # indent=4缩进保存json文件




