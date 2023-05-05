import os

path = r'D:\CC\Hot_5classes\json'

json_file = os.listdir(path)

os.system("activate labelme")

for file in json_file:
    os.system(r"E:\caixin\Anaconda3\Scripts\labelme_json_to_dataset.exe %s"%(path + '\\' + file))
