import os
import glob
import shutil


files = r"F:\data\mountain0727\88083\split-hot_group"
save_dir = r"F:\data\mountain0727\88083\split-hot_group\data"
os.mkdir(save_dir)

for i,j,k in os.walk(files):
    print(i)
    print(j)
    print(k)
    for y in j:
        for x in os.listdir(os.path.join(i,y)):
            shutil.move(os.path.join(i,y,x),os.path.join(save_dir,x))