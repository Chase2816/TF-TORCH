import os
import glob
import shutil

mask_folder = r"E:\data\total_Plant_inspection\nul_lg_800\ann"
save_img = r"E:\data\image_segmentation_test\train\Images"
save_mask = r"E:\data\image_segmentation_test\train\SegmentationClass800"

if not os.path.exists(save_mask):os.mkdir(save_mask)
mask_folderes = os.listdir(mask_folder)

for i in mask_folderes:
    print(i)
    flag = os.path.isdir(os.path.join(mask_folder,i))
    if flag:
        print("-----------------------------")
        for j in os.listdir(os.path.join(mask_folder,i)):
            print(j)
            print(i.split("_json")[0])
            # exit()
            if j =="label.png":
                src_label = os.path.join(mask_folder,i,j)
                dst_label = os.path.join(save_mask,i.split("_json")[0]+".png")
                print(src_label)
                print(dst_label)
                shutil.copyfile(src_label,dst_label)
                # exit()