import os
import cv2
import glob
import json
import numpy as np

class_names = ["light_group"]
out_dir = "D:/data/Temperature/test/total_detect"
txt_dir = "0403_train/txt"
# os.makedirs(txt_dir, exist_ok=True)

json_pths = glob.glob(out_dir + "/*.json")

for json_pth in json_pths:
    f1 = open(json_pth, "r")
    json_data = json.load(f1)

    img_pth = os.path.join(json_pth.replace("json", "jpg"))
    img = cv2.imread(img_pth)
    h, w = img.shape[:2]

    tag = os.path.basename(json_pth)
    out_file = open(os.path.join(json_pth.replace("json", "txt")), "w")
    print(json_data)
    exit()
    label_infos = json_data["shapes"]
    for label_info in label_infos:
        label = label_info["label"]
        points = label_info["points"]
        parts = np.array(points).reshape(8)
        if len(parts) < 8:
            continue
        coords = [float(p) for p in parts]
        normalized_coords = [
                    coords[i] / w if i % 2 == 0 else coords[i] / h for i in range(8)
                ]
        formatted_coords = ["{:.6g}".format(coord) for coord in normalized_coords]
        print(f"0 {' '.join(formatted_coords)}\n")
        out_file.write(f"0 {' '.join(formatted_coords)}\n")