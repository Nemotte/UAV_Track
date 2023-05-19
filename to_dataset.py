import os
import json
import fnmatch
import cv2
import numpy as np
import random


def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)


def get_all_jpg(dir):
    res = []
    for folderName, subFolders, fileNames in os.walk(dir):
        for filename in fileNames:
            if fnmatch.fnmatch(filename, "*.jpg"):
                res.append(filename)
    return res


del_file("./data/sirst_aug/test/images")
del_file("./data/sirst_aug/test/masks")
del_file("./data/sirst_aug/trainval/images")
del_file("./data/sirst_aug/trainval/masks")

dirs = "./original_data/train"
label = "/IR_label.json"
dirs_set = os.listdir(dirs)

num = 0

train_dir = "./data/sirst_aug/trainval/"
test_dir = "./data/sirst_aug/test/"

for dir in dirs_set:
    pics = get_all_jpg(dirs + "/" + dir)

    with open(dirs + "/" + dir + label, "r", encoding="utf-8") as f:
        content = json.load(f)

    for i in range(len(content["exist"])):
        img = cv2.imread(dirs + "/" + dir + "/" + pics[i], cv2.IMREAD_GRAYSCALE)
        mask = np.zeros_like(img)
        x_center = 0
        y_center = 0
        width = 0
        height = 0

        if content["exist"][i] == 1:
            y = content["gt_rect"][i][0]
            x = content["gt_rect"][i][1]
            width = content["gt_rect"][i][2]
            height = content["gt_rect"][i][3]
            tmp = img[x : x + width, y : y + height]
            tmp_mean = tmp.mean()

            for j in range(width):
                for k in range(height):
                    if (
                        x + j < img.shape[0]
                        and y + k < img.shape[1]
                        and img[x + j][y + k] > tmp_mean + 10
                    ):
                        mask[x + j][y + k] = 255

            if mask.max() == 0:
                for j in range(width):
                    for k in range(height):
                        if x + j < img.shape[0] and y + k < img.shape[1]:
                            mask[x + j][y + k] = 255

        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256))
        ran = random.uniform(0, 1)
        if ran < 0.8:
            cv2.imwrite(train_dir + "images/Misc_" + str(num) + ".png", img)
            cv2.imwrite(train_dir + "masks/Misc_" + str(num) + ".png", mask)
        else:
            cv2.imwrite(test_dir + "images/Misc_" + str(num) + ".png", img)
            cv2.imwrite(test_dir + "masks/Misc_" + str(num) + ".png", mask)
        num += 1
        if i > 60:
            break
