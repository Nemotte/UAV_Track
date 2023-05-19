import torch
from torch.autograd import Variable

import numpy as np
import cv2
import os
import fnmatch
import json
import math


def get_all_jpg(dir):
    res = []
    for folderName, subFolders, fileNames in os.walk(dir):
        for filename in fileNames:
            if fnmatch.fnmatch(filename, "*.jpg"):
                res.append(filename)
    return res


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad=True)
    return input


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def blur_process(img):
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    thresh = np.uint8(thresh)
    img_blur = cv2.GaussianBlur(thresh, (3, 3), 1)
    _, thresh = cv2.threshold(img_blur, 1, 255, cv2.THRESH_BINARY)
    return thresh


def get_contours(img):
    thresh = blur_process(img)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    return x, y, w, h


def main():
    input_dirs = "./original_data/train"
    input_dirs_set = os.listdir(input_dirs)
    label = "/IR_label.json"
    net = torch.load(
        "./weights/UAV_mIoU-0.6339_fmeasure-0.7759.pkl",
        # map_location=torch.device("cpu"),
    )
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for dir in input_dirs_set:
        pics = get_all_jpg(input_dirs + "/" + dir)
        with open(input_dirs + "/" + dir + label, "r", encoding="utf-8") as f:
            content = json.load(f)
        T = 0
        T_Convert = 0
        expr1 = 0
        expr2 = 0
        for i in range(len(content["exist"])):
            img1 = cv2.imread(input_dirs + "/" + dir + "/" + pics[i], 1)
            img_temp = cv2.resize(img1, (256, 256))
            img = np.float32(cv2.resize(img1, (256, 256))) / 255
            input_img = preprocess_image(img).to(device)
            with torch.no_grad():
                output_img = net(input_img)

            output_img = output_img.cpu().detach().numpy().reshape(256, 256)

            x1 = 25565
            y1 = -1
            w1 = 0
            h1 = 0

            T += 1
            iou = 0
            pt = 1
            vt = 0
            if output_img.max() > 0:
                x1, y1, w1, h1 = get_contours(output_img)

            if y1 > 0:
                pt = 0

            if content["exist"][i] == 1:
                vt = 1
                x2 = content["gt_rect"][i][0]
                y2 = content["gt_rect"][i][1]
                w2 = content["gt_rect"][i][2]
                h2 = content["gt_rect"][i][3]

                y2 = int(float(y2) / float(img1.shape[0]) * 256)
                x2 = int(float(x2) / float(img1.shape[1]) * 256)
                w2 = int(float(w2) / float(img1.shape[1]) * 256)
                h2 = int(float(h2) / float(img1.shape[0]) * 256)

                T_Convert += 1
                if pt == 0:
                    temp_x1 = max(x1, x2)
                    temp_x2 = min(x1 + h1, x2 + h2)
                    temp_y1 = max(y1, y2)
                    temp_y2 = min(y1 + w1, y2 + w2)
                    res = (temp_y2 - temp_y1) * (temp_x2 - temp_x1)
                    iou = float(res) / float(w2 * h2 + w1 * h1 - res)
                expr2 = expr2 + pt * vt
            else:
                vt = 0

            expr1 = expr1 + iou * vt + pt * (1 - vt)
            expr1_tmp = expr1 / T
            expr2_tmp = 0.2 * math.pow((expr2 / T_Convert), 0.3)
            acc = expr1_tmp - expr2_tmp
            print(str(i + 1) + "/" + str(len(content["exist"])) + " acc:" + str(acc))


main()
