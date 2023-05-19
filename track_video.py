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
    input_dirs = "./original_data/test"
    input_dirs_set = os.listdir(input_dirs)
    label = "/IR_label.json"
    label1 = "/IR_label1.json"
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
        im = cv2.imread(input_dirs + "/" + dir + "/" + pics[0], 1)
        out = cv2.VideoWriter(
            "test_result/video/" + dir + ".avi",
            cv2.VideoWriter_fourcc(*"DIVX"),
            30,
            (im.shape[1], im.shape[0]),
        )
        for i in range(len(pics)):
            L = []
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
            has = False

            if output_img.max() > 0:
                x1, y1, w1, h1 = get_contours(output_img)
                y1 = int(float(y1) / float(256) * img1.shape[0])
                x1 = int(float(x1) / float(256) * img1.shape[1])
                w1 = int(float(w1) / float(256) * img1.shape[1])
                h1 = int(float(h1) / float(256) * img1.shape[0])
                has = True

                cv2.rectangle(img1, (x1, y1), (x1 + w1, y1 + h1), 255)

            out.write(img1)
            print(i)

        out.release()

        #     if has and i > 0:
        #         L = [x1, y1, w1, h1]
        #         content["res"].append(L)
        #     else:
        #         content["res"].append([])
        #     print(i)

        # data2 = json.dumps(content)
        # with open(input_dirs + "/" + dir + label1, "w", encoding="utf-8") as f:
        #     f.write(data2)


main()
