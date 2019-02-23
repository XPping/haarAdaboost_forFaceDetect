from __future__ import division, print_function

import numpy as np
import cv2
from matplotlib import pyplot as plt
from haar_feature import type1, type2, type3, type4, type5, \
    generateHaarFeatures, calculateHaarFeature
from utils import integralImage


def test():
    width = 24
    height = 24
    t1 = type1(width, height, 1, 1)
    t2 = type2(width, height, 1, 1)
    t3 = type3(width, height, 1, 1)
    t4 = type4(width, height, 1, 1)
    t5 = type5(width, height, 1, 1)
    all = generateHaarFeatures(width, height, 1, 1)
    print(len(t1), len(t2), len(t3), len(t4), len(t5))
    print(len(all))
    images = []
    img = cv2.imread(r'images/type1.jpg', 0)
    img = cv2.resize(img, (width, height))
    images.append(img)
    img = cv2.imread(r'images/type2.jpg', 0)
    img = cv2.resize(img, (width, height))
    images.append(img)
    inte_images = integralImage(np.array(images))
    # print(inte_images[0])
    # print(inte_images[1])
    print(all[-1])
    feats = []
    for i, feat in enumerate(all):
        haar = calculateHaarFeature(inte_images, feat)
        feats.append(haar)
        # print(haar)
        # if i==3:break
    feats = np.array(feats, dtype=np.int32)
    print(np.min(feats), np.max(feats))
    plt.hist(feats, bins=[i for i in range(np.min(feats), np.max(feats)+1, 100)])
    plt.title("histogram")
    plt.show()


if __name__ == "__main__":
    test()