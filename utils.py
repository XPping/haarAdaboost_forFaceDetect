from __future__ import division, print_function

import numpy as np
import cv2
import os

def ht(x, polarity, theta):
    return np.where(polarity*x < polarity*theta, 1, -1)

def integralImage(images):
    """
    calculate integral image
    :param images: (batch_size, height, width);
    :return: (batch_size, height, width)
    """
    return images.cumsum(axis=-1).cumsum(axis=-2)
def squareIntegralImage(images):
    """
    calculate integral image
    :param images: (batch_size, height, width);
    :return: (batch_size, height, width)
    """
    images *= images
    return images.cumsum(axis=-1).cumsum(axis=-2)

def dataLoader(face_path, noface_path, width=24, height=24):
    faces = []
    nofaces = []

    dir1 = os.listdir(face_path)
    perm = np.random.permutation(len(dir1))
    dir1 = np.array(dir1)[perm]
    for i, d1 in enumerate(dir1):
        if i == 8000: break
        try:
            image = cv2.imread(os.path.join(face_path, d1), 0)
            image = cv2.equalizeHist(image)
            image = cv2.resize(image, (height, width))
            faces.append(image)
        except:
            print(os.path.join(face_path, d1))
            continue
    dir1 = os.listdir(noface_path)
    perm = np.random.permutation(len(dir1))
    dir1 = np.array(dir1)[perm]
    for i, d1 in enumerate(dir1):
        if i == 8000: break
        try:
            image = cv2.imread(os.path.join(noface_path, d1), 0)
            image = cv2.equalizeHist(image)
            # print(image.shape)
            image = cv2.resize(image, (height, width))
            nofaces.append(image)
        except:
            print(os.path.join(noface_path, d1))
            continue
    faces = np.array(faces, dtype=np.float32)
    nofaces = np.array(nofaces, dtype=np.float32)

    mean = (np.mean(np.concatenate((faces, nofaces))))
    std = np.std(np.concatenate((faces, nofaces)))
    print(mean, std)
    # faces = (faces - mean) / std
    # nofaces = (nofaces - mean) / std
    # print(faces, nofaces)
    # raise
    return faces, nofaces