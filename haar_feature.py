from __future__ import division, print_function

import numpy as np

# the sum of pixels of white sub the sum of pixels of black

def type1(width, height, stride=1, increment=1):
    """
    func:  2D+E+A-F-B-2C
    :param width:
    :param height:
    :param stride:
    :param increment:
    :return:
    """
    features = []
    for w in range(1, width, 1*increment):
        for h in range(2, height, 2*increment):
            for x in range(0, (width - w), stride):
                for y in range(0, (height - h), stride):
                    A = (x, y)
                    B = (x+w, y)
                    C = (x, y+(h//2))
                    D = (x+w, y+(h//2))
                    E = (x, y+h)
                    F = (x+w, y+h)
                    # if x+w >= width or y+h >= height: continue
                    # if x+1 >= width or y+1 >= height or x+w-1<0 or y+h-1<0: continue
                    tl = (x, y)
                    br = (x+w, y+h)
                    sub = [F, B, C, C]
                    add = [E, A, D, D]
                    features.append((tl, br, w, h, add, sub))
    return features

def type2(width, height, stride=1, increment=1):
    """
    func: 2E+A+C-F-D-2B
    :param width:
    :param height:
    :param stride:
    :param increment:
    :return:
    """
    features = []
    for w in range(2, width, 2 * increment):
        for h in range(1, height, 1 * increment):
            for x in range(0, (width - w), stride):
                for y in range(0, (height - h), stride):
                    A = (x, y)
                    B = (x + (w//2), y)
                    C = (x + w, y)
                    D = (x, y + h)
                    E = (x + (w//2), y + h)
                    F = (x + w, y + h)
                    # if x+w >= width or y+h >= height: continue
                    # if x+1 >= width or y+1 >= height or x+w-1<0 or y+h-1<0: continue
                    tl = (x, y)
                    br = (x+w, y+h)
                    sub = [F, D, B, B]
                    add = [A, C, E, E]
                    features.append((tl, br, w, h, add, sub))
    return features

def type3(width, height, stride=1, increment=1):
    """
    func: 2E+2D+G+A-2F-2C-H-B
    :param width:
    :param height:
    :param stride:
    :param increment:
    :return:
    """
    features = []
    for w in range(1, width, 1 * increment):
        for h in range(3, height, 3 * increment):
            for x in range(0, (width - w), stride):
                for y in range(0, (height - h), stride):
                    A = (x, y)
                    B = (x + w, y)
                    C = (x, y + (h//3))
                    D = (x + w, y + (h//3))
                    E = (x, y + 2 * (h//3))
                    F = (x + w, y + 2 * (h//3))
                    H = (x, y + h)
                    G = (x + w, y + h)
                    # if x+w >= width or y+h >= height: continue
                    # if x+1 >= width or y+1 >= height or x+w-1<0 or y+h-1<0: continue
                    tl = (x, y)
                    br = (x+w, y+h)
                    sub = [F, F, C, C, H, B]
                    add = [G, A, E, E, D, D]
                    features.append((tl, br, w, h, add, sub))
    return features

def type4(width, height, stride=1, increment=1):
    """
    func: 2F+2C+A+H-2G-2B-E-D
    :param width:
    :param height:
    :param stride:
    :param increment:
    :return:
    """
    features = []
    for w in range(3, width, 3 * increment):
        for h in range(1, height, 1 * increment):
            for x in range(0, (width - w), stride):
                for y in range(0, (height - h), stride):
                    A = (x, y)
                    B = (x + (w//3), y)
                    C = (x + 2 * (w//3), y)
                    D = (x + w, y)
                    E = (x, y + h)
                    F = (x + (w//3), y + h)
                    G = (x + 2 * (w//3), y + h)
                    H = (x + w, y + h)
                    # if x+w >= width or y+h >= height: continue
                    # if x+1 >= width or y+1 >= height or x+w-1<0 or y+h-1<0: continue
                    tl = (x, y)
                    br = (x+w, y+h)
                    sub = [E, D, G, G, B, B]
                    add = [A, H, F, F, C, C]
                    features.append((tl, br, w, h, add, sub))
    return features

def type5(width, height, stride=1, increment=1):
    """
    func: G+C+A+L+4E-2H-2D-2B-2F
    :param width:
    :param height:
    :param stride:
    :param increment:
    :return:
    """
    features = []
    for w in range(2, width, 2 * increment):
        for h in range(2, height, 2 * increment):
            for x in range(0, (width - w), stride):
                for y in range(0, (height - h), stride):
                    A = (x, y)
                    B = (x + (w//2), y)
                    C = (x + w, y)
                    D = (x, y + (h//2))
                    E = (x + (w//2), y + (h//2))
                    F = (x + w, y + (h//2))
                    G = (x, y + h)
                    H = (x + (w//2), y + h)
                    L = (x + w, y + h)
                    # if x+w >= width or y+h >= height: continue
                    # if x+1 >= width or y+1 >= height or x+w-1<0 or y+h-1<0: continue
                    tl = (x, y)
                    br = (x+w, y+h)
                    sub = [H, H, D, D, B, B, F, F]
                    add = [G, C, A, L, E, E, E, E]
                    features.append((tl, br, w, h, add, sub))
    return features

def generateHaarFeatures(width, height, stride=1, increment=1):
    features = []
    features.extend(type1(width, height, stride, increment))
    features.extend(type2(width, height, stride, increment))
    features.extend(type3(width, height, stride, increment))
    features.extend(type4(width, height, stride, increment))
    features.extend(type5(width, height, stride, increment))
    return features

def calculateHaarFeature(integral_image, square_integral_image, feature):
    """
    calculate a type haar feature
    :param integral_image: integral image, (batch_size, width, height)
    :param square_integral_image: integral image of square image, (batch_size, width, height)
    :param feature: tupe, (top-left, bottom-right, w, h, add, sub), one type haar feature
    :return: (batch_size)
    """
    tl, br, w, h, add, sub = feature[0], feature[1], feature[2], feature[3], feature[4], feature[5]
    output = np.zeros(len(integral_image), dtype=np.float32)
    for add_ in add:
        output += integral_image[:, add_[0], add_[1]]
    for sub_ in sub:
        output -= integral_image[:, sub_[0], sub_[1]]

    sum = np.zeros(len(integral_image), dtype=np.float32)
    sum += integral_image[:, br[0], br[1]]
    sum -= integral_image[:, tl[0], tl[1]]
    sqsum = np.zeros(len(integral_image), dtype=np.float32)
    sqsum += square_integral_image[:, br[0], br[1]]
    sqsum -= square_integral_image[:, tl[0], tl[1]]
    sum /= (w * h)
    sqsum /= (w * h)
    # print(np.nan_to_num(np.sqrt(sqsum - sum * sum)))
    # Normalization
    # output = output / (np.sqrt(np.abs(sqsum - sum * sum)) + 1.0)
    output = output / (w * h)
    # to_add = data[:, [x[0] for x in add], [y[1] for y in add]].sum(axis=-1)
    # to_sub = data[:, [x[0] for x in sub], [y[1] for y in sub]].sum(axis=-1)
    return output  # to_add - to_sub
