from __future__ import division, print_function

import numpy as np
import cv2
import os
import json
import pickle
import config

from utils import integralImage, ht
from haar_feature import calculateHaarFeature

def bbox_iou(box1, box2):
    """
    box1: (x1, y1, x2, y2)
    box2: (x1, y1, x2, y2)
    Returns the IoU of two bounding boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    # Intersection area
    width = max(inter_rect_x2 - inter_rect_x1 + 1.0, 0.0)
    height = max(inter_rect_y2 - inter_rect_y1 + 1.0, 0.0)
    inter_area = width * height
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1.0) * (b1_y2 - b1_y1 + 1.0)
    b2_area = (b2_x2 - b2_x1 + 1.0) * (b2_y2 - b2_y1 + 1.0)
    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    return iou

def non_max_suppression(boxes):
    """
    if bbox_iou(box1, box2)>0.6, combine two boxes
    """
    if len(boxes) == 0:
        return None
    group = [i for i in range(len(boxes))]  # if bbox_iou(boxes[i], boxes[j]) > 0.6, group[i]=group[j]
    size = [1 for i in range(len(boxes))]   # The number of boxes of group[i]

    def findGroup(p):
        if p != group[p]:
            group[p] = findGroup(group[p])
        return group[p]
    def unionOne(p, q):
        pid = findGroup(p)
        qid = findGroup(q)
        if pid == qid:
            return
        group[pid] = qid
        size[qid] += size[pid]
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            if bbox_iou(boxes[i], boxes[j]) > config.IOU_threshold:
                unionOne(i, j)          # The union find algorithm
    done = {}
    ret_boxes = []

    for i in range(len(boxes)):
        p_id = findGroup(i)
        if size[p_id] < 10: continue  # The number of boxes of group[id] < 50, remove it
        if p_id in done: continue
        done[p_id] = 0
        ret_boxes.append(boxes[i])
        for j in range(i+1, len(boxes)):
            id = findGroup(j)
            if id == p_id:
                tmp_box = boxes[j]
                tmp_box[0] = min(tmp_box[0], ret_boxes[-1][0])     # x1
                tmp_box[1] = min(tmp_box[1], ret_boxes[-1][1])     # y1
                tmp_box[2] = max(tmp_box[2], ret_boxes[-1][2])     # x2
                tmp_box[3] = max(tmp_box[3], ret_boxes[-1][3])     # y2
                ret_boxes[-1] = tmp_box
    return ret_boxes

def calculateHardClassifierError(faces, square_faces, classifiers, alphas):
    threshold = 0.0
    faces_score = np.zeros(len(faces))
    for id, classifier in enumerate(classifiers):
        feature, polarity, theta, _ = classifier
        faces_score += alphas[id] * ht(calculateHaarFeature(faces, square_faces, feature), polarity, theta)
        threshold += alphas[id]
    threshold = 0.5 * threshold

    # faces_score -= threshold

    nofaces = faces_score < 0
    return nofaces

def predict(imagename):
    src_image = cv2.imread(imagename)
    image = cv2.cvtColor(src_image, cv2.COLOR_RGB2GRAY)
    mean = 128.643
    std = 73.2
    height, width = image.shape[0:2]
    image = np.array(image, dtype=np.float32)
    # image = (image - mean) / std

    scale_w = 40
    scale_h = 40
    areas = []
    boxes = []
    while scale_w < width/2 and scale_h < height/2:
        stride = 5
        for x in range(0, width-int(scale_w), stride):
            for y in range(0, height-int(scale_h), stride):
                x2 = x + int(scale_w)
                y2 = y + int(scale_h)
                tmp_img = image[y:y2, x:x2]
                tmp_img = cv2.resize(tmp_img, (config.HEIGHT, config.WIDTH))
                areas.append(tmp_img)
                boxes.append((x, y, x2, y2))
        scale_h *= config.ScaleFactor
        scale_w *= config.ScaleFactor

    areas = np.array(areas)
    boxes = np.array(boxes)
    # print("boxes length: ", len(boxes))
    with open('haar.pickle', 'rb') as fp:
        cascades = pickle.load(fp)

    # with open('cascade_2th_dataset.json', 'r') as fp:
    #     cascades = json.load(fp)

    for id, cascade in enumerate(cascades):
        classifiers, alphas, threshold = cascade
        faces_false = calculateHardClassifierError(integralImage(areas), integralImage(areas * areas),
                                                   classifiers, alphas)
        areas = areas[~faces_false]
        boxes = boxes[~faces_false]

    print("length of boxes having face: ", len(boxes))
    predict_boxes = non_max_suppression(boxes)
    print("number of  face: ", len(predict_boxes))
    for x1, y1, x2, y2 in predict_boxes:
        cv2.rectangle(src_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imwrite("predict.jpg", src_image)
    # x1, y1, x2, y2 = boxes[0]
    # cv2.imwrite("ha.jpg", src_image[y1:y2, x1:x2])

predict('./images/1.jpg')
# predict('./images/2.jpg')