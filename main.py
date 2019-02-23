from __future__ import division, print_function

from train import train
import config


if __name__ == "__main__":

    train("/home/xpp/data/face-detect/faces", "/home/xpp/data/face-detect/no_faces_val2014",
    # train("/home/xpp/code/github-code/face-detector/face-detection-master/train/face", "/home/xpp/code/github-code/face-detector/face-detection-master/train/non_face",
    #train("datasets/dataV2/face", "datasets/dataV2/non_face",
    #train("datasets/faces", "datasets/no_faces_val2014",
          width=config.WIDTH, height=config.HEIGHT, stride=config.STRIDE, increment=config.INCREMENT)
    # predict('./images/31.jpg')