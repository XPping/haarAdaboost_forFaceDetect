from __future__ import division, print_function


import numpy as np
from multiprocessing import cpu_count, Pool
import json
import pickle

from haar_feature import generateHaarFeatures, calculateHaarFeature
from utils import integralImage, dataLoader, squareIntegralImage, ht
import config

def _trainSoftClassifier(faces, nofaces, square_faces, square_nofaces, haar_features,
                         faces_weight, nofaces_weight):
    #print(faces.shape, nofaces.shape, square_faces.shape, square_nofaces.shape, len(haar_features),
    # faces_weight.shape, nofaces_weight.shape)
    all_data = np.concatenate((faces, nofaces))
    all_square_data = np.concatenate((square_faces, square_nofaces))
    all_weight = np.concatenate((faces_weight, nofaces_weight * -1)) #  < 0 represent nofaces

    T_add = faces_weight.sum()
    T_sub = nofaces_weight.sum()
    # The best soft classifier is (feature, polarity, threshold, error)
    best_soft_classifier = (0, 0, 0, np.inf)
    for id, feature in enumerate(haar_features):
        haar_values = calculateHaarFeature(all_data, all_square_data, feature)
        # Descent the haar value
        sort_value_perm = haar_values.argsort()
        haar_values, all_weight = haar_values[sort_value_perm], all_weight[sort_value_perm]

        S_add = 0.0 if all_weight[0] < 0 else all_weight[0]
        S_sub = 0.0 if all_weight[0] >= 0 else all_weight[0] * -1
        # The error of current haar-value
        error_min = min(S_add + (T_sub - S_sub), S_sub + (T_add - S_add))
        polarity_min = 1 if error_min == S_add + T_sub - S_sub else -1
        threshold = haar_values[0]      # The threshold of soft classifier is some kind of haar-value
        for i in range(1, len(haar_values)):
            half_haar = (haar_values[i-1] + haar_values[i]) / 2.0
            left = S_add + T_sub - S_sub
            right = S_sub + T_add - S_add
            error_tmp = min(left, right)
            if error_tmp < error_min:
                error_min = error_tmp
                polarity_min = +1 if left < right else -1
                threshold = half_haar

            if all_weight[i] < 0:
                S_sub -= all_weight[i]
            else:
                S_add += all_weight[i]
        if error_min < best_soft_classifier[-1]:
            best_soft_classifier = tuple((feature, polarity_min, threshold, error_min))
        # print(id, error_min)
        # error.append(tuple((add, sub, polarity_min, threshold, abs(error_min))))
    # error.sort(key=lambda x: x[-1])
    return best_soft_classifier

def _trainWeakClassifier(faces, nofaces, square_faces, square_nofaces, haar_features,
                         faces_weight, nofaces_weight):
    #print(faces.shape, nofaces.shape, square_faces.shape, square_nofaces.shape, len(haar_features),
    # faces_weight.shape, nofaces_weight.shape)
    face_weight_sum = faces_weight.sum()
    noface_weight_sum = nofaces_weight.sum()

    min_feature = 0
    min_error = np.inf
    polarity = 0
    min_threshold = 0

    for id, feature in enumerate(haar_features):
        face_haar_values = calculateHaarFeature(faces, square_faces, feature)
        noface_haar_value = calculateHaarFeature(nofaces, square_nofaces, feature)

        face_weight_val_sum = (faces_weight * face_haar_values).sum()
        noface_weight_val_sum = (nofaces_weight * noface_haar_value).sum()
        # print(face_weight_val_sum.shape, face_weight_sum.shape, noface_weight_val_sum.shape, noface_weight_sum.shape)
        threshold = (face_weight_val_sum / face_weight_sum + noface_weight_val_sum / noface_weight_sum) / 2
        for direction in [1, -1]:
            face_false = np.where(direction * face_haar_values < direction * threshold, True, False)
            noface_false = np.where(direction * noface_haar_value >= direction * threshold, True, False)
            temp_error = faces_weight[face_false==False].sum() + nofaces_weight[noface_false==False].sum()
            if temp_error < min_error:
                min_feature = feature
                min_error = temp_error
                polarity = direction
                min_threshold = threshold
    return (min_feature, polarity, min_threshold, min_error)

def trainBestSoftClassifier(faces, nofaces, square_faces, square_nofaces, haar_features, pool_thread,
                            faces_weight, nofaces_weight):
    # Multiprocess
    process_num = cpu_count() * config.PerCPUProcessNum
    args = []
    chunk = len(haar_features) // process_num
    for cpu in range(process_num):
        if cpu+1 == process_num:
            args.append(
                (faces, nofaces, square_faces, square_nofaces,
                 haar_features[cpu * chunk::], faces_weight, nofaces_weight)
            )
        else:
            args.append(
                (faces, nofaces, square_faces, square_nofaces,
                 haar_features[cpu * chunk: (cpu + 1) * chunk], faces_weight, nofaces_weight)
            )
    result = [x for x in pool_thread.starmap_async(_trainWeakClassifier, args).get()]
    # The best soft classifier is (feature, polarity, threshold, error)
    best_soft_classifier = (0, 0, 0, np.inf)
    for ret in result:
        # print(ret)
        if ret[-1] < best_soft_classifier[-1]:
            best_soft_classifier = ret
    return best_soft_classifier

def calculateHardClassifierError(faces, nofaces, square_faces, square_nofaces, classifiers, alphas):
    faces_score = np.zeros(len(faces), dtype=np.float32)
    nofaces_score = np.zeros(len(nofaces), dtype=np.float32)

    threshold = 0.0
    for id, classifier in enumerate(classifiers):
        feature, polarity, theta, _ = classifier
        faces_score += alphas[id] * ht(calculateHaarFeature(faces, square_faces, feature), polarity, theta)
        nofaces_score += alphas[id] * ht(calculateHaarFeature(nofaces, square_nofaces, feature), polarity, theta)
        threshold += alphas[id]

    threshold = 0.5 * threshold

    # faces_score -= threshold
    # nofaces_score -= threshold

    faces_true = faces_score >= 0
    nofaces_true = nofaces_score < 0

    true_positive_rate = faces_true.sum() / (len(faces) + 0.0)
    false_positive_rate = 1 - nofaces_true.sum() / (len(nofaces) + 0.0)
    return true_positive_rate, false_positive_rate, threshold, faces_true, nofaces_true

def trainHardClassifier(faces, nofaces, square_faces, square_nofaces, haar_features, pool_thread, name):
    faces_weight = np.full(len(faces), 1.0, dtype=np.float32)
    nofaces_weight = np.full(len(nofaces), 1.0, dtype=np.float32)

    classifiers = []
    alphas = []
    eps = 1E-10  # in case fo zero
    for i in range(config.SoftPerHardClassifeirNum):
        sum = faces_weight.sum() + nofaces_weight.sum()
        faces_weight /= sum
        nofaces_weight /= sum

        feature, polarity, theta, error_m = trainBestSoftClassifier(
            faces, nofaces, square_faces, square_nofaces, haar_features, pool_thread, faces_weight, nofaces_weight
        )
        alpha_m = (1/2) * np.log((1-error_m) / error_m)
        faces_Gm = ht(calculateHaarFeature(faces, square_faces, feature), polarity, theta)
        nofaces_Gm = ht(calculateHaarFeature(nofaces, square_nofaces, feature), polarity, theta)
        faces_weight = faces_weight * np.exp(-alpha_m * 1 * faces_Gm)
        nofaces_weight = nofaces_weight * np.exp(-alpha_m * -1 * nofaces_Gm)

        """
        # error_m = max(error_m, 0.0001)
        beta_m = error_m / (1 - error_m)
        alpha_m = np.log(1 / beta_m)

        faces_Gm = ht(calculateHaarFeature(faces, square_faces, feature), polarity, theta)
        nofaces_Gm = ht(calculateHaarFeature(nofaces, square_nofaces, feature), polarity, theta)

        faces_weight = faces_weight * np.where(faces_Gm>=1, beta_m, 1.) #np.exp(-1 * alpha_m * 1 * faces_Gm)
        nofaces_weight = nofaces_weight * np.where(nofaces_Gm>=1, 1., beta_m)# np.exp(-1 * alpha_m * -1 * nofaces_Gm)
        """
        # Zm = faces_weight.sum() + nofaces_weight.sum()
        # faces_weight /= Zm
        # nofaces_weight /= Zm

        alphas.append(alpha_m)
        classifiers.append((feature, polarity, theta, error_m))
        true_positive_rate, false_positive_rate, threshold, faces_true, nofaces_true = \
            calculateHardClassifierError(faces, nofaces,
                                         square_faces, square_nofaces,
                                         classifiers, alphas)
        acc = (faces_true.sum() + nofaces_true.sum()) / (len(faces) + len(nofaces))
        print(name, "the {}-th features".format(len(classifiers)),
              feature[4], feature[5], error_m, alpha_m, true_positive_rate, false_positive_rate, acc)
        if true_positive_rate >= config.TruePositiveRatePerHard and \
                false_positive_rate <= config.FalsePositiveRatePerHard:    # must less than the HardClassifierErrorRate
            break
    return classifiers, alphas, threshold

def calculateCascadeError(faces, nofaces, square_faces, square_nofaces, cascades):
    true_positive_rate = np.ones(len(faces)) < 0
    for cascade in cascades:
        classifiers, alphas, threshold = cascade

        _, _, threshold, faces_true, nofaces_true = calculateHardClassifierError(faces, nofaces,
                                                                               square_faces, square_nofaces,
                                                                               classifiers, alphas)

        true_positive_rate = true_positive_rate ^ faces_true
        # faces = faces[faces_true]         # keep the right samples of faces
        # square_faces = square_faces[faces_true]
        nofaces = nofaces[~nofaces_true]    # keep the wrong samples of nofaces
        square_nofaces = square_nofaces[~nofaces_true]
    return true_positive_rate.sum()/(len(faces)+0.0), faces, nofaces, square_faces, square_nofaces

def cascadeClassifier(faces, nofaces, square_faces, square_nofaces, haar_features, pool_thread):

    cascades = []
    nofaces_num = len(nofaces) + 0.0
    for i in range(config.HardPerCascadesNum):
        print("Cascade length {}, faces number {}, nofaces_number {}. ".format(len(cascades)+1, len(faces), len(nofaces)))
        classifiers, alphas, threshold = trainHardClassifier(
            faces, nofaces, square_faces, square_nofaces, haar_features, pool_thread,
            name="The {} th cascade".format(len(cascades)+1)
        )
        cascades.append((classifiers, alphas, threshold))

        true_positive_rate, faces, nofaces, square_faces, square_nofaces = calculateCascadeError(faces, nofaces, square_faces, square_nofaces, cascades)

        false_positive_rate = len(nofaces) / nofaces_num
        print("Cascade length {}, true_positive_rate {}, false_positive_rate {}.".format(
            len(cascades), true_positive_rate, false_positive_rate))
        if true_positive_rate >= config.TargetTruePositiveRate \
                and false_positive_rate <= config.TargetFalsePositiveRate:  # the cascade classifier's error rate < config.FinalErrorRate
            break
    return cascades


def train(face_path, noface_path, width=20, height=20, stride=1, increment=1):
    _faces, _nofaces = dataLoader(face_path, noface_path, width=width, height=height)
    print("faces:", _faces.shape, "nofaces:", _nofaces.shape)
    haar_features = generateHaarFeatures(width=width, height=height,
                                         stride=stride, increment=increment)
    print("haar features length: ", len(haar_features))
    faces = integralImage(_faces)
    nofaces = integralImage(_nofaces)
    square_faces = integralImage(_faces * _faces)
    square_nofaces = integralImage(_nofaces * _nofaces)

    process_num = cpu_count() * config.PerCPUProcessNum
    pool_thread = Pool(processes=process_num)

    print("Start training cascade classifier")
    cascade = cascadeClassifier(faces, nofaces, square_faces, square_nofaces, haar_features, pool_thread)

    with open('haar.pickle', 'wb') as fw:
        pickle.dump(cascade, fw)

    try:
        with open('cascade_2th_dataset.json', 'w') as fw:
            json.dump(cascade, fw)
    except:
        for i, casc in enumerate(cascade):
            classifiers, alphas, threshold = casc
            for j, clas in enumerate(classifiers):
                feature, polarity, theta, error_m = clas
                print(i, j, feature, polarity, theta, error_m)
    print("Training end")