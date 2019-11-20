# -*- coding: utf-8 -*-
import numpy as np


def generate_priors(feature_map_list, shrinkage_list, image_size, min_boxes, clamp=True):
    priors = []
    for index in range(0, len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]
        for j in range(0, feature_map_list[1][index]):
            for i in range(0, feature_map_list[0][index]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h

                for min_box in min_boxes[index]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([x_center, y_center, w, h])
    print("priors nums:{}".format(len(priors)))
    priors = np.array(priors)
    if clamp:
        np.clip(priors, 0.0, 1.0, out=priors)
    return priors


def define_img_size(size):
    min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
    img_size_dict = {128: [128, 96],
                     160: [160, 120],
                     320: [320, 240],
                     480: [480, 360],
                     640: [640, 480],
                     1280: [1280, 960]}
    image_size = img_size_dict[size]  # image_size = 320x240

    feature_map_w_h_list_dict = {128: [[16, 8, 4, 2], [12, 6, 3, 2]],
                                 160: [[20, 10, 5, 3], [15, 8, 4, 2]],
                                 320: [[40, 20, 10, 5], [30, 15, 8, 4]],
                                 480: [[60, 30, 15, 8], [45, 23, 12, 6]],
                                 640: [[80, 40, 20, 10], [60, 30, 15, 8]],
                                 1280: [[160, 80, 40, 20], [120, 60, 30, 15]]}
    feature_map_w_h_list = feature_map_w_h_list_dict[size]  # [[40, 20, 10, 5], [30, 15, 8, 4]]

    shrinkage_list = []
    for i in range(0, len(image_size)):
        item_list = []
        for k in range(0, len(feature_map_w_h_list[i])):
            item_list.append(image_size[i] / feature_map_w_h_list[i][k])
        shrinkage_list.append(item_list)
    return generate_priors(feature_map_w_h_list, shrinkage_list, image_size, min_boxes)

