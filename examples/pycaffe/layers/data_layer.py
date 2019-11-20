# -*- coding: utf-8 -*-

import os
from random import shuffle

import numpy as np
import cv2
import xml.etree.ElementTree as ET
import caffe

from data_transforms import *
from ssd_data import *
from generate_prior import *


def check_params(params):
    assert 'batch_size' in params.keys(), 'Params must include batch_size.'
    assert 'file_source' in params.keys(), 'Params must include file_source (train, val, or test).'


class BoxLandmarkDataLayer(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        check_params(params)

        if 'data_root' in params:
            self.root = params['data_root']
        else:
            self.root = ""
        self.batch_size = params['batch_size']
        file_source = params['file_source']
        self.image_id = [line.rstrip('\n') for line in open(file_source)]

        if 'im_shape' in params:
            self.im_shape = params['im_shape']
        else:
            self.im_shape = [320, 240]
        if 'mean' in params:
            self.mean = np.array(params['mean'])
        else:
            self.mean = np.array([127, 127, 127])
        if 'std' in params:
            self.std = params['std']
        else:
            self.std = 128.0
        if 'variance' in params:
            self.variance = params['variance']
        else:
            self.variance = [0.1, 0.2]

        if 'keep_difficult' in params:
            self.keep_difficult = params['keep_difficult']
        else:
            self.keep_difficult = 0

        self.transform_img = Compose([ConvertFromInts(),
                                      PhotometricDistort(),
                                      RandomSampleCrop(),
                                      RandomMirror(),
                                      ToPercentCoords(),
                                      Resize(self.im_shape),
                                      SubtractMeans(self.mean),
                                      DivStd(self.std),
                                      ChannelFirst()])
        self.priors = define_img_size(self.im_shape[0])
        self.target_transform = MatchPrior(self.priors, self.variance[0], self.variance[1], 0.35)

        self.num = 0
        self.indexs = list(range(len(self.image_id)))

    def reshape(self, bottom, top):
        pass

    def load_data(self, idx):
        boxes, labels, landmarks, is_difficult = self._get_annotation(self.annotation_files[idx])
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
            landmarks = landmarks[is_difficult == 0]
        image = self._read_image(idx)
        # transform
        image, boxes, labels, landmarks = self.transform_img(image, boxes, labels, landmarks)
        boxes, labels, landmarks = self.target_transform(boxes, labels, landmarks)
        return image, boxes, labels, landmarks

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        end_idx = min(self.num + self.batch_size, len(self.image_id))
        db_inds = self.indexs[self.num: end_idx]
        self.num = self.num + self.batch_size
        if self.num >= len(self.im_file_names):
            shuffle(self.indexs)
            self.num -= len(self.im_file_names)
            db_inds += self.indexs[:self.num]
        db_inds = np.asarray(db_inds)
        return db_inds

    def forward(self, bottom, top):
        db_inds = self._get_next_minibatch_inds()
        for i in range(self.batch_size):
            # image: self.im_shape, default: 320 x 240
            # boxes: 4420 x 4
            # labels: 4420
            # landmarks: 4420 x 10
            image, boxes, labels, landmarks = self.load_data(db_inds[i])
            top[0].data[i, ...] = image
            top[1].data[i, ...] = boxes
            top[2].data[i, ...] = labels
            top[3].data[i, ...] = landmarks

    def backward(self, top, propagate_down, bottom):
        pass

    def _get_annotation(self, idx):
        annotation_file = os.path.join(self.root, "Annotations", self.image_id[idx] + ".xml")
        objects = ET.parse(str(annotation_file)).findall("object")
        boxes = []
        labels = []
        landmarks = []
        is_difficult = []
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            # we're only concerned with clases in our list
            if class_name in self.class_dict:
                if (object.find('has_lm') is None) or (int(object.find('has_lm').text) != 1):
                    continue

                bbox = object.find('bndbox')

                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes.append([x1, y1, x2, y2])

                lm = object.find('lm')
                kp_x1 = float(lm.find('x1').text) - 1
                kp_y1 = float(lm.find('y1').text) - 1
                kp_x2 = float(lm.find('x2').text) - 1
                kp_y2 = float(lm.find('y2').text) - 1
                kp_x3 = float(lm.find('x3').text) - 1
                kp_y3 = float(lm.find('y3').text) - 1
                kp_x4 = float(lm.find('x4').text) - 1
                kp_y4 = float(lm.find('y4').text) - 1
                kp_x5 = float(lm.find('x5').text) - 1
                kp_y5 = float(lm.find('y5').text) - 1
                landmarks.append([kp_x1, kp_y1, kp_x2, kp_y2, kp_x3, kp_y3, kp_x4, kp_y4, kp_x5, kp_y5])

                labels.append(self.class_dict[class_name])
                is_difficult_str = object.find('difficult').text
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(landmarks, dtype=np.float32),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, idx):
        image_file = os.path.join(self.root, "JPEGImages", self.image_id[idx] + ".jpg")
        image = cv2.imread(str(image_file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
