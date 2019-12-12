# -*- coding: utf-8 -*-

import os
from random import shuffle
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy import random
import cv2
import caffe


def check_params(params):
    assert 'is_train' in params.keys(), 'Params must include is_train (train or test).'

    required = ['batch_size', 'img_root', 'img_size', 'label_file']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)


class Landmark5Data(caffe.Layer):
    def setup(self, bottom, top):
        params = eval(self.param_str)
        self.root_path = params["img_root"]
        self.img_size = tuple(params["img_size"])
        self.batch_size = params["batch_size"]
        self.label_file = params["label_file"]
        self.is_train = params["is_train"] in ["TRAIN", "Train", "train"]
        self.num_workers = params["num_workers"] if "num_workers" in params else 4
        self.is_std = params["is_std"] if "is_std" in params else True
        self.img_list = []
        self.pts_list = []
        self.box_list = []

        for line in open(self.label_file):
            line_list = line.strip().split()
            self.img_list.append(line_list[0])
            self.pts_list.append([float(item) for item in line_list[1:-4]])
            self.box_list.append([int(item) for item in line_list[-4:]])
        self.pts_list = np.array(self.pts_list, dtype=np.float32)
        self.box_list = np.array(self.box_list)

        top[0].reshape(self.batch_size, 3, self.img_size[0], self.img_size[1])
        top[1].reshape(self.batch_size, 10)  # landmark
        self.num = 0
        self.indexs = list(range(len(self.img_list)))
        self.angle = [0, 0, 0, -12, -8, -4, 4, 8, 12]
        self.box_scale_min = 0.9
        self.box_scale_max = 1.3
        self.mean = np.array([127, 127, 127])
        self.std = 128.0

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        db_inds = self._get_next_minibatch_inds()
        with ThreadPoolExecutor(self.num_workers) as executor:
            for i in range(self.batch_size):
                executor.submit(self.thread_load_data, i, db_inds[i], top)
        """
        for i in range(self.batch_size):
            image, landmark5 = self.load_data(db_inds[i])
            top[0].data[i, ...] = image.transpose((2, 0, 1))
            top[1].data[i, ...] = landmark5
        """

    def backward(self, top, propagate_down, bottom):
        pass

    def thread_load_data(self, i, idx, top):
        image, landmark5 = self.load_data(idx)
        top[0].data[i, ...] = image.transpose((2, 0, 1))
        top[1].data[i, ...] = landmark5

    def _get_next_minibatch_inds(self):
        """Return the roidb indices for the next minibatch."""
        end_idx = min(self.num + self.batch_size, len(self.img_list))
        db_inds = self.indexs[self.num: end_idx]
        self.num = self.num + self.batch_size
        if self.num >= len(self.img_list):
            shuffle(self.indexs)
            self.num -= len(self.img_list)
            db_inds += self.indexs[:self.num]
        db_inds = np.asarray(db_inds)
        return db_inds

    def load_data(self, idx):
        img = cv2.imread(os.path.join(self.root_path, self.img_list[idx]))
        landmark5 = self.pts_list[idx, :].copy()   # copy是必须的, np.array结构类似C++的传地址操作
        box = self.box_list[idx, :].copy()         # 如果不加copy, 则第二次epoch后均会出错
        box[0] = min(max(box[0], 0), img.shape[1]-1)
        box[1] = min(max(box[1], 0), img.shape[0]-1)
        box[2] = max(min(img.shape[1]-1, box[2]), 0)
        box[3] = max(min(img.shape[0]-1, box[3]), 0)
        face_img, landmark5 = self.box_process(img, landmark5, box)
        if self.is_train:
            face_img = self.random_bright(face_img)
            face_img, landmark5 = self.random_flip(face_img, landmark5)
        if self.is_std:
            face_img = self.standardization(face_img)

        """
        # debug for loss error that started at the second epoch 
        out_path = os.path.join("/home/ubuntu/tanghy/landmark68_pfld/dataset/WFLW_300W/out", str(idx) + ".jpg")
        for i in range(int(len(landmark5) / 2)):
            x = int(landmark5[2 * i] * face_img.shape[0])
            y = int(landmark5[2 * i + 1] * face_img.shape[1])
            cv2.circle(img=face_img, center=(x, y), radius=2, color=(0, 0, 255), thickness=-1)
        cv2.imwrite(out_path, face_img)
        print(idx, face_img.shape, landmark5.shape)
        """

        return face_img.astype(np.float32), landmark5

    def box_process(self, img, landmark5, box):
        centor_x = (box[0] + box[2]) / 2.0
        centor_y = (box[1] + box[3]) / 2.0

        if self.is_train:
            random_angle = random.randint(len(self.angle))
            if self.angle[random_angle] != 0:
                # 图像旋转
                img, center, fill = self.img_rotate(img, [centor_x, centor_y], self.angle[random_angle])
                # 关键点旋转
                landmark5 = self.landmark_rotate(landmark5, center, fill, self.angle[random_angle])
                # face box旋转
                box4 = [box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3]]
                box4 = self.landmark_rotate(box4, center, fill, self.angle[random_angle])
                box[0] = min(box4[0::2])
                box[1] = min(box4[1::2])
                box[2] = max(box4[0::2])
                box[3] = max(box4[1::2])

                centor_x = (box[0] + box[2]) / 2.0
                centor_y = (box[1] + box[3]) / 2.0

        w = box[2] - box[0]
        h = box[3] - box[1]

        scale = random.uniform(self.box_scale_min, self.box_scale_max)
        if (not self.is_train) or random.random() < 0.3:  # 方形face box的概率
            w = h = max(w, h) * scale
        else:
            if w > h:
                h += int((w - h) * random.random())
            else:
                w += int((h - w) * random.random())
            w *= scale
            h *= scale
        box[0] = int(centor_x - w / 2.0)
        box[1] = int(centor_y - h / 2.0)
        box[2] = int(centor_x + w / 2.0)
        box[3] = int(centor_y + h / 2.0)

        top = bottom = left = right = 0
        if box[1] < 0:
            top = int(-box[1])
            box[1] = 0
        if box[3] > img.shape[0]:
            bottom = int(box[3] - img.shape[0] + 1)
        if box[0] < 0:
            left = int(-box[0])
            box[0] = 0
        if box[2] > img.shape[1]:
            right = int(box[2] - img.shape[1] + 1)
        if top > 0 or bottom > 0 or left > 0 or right > 0:
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)
        for i in range(int(len(landmark5) / 2)):
            landmark5[2 * i] += left - box[0]
            landmark5[2 * i] = min(landmark5[2 * i], box[2])
            landmark5[2 * i] = max(landmark5[2 * i], 0)
            landmark5[2 * i] /= (box[2] - box[0])
            landmark5[2 * i + 1] += top - box[1]
            landmark5[2 * i + 1] = min(landmark5[2 * i + 1], box[3])
            landmark5[2 * i + 1] = max(landmark5[2 * i + 1], 0)
            landmark5[2 * i + 1] /= (box[3] - box[1])
        face_img = img[box[1]:box[3], box[0]:box[2], :]
        face_img = cv2.resize(face_img, self.img_size)
        return face_img, landmark5

    def random_bright(self, img, delta=16):
        alpha = random.random()
        if alpha > 0.3:
            img = img * alpha + random.uniform(-delta, delta)
            img = img.clip(min=0, max=255).astype(np.uint8)
        return img

    def random_flip(self, img, landmarks):
        if random.randint(2):
            img = img[:, ::-1].copy()
            landmarks[0::2] = 1 - landmarks[0::2]
            # 关键点的顺序将会变化, 1 <-> 2, 4 <-> 5
            temp = landmarks[0:2].copy()
            landmarks[0:2] = landmarks[2:4].copy()
            landmarks[2:4] = temp.copy()
            temp = landmarks[6:8].copy()
            landmarks[6:8] = landmarks[8:10].copy()
            landmarks[8:10] = temp.copy()
        return img, landmarks

    def standardization(self, img):
        img = img.astype(np.float32)
        img -= self.mean
        img = img / self.std
        return img

    def img_rotate(self, im, center, angle):
        """center: list/tuple, (x, y)"""
        radian = angle / 180.0 * np.pi
        # top, right, bottom, left
        trbl = [center[1], (im.shape[1] - center[0] + 1), (im.shape[0] - center[1] + 1), center[0]]
        fill = []
        if angle > 0:
            tag = 1
        else:
            tag = -1
        for i in range(4):
            fill.append(int(np.abs(np.sin(radian)) * trbl[(i + tag + 4) % 4]))
        if fill[0] > 0 or fill[2] > 0 or fill[3] > 0 or fill[1] > 0:
            image = cv2.copyMakeBorder(im, fill[0], fill[2], fill[3], fill[1], cv2.BORDER_CONSTANT)
        center[0] += fill[3]
        center[1] += fill[0]

        affine_matrix = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
        image = cv2.warpAffine(image, affine_matrix, (image.shape[1], image.shape[0]))
        return image, center, (fill[3], fill[0])

    def landmark_rotate(self, landmark, center, fill, angle):
        radian = angle / 180.0 * np.pi
        landmark_new = []
        for i in range(int(len(landmark) / 2)):
            x = landmark[i * 2] + fill[0]
            y = landmark[i * 2 + 1] + fill[1]
            x_new = (x - center[0]) * np.cos(radian) + (y - center[1]) * np.sin(radian) + center[0]
            y_new = -(x - center[0]) * np.sin(radian) + (y - center[1]) * np.cos(radian) + center[1]
            landmark_new.append(x_new)
            landmark_new.append(y_new)
        return np.array(landmark_new)
