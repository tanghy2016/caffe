# -*- coding: utf-8 -*-

import numpy as np
import caffe


class LandmarkAccuracyLayer(caffe.Layer):
    """关键点的预测值到真实值的平均距离"""
    def setup(self, bottom, top):
        params = eval(self.param_str)
        assert 'img_size' in params.keys(), 'Params must include img_size (int).'
        self.img_size = params["img_size"]

        if len(bottom) != 2:
            raise Exception("Need 2 inputs to compute Landmark5 loss.")

    def reshape(self, bottom, top):
        if bottom[0].count != bottom[1].count:
            raise Exception("bottom[0] and bottom[1] must have the same dimension.")

        top[0].reshape(1)

    def forward(self, bottom, top):
        diff = (bottom[0].data - bottom[1].data) * self.img_size  # (N, 136) for landmark68
        diff = diff**2  # (N, 136) for landmark68
        diff_x = diff[:, 0::2]  # (N, 68) for landmark68
        diff_y = diff[:, 1::2]  # (N, 68) for landmark68
        shape_ = bottom[0].data.shape[0] * bottom[0].data.shape[1]/2
        top[0].data[...] = np.sum(np.sqrt(diff_x + diff_y)) / shape_

    def backward(self, top, propagate_down, bottom):
        pass
