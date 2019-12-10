# -*- coding: utf-8 -*-

import os

from tools.create_prototxt.data_layer import *
from tools.create_prototxt.backbone import half_net_backbone_for_landmark5
from tools.create_prototxt.loss_layer import euclidean_loss, landmark_accuracy


def landmark5_net():
    net_train = "name: \"Landmark5Net\"\n"
    net_val = "name: \"Landmark5Net\"\n"
    net_deploy = "name: \"Landmark5Net\"\n"

    train_root_folder = "/home/ubuntu/tanghy/landmark68_pfld/dataset/WFLW_300W/imgs"
    train_label_file = "/home/ubuntu/tanghy/landmark68_pfld/dataset/WFLW_300W/labels/train.txt"
    batch_size = 128
    img_w = img_h = 56
    data_layer, top_layer = landmark5_data(img_w=img_w, img_h=img_h,
                                           label_file=train_label_file, root_folder=train_root_folder,
                                           batch_size=batch_size, phase="TRAIN")
    net_train += data_layer + "\n"

    test_root_folder = "/home/ubuntu/tanghy/landmark68_pfld/dataset/WFLW_300W/imgs"
    test_label_file = "/home/ubuntu/tanghy/landmark68_pfld/dataset/WFLW_300W/labels/test.txt"
    data_layer, top_layer = landmark5_data(img_w=img_w, img_h=img_h,
                                           label_file=test_label_file, root_folder=test_root_folder,
                                           batch_size=batch_size, phase="TEST")
    net_val += data_layer + "\n"

    temp_layer_deploy, top_name_deploy = deploy_data(shape=[1, 3, img_w, img_h])
    net_deploy += temp_layer_deploy + "\n"

    net_backbone_train_val, top_name_train_val = half_net_backbone_for_landmark5(train=True)
    net_backbone_deploy, _ = half_net_backbone_for_landmark5(train=False)

    net_deploy += net_backbone_deploy + "\n"
    net_train += net_backbone_train_val + "\n"
    net_val += net_backbone_train_val + "\n"

    temp_layer, _ = euclidean_loss("landmark5_loss", [top_name_train_val, "landmark"])
    net_train += temp_layer + "\n"

    temp_layer, _ = landmark_accuracy("accuracy", [top_name_train_val, "landmark"], img_w)
    net_val += temp_layer + "\n"

    return net_train, net_val, net_deploy


def main():
    root_dir = "../../../caffe_model/landmark5"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    train_path = os.path.join(root_dir, "train.prototxt")
    val_path = os.path.join(root_dir, "val.prototxt")
    deploy_path = os.path.join(root_dir, "deploy.prototxt")
    net_train, net_val, net_deploy = landmark5_net()
    with open(train_path, "w", encoding='utf-8') as fp:
        fp.write(net_train)
    with open(val_path, "w", encoding='utf-8') as fp:
        fp.write(net_val)
    with open(deploy_path, "w", encoding='utf-8') as fp:
        fp.write(net_deploy)


if __name__ == '__main__':
    main()
