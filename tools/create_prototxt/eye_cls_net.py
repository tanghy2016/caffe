# -*- coding: utf-8 -*-

from .data_layer import *
from .loss_layer import *
from .backbone import *


def eye_cls_net():
    net_train_val = "name: \"EYENet\"\n"
    net_deploy = "name: \"EYENet\"\n"

    temp_layer_deploy, top_name_deploy = deploy_data(shape=[1, 1, 24, 24])
    net_deploy += temp_layer_deploy + "\n"

    source_train = "/home/ubuntu/disk_b/tanghy/blink_datasets/train.txt"
    source_test = "/home/ubuntu/disk_b/tanghy/blink_datasets/test.txt"
    temp_layer, top_name = image_data(source_train, 24, 24, mirror=True, batch_size=256, is_color=False, shuffle=True,
                                      phase="TRAIN")
    net_train_val += temp_layer + "\n"
    temp_layer, top_name = image_data(source_test, 24, 24, batch_size=100, is_color=False, shuffle=False, phase="TEST")
    net_train_val += temp_layer + "\n"

    net_backbone_train_val, top_name_train_val = eye_cls_backbone(train=True)
    net_backbone_deploy, top_name_deploy = eye_cls_backbone(train=False)

    net_deploy += net_backbone_deploy + "\n"
    temp_layer_deploy, _ = softmax("prob", top_name_deploy)
    net_deploy += temp_layer_deploy + "\n"

    net_train_val += net_backbone_train_val + "\n"
    temp_layer, _ = accuracy("accuracy", [top_name_train_val, "label"])
    net_train_val += temp_layer + "\n"
    temp_layer, _ = softmax_with_loss("loss", [top_name_train_val, "label"])
    net_train_val += temp_layer + "\n"
    return net_train_val, net_deploy


if __name__ == '__main__':
    train_val, deploy = eye_cls_net()
    train_val_path = "./eye_cls/train_val.prototxt"
    deploy_path = "./eye_cls/deploy.prototxt"
    with open(train_val_path, "w") as fp:
        fp.write(train_val)
    with open(deploy_path, "w") as fp:
        fp.write(deploy)

