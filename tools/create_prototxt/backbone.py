# -*- coding: utf-8 -*-

from tools.create_prototxt.block import *


def shuffle_net_v2_backbone(train=True):
    temp_layer, top_name = conv_block("conv1", "data", 24, 3, blob_same=True,
                                      bias_term_conv=False, bias_term_scale=True,
                                      pad=1, stride=2, train=train, relu_type="ReLU6", weight_filler="msra")
    net = temp_layer + "\n"
    temp_layer, top_name = pool("pool1", top_name, 3, pool="MAX", stride=2)
    net += temp_layer + "\n"
    temp_layer, top_name = shufflev2_block(top_name, 1, 24, 58, stride=2,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = shufflev2_block(top_name, 2, 58 * 2, 58, stride=1,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = shufflev2_block(top_name, 3, 58 * 2, 58, stride=1,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = shufflev2_block(top_name, 4, 58 * 2, 58, stride=1,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = shufflev2_block(top_name, 5, 58 * 2, 116, stride=2,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = shufflev2_block(top_name, 6, 116 * 2, 116, stride=1,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = shufflev2_block(top_name, 7, 116 * 2, 116, stride=1,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = shufflev2_block(top_name, 8, 116 * 2, 116, stride=1,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = shufflev2_block(top_name, 9, 116 * 2, 116, stride=1,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = shufflev2_block(top_name, 10, 116 * 2, 116, stride=1,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = shufflev2_block(top_name, 11, 116 * 2, 116, stride=1,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = shufflev2_block(top_name, 12, 116 * 2, 116, stride=1,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = shufflev2_block(top_name, 13, 116 * 2, 232, stride=2,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = shufflev2_block(top_name, 14, 232 * 2, 232, stride=1,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = shufflev2_block(top_name, 15, 232 * 2, 232, stride=1,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = shufflev2_block(top_name, 16, 232 * 2, 232, stride=1,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = conv_block("conv5", top_name, 512, 1, blob_same=True,
                                      bias_term_conv=False, bias_term_scale=True,
                                      pad=0, stride=1, train=train, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = conv_block("conv5/dw", top_name, 512, 7, blob_same=True,
                                      bias_term_conv=False, bias_term_scale=True,
                                      pad=0, stride=1, group=512, train=train, relu_type="", weight_filler="msra")
    net += temp_layer + "\n"
    return net, top_name


def eye_cls_backbone(train=True):
    temp_layer, top_name = conv_block("conv1", "data", 16, 3, bias_term_conv=False, bias_term_scale=True,
                                      relu_type="ReLU6", pad=1, train=train, weight_filler="msra")
    net = temp_layer + "\n"
    temp_layer, top_name = conv_block("conv2_1", top_name, 32, 3, bias_term_conv=False, bias_term_scale=True,
                                      relu_type="ReLU6", pad=1, stride=2, train=train, weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = conv_block("conv2_2", top_name, 32, 3, bias_term_conv=False, bias_term_scale=True,
                                      relu_type="ReLU6", pad=1, train=train, weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = conv_block("conv2_3", top_name, 32, 3, bias_term_conv=False, bias_term_scale=True,
                                      relu_type="ReLU6", pad=1, train=train, weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = conv_block("conv3_1", top_name, 64, 3, bias_term_conv=False, bias_term_scale=True,
                                      relu_type="ReLU6", pad=1, stride=2, train=train, weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = conv_block("conv3_2", top_name, 64, 3, bias_term_conv=False, bias_term_scale=True,
                                      relu_type="ReLU6", pad=1, train=train, weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = conv_block("conv3_3", top_name, 64, 3, bias_term_conv=False, bias_term_scale=True,
                                      relu_type="ReLU6", pad=1, train=train, weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = conv_block("conv4_1", top_name, 64, 3, bias_term_conv=False, bias_term_scale=True,
                                      relu_type="ReLU6", pad=1, stride=2, train=train, weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = conv_block("conv4_2", top_name, 64, 3, bias_term_conv=False, bias_term_scale=True,
                                      relu_type="ReLU6", pad=1, train=train, weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = conv_block("conv4_3", top_name, 64, 3, bias_term_conv=False, bias_term_scale=True,
                                      relu_type="ReLU6", pad=1, train=train, weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = conv("conv5", top_name, 2, 3, bias_term=True, weight_filler="msra")
    net += temp_layer + "\n"
    return net, top_name


def half_net_backbone(train=True, relu_type="ReLU"):
    temp_layer, top_name = conv_block("conv1_1", "data", 32, 3, pad=1, stride=1, train=train, relu_type=relu_type)
    net_base = temp_layer + "\n"
    temp_layer, top_name = conv_block("conv1_2", top_name, 32, 3, pad=1, stride=2, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"

    temp_layer, top_name = half_conv_block1("stage2_1", top_name, 32, 64, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"
    temp_layer, top_name = half_conv_block1("stage2_2", top_name, 64, 64, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"
    temp_layer, top_name = conv_block("stage2_3", top_name, 64, 3, pad=1, stride=2, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"

    temp_layer, top_name = half_conv_block1("stage3_1", top_name, 64, 64, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"
    temp_layer, top_name = half_conv_block1("stage3_2", top_name, 64, 64, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"
    temp_layer, top_name = conv_block("stage3_3", top_name, 64, 3, pad=1, stride=2, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"

    temp_layer, top_name = half_conv_block1("stage4_1", top_name, 64, 64, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"
    temp_layer, top_name = half_conv_block1("stage4_2", top_name, 64, 64, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"
    temp_layer, top_name = conv_block("stage4_3", top_name, 64, 3, pad=1, stride=2, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"

    temp_layer, top_name = half_conv_block1("stage5_1", top_name, 64, 64, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"
    temp_layer, top_name = half_conv_block1("stage5_2", top_name, 64, 64, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"
    temp_layer, top_name = conv_block("stage5_3", top_name, 64, 3, pad=1, stride=2, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"

    temp_layer, top_name = half_conv_block1("stage6_1", top_name, 64, 64, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"
    temp_layer, top_name = half_conv_block1("stage6_2", top_name, 64, 64, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"
    temp_layer, top_name = conv_block("stage6_3", top_name, 64, 3, pad=1, stride=2, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"

    temp_layer, top_name = half_conv_block1("stage7_1", top_name, 64, 64, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"
    temp_layer, top_name = half_conv_block1("stage7_2", top_name, 64, 64, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"
    temp_layer, top_name = conv_block("stage7_3", top_name, 64, 5, pad=0, stride=1, train=train, relu_type=relu_type,
                                      c_w_decay_mult=10)
    net_base += temp_layer + "\n"
    return net_base, top_name


def half_net_backbone_for_landmark5(train=True, relu_type="ReLU"):
    temp_layer, top_name = conv_block("conv1_1", "data", 16, 3, pad=1, stride=1, train=train, relu_type=relu_type)
    net_base = temp_layer + "\n"
    temp_layer, top_name = conv_block("conv1_2", top_name, 16, 3, pad=1, stride=2, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"

    temp_layer, top_name = half_conv_block1("stage2_1", top_name, 16, 32, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"
    temp_layer, top_name = half_conv_block1("stage2_2", top_name, 32, 32, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"
    temp_layer, top_name = conv_block("stage2_3", top_name, 64, 3, pad=1, stride=2, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"

    temp_layer, top_name = half_conv_block1("stage3_1", top_name, 64, 64, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"
    temp_layer, top_name = half_conv_block1("stage3_2", top_name, 64, 64, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"
    temp_layer, top_name = half_conv_block1("stage3_3", top_name, 64, 64, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"
    temp_layer, top_name = conv_block("stage3_4", top_name, 128, 3, pad=1, stride=2, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"

    temp_layer, top_name = half_conv_block1("stage4_1", top_name, 128, 128, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"
    temp_layer, top_name = half_conv_block1("stage4_2", top_name, 128, 128, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"
    temp_layer, top_name = conv_block("stage4_3", top_name, 10, 7, pad=0, stride=7, train=train, relu_type=relu_type)
    net_base += temp_layer + "\n"
    temp_layer, top_name = sigmoid("sigmoid", top_name)
    net_base += temp_layer + "\n"
    return net_base, top_name


def ssd_top_backbone(top_name, min_size, max_size, aspect_ratio, step, box_num, class_num, train=True):
    top_net = ""
    for i in range(len(top_name)):
        temp_layer, top_layer = conv_block(top_name[i] + "/mbox_loc", top_name[i], box_num[i] * 4, 3, pad=1,
                                           stride=1, train=True, relu_type="ReLU")
        top_net += temp_layer + "\n"
        temp_layer, top_layer = permute(top_name[i] + "/mbox_loc_perm", top_layer, [0, 2, 3, 1], top=None)
        top_net += temp_layer + "\n"
        temp_layer, top_layer = flatten(top_name[i] + "/mbox_loc_flat", top_layer, top=None)
        top_net += temp_layer + "\n"

        temp_layer, top_layer = conv_block(top_name[i] + "/mbox_conf", top_name[i], box_num[i] * class_num, 3, pad=1,
                                           stride=1, train=True, relu_type="ReLU")
        top_net += temp_layer + "\n"
        temp_layer, top_layer = permute(top_name[i] + "/mbox_conf_perm", top_layer, [0, 2, 3, 1], top=None)
        top_net += temp_layer + "\n"
        temp_layer, top_layer = flatten(top_name[i] + "/mbox_conf_flat", top_layer, top=None)
        top_net += temp_layer + "\n"

        temp_layer, top_layer = prior_box(top_name[i] + "/mbox_priorbox", [top_name[i], "data"],
                                          min_size[i], max_size[i], aspect_ratio[i],
                                          variance=[0.1, 0.1, 0.2, 0.2], step=step[i], top=None)
        top_net += temp_layer + "\n"

    temp_layer, top_name_loc = concat("mbox_loc", [item + "/mbox_loc_flat" for item in top_name], top=None, axis=1)
    top_net += temp_layer + "\n"
    temp_layer, top_name_conf = concat("mbox_conf", [item + "/mbox_conf_flat" for item in top_name], top=None, axis=1)
    top_net += temp_layer + "\n"
    temp_layer, top_name_priorbox = concat("mbox_priorbox", [item + "/mbox_priorbox" for item in top_name], top=None,
                                           axis=2)
    top_net += temp_layer + "\n"
    if not train:
        temp_layer, top_name_conf = reshape("mbox_conf_reshape", top_name_conf, [0, -1, 2])
        top_net += temp_layer + "\n"
        temp_layer, top_name_conf = softmax("mbox_conf_softmax", top_name_conf, axis=2)
        top_net += temp_layer + "\n"
        temp_layer, top_name_conf = flatten("mbox_conf_flatten", top_name_conf)
        top_net += temp_layer + "\n"
    return top_net, [top_name_loc, top_name_conf, top_name_priorbox]


def test_backbone():
    # net, top_name = shuffle_net_v2_backbone(train=True)
    # net, top_name = eye_cls_backbone(train=True)
    net, top_name = half_net_backbone(train=True, relu_type="ReLU")
    print("<====================base_net=====================>")
    print(net)
    print("<====================top_name=====================>")
    print(top_name)


if __name__ == '__main__':
    test_backbone()

