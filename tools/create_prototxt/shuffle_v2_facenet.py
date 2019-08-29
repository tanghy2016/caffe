# -*- coding: utf-8 -*-

from tools.create_prototxt.data_layer import *
from tools.create_prototxt.block import *
from tools.create_prototxt.loss_layer import *


def shuffle_facenet_backbone(train=True):
    # input: 3*112, output: 64*56
    temp_layer, top_name = conv_block("conv1", "data", 64, 3, blob_same=True,
                                      bias_term_conv=False, bias_term_scale=True,
                                      pad=1, stride=2, train=train, relu_type="ReLU6", weight_filler="msra")
    net = temp_layer + "\n"
    # input: 64*56, output: 64*28
    temp_layer, top_name = pool("pool1", top_name, 3, pool="MAX", stride=2)
    net += temp_layer + "\n"
    # input: 64*28, output: 64*28
    temp_layer, top_name = shufflev2_block(top_name, 1, 64, 32, stride=1,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    # input: 64*28, output: 64*28
    temp_layer, top_name = shufflev2_block(top_name, 2, 64, 32, stride=1,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    # input: 64*28, output: 64*28
    temp_layer, top_name = shufflev2_block(top_name, 3, 64, 32, stride=1,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    # input: 64*28, output: 128*14
    temp_layer, top_name = shufflev2_block(top_name, 4, 64, 64, stride=2,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")

    net += temp_layer + "\n"
    # input: 128*14, output: 128*14
    temp_layer, top_name = shufflev2_block(top_name, 5, 128, 64, stride=1,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    # input: 128*14, output: 128*14
    temp_layer, top_name = shufflev2_block(top_name, 6, 128, 64, stride=1,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    # input: 128*14, output: 128*14
    temp_layer, top_name = shufflev2_block(top_name, 7, 128, 64, stride=1,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    # input: 128*14, output: 128*14
    temp_layer, top_name = shufflev2_block(top_name, 8, 128, 64, stride=1,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    # input: 128*14, output: 128*14
    temp_layer, top_name = shufflev2_block(top_name, 9, 128, 64, stride=1,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    # input: 128*14, output: 128*14
    temp_layer, top_name = shufflev2_block(top_name, 10, 128, 64, stride=1,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    # input: 128*14, output: 256*7
    temp_layer, top_name = shufflev2_block(top_name, 11, 128, 128, stride=2,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    # input: 256*7, output: 256*7
    temp_layer, top_name = shufflev2_block(top_name, 12, 256, 128, stride=1,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    # input: 256*7, output: 256*7
    temp_layer, top_name = shufflev2_block(top_name, 13, 256, 128, stride=1,
                                           bias_term_conv=False, bias_term_scale=True,
                                           train=True, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    # input: 256*7, output: 512*7
    temp_layer, top_name = conv_block("conv5", top_name, 512, 1, blob_same=True,
                                      bias_term_conv=False, bias_term_scale=True,
                                      pad=0, stride=1, train=train, relu_type="ReLU", weight_filler="msra")
    net += temp_layer + "\n"
    # input: 512*7, output: 512*1
    temp_layer, top_name = conv_block("conv5/dw", top_name, 512, 7, blob_same=True,
                                      bias_term_conv=False, bias_term_scale=True,
                                      pad=0, stride=1, group=512, train=train, relu_type="",
                                      c_w_decay_mult=10, weight_filler="msra")
    net += temp_layer + "\n"
    # input: 512*1, output: 128
    temp_layer, top_name = fc("fc5", top_name, 128, top=None, bias_term=False, w_decay_mult=1,
                              weight_filler="msra", normalize=False)
    net += temp_layer + "\n"
    return net, top_name


def shuffle_facenet(source_train, source_test, class_num, type_margin):
    net_train = "name: \"ShuffleV2FaceNet\"\n"
    net_val = "name: \"ShuffleV2FaceNet\"\n"
    net_deploy = "name: \"ShuffleV2FaceNet\"\n"

    temp_layer_deploy, top_name_deploy = deploy_data(shape=[1, 3, 112, 112])
    net_deploy += temp_layer_deploy + "\n"

    # source_train = "examples/face_recognition/train_faceid_list.txt"
    # source_test = "examples/face_recognition/test_faceid_list.txt"
    # class_num = 85742
    # type_margin = "sv-x"

    temp_layer, top_name = image_data(source_train, 112, 112, name="data", top=["data", "label"],
                                      batch_size=64, root_folder="",
                                      crop_size=0, mirror=True, mean_file="", mean_value=[128], scale=0.0078125,
                                      is_color=True, shuffle=True, phase="TRAIN")
    net_train += temp_layer + "\n"
    temp_layer, top_name = image_data(source_test, 112, 112, name="data", top=["data", "label"],
                                      batch_size=50, root_folder="",
                                      crop_size=0, mirror=False, mean_file="", mean_value=[128], scale=0.0078125,
                                      is_color=True, shuffle=False, phase="TEST")
    net_val += temp_layer + "\n"

    net_backbone_train_val, top_name_train_val = shuffle_facenet_backbone(train=True)
    net_backbone_deploy, _ = shuffle_facenet_backbone(train=False)

    net_deploy += net_backbone_deploy + "\n"
    net_train += net_backbone_train_val + "\n"
    net_val += net_backbone_train_val + "\n"

    temp_layer, top_name = normalize("norm1", top_name_train_val, top=None)
    net_train += temp_layer + "\n"
    net_val += temp_layer + "\n"
    temp_layer, top_name = fc("fc6_l2", top_name, class_num, top=None, bias_term=False,
                              w_decay_mult=1, weight_filler="msra", normalize=True)
    net_train += temp_layer + "\n"
    net_val += temp_layer + "\n"

    temp_layer, _ = accuracy("accuracy", [top_name, "label"])
    net_val += temp_layer + "\n"

    if type_margin == "sv-x":
        temp_layer, top_name = sv_x(type_margin, [top_name, "label"], top="fc6_margin", m1=1, m2=0, m3=0.5, t=1.2)
    elif type_margin == "softmax":
        pass
    elif type_margin == "arc":
        temp_layer, top_name = add_margin(type_margin, [top_name, "label"], top="fc6_margin", m1=1, m2=0, m3=0.5)
    elif type_margin == "am":
        temp_layer, top_name = add_margin(type_margin, [top_name, "label"], top="fc6_margin", m1=1, m2=0.35, m3=0)
    else:
        raise Exception("unkown type for margin %s" % type_margin)
    if type_margin != "softmax":
        net_train += temp_layer + "\n"
        net_val += temp_layer + "\n"

    temp_layer, top_name = scale("fc6_scale", top_name, "fc6_scale", type_s="constant", w_value=64,
                                 w_lr_mult=0, w_decay_mult=0,
                                 bias_term=False, b_lr_mult=0, b_decay_mult=0)
    net_train += temp_layer + "\n"
    net_val += temp_layer + "\n"

    temp_layer, _ = softmax_with_loss("loss", [top_name, "label"])
    net_train += temp_layer + "\n"
    net_val += temp_layer + "\n"

    return net_train, net_val, net_deploy


def get_shuffle_v2_facenet():
    # for big dataset
    source_train = "examples/face_recognition/train_faceid_list.txt"
    source_test = "examples/face_recognition/test_faceid_list.txt"
    class_num = 85742
    net_train, net_val, net_deploy = shuffle_facenet(source_train, source_test, class_num, type_margin="sv-x")
    train_path = "../../caffe_model/ShuffleFaceNet/train.prototxt"
    val_path = "../../caffe_model/ShuffleFaceNet/val.prototxt"
    deploy_path = "../../caffe_model/ShuffleFaceNet/deploy.prototxt"
    with open(train_path, "w") as fp:
        fp.write(net_train)
    with open(val_path, "w") as fp:
        fp.write(net_val)
    with open(deploy_path, "w") as fp:
        fp.write(net_deploy)
    train_path_soft = "../../caffe_model/ShuffleFaceNet/train_soft.prototxt"
    val_path_soft = "../../caffe_model/ShuffleFaceNet/val_soft.prototxt"
    net_train_soft, net_val_soft, _ = shuffle_facenet(source_train, source_test, class_num, type_margin="softmax")
    with open(train_path_soft, "w") as fp:
        fp.write(net_train_soft)
    with open(val_path_soft, "w") as fp:
        fp.write(net_val_soft)

    # for small dataset
    source_train_small = "examples/face_recognition/train_small_faceid_list.txt"
    source_test_small = "examples/face_recognition/test_small_faceid_list.txt"
    class_num_small = 100
    net_train_small_soft, net_val_small_soft, _ = shuffle_facenet(source_train_small, source_test_small,
                                                                  class_num_small, type_margin="softmax")
    train_path_small_soft = "../../caffe_model/ShuffleFaceNet/train_small_soft.prototxt"
    val_path_small_soft = "../../caffe_model/ShuffleFaceNet/val_small_soft.prototxt"
    with open(train_path_small_soft, "w") as fp:
        fp.write(net_train_small_soft)
    with open(val_path_small_soft, "w") as fp:
        fp.write(net_val_small_soft)

    net_train_small_svx, net_val_small_svx, _ = shuffle_facenet(source_train_small, source_test_small,
                                                                class_num_small, type_margin="sv-x")
    train_path_small_svx = "../../caffe_model/ShuffleFaceNet/train_small_svx.prototxt"
    val_path_small_svx = "../../caffe_model/ShuffleFaceNet/val_small_svx.prototxt"
    with open(train_path_small_svx, "w") as fp:
        fp.write(net_train_small_svx)
    with open(val_path_small_svx, "w") as fp:
        fp.write(net_val_small_svx)

    net_train_small_arc, net_val_small_arc, _ = shuffle_facenet(source_train_small, source_test_small,
                                                                class_num_small, type_margin="arc")
    train_path_small_arc = "../../caffe_model/ShuffleFaceNet/train_small_arc.prototxt"
    val_path_small_arc = "../../caffe_model/ShuffleFaceNet/val_small_arc.prototxt"
    with open(train_path_small_arc, "w") as fp:
        fp.write(net_train_small_arc)
    with open(val_path_small_arc, "w") as fp:
        fp.write(net_val_small_arc)


if __name__ == '__main__':
    get_shuffle_v2_facenet()

