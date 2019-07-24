# -*- coding: utf-8 -*-

import os


def deploy_data(shape, name="data", top=None):
    if not top:
        top = name
    if not isinstance(shape, list):
        raise Exception("shape must be list")
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"Input\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "  input_param {\n"
    layer += "    shape: {\n"
    for dim in shape:
        layer += "      dim: " + str(dim) + "\n"
    layer += "    }\n"
    layer += "  }\n"
    layer += "}"
    return layer, top


def image_data(source, new_height, new_width, root_folder="", crop_size=0, mirror=False, batch_size=64, mean_file="", mean_value=[], is_color=True, shuffle=False, name="data", top=["data", "label"], phase="TRAIN"):
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"ImageData\"\n"
    for top_name in top:
        layer += "  top: \"" + top_name + "\"\n"
    layer += "  include {\n"
    layer += "    phase: " + phase + "\n"
    layer += "  }\n"
    if not mirror and crop_size==0 and mean_file=="" and len(mean_value)==0:
        print("Don't have any pre-process for input image!")
    else:
        layer += "  transform_param {\n"
        if mirror:
            layer += "    mirror: true\n"
        if crop_size > 0:
            layer += "    crop_size: " + str(crop_size) + "\n"
        if mean_file != "":
            layer += "    mean_file: \"" + mean_file + "\"\n"
        if len(mean_value) > 0:
            for value in mean_value:
                layer += "    mean_value: " + str(value) + "\n"
        layer += "  }\n"
    layer += "  image_data_param {\n"
    layer += "    source: \"" + source + "\"\n"
    if root_folder != "":
        layer += "    root_folder: \"" + root_folder + "\"\n"
    layer += "    batch_size: " + str(batch_size) + "\n"
    if not is_color:
        layer += "    is_color: false\n"
    layer += "    new_height: " + str(new_height) + "\n"
    layer += "    new_width: " + str(new_width) + "\n"
    if shuffle:
        layer += "    shuffle: true\n"
    layer += "  }\n"
    layer += "}"
    return layer, top


def conv(name, bottom, num_output, kernel_size, top=None, bias_term=True, pad=0, stride=1, group=1, weight_filler="msra"):
    if not top:
        top=name
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"Convolution\"\n"
    layer += "  bottom: \"" + bottom + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "  param {\n"
    layer += "    lr_mult: 1\n"
    layer += "    decay_mult: 1\n"
    layer += "  }\n"
    if bias_term:
        layer += "  param {\n"
        layer += "    lr_mult: 2\n"
        layer += "    decay_mult: 0\n"
        layer += "  }\n"
    layer += "  convolution_param {\n"
    layer += "    num_output: " + str(num_output) + "\n"
    layer += "    kernel_size: " + str(kernel_size) + "\n"
    if not bias_term:
        layer += "    bias_term: false\n"
    if pad != 0:
        layer += "    pad: " + str(pad) + "\n"
    if stride != 1:
        layer += "    stride: " + str(stride) + "\n"
    if group != 1:
        layer += "    group: " + str(group) + "\n"
    if weight_filler=="msra":
        layer += "    weight_filler {\n"
        layer += "      type: \"msra\"\n"
        layer += "    }\n"
    elif weight_filler=="gaussian":
        layer += "    weight_filler {\n"
        layer += "      type: \"gaussian\"\n"
        layer += "      std: 0.01\n"
        layer += "    }\n"
    else:
        raise Exception("unknown weight_filler: %s" % weight_filler)
    if bias_term:
        layer += "    bias_filler {\n"
        layer += "      type: \"constant\"\n"
        layer += "      value: 0\n"
        layer += "    }\n"
    layer += "  }\n"
    layer += "}"
    return layer, top


def dwconv(name, bottom, num_output, kernel_size, top=None, bias_term=True, pad=0, stride=1, group=None, weight_filler="msra"):
    if not top:
        top=name
    if not group:
        group = num_output
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"DepthwiseConvolution\"\n"
    layer += "  bottom: \"" + bottom + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "  param {\n"
    layer += "    lr_mult: 1.0\n"
    layer += "    decay_mult: 1.0\n"
    layer += "  }\n"
    if bias_term:
        layer += "  param {\n"
        layer += "    lr_mult: 2\n"
        layer += "    decay_mult: 0\n"
        layer += "  }\n"
    layer += "  convolution_param {\n"
    layer += "    num_output: " + str(num_output) + "\n"
    layer += "    kernel_size: " + str(kernel_size) + "\n"
    if not bias_term:
        layer += "    bias_term: false\n"
    if pad != 0:
        layer += "    pad: " + str(pad) + "\n"
    if stride != 1:
        layer += "    stride: " + str(stride) + "\n"
    if group != 1:
        layer += "    group: " + str(group) + "\n"
    if weight_filler=="msra":
        layer += "    weight_filler {\n"
        layer += "      type: \"msra\"\n"
        layer += "    }\n"
    elif weight_filler=="gaussian":
        layer += "    weight_filler {\n"
        layer += "      type: \"gaussian\"\n"
        layer += "      std: 0.01\n"
        layer += "    }\n"
    else:
        raise Exception("unknown weight_filler: %s" % weight_filler)
    if bias_term:
        layer += "    bias_filler {\n"
        layer += "      type: \"constant\"\n"
        layer += "      value: 0\n"
        layer += "    }\n"
    layer += "  }\n"
    layer += "}"
    return layer, top


def pool(name, bottom, kernel_size, pool="MAX", top=None, pad=0, stride=1, global_pooling=False):
    if not top:
        top=name
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"Pooling\"\n"
    layer += "  bottom: \"" + bottom + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "  pooling_param {\n"
    if pool not in ["MAX", "AVG", "STOCHASTIC"]:
        raise Exception("unknown pool: %s" % pool)
    layer += "    pool: " + pool + "\n"
    layer += "    kernel_size: " + str(kernel_size) + "\n"
    if pad != 0:
        layer += "    pad: " + str(pad) + "\n"
    if stride != 1 and not global_pooling:
        layer += "    stride: " + str(stride) + "\n"
    if global_pooling:
        layer += "    global_pooling: true\n"
    layer += "  }\n"
    layer += "}"
    return layer, top


def fc(name, bottom, num_output, top=None, bias_term=True, weight_filler="msra"):
    if not top:
        top=name
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"InnerProduct\"\n"
    layer += "  bottom: \"" + bottom + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "  param {\n"
    layer += "    lr_mult: 1\n"
    layer += "    decay_mult: 1\n"
    layer += "  }\n"
    if bias_term:
        layer += "  param {\n"
        layer += "    lr_mult: 2\n"
        layer += "    decay_mult: 0\n"
        layer += "  }\n"
    layer += "  inner_product_param {\n"
    layer += "    num_output: " + str(num_output) + "\n"
    if not bias_term:
        layer += "    bias_term: false\n"
    if weight_filler=="msra":
        layer += "    weight_filler {\n"
        layer += "      type: \"msra\"\n"
        layer += "    }\n"
    elif weight_filler=="gaussian":
        layer += "    weight_filler {\n"
        layer += "      type: \"gaussian\"\n"
        layer += "      std: 0.01\n"
        layer += "    }\n"
    else:
        raise Exception("unknown weight_filler: %s" % weight_filler)
    if bias_term:
        layer += "    bias_filler {\n"
        layer += "      type: \"constant\"\n"
        layer += "      value: 0\n"
        layer += "    }\n"
    layer += "  }\n"
    layer += "}"
    return layer, top


def relu(name, bottom, top, type="ReLU"):
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    if type not in ["ReLU", "ReLU6", "CReLU"]:
        raise Exception("unknown relu: %s" % type)
    layer += "  type: \"" + type + "\"\n"
    layer += "  bottom: \"" + bottom + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "}"
    return layer, top


def bn(name, bottom, top, train=True):
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"BatchNorm\"\n"
    layer += "  bottom: \"" + bottom + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "  param {\n"
    layer += "    lr_mult: 0\n"
    layer += "    decay_mult: 0\n"
    layer += "  }\n"
    layer += "  param {\n"
    layer += "    lr_mult: 0\n"
    layer += "    decay_mult: 0\n"
    layer += "  }\n"
    layer += "  param {\n"
    layer += "    lr_mult: 0\n"
    layer += "    decay_mult: 0\n"
    layer += "  }\n"
    layer += "  batch_norm_param {\n"
    if train:
        layer += "    use_global_stats: false\n"
    else:
        layer += "    use_global_stats: true\n"
    layer += "    eps: 1e-5\n"
    layer += "  }\n"
    layer += "}"
    return layer, top


def scale(name, bottom, top, bias_term=False):
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"Scale\"\n"
    layer += "  bottom: \"" + bottom + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "  scale_param {\n"
    layer += "    filler {\n"
    layer += "      value: 1.0\n"
    layer += "    }\n"
    if bias_term:
        layer += "    bias_term: true\n"
        layer += "    bias_filler {\n"
        layer += "      value: 0.0\n"
        layer += "    }\n"
    layer += "  }\n"
    layer += "}"
    return layer, top


def eltwise(name, bottom, top=None):
    if not top:
        top=name
    if not isinstance(bottom, list):
        raise Exception("bottom must be list")
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"Eltwise\"\n"
    for bottom_name in bottom:
        layer += "  bottom: \"" + bottom_name + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "}"
    return layer, top


def mobilenetv2block(name, bottom, num_input, num_output, kernel_size=3, expansion=6, top=None, bias_term_conv=False, bias_term_scale=True, pad=0, stride=1, group=None, train=True, relu_type="ReLU6", weight_filler="msra"):
    layer = ""
    # exp
    if not top:
        top_conv = name + "/exp"
    else:
        top_conv = top + "/exp"
    name_exp = name + "/exp"
    name_exp_bn = name + "/exp/bn"
    name_exp_scale = name + "/exp/scale"
    name_exp_relu = name + "/exp/relu"
    temp_layer, top_name = conv(name_exp, bottom, num_input*expansion, 1, top=top_conv, bias_term=bias_term_conv, weight_filler=weight_filler)
    layer += temp_layer + "\n"
    temp_layer, top_name = bn(name_exp_bn, top_name, top=name_exp_bn, train=train)
    layer += temp_layer + "\n"
    temp_layer, top_name = scale(name_exp_scale, top_name, top=name_exp_bn, bias_term=bias_term_scale)
    layer += temp_layer + "\n"
    temp_layer, top_name = relu(name_exp_relu, top_name, top=name_exp_bn, type=relu_type)
    layer += temp_layer + "\n"
    # dw
    if not top:
        top_dw = name + "/dw"
    else:
        top_dw = top + "/dw"
    if not group:
        group = num_input*expansion
    name_dw = name + "/dw"
    name_dw_bn = name + "/dw/bn"
    name_dw_scale = name + "/dw/scale"
    name_dw_relu = name + "/dw/relu"
    temp_layer, top_name = dwconv(name_dw, top_name, num_input*expansion, kernel_size, top=top_dw, bias_term=bias_term_conv, pad=1, stride=stride, group=group, weight_filler=weight_filler)
    layer += temp_layer + "\n"
    temp_layer, top_name = bn(name_dw_bn, top_name, top=name_dw_bn, train=train)
    layer += temp_layer + "\n"
    temp_layer, top_name = scale(name_dw_scale, top_name, top=name_dw_bn, bias_term=bias_term_scale)
    layer += temp_layer + "\n"
    temp_layer, top_name = relu(name_dw_relu, top_name, top=name_dw_bn, type=relu_type)
    layer += temp_layer + "\n"
    # line
    if not top:
        top_line = name + "/line"
    else:
        top_line = top + "/line"
    name_line = name + "/line"
    name_line_bn = name + "/line/bn"
    name_line_scale = name + "/line/scale"
    temp_layer, top_name = conv(name_line, top_name, num_output, 1, top=top_line, bias_term=bias_term_conv, weight_filler=weight_filler)
    layer += temp_layer + "\n"
    temp_layer, top_name = bn(name_line_bn, top_name, top=name_line_bn, train=train)
    layer += temp_layer + "\n"
    temp_layer, top_name = scale(name_line_scale, top_name, top=name_line_bn, bias_term=bias_term_scale)
    layer += temp_layer + "\n"
    if stride == 1:
        temp_layer, top_name = eltwise(name + "/add", [bottom, top_name])
        layer += temp_layer + "\n"
    return layer, top_name


def conv_block(name, bottom, num_output, kernel_size, top=None, bias_term_conv=False, bias_term_scale=True, pad=0, stride=1, group=1, train=True, relu_type="ReLU6", weight_filler="msra"):
    layer = ""
    if not top:
        top_conv = name
    name_bn = name + "/bn"
    name_scale = name + "/scale"
    name_relu = name + "/relu"
    temp_layer, top_name = conv(name, bottom, num_output, kernel_size, top=top_conv, bias_term=bias_term_conv, pad=pad, stride=stride, weight_filler=weight_filler)
    layer += temp_layer + "\n"
    temp_layer, top_name = bn(name_bn, top_name, top=name_bn, train=train)
    layer += temp_layer + "\n"
    temp_layer, top_name = scale(name_scale, top_name, top=name_bn, bias_term=bias_term_scale)
    layer += temp_layer + "\n"
    temp_layer, top_name = relu(name_relu, top_name, top=name_bn, type=relu_type)
    layer += temp_layer + "\n"
    return layer, top_name


def accuracy(name, bottom, top=None):
    if not top:
        top=name
    if not isinstance(bottom, list):
        raise Exception("bottom must be list")
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"Accuracy\"\n"
    for bottom_name in bottom:
        layer += "  bottom: \"" + bottom_name + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "  include {\n"
    layer += "    phase: TEST\n"
    layer += "  }\n"
    layer += "}"
    return layer, top


def softmax_with_loss(name, bottom, top=None):
    if not top:
        top=name
    if not isinstance(bottom, list):
        raise Exception("bottom must be list")
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"SoftmaxWithLoss\"\n"
    for bottom_name in bottom:
        layer += "  bottom: \"" + bottom_name + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "}"
    return layer, top


def softmax(name, bottom, top=None):
    if not top:
        top=name
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"Softmax\"\n"
    layer += "  bottom: \"" + bottom + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "}"
    return layer, top


def eye_cls_backbone(train=True):
    temp_layer, top_name = conv_block("conv1", "data", 16, 3, bias_term_conv=False, bias_term_scale=True, relu_type="ReLU6", pad=1, train=train, weight_filler="msra")
    net = temp_layer + "\n"
    temp_layer, top_name = conv_block("conv2_1", top_name, 32, 3, bias_term_conv=False, bias_term_scale=True, relu_type="ReLU6", pad=1, stride=2, train=train, weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = conv_block("conv2_2", top_name, 32, 3, bias_term_conv=False, bias_term_scale=True, relu_type="ReLU6", pad=1, train=train, weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = conv_block("conv2_3", top_name, 32, 3, bias_term_conv=False, bias_term_scale=True, relu_type="ReLU6", pad=1, train=train, weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = conv_block("conv3_1", top_name, 64, 3, bias_term_conv=False, bias_term_scale=True, relu_type="ReLU6", pad=1, stride=2, train=train, weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = conv_block("conv3_2", top_name, 64, 3, bias_term_conv=False, bias_term_scale=True, relu_type="ReLU6", pad=1, train=train, weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = conv_block("conv3_3", top_name, 64, 3, bias_term_conv=False, bias_term_scale=True, relu_type="ReLU6", pad=1, train=train, weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = conv_block("conv4_1", top_name, 64, 3, bias_term_conv=False, bias_term_scale=True, relu_type="ReLU6", pad=1, stride=2, train=train, weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = conv_block("conv4_2", top_name, 64, 3, bias_term_conv=False, bias_term_scale=True, relu_type="ReLU6", pad=1, train=train, weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = conv_block("conv4_3", top_name, 64, 3, bias_term_conv=False, bias_term_scale=True, relu_type="ReLU6", pad=1, train=train, weight_filler="msra")
    net += temp_layer + "\n"
    temp_layer, top_name = conv("conv5", top_name, 2, 3, bias_term=True, weight_filler="msra")
    net += temp_layer + "\n"
    return net, top_name


def eye_cls_net():
    net_train_val = "name: \"EYENet\"\n"
    net_deploy = "name: \"EYENet\"\n"

    temp_layer_deploy, top_name_deploy = deploy_data(shape=[1, 1, 24, 24])
    net_deploy += temp_layer_deploy + "\n"

    source_train = "/home/ubuntu/disk_b/tanghy/blink_datasets/train.txt"
    source_test = "/home/ubuntu/disk_b/tanghy/blink_datasets/test.txt"
    temp_layer, top_name = image_data(source_train, 24, 24, mirror=True, batch_size=256, is_color=False, shuffle=True, phase="TRAIN")
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


def face_live_cls_net():
    net_train_val = "name: \"FaceLiveNet\"\n"
    net_deploy = "name: \"FaceLiveNet\"\n"

    temp_layer_deploy, top_name_deploy = deploy_data(shape=[1, 1, 24, 24])
    net_deploy += temp_layer_deploy + "\n"

    source_train = "/home/ubuntu/disk_b/tanghy/BGR_IR_FaceDataSet/train.txt"
    source_test = "/home/ubuntu/disk_b/tanghy/BGR_IR_FaceDataSet/test.txt"
    temp_layer, top_name = image_data(source_train, 24, 24, mirror=True, batch_size=256, is_color=False, shuffle=True, phase="TRAIN")
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


def main():
    # print(conv("conv1", "data", 32, 3, top=None, bias_term=True, pad=0, stride=1, group=1, weight_filler="msra"))
    # print(dwconv("conv1", "data", 32, 3, top=None, bias_term=True, pad=0, stride=1, group=None, weight_filler="msra"))
    # print(pool("pool1", "conv1", 3, pool="MAX", global_pooling=False))
    # print(fc("fc1", "conv1", 136, bias_term=True, weight_filler="gaussian"))
    # print(relu("relu1", "conv1", "conv1", type="ReLU6"))
    # print(bn("bn1", "conv1", "conv1"))
    # print(scale("scale1", "conv1", "conv1", bias_term=True))
    # print(eltwise("add1", ["conv1", "conv2"], top=None))
    # print(mobilenetv2block("block1", "conv1", 64, 64, kernel_size=3, expansion=2, top=None, bias_term_conv=False, bias_term_scale=True, pad=0, stride=1, group=None, train=True, relu_type="ReLU6", weight_filler="msra"))
    # train_val, deploy = eye_cls_net()
    train_val, deploy = face_live_cls_net()
    train_val_path = "./face_live/train_val.prototxt"
    deploy_path = "./face_live/deploy.prototxt"
    with open(train_val_path, "w") as fp:
        fp.write(train_val)
    with open(deploy_path, "w") as fp:
        fp.write(deploy)


if __name__ == '__main__':
    main()

