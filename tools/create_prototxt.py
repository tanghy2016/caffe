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


def image_data(source, new_height, new_width, name="data", top=["data", "label"], batch_size=64, root_folder="",
               crop_size=0, mirror=False, mean_file="", mean_value=[], scale=-1.0,
               is_color=True, shuffle=False, phase="TRAIN"):
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"ImageData\"\n"
    for top_name in top:
        layer += "  top: \"" + top_name + "\"\n"
    layer += "  include {\n"
    layer += "    phase: " + phase + "\n"
    layer += "  }\n"
    if not mirror and crop_size == 0 and mean_file == "" and len(mean_value) == 0:
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
        if scale > 0:
            layer += "    scale: " + str(scale) + "\n"
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


def conv(name, bottom, num_output, kernel_size, top=None, bias_term=True, pad=0, stride=1, group=1,
         w_decay_mult=1, weight_filler="msra"):
    if not top:
        top = name
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"Convolution\"\n"
    layer += "  bottom: \"" + bottom + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "  param {\n"
    layer += "    lr_mult: 1\n"
    if w_decay_mult == 1:
        layer += "    decay_mult: 1\n"
    else:
        layer += "    decay_mult: " + str(w_decay_mult) + "\n"
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


def dwconv(name, bottom, num_output, kernel_size,
           top=None, bias_term=True, pad=0, stride=1,
           group=None, w_decay_mult=1, weight_filler="msra"):
    if not top:
        top = name
    if not group:
        group = num_output
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"DepthwiseConvolution\"\n"
    layer += "  bottom: \"" + bottom + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "  param {\n"
    layer += "    lr_mult: 1\n"
    if w_decay_mult == 1:
        layer += "    decay_mult: 1\n"
    else:
        layer += "    decay_mult: " + str(w_decay_mult) + "\n"
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
    if weight_filler == "msra":
        layer += "    weight_filler {\n"
        layer += "      type: \"msra\"\n"
        layer += "    }\n"
    elif weight_filler == "gaussian":
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


def fc(name, bottom, num_output, top=None, bias_term=True, w_decay_mult=1, weight_filler="msra", normalize=False):
    if not top:
        top = name
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"InnerProduct\"\n"
    layer += "  bottom: \"" + bottom + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "  param {\n"
    layer += "    lr_mult: 1\n"
    if w_decay_mult == 1:
        layer += "    decay_mult: 1\n"
    else:
        layer += "    decay_mult: " + str(w_decay_mult) + "\n"
    layer += "  }\n"
    if bias_term:
        layer += "  param {\n"
        layer += "    lr_mult: 2\n"
        layer += "    decay_mult: 0\n"
        layer += "  }\n"
    layer += "  inner_product_param {\n"
    layer += "    num_output: " + str(num_output) + "\n"
    if normalize:
        layer += "    normalize: true\n"
    if not bias_term:
        layer += "    bias_term: false\n"
    if weight_filler == "msra":
        layer += "    weight_filler {\n"
        layer += "      type: \"msra\"\n"
        layer += "    }\n"
    elif weight_filler == "gaussian":
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


def scale(name, bottom, top, type_s=None, w_value=1, w_lr_mult=1, w_decay_mult=1,
          bias_term=False, b_lr_mult=1, b_decay_mult=1):
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"Scale\"\n"
    layer += "  bottom: \"" + bottom + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    if w_lr_mult != 1 or w_decay_mult != 1:
        layer += "  param {\n"
        layer += "    lr_mult: " + str(w_lr_mult) + "\n"
        layer += "    decay_mult: " + str(w_decay_mult) + "\n"
        layer += "  }\n"
    if bias_term and (b_lr_mult != 1 or b_decay_mult != 1):
        layer += "  param {\n"
        layer += "    lr_mult: " + str(b_lr_mult) + "\n"
        layer += "    decay_mult: " + str(b_decay_mult) + "\n"
        layer += "  }\n"
    layer += "  scale_param {\n"
    layer += "    filler {\n"
    if type_s == "constant":
        layer += "      type: \"" + type_s + "\"\n"
    layer += "      value: " + str(w_value) + "\n"
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


def normalize(name, bottom, top=None):
    if not top:
        top = name
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"Normalize\"\n"
    layer += "  bottom: \"" + bottom + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "}"
    return layer, top


def concat(name, bottom, top=None):
    if not top:
        top = name
    if not isinstance(bottom, list):
        raise Exception("bottom must be list")
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"Concat\"\n"
    for bottom_name in bottom:
        layer += "  bottom: \"" + bottom_name + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "}"
    return layer, top


def slice(name, bottom, top, slice_point, axis=1):
    if not isinstance(top, list):
        raise Exception("bottom must be list")
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"Slice\"\n"
    layer += "  bottom: \"" + bottom + "\"\n"
    for top_name in top:
        layer += "  top: \"" + top_name + "\"\n"
    layer += "  slice_param {\n"
    layer += "    slice_point: " + str(slice_point) + "\n"
    if axis != 1:
        layer += "    axis: " + str(axis) + "\n"
    layer += "  }\n"
    layer += "}"
    return layer, top


def shuffle_channel(name, bottom, top=None, group=2):
    if not top:
        top = name
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"ShuffleChannel\"\n"
    layer += "  bottom: \"" + bottom + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "  shuffle_channel_param {\n"
    layer += "    group: " + str(group) + "\n"
    layer += "  }\n"
    layer += "}"
    return layer, top


def mobilenetv2block(name, bottom, num_input, num_output, kernel_size=3, expansion=6,
                     top=None, bias_term_conv=False, bias_term_scale=True,
                     stride=1, group=None, train=True, relu_type="ReLU6", weight_filler="msra"):
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


def conv_block(name, bottom, num_output, kernel_size, top=None, blob_same=True,
               bias_term_conv=False, bias_term_scale=True,
               pad=0, stride=1, group=1, train=True, relu_type="ReLU6", c_w_decay_mult=1, weight_filler="msra",
               type_s=None):
    layer = ""
    if not top:
        top_conv = name
    name_bn = name + "/bn"
    name_scale = name + "/scale"
    name_relu = name + "/relu"
    if group == 1:
        temp_layer, top_name = conv(name, bottom, num_output, kernel_size, top=top_conv, bias_term=bias_term_conv,
                                    pad=pad, stride=stride, w_decay_mult=c_w_decay_mult, weight_filler=weight_filler)
    elif group > 1:
        temp_layer, top_name = dwconv(name, bottom, num_output, kernel_size, top=top_conv, bias_term=bias_term_conv,
                                      pad=pad, stride=stride, w_decay_mult=c_w_decay_mult, weight_filler=weight_filler)
    layer += temp_layer + "\n"
    if blob_same:
        temp_layer, top_name = bn(name_bn, top_name, top=name, train=train)
    else:
        temp_layer, top_name = bn(name_bn, top_name, top=name_bn, train=train)
    layer += temp_layer + "\n"
    if blob_same:
        temp_layer, top_name = scale(name_scale, top_name, top=name,
                                     type_s=type_s, w_value=1, w_lr_mult=1, w_decay_mult=0,
                                     bias_term=bias_term_scale, b_lr_mult=1, b_decay_mult=0)
    else:
        temp_layer, top_name = scale(name_scale, top_name, top=name_scale,
                                     type_s=type_s, w_value=1, w_lr_mult=1, w_decay_mult=0,
                                     bias_term=bias_term_scale, b_lr_mult=1, b_decay_mult=0)
    layer += temp_layer + "\n"
    if len(relu_type) > 0:
        if blob_same:
            temp_layer, top_name = relu(name_relu, top_name, top=name, type=relu_type)
        else:
            temp_layer, top_name = relu(name_relu, top_name, top=name_relu, type=relu_type)
        layer += temp_layer + "\n"
    return layer, top_name


# param:
#   num_input: number of input channel
#   num_output: number of right branch channel
def shufflev2_block(bottom, top_num, num_input, num_output, stride=1,
                    bias_term_conv=False, bias_term_scale=True,
                    train=True, relu_type="ReLU", weight_filler="msra"):
    layer = ""
    layer_left = ""
    layer_right = ""

    # get left branch
    if stride == 1:
        if int(num_input/2)*2 != num_input:
            raise Exception("num_input must be an integer multiple of 2")
        temp_top = ["branch"+str(top_num)+"_1", "branch"+str(top_num)+"_2"]
        slice_name = "slice" + str(top_num)
        temp_layer, top = slice(slice_name, bottom, temp_top, int(num_input/2), axis=1)
        layer += temp_layer + "\n"
        top_left = top[0]
        top_right = top[1]
    elif stride == 2:
        top_left = bottom
        top_right = bottom
        conv_name = "branch" + str(top_num) + "_1_conv1"
        temp_layer, top_left = conv_block(conv_name, top_left, num_input, 3, pad=1, stride=stride, group=num_input,
                                          bias_term_conv=bias_term_conv, bias_term_scale=bias_term_scale,
                                          blob_same=True, train=train, relu_type="", weight_filler=weight_filler)
        layer_left += temp_layer + "\n"
        conv_name = "branch" + str(top_num) + "_1_conv2"
        temp_layer, top_left = conv_block(conv_name, top_left, num_output, 1,
                                          bias_term_conv=bias_term_conv, bias_term_scale=bias_term_scale,
                                          blob_same=True, train=train, relu_type=relu_type, weight_filler=weight_filler)
        layer_left += temp_layer + "\n"
    else:
        raise Exception("stride error in shuffle-v2 block, must in {1, 2}")
    if len(layer_left) > 0:
        layer += layer_left + "\n"

    # get right branch
    conv_name = "branch" + str(top_num) + "_2_conv1"
    temp_layer, top_right = conv_block(conv_name, top_right, num_output, 1,
                                       bias_term_conv=bias_term_conv, bias_term_scale=bias_term_scale,
                                       blob_same=True, train=train, relu_type=relu_type, weight_filler=weight_filler)
    layer_right += temp_layer + "\n"
    conv_name = "branch" + str(top_num) + "_2_conv2"
    temp_layer, top_right = conv_block(conv_name, top_right, num_output, 3, pad=1, stride=stride, group=num_output,
                                       bias_term_conv=bias_term_conv, bias_term_scale=bias_term_scale,
                                       blob_same=True, train=train, relu_type="", weight_filler=weight_filler)
    layer_right += temp_layer + "\n"
    conv_name = "branch" + str(top_num) + "_2_conv3"
    temp_layer, top_right = conv_block(conv_name, top_right, num_output, 1,
                                       bias_term_conv=bias_term_conv, bias_term_scale=bias_term_scale,
                                       blob_same=True, train=train, relu_type=relu_type, weight_filler=weight_filler)
    layer_right += temp_layer + "\n"
    layer += layer_right + "\n"

    concat_name = "concat" + str(top_num)
    temp_layer, top = concat(concat_name, [top_left, top_right])
    layer += temp_layer + "\n"
    shuffle_name = "shuffle" + str(top_num)
    temp_layer, top = shuffle_channel(shuffle_name, top, group=2)
    layer += temp_layer + "\n"
    return layer, top


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
        top = name
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
        top = name
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"Softmax\"\n"
    layer += "  bottom: \"" + bottom + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "}"
    return layer, top


def sv_x(name, bottom, top=None, m1=1, m2=0.35, m3=0.5, t=1.2):
    if not top:
        top = name
    if not isinstance(bottom, list):
        raise Exception("bottom must be list")
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"SVX\"\n"
    for bottom_name in bottom:
        layer += "  bottom: \"" + bottom_name + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "  sv_x_param: {\n"
    if abs(m1 - 1) > 0.00001:
        layer += "    m1: " + str(m1) + "\n"
    if abs(m2 - 0.35) > 0.00001:
        layer += "    m2: " + str(m2) + "\n"
    if abs(m3 - 0.5) > 0.00001:
        layer += "    m3: " + str(m3) + "\n"
    if abs(t - 1.2) > 0.00001:
        layer += "    t: " + str(t) + "\n"
    layer += "  }\n"
    layer += "}"
    return layer, top


def add_margin(name, bottom, top=None, m1=1, m2=0.35, m3=0.5, t=1.2):
    if not top:
        top = name
    if not isinstance(bottom, list):
        raise Exception("bottom must be list")
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"AddMargin\"\n"
    for bottom_name in bottom:
        layer += "  bottom: \"" + bottom_name + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "  add_margin_param: {\n"
    if abs(m1 - 1) > 0.00001:
        layer += "    m1: " + str(m1) + "\n"
    if abs(m2 - 0.35) > 0.00001:
        layer += "    m2: " + str(m2) + "\n"
    if abs(m3 - 0.5) > 0.00001:
        layer += "    m3: " + str(m3) + "\n"
    layer += "  }\n"
    layer += "}"
    return layer, top


def test_base_layer():
    print(conv("conv1", "data", 32, 3, top=None, bias_term=True, pad=0, stride=1, group=1, weight_filler="msra"))
    print(dwconv("conv1", "data", 32, 3, top=None, bias_term=True, pad=0, stride=1, group=None, weight_filler="msra"))
    print(pool("pool1", "conv1", 3, pool="MAX", global_pooling=False))
    print(fc("fc1", "conv1", 136, bias_term=True, weight_filler="gaussian"))
    print(relu("relu1", "conv1", "conv1", type="ReLU6"))
    print(bn("bn1", "conv1", "conv1"))
    print(scale("scale1", "conv1", "conv1", bias_term=True))
    print(eltwise("add1", ["conv1", "conv2"], top=None))
    print(slice("slice1", "conv1", ["conv1_1", "conv1_2"], 58, axis=1)[0])
    print(shuffle_channel("shuffle1", "resx2_conv1", "shuffle1", group=2)[0])
    print(shufflev2_block("shuffle5", 6, 116*2, stride=2,
                          bias_term_conv=False, bias_term_scale=True,
                          train=True, relu_type="ReLU", weight_filler="msra")[0])
    print(mobilenetv2block("block1", "conv1", 64, 64, kernel_size=3, expansion=2, top=None,
                           bias_term_conv=False, bias_term_scale=True, pad=0, stride=1,
                           group=None, train=True, relu_type="ReLU6", weight_filler="msra"))


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
    temp_layer, top_name = fc("fc5", top_name, 128, top=None, bias_term=False, w_decay_mult=10,
                              weight_filler="msra", normalize=False)
    net += temp_layer + "\n"
    return net, top_name


def shuffle_facenet(source_train, source_test, class_num, type_margin):
    net_train = "name: \"EYENet\"\n"
    net_val = "name: \"EYENet\"\n"
    net_deploy = "name: \"EYENet\"\n"

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
    temp_layer, top_name = fc("fc6_l2", top_name, class_num, top=None, bias_term=True,
                              w_decay_mult=10, weight_filler="msra", normalize=True)
    net_train += temp_layer + "\n"
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

    temp_layer, _ = accuracy("accuracy", [top_name, "label"])
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


def main():
    # test_base_layer()

    # train_val, deploy = eye_cls_net()

    # train_val, deploy = face_live_cls_net()
    # train_val_path = "./face_live/train_val.prototxt"
    # deploy_path = "./face_live/deploy.prototxt"
    # with open(train_val_path, "w") as fp:
    #     fp.write(train_val)
    # with open(deploy_path, "w") as fp:
    #     fp.write(deploy)

    # -------------ShuffleFaceNet-----------------
    get_shuffle_v2_facenet()


if __name__ == '__main__':
    main()

