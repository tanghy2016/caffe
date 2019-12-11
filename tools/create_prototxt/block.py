# -*- coding: utf-8 -*-

from tools.create_prototxt.base_layer import *
from tools.create_prototxt.active_func import *


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
    temp_layer, top_name = conv(name_exp, bottom, num_input * expansion, 1, top=top_conv, bias_term=bias_term_conv,
                                weight_filler=weight_filler)
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
        group = num_input * expansion
    name_dw = name + "/dw"
    name_dw_bn = name + "/dw/bn"
    name_dw_scale = name + "/dw/scale"
    name_dw_relu = name + "/dw/relu"
    temp_layer, top_name = dwconv(name_dw, top_name, num_input * expansion, kernel_size, top=top_dw,
                                  bias_term=bias_term_conv, pad=1, stride=stride, group=group,
                                  weight_filler=weight_filler)
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
    temp_layer, top_name = conv(name_line, top_name, num_output, 1, top=top_line, bias_term=bias_term_conv,
                                weight_filler=weight_filler)
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
               pad=0, stride=1, group=1, bias_term_conv=False,
               has_bn=True, train=True, bias_term_scale=True, type_s=None,
               relu_type="ReLU6", c_w_decay_mult=1, weight_filler="msra"):
    layer = ""
    if not top:
        top_conv = name
    name_bn = name + "/bn"
    name_scale = name + "/scale"
    name_relu = name + "/relu"
    if not has_bn:
        bias_term_conv = True
    if group == 1:
        temp_layer, top_name = conv(name, bottom, num_output, kernel_size, top=top_conv, bias_term=bias_term_conv,
                                    pad=pad, stride=stride, w_decay_mult=c_w_decay_mult, weight_filler=weight_filler)
    elif group > 1:
        temp_layer, top_name = dwconv(name, bottom, num_output, kernel_size, top=top_conv, bias_term=bias_term_conv,
                                      pad=pad, stride=stride, w_decay_mult=c_w_decay_mult, weight_filler=weight_filler)
    layer += temp_layer + "\n"
    if has_bn:
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
        if int(num_input / 2) * 2 != num_input:
            raise Exception("num_input must be an integer multiple of 2")
        temp_top = ["branch" + str(top_num) + "_1", "branch" + str(top_num) + "_2"]
        slice_name = "slice" + str(top_num)
        temp_layer, top = slice(slice_name, bottom, temp_top, int(num_input / 2), axis=1)
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


def half_conv_block1(name, bottom, input_num, output_num, train=True, relu_type="ReLU"):
    """branch1_2 include 1x conv3x3"""
    if (input_num / 2) * 2 != input_num:
        raise Exception("input error!")
    top_slice = [name + "/branch1_1", name + "/branch1_2"]
    temp_layer, top_name_branch = slice(name + "/slice", bottom, top_slice, int(input_num / 2))
    net_block = temp_layer + "\n"

    temp_layer, top_name = conv_block(name + "/conv3", top_name_branch[-1], int(input_num / 2), 3, pad=1, stride=1,
                                      train=train, relu_type=relu_type)
    net_block += temp_layer + "\n"

    temp_layer, top_name = concat(name + "/cat", [top_name_branch[0], top_name])
    net_block += temp_layer + "\n"

    temp_layer, top_name = conv_block(name + "/conv1", top_name, output_num, 1, stride=1,
                                      train=train, relu_type=relu_type)
    net_block += temp_layer
    return net_block, top_name


def half_conv_block2(name, bottom, input_num, output_num, train=True, relu_type="ReLU"):
    """branch1_2 include 2x conv3x3"""
    if (input_num / 2) * 2 != input_num:
        raise Exception("input error!")
    top_slice = [name + "/branch1_1", name + "/branch1_2"]
    temp_layer, top_name_branch = slice(name + "/slice", bottom, top_slice, int(input_num / 2))
    net_block = temp_layer + "\n"

    temp_layer, top_name = conv_block(name + "/conv3_1", top_name_branch[-1], int(input_num / 2), 3, pad=1, stride=1,
                                      train=train, relu_type=relu_type)
    net_block += temp_layer + "\n"

    temp_layer, top_name = conv_block(name + "/conv3_2", top_name, int(input_num / 2), 3, pad=1, stride=1,
                                      train=train, relu_type=relu_type)
    net_block += temp_layer + "\n"

    temp_layer, top_name = concat(name + "/cat", [top_name_branch[0], top_name])
    net_block += temp_layer + "\n"

    temp_layer, top_name = conv_block(name + "/conv1", top_name, output_num, 1, stride=1,
                                      train=train, relu_type=relu_type)
    net_block += temp_layer
    return net_block, top_name


def test_layer():
    print(shuffle_channel("shuffle1", "resx2_conv1", "shuffle1", group=2)[0])
    # print(shufflev2_block("shuffle5", 6, 116*2, stride=2,
    #                       bias_term_conv=False, bias_term_scale=True,
    #                       train=True, relu_type="ReLU", weight_filler="msra")[0])
    # print(mobilenetv2block("block1", "conv1", 64, 64, kernel_size=3, expansion=2, top=None,
    #                        bias_term_conv=False, bias_term_scale=True, pad=0, stride=1,
    #                        group=None, train=True, relu_type="ReLU6", weight_filler="msra")[0])
    print(half_conv_block1("stage2", "stage2", 64, 64)[0])
    print(half_conv_block2("stage2", "stage2", 64, 64)[0])


if __name__ == '__main__':
    test_layer()
