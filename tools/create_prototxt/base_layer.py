# -*- coding: utf-8 -*-


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
        top = name
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
        top = name
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


def concat(name, bottom, top=None, axis=1):
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
    layer += "  concat_param {\n"
    layer += "    axis: " + str(axis) + "\n"
    layer += "  }\n"
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


def permute(name, bottom, order, top=None):
    if not top:
        top = name
    if not isinstance(order, list):
        raise Exception("bottom must be list")
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"Permute\"\n"
    layer += "  bottom: \"" + bottom + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "  permute_param {\n"
    for order_item in order:
        layer += "    order: \"" + str(order_item) + "\"\n"
    layer += "  }\n"
    layer += "}"
    return layer, top


def flatten(name, bottom, top=None):
    if not top:
        top = name
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"Flatten\"\n"
    layer += "  bottom: \"" + bottom + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "  flatten_param {\n"
    layer += "    axis: 1\n"
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


def prior_box(name, bottom, min_size, max_size, aspect_ratio, variance, step, top=None):
    if not isinstance(bottom, list):
        raise Exception("bottom must be list")
    if not isinstance(aspect_ratio, list):
        raise Exception("aspect_ratio must be list")
    if not isinstance(variance, list):
        raise Exception("variance must be list")
    if not top:
        top = name
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"PriorBox\"\n"
    for bottom_name in bottom:
        layer += "  bottom: \"" + bottom_name + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "  prior_box_param {\n"
    layer += "    min_size: " + str(min_size) + "\n"
    layer += "    max_size: " + str(max_size) + "\n"
    for aspect_ratio_item in aspect_ratio:
        layer += "    aspect_ratio: " + str(aspect_ratio_item) + "\n"
    layer += "    flip: true\n"
    layer += "    clip: false\n"
    for variance_item in variance:
        layer += "    variance: " + str(variance_item) + "\n"
    layer += "    step: " + str(step) + "\n"
    layer += "    offset: 0.5\n"
    layer += "  }\n"
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
    layer += "    t: " + str(t) + "\n"
    layer += "  }\n"
    layer += "}"
    return layer, top


def add_margin(name, bottom, top=None, m1=1, m2=0.35, m3=0.5):
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


def detection_output():
    pass


def test_layer():
    print(conv("conv1", "data", 32, 3, top=None, bias_term=True, pad=0, stride=1, group=1, weight_filler="msra")[0])
    print(dwconv("conv1", "data", 32, 3, top=None, bias_term=True, pad=0, stride=1, group=None, weight_filler="msra")[0])
    print(pool("pool1", "conv1", 3, pool="MAX", global_pooling=False)[0])
    print(fc("fc1", "conv1", 136, bias_term=True, weight_filler="gaussian")[0])
    print(bn("bn1", "conv1", "conv1")[0])
    print(scale("scale1", "conv1", "conv1", bias_term=True)[0])
    print(eltwise("add1", ["conv1", "conv2"], top=None)[0])
    print(slice("slice1", "conv1", ["conv1_1", "conv1_2"], 58, axis=1)[0])


if __name__ == '__main__':
    test_layer()
