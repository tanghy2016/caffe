# -*- coding: utf-8 -*-


def accuracy(name, bottom, top=None):
    if not top:
        top = name
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


def landmark_accuracy(name, bottom, img_size, top=None):
    if not top:
        top = name
    if not isinstance(bottom, list):
        raise Exception("bottom must be list")
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"Python\"\n"
    for bottom_name in bottom:
        layer += "  bottom: \"" + bottom_name + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "  include {\n"
    layer += "    phase: TEST\n"
    layer += "  }\n"
    layer += "  python_param {\n"
    layer += "    module: 'landmark_loss'\n"
    layer += "    layer: 'LandmarkAccuracyLayer'\n"
    layer += "    param_str: \"{'img_size': %d}\"\n" % img_size
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


def euclidean_loss(name, bottom, top=None):
    if not top:
        top = name
    if not isinstance(bottom, list):
        raise Exception("bottom must be list")
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"EuclideanLoss\"\n"
    for bottom_name in bottom:
        layer += "  bottom: \"" + bottom_name + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "}"
    return layer, top


def multi_box_loss(name, bottom, propagate_down, num_classes, top=None):
    if not top:
        top = name
    if not isinstance(bottom, list):
        raise Exception("bottom must be list")
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"MultiBoxLoss\"\n"
    for bottom_name in bottom:
        layer += "  bottom: \"" + bottom_name + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "  include: {\n"
    layer += "    phase: TRAIN\n"
    layer += "  }\n"
    for propagate_down_item in propagate_down:
        layer += "  propagate_down: \"" + propagate_down_item + "\"\n"
    layer += "  loss_param: {\n"
    layer += "    normalization: VALID\n"
    layer += "  }\n"
    layer += "  multibox_loss_param: {\n"
    layer += "    loc_loss_type: SMOOTH_L1\n"
    layer += "    conf_loss_type: SOFTMAX\n"
    layer += "    loc_weight: 1.0\n"
    layer += "    num_classes: " + str(num_classes) + "\n"
    layer += "    share_location: true\n"
    layer += "    match_type: PER_PREDICTION\n"
    layer += "    overlap_threshold: 0.5\n"
    layer += "    use_prior_for_matching: true\n"
    layer += "    background_label_id: 0\n"
    layer += "    use_difficult_gt: true\n"
    layer += "    neg_pos_ratio: 3.0\n"
    layer += "    neg_overlap: 0.5\n"
    layer += "    code_type: CENTER_SIZE\n"
    layer += "    ignore_cross_boundary_bbox: false\n"
    layer += "    mining_type: MAX_NEGATIVE\n"
    layer += "  }\n"
    layer += "}"
    return layer, top
