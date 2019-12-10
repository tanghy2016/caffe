# -*- coding: utf-8 -*-


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
               crop_size=0, mirror=False, mean_file="", mean_value=[], scale_v=-1.0,
               is_color=True, shuffle=False, phase="TRAIN"):
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"ImageData\"\n"
    for top_name in top:
        layer += "  top: \"" + top_name + "\"\n"
    layer += "  include {\n"
    layer += "    phase: " + phase + "\n"
    layer += "  }\n"
    if not mirror and crop_size == 0 and mean_file == "" and len(mean_value) == 0 and scale_v < 0:
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
        if scale_v > 0:
            layer += "    scale: " + str(scale_v) + "\n"
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


def landmark5_data(img_w, img_h, label_file,
                   name="data", top=["data", "landmark"], batch_size=64, root_folder="", phase="TRAIN"):
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"Python\"\n"
    for top_name in top:
        layer += "  top: \"" + top_name + "\"\n"
    layer += "  include {\n"
    layer += "    phase: " + phase + "\n"
    layer += "  }\n"
    layer += "  python_param {\n"
    layer += "    module: landmark_loss\n"
    layer += "    layer: LandmarkAccuracyLayer\n"
    param_str = "'img_root': \"" + root_folder + "\""
    param_str += "'label_file': \"" + label_file + "\""
    param_str += "'batch_size': %d" % batch_size
    param_str += "'img_size': [%d, %d]" % (img_w, img_h)
    if phase == "TRAIN":
        param_str += "'is_train': train"
    else:
        param_str += "'is_train': test"
    layer += "    param_str: " + param_str + "\n"
    layer += "  }\n"
    layer += "}"
    return layer, top


def annotated_data(source, label_map_file, batch_sampler, new_height, new_width, name="data", data_type="LMDB",
                   top=["data", "label"], batch_size=64, phase="TRAIN",
                   crop_size=0, mirror=False, mean_file="", mean_value=[], scale_v=-1.0,
                   resize_b=True, emit_b=True, distort_b=True, expand_b=True):
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"AnnotatedData\"\n"
    for top_name in top:
        layer += "  top: \"" + top_name + "\"\n"
    layer += "  include {\n"
    layer += "    phase: " + phase + "\n"
    layer += "  }\n"
    if not mirror and crop_size == 0 and mean_file == "" and len(mean_value) == 0 and scale_v < 0\
            and not resize_b and not emit_b and not distort_b:
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
        if scale_v > 0:
            layer += "    scale: " + str(scale_v) + "\n"
        if resize_b:
            layer += "    resize_param {\n"
            layer += "      prob: 1\n"
            layer += "      resize_mode: WARP\n"
            layer += "      height: " + str(new_height) + "\n"
            layer += "      width: " + str(new_width) + "\n"
            layer += "      interp_mode: LINEAR\n"
            layer += "      interp_mode: AREA\n"
            layer += "      interp_mode: NEAREST\n"
            layer += "      interp_mode: CUBIC\n"
            layer += "      interp_mode: LANCZOS4\n"
            layer += "    }\n"
        if emit_b:
            layer += "    emit_constraint {\n"
            layer += "      emit_type: CENTER\n"
            layer += "    }\n"
        if distort_b:
            layer += "    distort_param {\n"
            layer += "      brightness_prob: 0.5\n"
            layer += "      brightness_delta: 32\n"
            layer += "      contrast_prob: 0.5\n"
            layer += "      contrast_lower: 0.5\n"
            layer += "      contrast_upper: 1.5\n"
            layer += "      hue_prob: 0.5\n"
            layer += "      hue_delta: 18\n"
            layer += "      saturation_prob: 0.5\n"
            layer += "      saturation_lower: 0.5\n"
            layer += "      saturation_upper: 1.5\n"
            layer += "      random_order_prob: 0.0\n"
            layer += "    }\n"
        if expand_b:
            layer += "    expand_param {\n"
            layer += "      prob: 0.5\n"
            layer += "      max_expand_ratio: 4.0\n"
            layer += "    }\n"
        layer += "  }\n"
        layer += "  data_param {\n"
        layer += "    source: \"" + source + "\"\n"
        layer += "    batch_size: " + str(batch_size) + "\n"
        layer += "    backend: " + data_type + "\n"
        layer += "  }\n"
        layer += "  annotated_data_param {\n"
        for item in batch_sampler:
            layer += "    batch_sampler {\n"
            if "sampler" in item:
                layer += "      sampler {\n"
                layer += "        min_scale: " + str(item["sampler"]["min_scale"]) + "\n"
                layer += "        max_scale: " + str(item["sampler"]["max_scale"]) + "\n"
                layer += "        min_aspect_ratio: " + str(item["sampler"]["min_aspect_ratio"]) + "\n"
                layer += "        max_aspect_ratio: " + str(item["sampler"]["max_aspect_ratio"]) + "\n"
                layer += "      }\n"
            if "sample_constraint" in item:
                layer += "      sample_constraint {\n"
                layer += "        min_jaccard_overlap: " + str(item["sample_constraint"]["min_jaccard_overlap"]) + "\n"
                layer += "      }\n"
            if "max_sample" in item:
                layer += "      max_sample: " + str(item["max_sample"]) + "\n"
            if "max_trials" in item:
                layer += "      max_trials: " + str(item["max_trials"]) + "\n"
            layer += "    }\n"
        layer += "    label_map_file: \"" + label_map_file + "\"\n"
        layer += "  }\n"
    layer += "}"
    return layer, top


def get_annotated_data(source, label_map_file, new_height, new_width, name="data", data_type="LMDB",
                       top=["data", "label"], batch_size=64, phase="TRAIN",
                       crop_size=0, mirror=False, mean_file="", mean_value=[], scale_v=-1.0,
                       resize_b=True, emit_b=True, distort_b=True, expand_b=True):
    batch_sampler = []
    batch_sampler_item = {
        "max_sample": 1,
        "max_trials": 1
    }
    batch_sampler.append(batch_sampler_item)

    for i in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        batch_sampler_item = {
            "sampler": {
                "min_scale": 0.3,
                "max_scale": 1.0,
                "min_aspect_ratio": 0.5,
                "max_aspect_ratio": 2.0
            },
            "sample_constraint": {
                "min_jaccard_overlap": i
            },
            "max_sample": 1,
            "max_trials": 50
        }
        batch_sampler.append(batch_sampler_item)
    return annotated_data(source, label_map_file, batch_sampler, new_height, new_width, name=name,
                          phase=phase, batch_size=batch_size, top=top, data_type=data_type,
                          crop_size=crop_size, mirror=mirror, mean_file=mean_file, mean_value=mean_value,
                          scale_v=scale_v, resize_b=resize_b, emit_b=emit_b, distort_b=distort_b, expand_b=expand_b)


def test_layer():
    source = ""
    label_map_file = ""
    new_height = 320
    new_width = 320
    batch_size = 64
    data_layer, top_layer = get_annotated_data(source, label_map_file, new_height, new_width, name="data",
                                               data_type="LMDB",
                                               top=["data", "label"], batch_size=batch_size, phase="TRAIN",
                                               crop_size=0, mirror=False, mean_file="", mean_value=[], scale_v=-1.0,
                                               resize_b=True, emit_b=True, distort_b=True, expand_b=True)
    print(data_layer)


if __name__ == '__main__':
    test_layer()
