# -*- coding: utf-8 -*-

from tools.create_prototxt.data_layer import *
from tools.create_prototxt.loss_layer import *
from tools.create_prototxt.backbone import *


def my_face_det_net():
    net_train = "name: \"FaceDetNet\"\n"
    net_val = "name: \"FaceDetNet\"\n"
    net_deploy = "name: \"FaceDetNet\"\n"

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
    net_train += data_layer + "\n"
    data_layer, top_layer = get_annotated_data(source, label_map_file, new_height, new_width, name="data",
                                               data_type="LMDB",
                                               top=["data", "label"], batch_size=batch_size, phase="TEST",
                                               crop_size=0, mirror=False, mean_file="", mean_value=[], scale_v=-1.0,
                                               resize_b=True, emit_b=True, distort_b=True, expand_b=True)
    net_val += data_layer + "\n"

    temp_layer_deploy, top_name_deploy = deploy_data(shape=[1, 3, 320, 320])
    net_deploy += temp_layer_deploy + "\n"

    net_backbone_train_val, top_name_train_val = half_net_backbone(train=True)
    net_backbone_deploy, _ = half_net_backbone(train=False)

    net_deploy += net_backbone_deploy + "\n"
    net_train += net_backbone_train_val + "\n"
    net_val += net_backbone_train_val + "\n"

    top_name = ["stage4_2/conv1", "stage5_2/conv1", "stage6_2/conv1", "stage7_2/conv1", "stage7_3"]
    # s = (0.1 + [0.2-1.05]) * 320
    min_size = [32, 64, 132, 200, 268]
    max_size = [64, 132, 200, 268, 336]
    aspect_ratio = [[2], [2], [2], [2], [2]]
    step = [8, 16, 32, 64, 320]
    box_num = [4, 4, 4, 4, 4]
    class_num = 2
    for i in range(len(top_name)):
        temp_layer, top_layer = conv_block(top_name[i] + "/mbox_loc", top_name[i], box_num[i] * 4, 3, pad=1,
                                           stride=1, train=True, relu_type="ReLU")
        net_train += temp_layer + "\n"
        net_val += temp_layer + "\n"
        net_deploy += temp_layer + "\n"
        temp_layer, top_layer = permute(top_name[i] + "/mbox_loc_perm", top_layer, [0, 2, 3, 1], top=None)
        net_train += temp_layer + "\n"
        net_val += temp_layer + "\n"
        net_deploy += temp_layer + "\n"
        temp_layer, top_layer = flatten(top_name[i] + "/mbox_loc_flat", top_layer, top=None)
        net_train += temp_layer + "\n"
        net_val += temp_layer + "\n"
        net_deploy += temp_layer + "\n"

        temp_layer, top_layer = conv_block(top_name[i] + "/mbox_conf", top_name[i], box_num[i] * class_num, 3, pad=1,
                                           stride=1, train=True, relu_type="ReLU")
        net_train += temp_layer + "\n"
        net_val += temp_layer + "\n"
        net_deploy += temp_layer + "\n"
        temp_layer, top_layer = permute(top_name[i] + "/mbox_conf_perm", top_layer, [0, 2, 3, 1], top=None)
        net_train += temp_layer + "\n"
        net_val += temp_layer + "\n"
        net_deploy += temp_layer + "\n"
        temp_layer, top_layer = flatten(top_name[i] + "/mbox_conf_flat", top_layer, top=None)
        net_train += temp_layer + "\n"
        net_val += temp_layer + "\n"
        net_deploy += temp_layer + "\n"

        temp_layer, top_layer = prior_box(top_name[i] + "/mbox_priorbox", [top_name[i], "data"],
                                          min_size[i], max_size[i], aspect_ratio[i],
                                          variance=[0.1, 0.1, 0.2, 0.2], step=step[i], top=None)
        net_train += temp_layer + "\n"
        net_val += temp_layer + "\n"
        net_deploy += temp_layer + "\n"

    temp_layer, top_name_loc = concat("mbox_loc", [item + "/mbox_loc_flat" for item in top_name], top=None, axis=1)
    net_train += temp_layer + "\n"
    net_val += temp_layer + "\n"
    net_deploy += temp_layer + "\n"
    temp_layer, top_name_conf = concat("mbox_conf", [item + "/mbox_conf_flat" for item in top_name], top=None, axis=1)
    net_train += temp_layer + "\n"
    net_val += temp_layer + "\n"
    net_deploy += temp_layer + "\n"
    temp_layer, top_name_conf = concat("mbox_priorbox", [item + "/mbox_priorbox" for item in top_name], top=None,
                                       axis=2)
    net_train += temp_layer + "\n"
    net_val += temp_layer + "\n"
    net_deploy += temp_layer + "\n"

    temp_layer, _ = multi_box_loss(name="mbox_loss", bottom=["mbox_loc", "mbox_conf", "mbox_priorbox", "label"],
                                   propagate_down=["true", "true", "false", "false"], num_classes=2)
    net_train += temp_layer + "\n"

    temp_layer, top_name = reshape("mbox_conf_reshape", "mbox_conf", [0, -1, 21])
    net_val += temp_layer + "\n"
    net_deploy += temp_layer + "\n"
    temp_layer, top_name = softmax("mbox_conf_softmax", top_name, axis=2)
    net_val += temp_layer + "\n"
    net_deploy += temp_layer + "\n"
    temp_layer, top_name = flatten("mbox_conf_flatten", top_name)
    net_val += temp_layer + "\n"
    net_deploy += temp_layer + "\n"

    from easydict import EasyDict
    val_parm = EasyDict()
    val_parm.output_directory = "/home-2/wliu/data/VOCdevkit/results/VOC2007/SSD_300x300/Main"
    val_parm.output_name_prefix = "comp4_det_test_"
    val_parm.output_format = "VOC"
    val_parm.label_map_file = "data/VOC0712/labelmap_voc.prototxt"
    val_parm.name_size_file = "data/VOC0712/test_name_size.txt"
    val_parm.num_test_image = 4952
    temp_layer, top_layer = detection_output("detection_out", ["mbox_loc", top_name, "mbox_priorbox"],
                                             num_classes=2, val_parm=val_parm)
    net_val += temp_layer + "\n"
    temp_layer, _ = detection_evaluate("detection_eval", [top_layer, "label"], num_classes=2,
                                       name_size_file="data/VOC0712/test_name_size.txt")
    net_val += temp_layer + "\n"

    temp_layer, _ = detection_output("detection_out", ["mbox_loc", top_name, "mbox_priorbox"], num_classes=2)
    net_deploy += temp_layer + "\n"

    return net_train, net_val, net_deploy


def main():
    train_path = "../../../caffe_model/FaceDet/train.prototxt"
    val_path = "../../../caffe_model/FaceDet/val.prototxt"
    deploy_path = "../../../caffe_model/FaceDet/deploy.prototxt"
    net_train, net_val, net_deploy = my_face_det_net()
    with open(train_path, "w") as fp:
        fp.write(net_train)
    with open(val_path, "w") as fp:
        fp.write(net_val)
    with open(deploy_path, "w") as fp:
        fp.write(net_deploy)


if __name__ == '__main__':
    main()
