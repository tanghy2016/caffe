# -*- coding: utf-8 -*-


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


def softmax(name, bottom, top=None, axis=-1):
    if not top:
        top = name
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"Softmax\"\n"
    layer += "  bottom: \"" + bottom + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    if axis > 0:
        layer += "  softmax_param {\n"
        layer += "    axis: " + str(axis) + "\n"
        layer += "  }\n"
    layer += "}"
    return layer, top


def sigmoid(name, bottom, top=None):
    if not top:
        top = name
    layer = "layer {\n"
    layer += "  name: \"" + name + "\"\n"
    layer += "  type: \"Sigmoid\"\n"
    layer += "  bottom: \"" + bottom + "\"\n"
    layer += "  top: \"" + top + "\"\n"
    layer += "}"
    return layer, top


def test_layer():
    layer, top = relu("relu1", "conv1", "conv1", type="ReLU6")
    print(layer)


if __name__ == '__main__':
    test_layer()

