# Extended Caffe

@tanghy2016

------

[toc]

------

## tools

### create_prototxt.py

- support layer
    - data_layer
        - deploy_data
        - image_data
    - Convolution
    - DepthwiseConvolution
    - Pooling
    - InnerProduct
    - ReLU
        - ReLU
        - ReLU6
        - CReLU
    - BatchNorm
    - Scale
    - Eltwise
    - SoftmaxWithLoss
    - Softmax
    - Accuracy
- support block
    - mobilenetv2block
    - conv_block

## InnerProductLayer

- add normalize [optional]

```
layer {
  ...
  inner_product_param{
    num_output: 10516
    normalize: true
    weight_filler {
      type: "xavier"
    }
    bias_term: false
  }
}
```

- add input layer [optional]
    - weight layer
    - bias layer
    
    
    ```
    layer {
      ...
      bottom: "input_x"
      bottom: "input_weight"
      bottom: "input_bias"
      ...
    }
    ```

- reference [caffe-windows][1]

## DepthwiseConvolutionLayer

- example

```
layer {
  name: "conv2_1/dw"
  type: "DepthwiseConvolution"
  bottom: "conv1"
  top: "conv2_1/dw"
  param {
    lr_mult: 1.0
    decay_mult: 1.0
  }
  convolution_param {
    num_output: 32
    bias_term: false
    pad: 1
    kernel_size: 3
    group: 32
    stride: 1
    weight_filler {
      type: "msra"
    }
  }
}
```

- reference [DepthwiseConvolution][2]

## SV-X-Layer

- example, for AM-based

```
layer {
  name: "sv_x"
  type: "SVX"
  bottom: "fc6"
  bottom: "label"
  top: "fc6_margin"
  sv_x_param {
    m1: 1
    m2: 0.35
    m3: 0
    t: 1.2
  }
}
```

- example, for Arc-based

```
layer {
  name: "sv_x"
  type: "SVX"
  bottom: "fc6"
  bottom: "label"
  top: "fc6_margin"
  sv_x_param {
    m1: 1
    m2: 0
    m3: 0.5
    t: 1.2
  }
}
```

- for gt class: $f(m, \theta_{gt}) = cos(m_1 \theta_{gt} + m_3) - m_2$
    - if $f(m, \theta_{gt}) = cos(\theta_{gt})$, then back to SV-Softmax
    - AM-based, if $m_1 = 1, m_2 = 0.35, m_3 = 0$
    - Arc-based, if $m_1 = 1, m_2 = 0, m_3 = 0.5$
- for other class:

$$
\left\{\begin{matrix} 
t cos(\theta_{other}) + t - 1 & f(m, \theta_{gt}) < cos(\theta_{other})\\ 
cos(\theta) & other
\end{matrix}\right.
$$

- param
    - m1: default = 1.0
        - don't code it
    - m2: best when m = 0.35 to m = 0.4 (paper), default = 0.35
    - m3: default = 0.5
    - t: > 1, default = 1.2
- reference [SV-X-Softmax][3]

## ReLU6Layer

- example

```
layer {
  name: "relu1"
  type: "ReLU6"
  bottom: "conv1"
  top: "conv2"
}
```

- $y = min(max(0, x), 6)$

## PFLDLossLayer

- example

```
layer {
  name: "pfld/loss"
  type: "PFLDLoss"
  bottom: "fc9/add"
  bottom: "fc2/aux"
  bottom: "landmark"
  bottom: "pose"
  bottom: "classes"
  top: "pfld/loss"
  pfld_loss_param {
    profile_face: 21.3653
    frontal_face: 1.0171
    head_up: 14.9402
    head_down: 2.5726
    expression: 6.9673
    occlusion: 1
  }
}
```

- reference [PFLD][4]

## NormalizeLayer

- L2 or L1
    - Normalize by channel
    - for fc layer, channel is num_output and spatial_dim is 1
    - for conv layer, spatial_dim is weight*height
- example

```
layer {
  name: "norm1"
  type: "Normalize"
  bottom: "fc5"
  top: "norm1"
  normalize_param {
    normalize_type: "L2"
  }
}
```

## MarginLayer

- $cos(m_1 \theta + m_3) - m_2$
- example, for arc-face

```
layer {
  name: "margin"
  type: "AddMargin"
  bottom: "fc6"
  bottom: "label"
  top: "fc6_margin"
  sv_x_param {
    m1: 1
    m2: 0
    m3: 0.5
  }
}
```

## ShuffleChannelLayer

- example

```
layer {
  name: "shuffle2"
  type: "ShuffleChannel"
  bottom: "resx2_conv1"
  top: "shuffle2"
  shuffle_channel_param {
    group: 3
  }
}
```

- the num output in bottom layer of ```ShuffleChannel``` should be divisible by param-group
- reference [ShuffleNet][5]

## SSD Layers

- 参考[caffe-ssd][6]
- ```util/bbox_util```
- ```util/sampler```
- ```util/im_transforms```
- ```util/io```
- 以及一些其他修改, 具体参考```SSD的提交```

### AnnotatedDataLayer

```
layer {
  name: "data"
  type: "AnnotatedData"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    scale: 0.007843
    mirror: true
    mean_value: 127.5
    mean_value: 127.5
    mean_value: 127.5
    resize_param {
      prob: 1.0
      resize_mode: WARP
      height: 300
      width: 300
      interp_mode: LINEAR
      interp_mode: AREA
      interp_mode: NEAREST
      interp_mode: CUBIC
      interp_mode: LANCZOS4
    }
    emit_constraint {
      emit_type: CENTER
    }
    distort_param {
      brightness_prob: 0.5
      brightness_delta: 32.0
      contrast_prob: 0.5
      contrast_lower: 0.5
      contrast_upper: 1.5
      hue_prob: 0.5
      hue_delta: 18.0
      saturation_prob: 0.5
      saturation_lower: 0.5
      saturation_upper: 1.5
      random_order_prob: 0.0
    }
    expand_param {
      prob: 0.5
      max_expand_ratio: 4.0
    }
  }
  data_param {
    source: "trainval_lmdb/"
    batch_size: 24
    backend: LMDB
  }
  annotated_data_param {
    batch_sampler {
      max_sample: 1
      max_trials: 1
    }
    batch_sampler {
      sampler {
        min_scale: 0.3
        max_scale: 1.0
        min_aspect_ratio: 0.5
        max_aspect_ratio: 2.0
      }
      sample_constraint {
        min_jaccard_overlap: 0.1
      }
      max_sample: 1
      max_trials: 50
    }
    batch_sampler {
      sampler {
        min_scale: 0.3
        max_scale: 1.0
        min_aspect_ratio: 0.5
        max_aspect_ratio: 2.0
      }
      sample_constraint {
        min_jaccard_overlap: 0.3
      }
      max_sample: 1
      max_trials: 50
    }
    batch_sampler {
      sampler {
        min_scale: 0.3
        max_scale: 1.0
        min_aspect_ratio: 0.5
        max_aspect_ratio: 2.0
      }
      sample_constraint {
        min_jaccard_overlap: 0.5
      }
      max_sample: 1
      max_trials: 50
    }
    batch_sampler {
      sampler {
        min_scale: 0.3
        max_scale: 1.0
        min_aspect_ratio: 0.5
        max_aspect_ratio: 2.0
      }
      sample_constraint {
        min_jaccard_overlap: 0.7
      }
      max_sample: 1
      max_trials: 50
    }
    batch_sampler {
      sampler {
        min_scale: 0.3
        max_scale: 1.0
        min_aspect_ratio: 0.5
        max_aspect_ratio: 2.0
      }
      sample_constraint {
        min_jaccard_overlap: 0.9
      }
      max_sample: 1
      max_trials: 50
    }
    batch_sampler {
      sampler {
        min_scale: 0.3
        max_scale: 1.0
        min_aspect_ratio: 0.5
        max_aspect_ratio: 2.0
      }
      sample_constraint {
        max_jaccard_overlap: 1.0
      }
      max_sample: 1
      max_trials: 50
    }
    label_map_file: "labelmap.prototxt"
  }
}
```

### PermuteLayer

```
layer {
  name: "conv11_mbox_loc_perm"
  type: "Permute"
  bottom: "conv11_mbox_loc"
  top: "conv11_mbox_loc_perm"
  permute_param {
    order: 0
    order: 2
    order: 3
    order: 1
  }
}
```

### PriorBoxLayer

```
layer {
  name: "conv11_mbox_priorbox"
  type: "PriorBox"
  bottom: "conv11"
  bottom: "data"
  top: "conv11_mbox_priorbox"
  prior_box_param {
    min_size: 60.0
    aspect_ratio: 2.0
    flip: true
    clip: false
    variance: 0.1
    variance: 0.1
    variance: 0.2
    variance: 0.2
    offset: 0.5
  }
}
```

### DetectionOutputLayer

```
layer {
  name: "detection_out"
  type: "DetectionOutput"
  bottom: "mbox_loc"
  bottom: "mbox_conf_flatten"
  bottom: "mbox_priorbox"
  top: "detection_out"
  include {
    phase: TEST
  }
  detection_output_param {
    num_classes: 21
    share_location: true
    background_label_id: 0
    nms_param {
      nms_threshold: 0.45
      top_k: 100
    }
    code_type: CENTER_SIZE
    keep_top_k: 100
    confidence_threshold: 0.25
  }
}
```

### DetectionEvaluateLayer

```
layer {
  name: "detection_eval"
  type: "DetectionEvaluate"
  bottom: "detection_out"
  bottom: "label"
  top: "detection_eval"
  include {
    phase: TEST
  }
  detection_evaluate_param {
    num_classes: 21
    background_label_id: 0
    overlap_threshold: 0.5
    evaluate_difficult_gt: false
  }
}
```

### MultiBoxLossLayer

```
layer {
  name: "mbox_loss"
  type: "MultiBoxLoss"
  bottom: "mbox_loc"
  bottom: "mbox_conf"
  bottom: "mbox_priorbox"
  bottom: "label"
  top: "mbox_loss"
  include {
    phase: TRAIN
  }
  propagate_down: true
  propagate_down: true
  propagate_down: false
  propagate_down: false
  loss_param {
    normalization: VALID
  }
  multibox_loss_param {
    loc_loss_type: SMOOTH_L1
    conf_loss_type: SOFTMAX
    loc_weight: 1.0
    num_classes: 21
    share_location: true
    match_type: PER_PREDICTION
    overlap_threshold: 0.5
    use_prior_for_matching: true
    background_label_id: 0
    use_difficult_gt: true
    neg_pos_ratio: 3.0
    neg_overlap: 0.5
    code_type: CENTER_SIZE
    ignore_cross_boundary_bbox: false
    mining_type: MAX_NEGATIVE
  }
}
```

### caffe.proto

- 增加```TransformationParameter```参数

```
optional uint32 crop_h = 11 [default = 0];
optional uint32 crop_w = 12 [default = 0];

// Resize policy
optional ResizeParameter resize_param = 8;
// Noise policy
optional NoiseParameter noise_param = 9;
// Distortion policy
optional DistortionParameter distort_param = 13;
// Expand policy
optional ExpansionParameter expand_param = 14;
// Constraint for emitting the annotation after transformation.
optional EmitConstraint emit_constraint = 10;
```

- 增加```SolverParameter```参数

### [BoxLandmarkDataLayer](./examples/pycaffe/layers/data_layer.py)

- Python Layer
- 输出: FaceImage, Box, Landmark
- 提供各种数据增加: XXXXXX

### [Landmark5Data](./examples/pycaffe/layers/landmark5_data.py)

- Python Layer
- 输出: FaceImage, Landmark5
- 提供数据增加: 旋转, 随机尺寸缩放, 明亮度, 水平翻转


[1]: https://github.com/happynear/caffe-windows/tree/504d8a85f552e988fabff88b026f2c31cb778329
[2]: https://github.com/yonghenglh6/DepthwiseConvolution
[3]: https://128.84.21.199/abs/1812.11317
[4]: https://128.84.21.199/abs/1902.10859
[5]: https://github.com/farmingyard/ShuffleNet
[6]: https://github.com/weiliu89/caffe/tree/ssd


