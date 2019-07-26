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
    profile_face: 1
    frontal_face: 1
    head_up: 1
    head_down: 1
    expression: 1
    occlusion: 1
  }
}
```

- reference [PFLD][4]

[1]: https://github.com/happynear/caffe-windows/tree/504d8a85f552e988fabff88b026f2c31cb778329
[2]: https://github.com/yonghenglh6/DepthwiseConvolution
[3]: https://128.84.21.199/abs/1812.11317
[4]: https://128.84.21.199/abs/1902.10859



