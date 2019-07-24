# Extended Caffe

@tanghy2016

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
- for other class:
    - when $f(m, \theta_{gt}) < cos(\theta_{other})$, then $t cos(\theta_{other}) + t - 1$
    - other, $cos(\theta)$
- param
    - m1: default = 1.0
        - don't code it
    - m2: best when m = 0.35 to m = 0.4 (paper), default = 0.35
    - m3: default = 0.5
    - t: > 1, default = 1.2


[1]: https://github.com/happynear/caffe-windows/tree/504d8a85f552e988fabff88b026f2c31cb778329
[2]: https://github.com/yonghenglh6/DepthwiseConvolution

