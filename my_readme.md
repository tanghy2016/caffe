# Extended Caffe

@tanghy2016

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

[1]: https://github.com/happynear/caffe-windows/tree/504d8a85f552e988fabff88b026f2c31cb778329
[2]: https://github.com/yonghenglh6/DepthwiseConvolution

