/*
Reproduce the loss of PFLD papers
*/

#ifndef CAFFE_PFLD_LOSS_LAYER_HPP_
#define CAFFE_PFLD_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

#define PI 3.1415926535

namespace caffe {

template <typename Dtype>
class PFLDLossLayer: public LossLayer<Dtype> {
public:
  explicit PFLDLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual inline const char* type() const { return "PFLDLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 5; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Dtype weights[6];  // profile-face, frontal-face, head-up, head-down, expression, and occlusion.
  Blob<Dtype> diff_pose;  // shape: (M, 3)
  Blob<Dtype> scale_pose;  // shape: (M, 3)
  Blob<Dtype> class_;  // shape: (M, 1)
  Blob<Dtype> lambda;  // shape: (M, 1)

  Blob<Dtype> diff_landmark;  // shape: (M, 136)
  Blob<Dtype> diff_landmark_2;  // shape: (M, 136)
  Blob<Dtype> diff_landmark_N;  // shape: (M, 1)
  Blob<Dtype> euler;  // \sum_{k=1}^{K}(1-cos \theta _m^k), shape: (M, 1)
};

}


#endif // CAFFE_PFLD_LOSS_LAYER_HPP_
