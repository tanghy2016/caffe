/*
Reproduce the loss of PFLD papers
*/

#include <vector>
#include <stdlib.h>
#include <math.h>

#include "caffe/layers/pfld_loss.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void PFLDLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  weights[0] = this->layer_param_.pfld_loss_param().profile_face();
  weights[1] = this->layer_param_.pfld_loss_param().frontal_face();
  weights[2] = this->layer_param_.pfld_loss_param().head_up();
  weights[3] = this->layer_param_.pfld_loss_param().head_down();
  weights[4] = this->layer_param_.pfld_loss_param().expression();
  weights[5] = this->layer_param_.pfld_loss_param().occlusion();
}

template <typename Dtype>
void PFLDLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[2]->count(1))
    << "bottom[0]&bottom[2], Inputs must have the same dimension.";
  CHECK_EQ(bottom[1]->count(1), bottom[3]->count(1))
    << "bottom[1]&bottom[3], Inputs must have the same dimension.";
  CHECK_EQ(bottom[4]->count(1), 6)
    << "bottom[4], Class of inputs categorize must be 6: profile-face, frontal-face, head-up, head-down, expression, and occlusion.";
  diff_landmark.ReshapeLike(*bottom[0]);  // shape: (M, 136)
  diff_landmark_2.ReshapeLike(*bottom[0]);  // shape: (M, 136)
  diff_pose.ReshapeLike(*bottom[1]);
  scale_pose.ReshapeLike(*bottom[1]);
 
  std::vector<int> lambda_shape;
  lambda_shape.push_back(bottom[0]->num());
  lambda.Reshape(lambda_shape);  // shape: (M, 1)
  euler.Reshape(lambda_shape);  // shape: (M, 1)
  diff_landmark_N.Reshape(lambda_shape);  // shape: (M, 1)
  class_.Reshape(lambda_shape);
}

template <typename Dtype>
void PFLDLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int batch = bottom[0]->num();
  int pose_num = bottom[1]->count(1);

  Dtype* euler_data = euler.mutable_cpu_data();
  Dtype* diff_pose_data = diff_pose.mutable_cpu_data();
  Dtype* scale_pose_data = scale_pose.mutable_cpu_data();

  caffe_cpu_scale(bottom[1]->count(), (Dtype)PI, bottom[1]->cpu_data(), scale_pose_data);
  caffe_sub(bottom[1]->count(), scale_pose.cpu_data(), bottom[3]->cpu_data(), diff_pose_data);
  for(int i = 0; i < batch; i++)
  {
    euler_data[i] = 0;
    for(int j = 0; j < pose_num; j++)
    {
      euler_data[i] += 1 - cos(diff_pose_data[i*pose_num + j]);
    }
  }
  // \sum_{c=1}^{C}w_m^c \sum_{k=1}^{K}(1-cos \theta _m^k), M*1
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, batch, 1, 6,
    (Dtype)1., bottom[4]->cpu_data(), weights, (Dtype)0., class_.mutable_cpu_data());
  caffe_mul(batch, class_.cpu_data(), euler_data, lambda.mutable_cpu_data());

  // \sum_{n=1}^{N} \left \| \mathbf{d}_n^m \right \|_2^2, M*1
  caffe_sub(
      bottom[0]->count(),
      bottom[0]->cpu_data(),
      bottom[2]->cpu_data(),
      diff_landmark.mutable_cpu_data());
  caffe_sqr(diff_landmark.count(), diff_landmark.cpu_data(), diff_landmark_2.mutable_cpu_data());

  Dtype* diff_N_data = diff_landmark_N.mutable_cpu_data();
  const Dtype* diff_2_data = diff_landmark_2.cpu_data();
  int cols = diff_landmark_2.count(1);
  for(int i = 0; i < batch; i++)
  {
    diff_N_data[i] = 0;
    for(int j = 0; j <= cols; j++)
      diff_N_data[i] += diff_2_data[i*cols+j];
  }

  // \frac{1}{2M}\sum_{m=1}^{M} \left ( \sum_{c=1}^{C}w_m^c \sum_{k=1}^{K}(1-cos \theta _m^k) \right )\sum_{n=1}^{N} \left \| \mathbf{d}_n^m \right \|_2^2
  Dtype dot = caffe_cpu_dot(batch, lambda.cpu_data(), diff_landmark_N.cpu_data());
  Dtype loss = dot / batch;
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void PFLDLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_diff_landmark = bottom[0]->mutable_cpu_diff();
  Dtype* bottom_diff_pose = bottom[1]->mutable_cpu_diff();
  const Dtype* lambda_data = lambda.cpu_data();
  const Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num();
  if (propagate_down[0]) {
    caffe_cpu_axpby(
        bottom[0]->count(),                 // count
        alpha,                              // alpha
        diff_landmark.cpu_data(),           // a
        Dtype(0),                           // beta
        bottom_diff_landmark);              // b
    int cols = bottom[0]->count(1);
    for(int i = 0; i < bottom[0]->num(); i++)
    {
      for(int j = 0; j < cols; j++)
        bottom_diff_landmark[i*cols + j] *= lambda_data[i];
    }
  }

  if (propagate_down[1]) {
    const Dtype* class_data = class_.cpu_data();
    const Dtype* diff_landmark_N_data = diff_landmark_N.cpu_data();
    const Dtype* diff_pose_data = diff_pose.cpu_data();
    int cols = bottom[1]->count(1);
    for(int i = 0; i < bottom[1]->num(); i++)
    {
      for(int j = 0; j < cols; j++)
        bottom_diff_pose[i*cols + j] *= alpha * class_data[i] * diff_landmark_N_data[i] * sin(diff_pose_data[i*cols + j]) * PI;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(PFLDLossLayer);
#endif

INSTANTIATE_CLASS(PFLDLossLayer);
REGISTER_LAYER_CLASS(PFLDLoss);

}  // namespace caffe

