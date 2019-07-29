#include <algorithm>
#include <vector>
#include <math.h>
#include "caffe/layers/add_margin_layer.hpp"

namespace caffe {

  template <typename Dtype>
  void AddMarginLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    const AddMarginParameter& param = this->layer_param_.add_margin_param();
    m1_ = param.m1();
    m2_ = param.m2();
    m3_ = param.m3();
    sin_m = sin(m3_);
    cos_m = cos(m3_);
    threshold = cos(M_PI - m3_);
  }

  template <typename Dtype>
  void AddMarginLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                                    const vector<Blob<Dtype>*>& top) {
    top[0]->ReshapeLike(*bottom[0]);
    top_flag.ReshapeLike(*bottom[0]);
    cos_theta.ReshapeLike(*bottom[0]);
  }

template <typename Dtype>
void AddMarginLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                                  const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* label_data = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* tpflag = top_flag.mutable_cpu_data();
  Dtype* cos_t = cos_theta.mutable_cpu_data();

  int num = bottom[0]->num();
  int count = bottom[0]->count();
  int dim = count / num;

  caffe_copy(count, bottom_data, top_data);
  caffe_copy(count, bottom_data, cos_t);
  caffe_set(count, Dtype(0), tpflag);

  //cosin add m
  for (int i = 0; i < num; ++i) {
    int gt = static_cast<int>(label_data[i]);
    if(gt < 0) continue;

    bi_data[i * dim + gt] = 0.f;

    Dtype cos_theta_2 = cos_t[i * dim + gt] * cos_t[i * dim + gt];
    Dtype sin_theta = sqrt(1.0f - cos_theta_2);
    if(cos_t[i * dim + gt] > 1.0f)
    {
        LOG(INFO) << "cos_theta > 1 ****** " << cos_t[i * dim + gt];
        cos_t[i * dim + gt] = 1.0f;
        cos_theta_2 = 1.0f;
        sin_theta = 0.0f;
    }

    if(cos_t[i * dim + gt] <= threshold)
    {
        top_data[i * dim + gt] = cos_t[i * dim + gt] - sin(M_PI - m3_) * m3_;
        tpflag[i * dim + gt] = 1.0f;
    }
    else
        top_data[i * dim + gt] = cos_t[i * dim + gt] * cos_m - sin_theta * sin_m - m2_;

//    if(count_num % 10 == 0)
//    {
//        LOG(INFO) << "top_data[" << i * dim + gt << "]: " << top_data[i * dim + gt] << "  cos_t: " << cos_t[i * dim + gt];
//    }
  }
}

template <typename Dtype>
void AddMarginLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
                                                   const vector<bool>& propagate_down,
                                                   const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* label_data = bottom[1]->cpu_data();
    const Dtype* cos_t = cos_theta.cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* tpflag = top_flag.cpu_data();
    int count = bottom[0]->count();

    caffe_copy(count, top_diff, bottom_diff);

    int num = bottom[0]->num();
    int dim = count / num;
    for (int i = 0; i < num; ++i)
    {
      int gt = static_cast<int>(label_data[i]);
      if(gt < 0) continue;
      Dtype cos_theta_2 = cos_t[i * dim + gt] * cos_t[i * dim + gt];
      if(abs(cos_t[i * dim + gt] - 1.0f) < 0.00001)
      {
          cos_theta_2 = 1.0f;
      }
      Dtype sin_theta = sqrt(1.0f - cos_theta_2);
      Dtype coffe = 0.0f;
      if(abs(sin_theta - 0.0f) < 0.00001)
        coffe = 1.0f;
      else
        coffe = cos_m + sin_m * cos_t[i * dim + gt] / sin_theta;

      if(tpflag[i * dim + gt] > 0.0f)
        coffe = 1.0f;
      bottom_diff[i * dim + gt] = coffe * bottom_diff[i * dim + gt];

//      if(count_num_back % 10 == 0)
//      {
//          LOG(INFO) << "top_diff: " << top_diff[i * dim + gt];
//          LOG(INFO) << "bottom_diff: " << bottom_diff[i * dim + gt];
//          LOG(INFO) << "cos_theta[ "<<i * dim + gt <<"]: "<< cos_t[i * dim + gt]<<  "    coffe: " << coffe;
//      }
    }
  }
}

INSTANTIATE_CLASS(AddMarginLayer);
REGISTER_LAYER_CLASS(AddMargin);

}  // namespace caffe

