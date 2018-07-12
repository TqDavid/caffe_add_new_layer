#include <algorithm>
#include <vector>
#include "iostream"

#include "caffe/layers/feat_reshape_common_layer.hpp"

namespace caffe {

template <typename Dtype>
void FeatReshapeCommonLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  this->feat_common_buf_ = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());
  sample_step_ = this->layer_param_.feat_reshape_common_param().sample_step();
  //if (sample_step_ & 1)
	//  std; cout << "sample_steo must need to be even��" << std::endl;
}

template <typename Dtype>
void FeatReshapeCommonLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  //read sampling step for reshape
  

  int src_num = bottom[0]->num(); //this is batch_size
  int src_channels = bottom[0]->channels();
  int src_height = bottom[0]->height();
  int src_width = bottom[0]->width();
  
  int dst_height = src_height / sample_step_;
  int dst_width = src_width / sample_step_;
  int dst_channels = src_channels * sample_step_ * sample_step_;
  int dst_num = src_num;

  // if odd add one
  if (src_height % 2 == 1) {
    dst_height += 1;
    src_height += 1;
  }
  if (src_width % 2 == 1) {
    dst_width += 1;
    src_width += 1;
  }   

  // reshape
  vector<int> top_shape(4);
  vector<int> buf_shape(4);
  top_shape[0] = dst_num;
  top_shape[1] = dst_channels;
  top_shape[2] = dst_height;
  top_shape[3] = dst_width;
  buf_shape[0] = src_num;
  buf_shape[1] = src_channels;
  buf_shape[2] = src_height;
  buf_shape[3] = src_width;
  top[0]->Reshape(top_shape);
  this->feat_common_buf_->Reshape(buf_shape);
 
  /*LOG(INFO) << "input data size: " << bottom[0]->num() << ","
      << bottom[0]->channels() << "," << bottom[0]->height() << ","
      << bottom[0]->width();

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();*/
}

template <typename Dtype>
void FeatReshapeCommonLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // not 
}

template <typename Dtype>
void FeatReshapeCommonLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // not
}


#ifdef CPU_ONLY
STUB_GPU(FeatReshapeCommonLayer);
#endif

INSTANTIATE_CLASS(FeatReshapeCommonLayer);
REGISTER_LAYER_CLASS(FeatReshapeCommon);

}  // namespace caffe
