#include <algorithm>
#include <vector>

#include "caffe/layers/feat_reshape_layer.hpp"

namespace caffe {

template <typename Dtype>
void FeatReshapeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  this->feat_buf_ = shared_ptr<Blob<Dtype> >(new Blob<Dtype>());

}

template <typename Dtype>
void FeatReshapeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  int src_num = bottom[0]->num(); //this is batch_size
  int src_channels = bottom[0]->channels();
  int src_height = bottom[0]->height();
  int src_width = bottom[0]->width();
  
  int dst_height = src_height / 2;
  int dst_width = src_width / 2;
  int dst_channels = src_channels * 4;
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
  this->feat_buf_->Reshape(buf_shape);
 
  /*LOG(INFO) << "input data size: " << bottom[0]->num() << ","
      << bottom[0]->channels() << "," << bottom[0]->height() << ","
      << bottom[0]->width();

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();*/
}

template <typename Dtype>
void FeatReshapeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // not 
}

template <typename Dtype>
void FeatReshapeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // not
}


#ifdef CPU_ONLY
STUB_GPU(FeatReshapeLayer);
#endif

INSTANTIATE_CLASS(FeatReshapeLayer);
REGISTER_LAYER_CLASS(FeatReshape);

}  // namespace caffe
