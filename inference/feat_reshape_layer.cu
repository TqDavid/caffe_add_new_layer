#include <algorithm>
#include <vector>

#include "caffe/layers/feat_reshape_layer.hpp"

namespace caffe {

// forward: fill zero
template <typename Dtype>
__global__ void ZeroFilling(const int n, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {

    out[index] = 0.0;

  }
}

// forward: copy bottom data to buffer data
template <typename Dtype>
__global__ void PixelCopying(const int n, const Dtype* in, Dtype* out,
    int src_num, int src_channels, int src_height, int src_width,
    int dst_num, int dst_channels, int dst_height, int dst_width) {
  CUDA_KERNEL_LOOP(index, n) {

    // mapping to the src data domain
    int spb = index / (src_width * src_height * src_channels);
    int spc = (index - spb * src_width * src_height * src_channels) / (src_width * src_height);
    int spy = (index - spb * src_width * src_height * src_channels - spc * src_width * src_height) / src_width;
    int spx = (index - spb * src_width * src_height * src_channels - spc * src_width * src_height - spy * src_width) % src_width;

    // mapping to the dst data domain
    int dpx = spx;
    int dpy = spy;
    int dpc = spc;
    int dpb = spb;
    int dst_index = dpx + dpy * dst_width + dpc * dst_width * dst_height + dpb * dst_width * dst_height * dst_channels;
    
    out[dst_index] = in[index];

  }
}

// backward: copy buffer diff to bottom diff
template <typename Dtype>
__global__ void PixelDiffCopying(const int n, const Dtype* in, Dtype* out,
    int src_num, int src_channels, int src_height, int src_width,
    int dst_num, int dst_channels, int dst_height, int dst_width) {
  CUDA_KERNEL_LOOP(index, n) {

    // mapping to the dst diff domain
    int dpb = index / (dst_width * dst_height * dst_channels);
    int dpc = (index - dpb * dst_width * dst_height * dst_channels) / (dst_width * dst_height);
    int dpy = (index - dpb * dst_width * dst_height * dst_channels - dpc * dst_width * dst_height) / dst_width;
    int dpx = (index - dpb * dst_width * dst_height * dst_channels - dpc * dst_width * dst_height - dpy * dst_width) % dst_width;

    // mapping to the src diff domain
    int spx = dpx;
    int spy = dpy;
    int spc = dpc;
    int spb = dpb;
    int src_index = spx + spy * src_width + spc * src_width * src_height + spb * src_width * src_height * src_channels;
    
    // copy
    out[index] = in[src_index];

  }
}

// forward: reshaping the bottom data to top data
template <typename Dtype>
__global__ void PixelReshaping(const int n, const Dtype* in, Dtype* out,
    int src_num, int src_channels, int src_height, int src_width, 
    int dst_num, int dst_channels, int dst_height, int dst_width) {
  CUDA_KERNEL_LOOP(index, n) {
    
    // pixel location decoding in the dst domain
    int dpb = index / (dst_width * dst_height * dst_channels);
    int dpc = (index - dpb * dst_width * dst_height * dst_channels) / (dst_width * dst_height);
    int dpy = (index - dpb * dst_width * dst_height * dst_channels - dpc * dst_width * dst_height) / dst_width;
    int dpx = (index - dpb * dst_width * dst_height * dst_channels - dpc * dst_width * dst_height - dpy * dst_width) % dst_width;

    // pixel location encoding in the src domain
    int spb = dpb;
    int spc = dpc / 4;
    int block_shift = dpc % 4;
    int spx = dpx * 2;
    int spy = dpy * 2;
    if (block_shift == 0) {
      spx += 0;
      spy += 0;
    }else if (block_shift == 1) {
      spx += 1;
      spy += 0;
    }else if (block_shift == 2) {
      spx += 0;
      spy += 1;
    }else {
      spx += 1;
      spy += 1;
    }
    int src_index = spx + spy * src_width + spc * src_width * src_height + spb * src_width * src_height * src_channels; 
    
    // copy
    out[index] = in[src_index];

  }
}

// backward: reshaping the top diff to buffer diff
template <typename Dtype>
__global__ void PixelDiffReshaping(const int n, const Dtype* in, Dtype* out,
    int src_num, int src_channels, int src_height, int src_width,
    int dst_num, int dst_channels, int dst_height, int dst_width) {
  CUDA_KERNEL_LOOP(index, n) {

    // pixel location decoding in the src domain
    int spb = index / (src_width * src_height * src_channels);
    int spc = (index - spb * src_width * src_height * src_channels) / (src_width * src_height);
    int spy = (index - spb * src_width * src_height * src_channels - spc * src_width * src_height) / src_width;
    int spx = (index - spb * src_width * src_height * src_channels - spc * src_width * src_height - spy * src_width) % src_width;

    // pixel location encoding in the dst domain
    int dpb = spb;
    int dpc = spc / 4;
    int block_shift = spc % 4;
    int dpx = spx * 2;
    int dpy = spy * 2;
    if (block_shift == 0) {
      dpx += 0;
      dpy += 0;
    }else if (block_shift == 1) {
      dpx += 1;
      dpy += 0;
    }else if (block_shift == 2) {
      dpx += 0;
      dpy += 1;
    }else {
      dpx += 1;
      dpy += 1;
    }
    int dst_index = dpx + dpy * dst_width + dpc * dst_width * dst_height + dpb * dst_width * dst_height * dst_channels;

    // copy
    out[dst_index] = in[index];

  }
}

// do forward
template <typename Dtype>
void FeatReshapeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  int src_num = bottom[0]->num();
  int src_channels = bottom[0]->channels();
  int src_height = bottom[0]->height();
  int src_width = bottom[0]->width();

  int dst_height = top[0]->height();
  int dst_width = top[0]->width();
  int dst_channels = top[0]->channels();
  int dst_num = top[0]->num();

  int buf_num = src_num;
  int buf_channels = src_channels;
  int buf_height = src_height;
  int buf_width = src_width;

  // if odd add one
  if (src_height % 2 == 1) {
    buf_height += 1;
  }
  if (src_width % 2 == 1) {
    buf_width += 1;
  }
  
  /*LOG(INFO) <<buf_num<<" "<<buf_channels<<" "<<buf_height<<" "<<buf_width;
  LOG(INFO) <<src_num<<" "<<src_channels<<" "<<src_height<<" "<<src_width;
  LOG(INFO) <<this->imgdata_buf_->num()<<" "<<this->imgdata_buf_->channels()<<" "<<this->imgdata_buf_->height()<<" "<<this->imgdata_buf_->width();*/
 

  const int buf_count = this->feat_buf_->count();
  const int src_count = bottom[0]->count();
  const int dst_count = top[0]->count();
 
  // filling zero to data buffer
  Dtype* feat_buf = this->feat_buf_->mutable_gpu_data();
  ZeroFilling<Dtype><<<CAFFE_GET_BLOCKS(buf_count), CAFFE_CUDA_NUM_THREADS>>>(
      buf_count, feat_buf);
  CUDA_POST_KERNEL_CHECK;

  // copy src data to buffer data
  const Dtype* bottom_data = bottom[0]->gpu_data();
  feat_buf = this->feat_buf_->mutable_gpu_data();
  PixelCopying<Dtype><<<CAFFE_GET_BLOCKS(src_count), CAFFE_CUDA_NUM_THREADS>>>(
      src_count, bottom_data, feat_buf, 
      src_num, src_channels, src_height, src_width,
      buf_num, buf_channels, buf_height, buf_width);
  CUDA_POST_KERNEL_CHECK;
  
  // pixel reshape
  const Dtype* feat_rbuf = this->feat_buf_->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  PixelReshaping<Dtype><<<CAFFE_GET_BLOCKS(dst_count), CAFFE_CUDA_NUM_THREADS>>>(
      dst_count, feat_rbuf, top_data, 
      buf_num, buf_channels, buf_height, buf_width, 
      dst_num, dst_channels, dst_height, dst_width);
  CUDA_POST_KERNEL_CHECK;
}

// do backward
template <typename Dtype>
void FeatReshapeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  
  int dst_num = bottom[0]->num();
  int dst_channels = bottom[0]->channels();
  int dst_height = bottom[0]->height();
  int dst_width = bottom[0]->width();

  int src_height = top[0]->height();
  int src_width = top[0]->width();
  int src_channels = top[0]->channels();
  int src_num = top[0]->num();

  int buf_num = dst_num;
  int buf_channels = dst_channels;
  int buf_height = dst_height;
  int buf_width = dst_width;

  // if odd add one
  if (dst_height % 2 == 1) {
    buf_height += 1;
  }
  if (dst_width % 2 == 1) {
    buf_width += 1;
  }

  const int buf_count = this->feat_buf_->count();
  const int dst_count = bottom[0]->count();
  const int src_count = top[0]->count();

  // pixel diff reshape
  Dtype* feat_diff_buf = this->feat_buf_->mutable_gpu_diff();
  const Dtype* top_diff = top[0]->gpu_diff();
  PixelDiffReshaping<Dtype><<<CAFFE_GET_BLOCKS(src_count), CAFFE_CUDA_NUM_THREADS>>>(
      src_count, top_diff, feat_diff_buf,
      src_num, src_channels, src_height, src_width,
      buf_num, buf_channels, buf_height, buf_width);
  CUDA_POST_KERNEL_CHECK;

  // pixel diff copy
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* feat_rbuf = this->feat_buf_->gpu_diff();
  PixelDiffCopying<Dtype><<<CAFFE_GET_BLOCKS(dst_count), CAFFE_CUDA_NUM_THREADS>>>(
      dst_count, feat_rbuf, bottom_diff,
      buf_num, buf_channels, buf_height, buf_width,
      dst_num, dst_channels, dst_height, dst_width);
  CUDA_POST_KERNEL_CHECK;

}

INSTANTIATE_LAYER_GPU_FUNCS(FeatReshapeLayer);

}  // namespace caffe
