#ifndef CAFFE_RESIZE_LAYER_HPP_  
#define CAFFE_RESIZE_LAYER_HPP_  
#include "caffe/blob.hpp"  
#include "../layer.hpp"
#include "../proto/caffe.pb.h"
namespace caffe {  

class ResizeLayer : public Layer {  
 public:  
  explicit ResizeLayer(const LayerParameter& param)  
      : Layer(param) {}  
  virtual void LayerSetUp(const vector<Blob*>& bottom,  
      const vector<Blob*>& top);  
  virtual void Reshape(const vector<Blob*>& bottom,  
      const vector<Blob*>& top);  
 virtual inline const char* type() const { return "Resize"; }
 protected:  
  virtual void Forward_gpu(const vector<Blob*>& bottom,  
      const vector<Blob*>& top);   
  virtual void Forward_cpu(const vector<Blob*>& bottom,
                             const vector<Blob*>& top);

  int out_height_;  
  int out_width_;  
  int height_;  
  int width_;  
  int num_images_;  
  int num_channels_;  
};  
}
#endif