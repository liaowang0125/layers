#include "./resize_layer.hpp"  
#include "../util/math_functions.hpp"  
#include <vector>
#include <opencv2/opencv.hpp>  
#include<iostream>
using namespace std; 
namespace caffe {  

void ResizeLayer::LayerSetUp(const vector<Blob*>& bottom,  
      const vector<Blob*>& top) {  
  // get parameters  
  const ResizeParameter& param = this->layer_param_.resize_param();  
  // get the output size  
  out_height_ = param.out_height();  
  out_width_ = param.out_width();   
   
  // get the input size  
  num_images_ = bottom[0]->num();  
  height_ = bottom[0]->height();  
  width_ = bottom[0]->width();  
  num_channels_ = bottom[0]->channels();  
 
  CHECK_GT(out_height_, 0);  
  CHECK_GT(out_height_, 0);  
   
}  
 
void ResizeLayer::Reshape(const vector<Blob*>& bottom,  
      const vector<Blob*>& top) {  
  // reshape the outputs  
  top[0]->Reshape(num_images_, num_channels_, out_height_, out_width_);  
}  

void BiLinearResizeMat_cpu(const float* src, const int src_height, const int src_width,
		float* dst, const int dst_height, const int dst_width)
{
	const float scale_w = src_width / (float)dst_width;
	const float scale_h = src_height / (float)dst_height;
	float* dst_data = dst;
	const float* src_data = src;

	for(int dst_h = 0; dst_h < dst_height; ++dst_h){
		float fh = dst_h * scale_h;

		int src_h = std::floor(fh);

		fh -= src_h;
		const float w_h0 = std::abs((float)1.0 - fh);
		const float w_h1 = std::abs(fh);

		const int dst_offset_1 =  dst_h * dst_width;
		const int src_offset_1 =  src_h * src_width;

		float* dst_data_ptr = dst_data + dst_offset_1;

		for(int dst_w = 0 ; dst_w < dst_width; ++dst_w){

			float fw = dst_w * scale_w;
			int src_w = std::floor(fw);
			fw -= src_w;
			const float w_w0 = std::abs((float)1.0 - fw);
			const float w_w1 = std::abs(fw);


			float dst_value = 0;

			const int src_idx = src_offset_1 + src_w;
			dst_value += (w_h0 * w_w0 * src_data[src_idx]);
			int flag = 0;
			if (src_w + 1 < src_width){
				dst_value += (w_h0 * w_w1 * src_data[src_idx + 1]);
				++flag;
			}
			if (src_h + 1 < src_height){
				dst_value += (w_h1 * w_w0 * src_data[src_idx + src_width]);
				++flag;
			}

			if (flag>1){
				dst_value += (w_h1 * w_w1 * src_data[src_idx + src_width + 1]);
//				++flag;
			}
			*(dst_data_ptr++) = dst_value;
		}
	}

}
void ResizeBlob_cpu(const Blob* src, const int src_n, const int src_c,
		Blob* dst, const int dst_n, const int dst_c) {


	const int src_channels = src->channels();
	const int src_height = src->height();
	const int src_width = src->width();
	const int src_offset = (src_n * src_channels + src_c) * src_height * src_width;

	const int dst_channels = dst->channels();
	const int dst_height = dst->height();
	const int dst_width = dst->width();
	const int dst_offset = (dst_n * dst_channels + dst_c) * dst_height * dst_width;


	const float* src_data = &(src->cpu_data()[src_offset]);
	float* dst_data = &(dst->mutable_cpu_data()[dst_offset]);
	BiLinearResizeMat_cpu(src_data,  src_height,  src_width,
			dst_data,  dst_height,  dst_width);
}

void ResizeLayer::Forward_cpu(  
    const vector<Blob*>& bottom, const vector<Blob*>& top) {   
	for(int n=0;n< bottom[0]->num();++n)
	{
		for(int c=0; c < bottom[0]->channels() ; ++c)
		{	
			ResizeBlob_cpu(bottom[0],n,c,top[0],n,c);
		}
	}
}  

// INSTANTIATE_CLASS(ResizeLayer);
 REGISTER_LAYER_CLASS(Resize);

}  // namespace caffe  