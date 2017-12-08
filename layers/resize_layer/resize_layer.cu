#include "./resize_layer.hpp"  
#include "../util/math_functions.hpp"  
#include <vector>
#include <opencv2/opencv.hpp>  
#include<iostream>
using namespace std;
namespace caffe {  


__global__ void kernel_ResizeBlob(const int nthreads,const int num,const int channels, const float* src, const int src_height, const int src_width,
		float* dst, const int dst_height, const int dst_width, const float scale_h, const float scale_w)
{
	CUDA_KERNEL_LOOP(index, nthreads) {
		int i = index %( dst_height * dst_width);
		int c = (index/(dst_height * dst_width))%channels;
		int n = (index/(dst_height * dst_width))/channels;
		int src_offset = (n * channels + c) * src_height * src_width;
		int dst_offset = (n * channels + c) * dst_height * dst_width;

		const float* src_data = src+src_offset;
		float* dst_data = dst+dst_offset;

		int dst_h = i /dst_width;
		float fh = dst_h * scale_h;
		const int src_h = floor(fh);
		fh -= src_h;
		const float w_h0 = std::abs(1.0f - fh);
		const float w_h1 = std::abs(fh);

		const int dst_offset_1 =  dst_h * dst_width;
		const int src_offset_1 =  src_h * src_width;

		int dst_w = i %dst_width;
		float fw = dst_w * scale_w;
		const int src_w = floor(fw);
		fw -= src_w;
		const float w_w0 = std::abs(1.0f - fw);
		const float w_w1 = std::abs(fw);

		const int dst_idx = dst_offset_1 + dst_w;


		const int src_idx = src_offset_1 + src_w;
		float res = (w_h0 * w_w0 * src_data[src_idx]);

		if (src_w + 1 < src_width)
			res += (w_h0 * w_w1 * src_data[src_idx + 1]);
		if (src_h + 1 < src_height)
			res += (w_h1 * w_w0 * src_data[src_idx + src_width]);

		if (src_w + 1 < src_width && src_h + 1 < src_height)
			res += (w_h1 * w_w1 * src_data[src_idx + src_width + 1]);

		dst_data[dst_idx] = res;
	}
}



void ResizeLayer::Forward_gpu(  
    const vector<Blob*>& bottom, const vector<Blob*>& top) {   
	const int src_num = bottom[0]->num();
	const int src_channels = bottom[0]->channels();
	const int src_height = bottom[0]->height();
	const int src_width = bottom[0]->width();

	const int dst_channels = top[0]->channels();
	const int dst_height = top[0]->height();
	const int dst_width = top[0]->width();
	
	const float scale_w = src_width / (float)dst_width;
	const float scale_h = src_height / (float)dst_height;
	int loop_n = dst_height * dst_width*dst_channels*src_num;
	const float* src_data = bottom[0]->gpu_data();
	float* dst_data = top[0]->mutable_gpu_data();
	kernel_ResizeBlob<<<CAFFE_GET_BLOCKS(loop_n), CAFFE_CUDA_NUM_THREADS >>>(loop_n,src_num,src_channels,src_data, src_height,src_width,dst_data, dst_height, dst_width,scale_h,scale_w);  
	CUDA_POST_KERNEL_CHECK;
}  

}  // namespace caffe  