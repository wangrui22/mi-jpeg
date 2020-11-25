#include <iostream>
#include <fstream>
#include <chrono>
#include <cuda_runtime.h>
#include "libgpujpeg/gpujpeg.h"
#include "libgpujpeg/gpujpeg_common.h"
#include "libgpujpeg/gpujpeg_encoder.h"

#define CHECK_CUDA_ERROR(err) \
{\
    if (err != cudaSuccess) {\
        std::cerr << "fn: " << __FUNCTION__ << " line: " << __LINE__ \
        << " catch cuda err(" << err << "): " << cudaGetErrorString(err) << std::endl; \
        throw std::runtime_error("DEVICE ERROR");\
    }\
}

using namespace std::chrono;

int main(int argc, char* argv[]) {
    const int width = 1024;
    const int height = 1024;
    unsigned char* rgb = new unsigned char[width*height*3];
    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
            const int idx = y*width+x;
            unsigned char r = (x/8 % 2 + y/8 %2) == 1 ? 255 : 10;
            unsigned char g = (x/8 % 2 + y/8 %2) == 1 ? 0 : 30;
            unsigned char b = (x/8 % 2 + y/8 %2) == 1 ? 0 : 110;
            rgb[idx*3] = r;
            rgb[idx*3+1] = g;
            rgb[idx*3+2] = b;
        }
    }

    unsigned char* d_rgb = nullptr;
    cudaError_t err = cudaMalloc(&d_rgb, width*height*3);
    CHECK_CUDA_ERROR(err)
    err = cudaMemcpy(d_rgb, rgb, width*height*3, cudaMemcpyDefault);
    CHECK_CUDA_ERROR(err)

    //create new param
    gpujpeg_parameters params;
    gpujpeg_set_default_parameters(&params);        //default parameter
    //params.verbose = 1;//gpujpeg log
    params.quality = 100;
    params.interleaved = 1;

    gpujpeg_image_parameters image_param;
    gpujpeg_image_set_default_parameters(&image_param);
    image_param.width = width;
    image_param.height = height;
    image_param.comp_count = 3;
    image_param.color_space = GPUJPEG_RGB;

    gpujpeg_encoder* encoder = gpujpeg_encoder_create(0);
    if (nullptr == encoder) {
        std::cerr << "create GPU compressor failed.";
        return -1;
    }

    gpujpeg_encoder_input encoder_input;
    encoder_input.image = (uint8_t*)d_rgb;
    encoder_input.type = GPUJPEG_ENCODER_INPUT_GPU_IMAGE;

    steady_clock::time_point _start = steady_clock::now();

    unsigned char* image_compressed = nullptr;
    int compress_size = 0;
    if (0 != gpujpeg_encoder_encode(encoder, &params, &image_param, &encoder_input,
        &image_compressed, &compress_size)) {
        std::cerr << "GPU compress failed.";
        return -1;
    }

    std::cout << "gpucompress cost " << duration_cast<duration<double>>(steady_clock::now()-_start).count()*1000 << " ms\n";

    std::ofstream out("/home/wangrui22/projects/mi-jpeg/bin/rgb-gpujpeg.jpeg", std::ios::out|std::ios::binary);
    if (out.is_open()) {
        out.write((char*)image_compressed, compress_size);
        out.close();
        return 0;
    }

    return 0;



}