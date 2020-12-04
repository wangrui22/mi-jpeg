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
    // const int width = 1024;
    // const int height = 1024;
    // unsigned char* rgb = new unsigned char[width*height*3];
    // for (int y=0; y<height; ++y) {
    //     for (int x=0; x<width; ++x) {
    //         const int idx = y*width+x;
    //         unsigned char r = (x/8 % 2 + y/8 %2) == 1 ? 255 : 10;
    //         unsigned char g = (x/8 % 2 + y/8 %2) == 1 ? 0 : 30;
    //         unsigned char b = (x/8 % 2 + y/8 %2) == 1 ? 0 : 110;
    //         rgb[idx*3] = r;
    //         rgb[idx*3+1] = g;
    //         rgb[idx*3+2] = b;
    //     }
    // }

    // int width = 1920;
    // int height = 1080;
    // unsigned char* rgb = new unsigned char[width*height*3];
    // std::string file_path = "/home/wangrui22/projects/mi-jpeg/data/gray1-1920-1080-rgb.raw";

    int width = 1024;
    int height = 1024;
    unsigned char* rgb = new unsigned char[width*height*3];
    std::string file_path = "/home/wangrui22/projects/mi-jpeg/data/gray3-1024-1024-rgb.raw";
    
    // int width = 4000;
    // int height = 2087;
    // unsigned char* rgb = new unsigned char[width*height*3];
    // std::string file_path = "/home/wangrui22/projects/mi-jpeg/data/color1-4000-2087.raw";

    std::ifstream in(file_path, std::ios::binary | std::ios::in);
    if (!in.is_open()) {
        return -1;
    }
    in.read((char*)(rgb), width*height*3);
    in.close();


    // unsigned char rgb[64*3] = {
    //     153,153,153, 123,123,123, 123,123,123, 123,123,123, 123,123,123, 123,123,123, 123,123,123, 136,136,136,
    //     192,192,192, 180,180,180, 136,136,136, 154,154,154, 154,154,154, 154,154,154, 136,136,136, 110,110,110,
    //     254,254,254, 198,198,198, 154,154,154, 154,154,154, 180,180,180, 154,154,154, 123,123,123, 123,123,123,
    //     239,239,239, 180,180,180, 136,136,136, 180,180,180, 180,180,180, 166,166,166, 123,123,123, 123,123,123,
    //     180,180,180, 154,154,154, 136,136,136, 167,167,167, 166,166,166, 149,149,149, 136,136,136, 136,136,136,
    //     128,128,128, 136,136,136, 123,123,123, 136,136,136, 154,154,154, 180,180,180, 198,198,198, 154,154,154,
    //     123,123,123, 105,105,105, 110,110,110, 149,149,149, 136,136,136, 136,136,136, 180,180,180, 166,166,166,
    //     110,110,110, 136,136,136, 123,123,123, 123,123,123, 123,123,123, 136,136,136, 154,154,154, 136,136,136
    // };
    // int width = 8;
    // int height = 8;

    unsigned char* d_rgb = nullptr;
    cudaError_t err = cudaMalloc(&d_rgb, width*height*3);
    CHECK_CUDA_ERROR(err)
    err = cudaMemcpy(d_rgb, rgb, width*height*3, cudaMemcpyDefault);
    CHECK_CUDA_ERROR(err)

    //create new param
    gpujpeg_parameters params;
    gpujpeg_set_default_parameters(&params);        //default parameter
    params.verbose = 0;//gpujpeg log
    params.quality = 80;
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
    const int loop = 100;
    for (int i=0; i<loop; ++i) {
        if (0 != gpujpeg_encoder_encode(encoder, &params, &image_param, &encoder_input,
            &image_compressed, &compress_size)) {
            std::cerr << "GPU compress failed.";
            return -1;
        }
    }
    std::cout << "gpucompress cost " << duration_cast<duration<double>>(steady_clock::now()-_start).count()*1000.0/loop << " ms\n";

    std::ofstream out("/home/wangrui22/projects/mi-jpeg/bin/gpujpeg.jpeg", std::ios::out|std::ios::binary);
    if (out.is_open()) {
        out.write((char*)image_compressed, compress_size);
        out.close();
        return 0;
    }

    return 0;



}