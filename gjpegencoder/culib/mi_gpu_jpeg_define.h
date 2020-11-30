#ifndef MI_GPU_JPEG_DEFINE_H
#define MI_GPU_JPEG_DEFINE_H

#include <vector>
#include <map>
#include <memory>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <exception>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERROR(err) \
{\
    if (err != cudaSuccess) {\
        std::cerr << "fn: " << __FUNCTION__ << " line: " << __LINE__ \
        << " catch cuda err(" << err << "): " << cudaGetErrorString(err) << std::endl; \
        throw std::runtime_error("DEVICE ERROR");\
    }\
}

struct Image {
    int width;
    int height;
    int component;
    unsigned char* buffer;

    Image():width(0),height(0),component(3),buffer(NULL) {}

    ~Image() {
        if (!buffer) {
            delete [] buffer;
        }
    }
};

struct BitString {
	short length;	
	short value;
};

struct DCTTable {
    unsigned char quant_tbl_luminance[64];
    unsigned char quant_tbl_chrominance[64];
    float *d_quant_tbl_luminance;//[64]
    float *d_quant_tbl_chrominance;//[64]
    unsigned char *d_zig_zag;//[64]
};

struct HuffmanTable {
    BitString *d_huffman_table_Y_DC;//[12]
    BitString *d_huffman_table_Y_AC;//[256]
    BitString *d_huffman_table_CbCr_DC;//[12]
    BitString *d_huffman_table_CbCr_AC;//[256]
    unsigned char *d_order_natural;//[80]
};

struct ImageInfo {
    int width;
    int height;
    int width_ext;
    int height_ext;
    int mcu_w;
    int mcu_h;
    int mcu_count;
    int segment_mcu_count;
    int segment_count;
    int component;
};

struct BlockUnit {
    unsigned int length;
    unsigned char* d_buffer;
};



#endif