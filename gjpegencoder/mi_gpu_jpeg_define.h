#ifndef MI_GPU_JPEG_DEFINE_H
#define MI_GPU_JPEG_DEFINE_H

#include <vector>
#include <map>
#include <memory>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define PRINT_CUDA_ERR(err) \
{ \
    std::cerr << "fn " < __FUNCTION__ << " line: " << __LINE__ \
    <<  " catch cuda err(" << err << "): " << cudaGetErrorString(err) << std::endl; \
}

struct Image {
    int width;
    int height;
    unsigned char* buffer;

    Image():width(0),height(0),buffer(nullptr) {}

    ~Image() {
        if (!buffer) {
            delete [] buffer;
        }
    }
};

struct BitString {
	int length;	
	int value;
};

struct Segment {
    unsigned char *d_segment;
    //64*3
    unsigned char *d_y;
    unsigned char *d_u;
    unsigned char *d_v;
    //64*2*3
    short *d_quat_y;
    short *d_quat_u;
    short *d_quat_v;
    //128*8*3
    BitString* d_huffman_code_y;
    BitString* d_huffman_code_u;
    BitString* d_huffman_code_v;

    int huffman_code_y_count;
    int huffman_code_u_count;
    int huffman_code_v_count;

    Segment():d_segment(nullptr), d_y(nullptr), d_u(nullptr), d_v(nullptr),
              d_quat_y(nullptr),d_quat_u(nullptr),d_quat_v(nullptr),
              d_huffman_code_y(nullptr),d_huffman_code_u(nullptr)d_huffman_code_v(nullptr)，
              huffman_code_y_count(0),huffman_code_u_count(0),huffman_code_v_count(0) {
        
    }

    ~Segment() {
        if (!d_segment) {
            cudaError_t err = cudaFree(d_segment);
            if (cudaSuccess != err) {
                PRINT_CUDA_ERR(err);
            }
        }
    }
    
    int allocate() {
        cudaError_t err = cudaMalloc(&d_segment, 3648);
        if (cudaSuccess != err) {
             PRINT_CUDA_ERR(err);
             return -1;
        } 
        d_y = d_segment;
        d_u = d_segment+64;
        d_v = d_segment+64*2;
        d_quat_y = (short*)(d_segment+64*3);
        d_quat_u = (short*)(d_segment+64*3+128);
        d_quat_v = (short*)(d_segment+64*3+128*2);
        d_huffman_code_y = (BitString*)(d_segment+64*3+128*3);
        d_huffman_code_u = (BitString*)(d_segment+64*3+128*3+128*8);
        d_huffman_code_v = (BitString*)(d_segment+64*3+128*3+128*8*2);

        return 0;
    }
};


#endif