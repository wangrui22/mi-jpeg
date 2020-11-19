#ifndef MI_GPU_JPEG_ENCODER_H
#define MI_GPU_JPEG_ENCODER_H

#include "mi_gpu_jpeg_define.h"

class GPUJpegEncoder {
public:
    GPUJpegEncoder();
    ~GPUJpegEncoder();
    int compress(std::shared_ptr<Image> rgb, int quality, unsigned char*& compress_buffer, unsigned int& buffer_len);
    
public:
    void init(int quality);
    void dct();
    void huffman_encode();

    void write_jpeg_header();
    void write_jpeg_segment();

private:
    unsigned char* _compress_buffer;
    unsigned int _compress_capacity;
    unsigned int _compress_byte;

    unsigned char _quality_quantization_table_luminance[64];
    unsigned char _quality_quantization_table_chrominance[64];

    float _d_quantization_table_luminance[64];
    float _d_quantization_table_chrominance[64];
    
    BitString _huffman_table_Y_DC[12];
	BitString _huffman_table_Y_AC[256];
	BitString _huffman_table_CbCr_DC[12];
	BitString _huffman_table_CbCr_AC[256];
};

#endif