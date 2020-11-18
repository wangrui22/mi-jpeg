#ifndef MI_JPEG_ENCODER_H
#define MI_JPEG_ENCODER_H

#include <vector>
#include <map>
#include <memory>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    unsigned char* rgb;
    unsigned char* y;
    unsigned char* u;
    unsigned char* v;
    short* quat_y;
    short* quat_u;
    short* quat_v;
    BitString* huffman_code_y;
    BitString* huffman_code_u;
    BitString* huffman_code_v;
    int huffman_code_y_count;
    int huffman_code_u_count;
    int huffman_code_v_count;

    Segment():rgb(nullptr), y(nullptr), u(nullptr), v(nullptr), 
    quat_y(nullptr), quat_u(nullptr), quat_v(nullptr),
    huffman_code_y(nullptr), huffman_code_u(nullptr), huffman_code_v(nullptr),
    huffman_code_y_count(0),huffman_code_u_count(0),huffman_code_v_count(0) {}

    ~Segment() {
        if (!rgb) {
            delete [] rgb;
            rgb = nullptr;
        }
        if (!y) {
            delete [] y;
            y = nullptr;
        }
        if (!u) {
            delete [] u;
            u = nullptr;
        }
        if (!v) {
            delete [] v;
            v = nullptr;
        }
        if (!quat_y) {
            delete [] quat_y;
            quat_y = nullptr;
        }
        if (!quat_u) {
            delete [] quat_u;
            quat_u = nullptr;
        }
        if (!quat_v) {
            delete [] quat_v;
            quat_v = nullptr;
        }
        if (!huffman_code_y) {
            delete [] huffman_code_y;
            huffman_code_y = nullptr;
        }
        if (!huffman_code_u) {
            delete [] huffman_code_u;
            huffman_code_u = nullptr;
        }
        if (!huffman_code_v) {
            delete [] huffman_code_v;
            huffman_code_v = nullptr;
        }
    }
};

class JpegEncoder {
public:
    JpegEncoder();
    ~JpegEncoder();
    int compress(std::shared_ptr<Image> rgb, unsigned char*& compress_buffer, unsigned int& buffer_len);
    
public:
    void init();

    void rgb_2_yuv_segment(std::shared_ptr<Image> rgb, std::vector<Segment>& segments);
    
    void dct_8x8(unsigned char* val, short* output, float* quant_table);
    void huffman_encode_8x8(short* quant, short preDC, const BitString* HTDC, const BitString* HTAC, BitString* output, int& output_count);

    void write_jpeg_header(std::shared_ptr<Image> rgb);

    void write_word(unsigned short val);
    void write_byte(unsigned char val);
    void write_byte_array(const unsigned char* buf, unsigned int buf_len);
    void write_bitstring(const BitString* bs, int counts, int& new_byte, int& new_byte_pos);

private:
    unsigned char* _compress_buffer;
    unsigned int _compress_capacity;
    unsigned int _compress_byte;

    float _quantization_table_luminance[64];
    float _quantization_table_chrominance[64];
    
    BitString _huffman_table_Y_DC[12];
	BitString _huffman_table_Y_AC[256];
	BitString _huffman_table_CbCr_DC[12];
	BitString _huffman_table_CbCr_AC[256];
};

#endif