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
    unsigned char rgb[64*3];
    unsigned char y[64];
    unsigned char u[64];
    unsigned char v[64];
    short quat_y[64];
    short quat_u[64];
    short quat_v[64];
    BitString* huffman_code_y;
    BitString* huffman_code_u;
    BitString* huffman_code_v;
    int huffman_code_y_count;
    int huffman_code_u_count;
    int huffman_code_v_count;

    Segment():huffman_code_y(nullptr), huffman_code_u(nullptr), huffman_code_v(nullptr),
    huffman_code_y_count(0),huffman_code_u_count(0),huffman_code_v_count(0) {
        memset(rgb, 0, 64*3);
        memset(y, 0, 64);
        memset(u, 0, 64);
        memset(v, 0, 64);
        memset(quat_y, 0, 64*2);
        memset(quat_u, 0, 64*2);
        memset(quat_v, 0, 64*2);
    }

    ~Segment() {
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


    //////////////////////////////////////////////////////////////////////////////
    void foword_fdc(const unsigned char* channel_data, short* fdc_data);
    void init_quality_tables(int quality_scale);
    //////////////////////////////////////////////////////////////////////////////

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

    //////////////////////////////////////////////////////////////////////////////
    unsigned char	_y_table[64];
	unsigned char	_cb_cr_table[64];
	BitString m_Y_DC_Huffman_Table[12];
	BitString m_Y_AC_Huffman_Table[256];
	BitString m_CbCr_DC_Huffman_Table[12];
	BitString m_CbCr_AC_Huffman_Table[256];
    //////////////////////////////////////////////////////////////////////////////
};

#endif