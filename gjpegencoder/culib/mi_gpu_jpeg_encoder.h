#ifndef MI_GPU_JPEG_ENCODER_H
#define MI_GPU_JPEG_ENCODER_H

#include "mi_gpu_jpeg_define.h"


class GPUJpegResourceMananger {
public:
    GPUJpegResourceMananger();
    ~GPUJpegResourceMananger();
    int allocate(unsigned int block_count, unsigned int block_length);
    const std::vector<BlockUnit>& get_blocks() const;
private:
    std::vector<BlockUnit> _blocks;
};



class GPUJpegEncoder {    
public:
    GPUJpegEncoder();
    ~GPUJpegEncoder();
    int init(std::vector<int> qualitys);
    int compress(std::shared_ptr<Image> rgb, int quality, unsigned char*& compress_buffer, unsigned int& buffer_len);
    
public:
    void dct();
    void huffman_encode();

    void write_jpeg_header();
    void write_jpeg_segment();

private:
    unsigned char* _compress_buffer;
    unsigned int _compress_capacity;
    unsigned int _compress_byte;

    ImageInfo _img_info;

    BlockUnit _raw_rgb;

    BlockUnit _yuv_ext;

    std::map<int, DCTTable> _dct_table;
    BlockUnit _dct_result;

    BitString _huffman_table_Y_DC[12];
	BitString _huffman_table_Y_AC[256];
	BitString _huffman_table_CbCr_DC[12];
	BitString _huffman_table_CbCr_AC[256];

    BlockUnit _huffman_result;
};

#endif