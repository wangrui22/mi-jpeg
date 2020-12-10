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

class CudaTimeQuery
{
public:
    CudaTimeQuery();
    ~CudaTimeQuery();

    void begin();
    float end();

private:
    cudaEvent_t _start;
    cudaEvent_t _end;
    float _time_elapsed;
};


class GPUJpegEncoder {    
public:
    GPUJpegEncoder();
    ~GPUJpegEncoder();
    int init(std::vector<int> qualitys, int restart_interval, std::shared_ptr<Image> rgb, bool gray=false);
    int compress(int quality, unsigned char*& compress_buffer, unsigned int& buffer_len);
    
public:
    void write_word(unsigned short val);
    void write_byte(unsigned char val);
    void write_byte_array(const unsigned char* buf, unsigned int buf_len);
    void write_bitstring(const BitString* bs, int counts, int& new_byte, int& new_byte_pos);

    void write_jpeg_header(int quality);
    void write_jpeg_segment();
    void write_jpeg_segment_gpu(int segment_compressed_byte);

private:
    std::shared_ptr<Image> _raw_image;

    unsigned char* _compress_buffer;
    unsigned int _compress_capacity;
    unsigned int _compress_byte;

    ImageInfo _img_info;

    BlockUnit _raw_rgb;

    std::map<int, DCTTable> _dct_table;
    BlockUnit _dct_result;

    BitString _huffman_table_Y_DC[12];
	BitString _huffman_table_Y_AC[256];
	BitString _huffman_table_CbCr_DC[12];
	BitString _huffman_table_CbCr_AC[256];
    HuffmanTable _huffman_table;

    BlockUnit _huffman_result;
    int *_d_huffman_code_count;

    BlockUnit _huffman_code;

    BlockUnit _segment_compressed;
    int *_d_segment_compressed_byte;
    int *_d_segment_compressed_offset;

    unsigned int *_d_segment_compressed_byte_sum;

    BlockUnit _segment_compressed_compact;
    
    
};

#endif