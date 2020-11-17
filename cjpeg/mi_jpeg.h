#ifndef MI_JPEG_H
#define MI_JPEG_H

#include "mi_jpeg_define.h"

class Jpeg {
public:
    Jpeg();
    ~Jpeg();
    void compress();
    int rgb_2_yuv(std::shared_ptr<Image> rgb, std::shared_ptr<Image> yuv);
    int expand_to_8(std::shared_ptr<Image> yuv, std::shared_ptr<Image> yuv_expand);
    void dct(std::shared_ptr<Image> yuv);

    void dct_8x8(int* inout);
    void dct_8x8_fast(int* inout);

    void init_quantization_table(float (&qt)[64]);
    void init_quantization_table_fast(float (&qt)[64]);
    
    void quantize_8x8(int* inout, float* qt);
    void zig_zag(int* input, int* output);

};

#endif