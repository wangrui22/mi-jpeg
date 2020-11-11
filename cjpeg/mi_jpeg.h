#ifndef MI_JPEG_H
#define MI_JPEG_H

#include "mi_jpeg_define.h"

class Jpeg {
public:
    Jpeg();
    ~Jpeg();
    void compress();
    int rgb_2_yuv(std::shared_ptr<RGBImage> raw_img);
    void dct();
};

#endif