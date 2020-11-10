#ifndef MI_JPEG_H
#define MI_JPEG_H

#include "mi_jpeg_define.h"

class Jpeg {
public:
    Jpeg();
    ~Jpeg();
    void compress();
    void dct();
};

#endif