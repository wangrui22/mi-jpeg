#ifndef MI_JPEG_DEFINE_H
#define MI_JPEG_DEFINE_H

#include <memory>

struct RGBImage {
    int width;
    int height;
    unsigned char* buffer;

    RGBImage():width(0),height(0),buffer(0) {}

    ~RGBImage() {
        if (!buffer) {
            delete [] buffer;
        }
    }
};



#endif