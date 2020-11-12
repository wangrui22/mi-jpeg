#ifndef MI_JPEG_DEFINE_H
#define MI_JPEG_DEFINE_H

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


inline int div_and_round_up(const int val, const int div) {
    return (val % div == 0) ? val : (val/div+1)*div;
}

#endif