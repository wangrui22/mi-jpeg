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

const char ZIGZAG_TABLE[64] = { 
	0, 1, 5, 6,14,15,27,28,
	2, 4, 7,13,16,26,29,42,
	3, 8,12,17,25,30,41,43,
	9,11,18,24,31,40,44,53,
	10,19,23,32,39,45,52,54,
	20,22,33,38,46,51,55,60,
	21,34,37,47,50,56,59,61,
	35,36,48,49,57,58,62,63 
};     



#endif