#include <iostream>
#include <fstream>
#include "jpeg_encoder.h"


int main(int argc, char* argv[]) {
    int width = 512;
    int height = 512;
    unsigned char* rgb = new unsigned char[width*height*3];
    for (int y=0; y<height; ++y) {
        for (int x=0; x<height; ++x) {
            const int idx = y*width+x;
            unsigned char r = (x/32 % 2 + y/32 %2) == 1 ? 255 : 10;
            rgb[idx*3] = r;
            rgb[idx*3+1] = r;
            rgb[idx*3+2] = r;
        }
    }
    // std::ofstream out("/home/wangrui22/projects/mi-jpeg2/rgb-512-512.raw", std::ios::out|std::ios::binary);
    // if (out.is_open()) {
    //     out.write((char*)rgb, width*height*3);
    //     out.close();
    //     return 0;
    // }

    JpegEncoder encoder;
    encoder.encode_to_jpeg(rgb, width, height, 100, "./rgb-512-512.raw.jpeg");

    return 0;

    
}