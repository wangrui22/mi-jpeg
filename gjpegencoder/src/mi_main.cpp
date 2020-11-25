#include "mi_gpu_jpeg_encoder.h"
#include <fstream>

int main(int argc, char *argv[]) {
    int width = 1024;
    int height = 1024;
    unsigned char* rgb = new unsigned char[width*height*3];
    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
            const int idx = y*width+x;
            unsigned char r = (x/8 % 2 + y/8 %2) == 1 ? 255 : 10;
            unsigned char g = (x/8 % 2 + y/8 %2) == 1 ? 0 : 30;
            unsigned char b = (x/8 % 2 + y/8 %2) == 1 ? 0 : 110;
            rgb[idx*3] = r;
            rgb[idx*3+1] = g;
            rgb[idx*3+2] = b;
        }
    }
    // {
    //     std::ofstream out("/home/wangrui22/projects/mi-jpeg/bin/rgb-520-518.raw", std::ios::out|std::ios::binary);
    //     if (out.is_open()) {
    //         out.write((char*)rgb, width*height*3);
    //         out.close();
    //         //return 0;
    //     }
    // }

    std::shared_ptr<Image> rgb_image(new Image());
    rgb_image->buffer = rgb;
    rgb_image->width = width;
    rgb_image->height = height;

    std::vector<int> qualitys;
    qualitys.push_back(50);
    qualitys.push_back(100);
    GPUJpegEncoder encoder;
    encoder.init(qualitys, rgb_image);
    unsigned char* compress_buffer = nullptr;
    unsigned int compress_buffer_len = 0;
    encoder.compress(100, compress_buffer, compress_buffer_len);
    std::ofstream out("/home/wangrui22/projects/mi-jpeg/gjpegencoder/bin/rgb-4000-512.2400-2.jpeg", std::ios::out|std::ios::binary);
    if (out.is_open()) {
        out.write((char*)compress_buffer, compress_buffer_len);
        out.close();
        return 0;
    }

    return 0;
}