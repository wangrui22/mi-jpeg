#include "mi_jpeg_encoder.h"
#include <fstream>

int main(int argc, char * argv[]) {

    int width = 512;
    int height = 512;
    unsigned char* rgb = new unsigned char[width*height*3];
    for (int y=0; y<height; ++y) {
        for (int x=0; x<height; ++x) {
            const int idx = y*width+x;
            unsigned char r = (x/8 % 2 + y/8 %2) == 1 ? 255 : 10;
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

    std::shared_ptr<Image> rgb_image(new Image());
    rgb_image->buffer = rgb;
    rgb_image->width = width;
    rgb_image->height = height;

    JpegEncoder encoder;
    unsigned char* compress_buffer = nullptr;
    unsigned int compress_buffer_len = 0;
    encoder.compress(rgb_image, compress_buffer, compress_buffer_len);
    std::ofstream out("/home/wangrui22/projects/mi-jpeg/bin/rgb-512-512.test.jpeg", std::ios::out|std::ios::binary);
    if (out.is_open()) {
        out.write((char*)compress_buffer, compress_buffer_len);
        out.close();
        return 0;
    }


    // JpegEncoder encoder;
    // encoder.encode_to_jpeg(rgb, width, height, 100, "./rgb-512-512.raw.jpeg");
    return 0;
}