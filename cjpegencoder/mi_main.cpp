#include "mi_jpeg_encoder.h"
#include <fstream>

int main(int argc, char * argv[]) {

    int width = 1024;
    int height = 1024;
    unsigned char* rgb = new unsigned char[width*height*3];
    for (int y=0; y<height; ++y) {
        for (int x=0; x<width; ++x) {
            const int idx = y*width+x;
            unsigned char r = (x/7 % 2 + y/7 %2) == 1 ? 255 : 10;
            unsigned char g = (x/7 % 2 + y/7 %2) == 1 ? 0 : 30;
            unsigned char b = (x/7 % 2 + y/7 %2) == 1 ? 0 : 110;
            rgb[idx*3] = r;
            rgb[idx*3+1] = g;
            rgb[idx*3+2] = b;
        }
    }
    {
        std::ofstream out("/home/wangrui22/projects/mi-jpeg/bin/rgb-520-518.raw", std::ios::out|std::ios::binary);
        if (out.is_open()) {
            out.write((char*)rgb, width*height*3);
            out.close();
            //return 0;
        }
    }

    // const std::string file = "/home/wangrui22/projects/mi-jpeg/data/img-4000-2400.raw";
    // const int width = 4000;
    // const int height = 2400;
    // std::fstream in(file.c_str(), std::ios::in|std::ios::binary);
    // if (!in.is_open()) {
    //     std::cerr << "invalid input rgb\n";
    //     return -1;
    // }
    // unsigned char* rgb = new unsigned char[width*height*3];
    // in.read((char*)rgb, width*height*3);
    // in.close();
    

    std::shared_ptr<Image> rgb_image(new Image());
    rgb_image->buffer = rgb;
    rgb_image->width = width;
    rgb_image->height = height;

    JpegEncoder encoder;
    unsigned char* compress_buffer = nullptr;
    unsigned int compress_buffer_len = 0;
    encoder.compress(rgb_image, 50, compress_buffer, compress_buffer_len);
    std::ofstream out("/home/wangrui22/projects/mi-jpeg/bin/rgb-4000-512.2400-2.jpeg", std::ios::out|std::ios::binary);
    if (out.is_open()) {
        out.write((char*)compress_buffer, compress_buffer_len);
        out.close();
        return 0;
    }


    // JpegEncoder encoder;
    // encoder.encode_to_jpeg(rgb, width, height, 100, "./rgb-512-512.raw.jpeg");
    return 0;
}