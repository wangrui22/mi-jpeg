#include <iostream>
#include <string>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include <jpeglib.h>

int main(int argc , char * argv[]) {
    if (argc < 4) {
        std::cerr << "invalid input rgb\n";
        return -1;
    }

    const std::string file = argv[1];
    const int width = atoi(argv[2]);
    const int height = atoi(argv[3]);
    const std::string fileout = file + ".jpeg";

    std::cout << "input: " << file << "\n";
    std::cout << "image width: " << width << " height: " << height << "\n";
    std::cout << "output: " << fileout << "\n";

    std::fstream in(file.c_str(), std::ios::in|std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "invalid input rgb\n";
        return -1;
    }
    unsigned char* rgb = new unsigned char[width*height*3];
    in.read((char*)rgb, width*height*3);
    in.close();
    
    FILE* outfile = fopen(fileout.c_str(), "wb");
    if (!outfile) {
        std::cerr << "open file: " << file << " failed.";
        return -1;
    }

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr       jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = width;
    cinfo.image_height = height;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;

    jpeg_set_defaults(&cinfo);
    /*set the quality [0..100]  */
    jpeg_set_quality(&cinfo, 75, static_cast<boolean>(true));

    jpeg_start_compress(&cinfo, static_cast<boolean>(true));

    JSAMPROW row_pointer;          /* pointer to a single row */

    int idx = 0;
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer = (JSAMPROW)&rgb[cinfo.next_scanline * 3 * width];
        jpeg_write_scanlines(&cinfo, &row_pointer, 1);
        ++idx;
    }
    std::cout << "loop: " << idx << std::endl;

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    fclose(outfile);

    return 0;
}