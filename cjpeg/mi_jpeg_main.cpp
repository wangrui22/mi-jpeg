#include <iostream>
#include "mi_jpeg.h"
//#include "mi_jpeg_fast_dct.h"
#include "mi_jpeg_slow_dct.h"

void test() {
    std::cout << "\n<><><><><><><><>test<><><><><><><><>\n";
    int input[64] = {
        52,55,61,66,70,61,64,73,
        63,59,55,90,109,85,69,72,
        62,59,68,113,144,104,66,73,
        63,58,71,122,154,106,70,69,
        67,61,68,104,126,88,68,70,
        79,65,60,70,77,68,58,75,
        85,71,64,59,55,61,65,83,
        87,79,69,68,65,76,78,94
    };

    std::cout << "\nDCT:";
    Jpeg jpeg;
    jpeg.dct_8x8(input);
    for (int i=0; i<64; ++i) {
        if (i%8 == 0) {
            std::cout << "\n";
        }
        std::cout << input[i] << " ";
    }
    
    std::cout << "\n\nQuatization:";
    float qt[64];
    jpeg.init_quantization_table(qt);
    jpeg.quantize_8x8(input, qt);

    for (int i=0; i<64; ++i) {
        if (i%8 == 0) {
            std::cout << "\n";
        }
        std::cout << input[i] << " ";
    }

    std::cout << "\n\nZigZag:";
    int output[64];
    jpeg.zig_zag(input, output);
    for (int i=0; i<64; ++i) {
        if (i%8 == 0) {
            std::cout << "\n";
        }
        std::cout << output[i] << " ";
    }
}

void test_fast() {
    std::cout << "\n<><><><><><><><>test fast<><><><><><><><>\n";
    int input[64] = {
        52,55,61,66,70,61,64,73,
        63,59,55,90,109,85,69,72,
        62,59,68,113,144,104,66,73,
        63,58,71,122,154,106,70,69,
        67,61,68,104,126,88,68,70,
        79,65,60,70,77,68,58,75,
        85,71,64,59,55,61,65,83,
        87,79,69,68,65,76,78,94
    };

    std::cout << "\nDCT:";
    Jpeg jpeg;
    jpeg.dct_8x8_fast(input);
    for (int i=0; i<64; ++i) {
        if (i%8 == 0) {
            std::cout << "\n";
        }
        std::cout << input[i] << " ";
    }
    
    std::cout << "\n\nQuatization:";
    float qt[64];
    jpeg.init_quantization_table_fast(qt);
    jpeg.quantize_8x8(input, qt);
    for (int i=0; i<64; ++i) {
        if (i%8 == 0) {
            std::cout << "\n";
        }
        std::cout << input[i] << " ";
    }

    std::cout << "\n\nZigZag:";
    int output[64];
    jpeg.zig_zag(input, output);
    for (int i=0; i<64; ++i) {
        if (i%8 == 0) {
            std::cout << "\n";
        }
        std::cout << output[i] << " ";
    }
}


int main(int argc , char *argv[]) {
    //std::cout << "jpeg test\n";

    // int input[64] = {
    //     52,55,61,66,70,61,64,73,
    //     63,59,55,90,109,85,69,72,
    //     62,59,68,113,144,104,66,73,
    //     63,58,71,122,154,106,70,69,
    //     67,61,68,104,126,88,68,70,
    //     79,65,60,70,77,68,58,75,
    //     85,71,64,59,55,61,65,83,
    //     87,79,69,68,65,76,78,94
    // };

    // unsigned char input2[64] = {
    //     52,55,61,66,70,61,64,73,
    //     63,59,55,90,109,85,69,72,
    //     62,59,68,113,144,104,66,73,
    //     63,58,71,122,154,106,70,69,
    //     67,61,68,104,126,88,68,70,
    //     79,65,60,70,77,68,58,75,
    //     85,71,64,59,55,61,65,83,
    //     87,79,69,68,65,76,78,94
    // };

    // std::cout << "\n\n";
    // int output[64];
    // jpeg_fdct_islow(output, input2, 0);
    // for (int i=0; i<64; ++i) {
    //     if (i%8 == 0) {
    //         std::cout << "\n";
    //     }
    //     std::cout << output[i] << " ";
    // }
    
    test();

    test_fast();

    
    return 0;
}