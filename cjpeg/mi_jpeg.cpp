#include "mi_jpeg.h"
#include "mi_jpeg_fast_dct_2.h"
#include "mi_jpeg_fast_dct.h"
#include "mi_jpeg_quantize.h"

Jpeg::Jpeg() {

}

Jpeg::~Jpeg() {

}

/** Specialization [color_space_from = GPUJPEG_RGB, color_space_to = GPUJPEG_YCBCR_BT601_256LVLS] */
// template<>
// struct gpujpeg_color_transform<GPUJPEG_RGB, GPUJPEG_YCBCR_BT601_256LVLS> {
//     /** RGB -> YCbCr (ITU-R Recommendation BT.601 with 256 levels) transform (8 bit) */
//     static __device__ void
//     perform(uint8_t & c1, uint8_t & c2, uint8_t & c3) {
//         GPUJPEG_COLOR_TRANSFORM_DEBUG(GPUJPEG_RGB, GPUJPEG_YCBCR_BT601_256LVLS, "Transformation");
//         // Source: http://www.ecma-international.org/publications/files/ECMA-TR/TR-098.pdf, page 3
//         /*const double matrix[] = {
//              0.299000,  0.587000,  0.114000,
//             -0.168700, -0.331300,  0.500000,
//              0.500000, -0.418700, -0.081300
//         };*/
//         const int matrix[] = {77, 150, 29, -43, -85, 128, 128, -107, -21};
//         gpujpeg_color_transform_to<8>(c1, c2, c3, matrix, 0, 128, 128);
//     }
// };

int Jpeg::rgb_2_yuv(std::shared_ptr<Image> rgb, std::shared_ptr<Image> yuv) {
    if (!rgb || !yuv) {
        std::cerr << "rgb_2_yuv null image.\n";
        return -1;
    }
    if (!rgb->buffer || !yuv->buffer) {
        std::cerr << "rgb_2_yuv null image buffer.\n";
        return -1;
    }
    if (rgb->width != yuv->width || rgb->height != yuv->height) {
        std::cerr << "rgb_2_yuv dismatch image size.\n";
        return -1;
    }

    for (int i=0; i<rgb->width*rgb->height; ++i) {
        const float r = (float)rgb->buffer[i*3];
        const float g = (float)rgb->buffer[i*3+1];
        const float b = (float)rgb->buffer[i*3+2];
        float y =  0.2990f*r + 0.5870f*g + 0.1140f*b ;
        float u = -0.1687f*r - 0.3313f*g + 0.5000f*b + 128.0f;
        float v =  0.5000f*r - 0.4187f*g - 0.0813f*b + 128.0f;
        y = y < 0.0f ? 0.0f : y;
        y = y > 255.0f ? 255.0f : y;
        u = u < 0.0f ? 0.0f : u;
        u = u > 255.0f ? 255.0f : u;
        v = v < 0.0f ? 0.0f : v;
        v = v > 255.0f ? 255.0f : v;
        yuv->buffer[i*3] = (unsigned char)y;
        yuv->buffer[i*3+1] = (unsigned char)u;
        yuv->buffer[i*3+2] = (unsigned char)v;
    }
    return 0;
}

int Jpeg::expand_to_8(std::shared_ptr<Image> yuv, std::shared_ptr<Image> yuv_expand) {
    if (!yuv_expand || !yuv) {
        std::cerr << "expand_to_8 null image.\n";
        return -1;
    }
    if (!yuv->buffer) {
        std::cerr << "expand_to_8 null image buffer.\n";
        return -1;
    }

    yuv_expand->width = div_and_round_up(yuv->width, 8);
    yuv_expand->height = div_and_round_up(yuv->height, 8);
    std::cout << "w: " << yuv->width << " h: " << yuv->height << " expand to w: " << yuv_expand->width << " h: " << yuv_expand->height << std::endl;
    yuv_expand->buffer = new unsigned char[yuv_expand->width * yuv_expand->height *3];
    memset(yuv_expand->buffer, 0, yuv_expand->width * yuv_expand->height *3);
    for (int y = 0; y < yuv->height; ++y) {
        memcpy(yuv_expand->buffer + yuv_expand->width*3, yuv->buffer + yuv->width*3, yuv->width*3);
    }

    return 0;
}

void Jpeg::dct_8x8_fast(int* inout) {
    //row 1d-dct
    for (int i=0; i<8; ++i) {
        int* i0 = inout + 8*i;
        int* o0 = inout + 8*i;
        dct_1d_8_fast<int>(i0[0], i0[1], i0[2], i0[3], i0[4], i0[5], i0[6], i0[7],
                      o0[0], o0[1], o0[2], o0[3], o0[4], o0[5], o0[6], o0[7], 128);
    }

    //collum 1d-dct
    for (int i=0; i<8; ++i) {
        int* i0 = inout + i;
        int* o0 = inout + i;
        dct_1d_8_fast<int>(i0[0], i0[1*8], i0[2*8], i0[3*8], i0[4*8], i0[5*8], i0[6*8], i0[7*8],
                      o0[0], o0[1*8], o0[2*8], o0[3*8], o0[4*8], o0[5*8], o0[6*8], o0[7*8], 0);
    }    
}

void Jpeg::dct_8x8(int* inout) {
    //row 1d-dct
    for (int i=0; i<8; ++i) {
        int* i0 = inout + 8*i;
        int* o0 = inout + 8*i;
        FastDct8_transform<int>(i0[0]-128, i0[1]-128, i0[2]-128, i0[3]-128, i0[4]-128, i0[5]-128, i0[6]-128, i0[7]-128,
                      o0[0], o0[1], o0[2], o0[3], o0[4], o0[5], o0[6], o0[7]);
    }

    //collum 1d-dct
    for (int i=0; i<8; ++i) {
        int* i0 = inout + i;
        int* o0 = inout + i;
        FastDct8_transform<int>(i0[0], i0[1*8], i0[2*8], i0[3*8], i0[4*8], i0[5*8], i0[6*8], i0[7*8],
                      o0[0], o0[1*8], o0[2*8], o0[3*8], o0[4*8], o0[5*8], o0[6*8], o0[7*8]);
    }    
}

void Jpeg::init_quantization_table(float (&qt)[64]) {
    init_qtable(qt);
}

void Jpeg::init_quantization_table_fast(float (&qt)[64]) {
    init_qtable_fast(qt);
}

void Jpeg::quantize_8x8(int* inout, float* qt) {
    quantize8(inout, qt);
}

void Jpeg::zig_zag(int* input, int* output) {
    for (int i=0; i<64; ++i) {
        output[i] = input[ZIGZAG_TABLE[i]];
    }
}