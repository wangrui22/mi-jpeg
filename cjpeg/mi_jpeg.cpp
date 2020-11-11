#include "mi_jpeg.h"

/***
template<>
struct gpujpeg_color_transform<GPUJPEG_RGB, GPUJPEG_YCBCR_BT601_256LVLS> {
RGB -> YCbCr (ITU-R Recommendation BT.601 with 256 levels) transform (8 bit) 
    static __device__ void
    perform(uint8_t & c1, uint8_t & c2, uint8_t & c3) {
        GPUJPEG_COLOR_TRANSFORM_DEBUG(GPUJPEG_RGB, GPUJPEG_YCBCR_BT601_256LVLS, "Transformation");
        // Source: http://www.ecma-international.org/publications/files/ECMA-TR/TR-098.pdf, page 3
        const double matrix[] = {
             0.299000,  0.587000,  0.114000,
            -0.168700, -0.331300,  0.500000,
             0.500000, -0.418700, -0.081300
        };
        const int matrix[] = {77, 150, 29, -43, -85, 128, 128, -107, -21};
        gpujpeg_color_transform_to<8>(c1, c2, c3, matrix, 0, 128, 128);
    }
};
***/

int Jpeg::rgb_2_yuv(std::shared_ptr<RGBImage> raw_img) {
    
    return 0;

}