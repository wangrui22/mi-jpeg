#include <vector_types.h>
#include <cuda_runtime.h>
#include "mi_gpu_jpeg_define.h"

__global__ void kernel_rgb_2_yuv(const BlockUnit rgb, const BlockUnit yuv, const ImageInfo img_info) {
    unsigned int mcu_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int mcu_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (mcu_x > img_info.mcu_w-1 || mcu_y > img_info.mcu_h-1) {
        return;
    }
    int width = img_info.width;
    int height = img_info.height;
    int width_ext = img_info.width_ext;
    //int height_ext = img_info.height_ext;

    int x0 = mcu_x * 8;
    int y0 = mcu_y * 8;
    int x1 = x0 + 8;
    int y1 = y0 + 8;
    y1 = y1 < height ? y1 : height;
    x1 = x1 < width ? x1 : width;

    int sidx = 0;
    int idx = 0;
    float r,g,b,y,u,v;
    for (int iy=y0; iy<y1; ++iy) {
        for (int ix=x0; ix<x1; ++ix) {
            idx = iy*width + ix;
            sidx = iy*width_ext + ix;
            r = (float)rgb.d_buffer[idx*3];
            g = (float)rgb.d_buffer[idx*3+1];
            b = (float)rgb.d_buffer[idx*3+2];
            y =  0.2990f*r + 0.5870f*g + 0.1140f*b ;
            u = -0.1687f*r - 0.3313f*g + 0.5000f*b + 128.0f;
            v =  0.5000f*r - 0.4187f*g - 0.0813f*b + 128.0f;
            y = y < 0.0f ? 0.0f : y;
            y = y > 255.0f ? 255.0f : y;
            u = u < 0.0f ? 0.0f : u;
            u = u > 255.0f ? 255.0f : u;
            v = v < 0.0f ? 0.0f : v;
            v = v > 255.0f ? 255.0f : v;

            yuv.d_buffer[sidx*3] = (unsigned char)y;
            yuv.d_buffer[sidx*3+1] = (unsigned char)u;
            yuv.d_buffer[sidx*3+2] = (unsigned char)v;

            if (sidx*3+2 > yuv.length) {
                printf("got err\n");
            }
        }
    }
}

extern "C"
cudaError_t rgb_2_yuv(const BlockUnit& rgb, const BlockUnit& yuv, const ImageInfo& img_info) {
    const int BLOCK_SIZEX = 4;
    const int BLOCK_SIZEY = 32;
    dim3 block(BLOCK_SIZEX, BLOCK_SIZEY, 1);
    dim3 grid(img_info.mcu_w / BLOCK_SIZEX, img_info.mcu_h / BLOCK_SIZEY);
    if (grid.x * BLOCK_SIZEX != img_info.mcu_w) {
        grid.x += 1;
    }
    if (grid.y * BLOCK_SIZEY != img_info.mcu_h) {
        grid.y += 1;
    }

    kernel_rgb_2_yuv << <grid, block >> >(rgb, yuv, img_info);
    
    return cudaDeviceSynchronize();
}
