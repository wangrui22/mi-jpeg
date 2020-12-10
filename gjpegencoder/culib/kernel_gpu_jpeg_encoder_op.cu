#include <vector_types.h>
#include <cuda_runtime.h>
#include "mi_gpu_jpeg_define.h"

#define gpujpeg_div_and_round_up(value, div) \
    ((((value) % (div)) != 0) ? ((value) / (div) + 1) : ((value) / (div)))

inline __device__ void dct_1d_8_fast(
    const float in0, const float in1, const float in2, const float in3, const float in4, const float in5, const float in6, const float in7,
    float & out0, float & out1, float & out2, float & out3, float & out4, float & out5, float & out6, float & out7, const float center_sample = 0.0f) {
    const float diff0 = in0 + in7;
    const float diff1 = in1 + in6;
    const float diff2 = in2 + in5;
    const float diff3 = in3 + in4;
    const float diff4 = in3 - in4;
    const float diff5 = in2 - in5;
    const float diff6 = in1 - in6;
    const float diff7 = in0 - in7;

    const float even0 = diff0 + diff3;
    const float even1 = diff1 + diff2;
    const float even2 = diff1 - diff2;
    const float even3 = diff0 - diff3;

    const float even_diff = even2 + even3;

    const float odd0 = diff4 + diff5;
    const float odd1 = diff5 + diff6;
    const float odd2 = diff6 + diff7;

    const float odd_diff5 = (odd0 - odd2) * 0.382683433f;
    const float odd_diff4 = 1.306562965f * odd2 + odd_diff5;
    const float odd_diff3 = diff7 - odd1 * 0.707106781f;
    const float odd_diff2 = 0.541196100f * odd0 + odd_diff5;
    const float odd_diff1 = diff7 + odd1 * 0.707106781f;

    out0 = even0 + even1 - 8 * center_sample;
    out1 = odd_diff1 + odd_diff4;
    out2 = even3 + even_diff * 0.707106781f;
    out3 = odd_diff3 - odd_diff2;
    out4 = even0 - even1;
    out5 = odd_diff3 + odd_diff2;
    out6 = even3 - even_diff * 0.707106781f;
    out7 = odd_diff1 - odd_diff4;
}

template <int WARP_COUNT>
__global__ void r_2_dct_op_kernel(const BlockUnit rgb, const BlockUnit dct_result, const ImageInfo img_info, const DCTTable dct_table) {
    const int COMPONET = 1;
    __shared__ unsigned char S_YUV[WARP_COUNT*4*64*COMPONET];
    __shared__ float S_QUANT[WARP_COUNT*4*64*COMPONET];
    __shared__ float S_DCT_TABLE[64];
    

    int tid = threadIdx.x;
    int wid = tid/32; //0~3
    int twid = tid - wid*32; //0~31
    int local_mcu_id = twid/8; //0~3
    int mcu_id = blockIdx.x*WARP_COUNT*4 + wid*4 + local_mcu_id;
    int cal_id = twid & 7; //0~7

    S_DCT_TABLE[twid*2] = dct_table.d_quant_tbl_luminance[twid*2];
    S_DCT_TABLE[twid*2+1] = dct_table.d_quant_tbl_luminance[twid*2+1];
    // _S_DCT_TABLE[64+twid*2] = dct_table.d_quant_tbl_chrominance[twid*2];
    // _S_DCT_TABLE[64+twid*2+1] = dct_table.d_quant_tbl_chrominance[twid*2+1];

    // printf("block_id: %d, thread_id: %d, warp_id: %d, wrap_thread_id: %d, local_mcu: %d, mcu: %d, cal_id: %d\n",
    // blockIdx.x, threadIdx.x, wid, twid, local_mcu_id, mcu_id, cal_id);

    if (mcu_id > img_info.mcu_count-1) {
        return; 
    }

    
    int mcu_y = mcu_id / img_info.mcu_w;
    int mcu_x = mcu_id - mcu_y*img_info.mcu_w;
    
    ///\1 rgb->yuv
    int width = img_info.width;
    int height = img_info.height;
    int x0 = mcu_x * 8;
    int y0 = mcu_y * 8;
    int x1 = x0 + 8;
    int y1 = y0 + 8;
    y1 = y1 < height+1 ? y1 : height;
    x1 = x1 < width+1 ? x1 : width;

    unsigned char* s_yuv_base = S_YUV + wid*4*64*COMPONET + local_mcu_id*64*COMPONET + cal_id*COMPONET*8;
    float* s_quant_base = S_QUANT + wid*4*64*COMPONET + local_mcu_id*64*COMPONET;

    int y = y0 + cal_id;
    ((uint*)(s_yuv_base))[0] = 0;
    ((uint*)(s_yuv_base))[1] = 0;
    //补齐0
    // if (y > height-1 ) {
    //     for (int i=0; i<8*COMPONET; ++i) {
    //         s_yuv_base[i] = 0;
    //     }
    // } else {
    //     if (x0 + 8 > height) {
    //         for (int i=0; i<8*COMPONET; ++i) {
    //             s_yuv_base[i] = 0;
    //         }   
    //     }
    //     //赋值
    //     int sidx = 0, idx = 0;
    //     for (int ix=x0; ix<x1; ++ix) {
    //         idx = y*width + ix;
    //         sidx = ix-x0;
    //         s_yuv_base[sidx] = rgb.d_buffer[3*idx];
    //     }
    // } 

    //赋值
    int sidx = 0, idx = 0;
    for (int ix=x0; ix<x1; ++ix) {
        idx = y*width + ix;
        sidx = ix-x0;
        s_yuv_base[sidx] = rgb.d_buffer[3*idx];
    }

    __syncthreads();

    ///\ 2 quantization
    //row 
    float *quant_out0 = s_quant_base + cal_id*COMPONET*8;
    dct_1d_8_fast((float)s_yuv_base[0], (float)s_yuv_base[1], (float)s_yuv_base[2], (float)s_yuv_base[3], 
                  (float)s_yuv_base[4], (float)s_yuv_base[5], (float)s_yuv_base[6], (float)s_yuv_base[7],
                  quant_out0[0], quant_out0[1], quant_out0[2], quant_out0[3], quant_out0[4], quant_out0[5], quant_out0[6], quant_out0[7], 128);
    
    //collumn
    float *quant_out1 = s_quant_base + cal_id;
    dct_1d_8_fast(quant_out1[0], quant_out1[1*8], quant_out1[2*8], quant_out1[3*8], quant_out1[4*8], quant_out1[5*8], quant_out1[6*8], quant_out1[7*8],
                  quant_out1[0], quant_out1[1*8], quant_out1[2*8], quant_out1[3*8], quant_out1[4*8], quant_out1[5*8], quant_out1[6*8], quant_out1[7*8]);

    //write 
    float* tbl = S_DCT_TABLE;
    const int id = cal_id*8;
    int out0 = rintf(quant_out0[0]*tbl[id]);
    int out1 = rintf(quant_out0[1]*tbl[id+1]);
    int out2 = rintf(quant_out0[2]*tbl[id+2]);
    int out3 = rintf(quant_out0[3]*tbl[id+3]);
    int out4 = rintf(quant_out0[4]*tbl[id+4]);
    int out5 = rintf(quant_out0[5]*tbl[id+5]);
    int out6 = rintf(quant_out0[6]*tbl[id+6]);
    int out7 = rintf(quant_out0[7]*tbl[id+7]);
    
    short* quant_write = (short*)dct_result.d_buffer + mcu_id*64*COMPONET + cal_id*COMPONET*8;
    ((uint4*)(quant_write))[0] = make_uint4(
        (out0 & 0xFFFF) + (out1 << 16),
        (out2 & 0xFFFF) + (out3 << 16),
        (out4 & 0xFFFF) + (out5 << 16),
        (out6 & 0xFFFF) + (out7 << 16)
    );
}


extern "C"
cudaError_t rgb_2_yuv_2_dct_op(const BlockUnit& rgb, const BlockUnit& dct_result, const ImageInfo& img_info, const DCTTable& dct_table) {
    // const int BLOCK_SIZEX = 4;
    // const int BLOCK_SIZEY = 4;
    // dim3 block(BLOCK_SIZEX, BLOCK_SIZEY, 1);
    // dim3 grid(img_info.mcu_w / BLOCK_SIZEX, img_info.mcu_h / BLOCK_SIZEY);
    // if (grid.x * BLOCK_SIZEX != img_info.mcu_w) {
    //     grid.x += 1;
    // }
    // if (grid.y * BLOCK_SIZEY != img_info.mcu_h) {
    //     grid.y += 1;
    // }

    // kernel_rgb_2_yuv_2_dct << <grid, block >> >(rgb, dct_result, img_info, dct_table);
    
    return cudaDeviceSynchronize();
}

extern "C"
cudaError_t r_2_dct_op(const BlockUnit& rgb, const BlockUnit& dct_result, const ImageInfo& img_info, const DCTTable& dct_table) {
    const int WARP_COUNT = 4;
    //一个warp32个线程计算4个mcu,一个block计算16个mcu
    dim3 block(32*WARP_COUNT);

    int mcu_count = img_info.mcu_w * img_info.mcu_h;
    dim3 grid = (mcu_count/16);
    if (grid.x * 16 != mcu_count) {
        grid.x += 1;
    }

    r_2_dct_op_kernel<WARP_COUNT> <<<grid, block>>>(rgb, dct_result, img_info, dct_table);
    
    return cudaDeviceSynchronize();
}
