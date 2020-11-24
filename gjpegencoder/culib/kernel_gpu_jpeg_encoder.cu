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

template<typename T0, typename T>
__device__ void dct_1d_8_fast(const T0 in0, const T0 in1, const T0 in2, const T0 in3, const T0 in4, const T0 in5, const T0 in6, const T0 in7,
    T & out0, T & out1, T & out2, T & out3, T & out4, T & out5, T & out6, T & out7, const float center_sample = 0.0f) {
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


__shared__ unsigned char _S_ZIG_ZAG[64];
__shared__ float _S_DCT_TABLE[128];

__global__ void kernel_rgb_2_yuv_2_dct(const BlockUnit rgb, const BlockUnit dct_result, const ImageInfo img_info, const DCTTable dct_table) {
    unsigned int mcu_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int mcu_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (mcu_x > img_info.mcu_w-1 || mcu_y > img_info.mcu_h-1) {
        return;
    }
    if (threadIdx.x == 0) {
        for (int i=0; i<64; ++i) {
            _S_ZIG_ZAG[i] = dct_table.d_zig_zag[i];
            _S_DCT_TABLE[i] = dct_table.d_quant_tbl_luminance[i];
            _S_DCT_TABLE[i+64] = dct_table.d_quant_tbl_chrominance[i];
        }
    }
    __syncthreads();

    int mcu_id = mcu_y*img_info.mcu_w + mcu_x;

    int width = img_info.width;
    int height = img_info.height;
    int width_ext = img_info.width_ext;
    int height_ext = img_info.height_ext;

    int x0 = mcu_x * 8;
    int y0 = mcu_y * 8;
    int x1 = x0 + 8;
    int y1 = y0 + 8;
    y1 = y1 < height ? y1 : height;
    x1 = x1 < width ? x1 : width;

    int sidx = 0;
    int idx = 0;
    float r,g,b,y,u,v;
    unsigned char yuv[64*3];
    for (int i=0; i<64*3; ++i) {
        yuv[i] = 0;
    }
    for (int iy=y0; iy<y1; ++iy) {
        for (int ix=x0; ix<x1; ++ix) {
            idx = iy*width + ix;
            sidx = (iy-y0)*8 + (ix-x0);
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

            yuv[sidx] = (unsigned char)y;
            yuv[sidx+64] = (unsigned char)u;
            yuv[sidx+128] = (unsigned char)v;
        }
    }

    float quant_local[64];
    short *quant_base = (short*)dct_result.d_buffer + mcu_id*64*3;
    // float *tbls[3] = {dct_table.d_quant_tbl_luminance, dct_table.d_quant_tbl_chrominance, dct_table.d_quant_tbl_chrominance};
    // unsigned char* ZIGZAG_TABLE = dct_table.d_zig_zag;
    float *tbls[3] = {_S_DCT_TABLE, _S_DCT_TABLE+64, _S_DCT_TABLE+64};
    unsigned char* ZIGZAG_TABLE = _S_ZIG_ZAG;

    for (int j=0; j<3; ++j) {
        short *quant_val = quant_base + 64*j;
        unsigned char *val = yuv + 64*j;
        float* tbl = tbls[j];

        for (int i=0; i<8; ++i) {
            unsigned char* i0 = val + 8*i;
            float* o0 = quant_local + 8*i;
            dct_1d_8_fast<unsigned char, float>(i0[0], i0[1], i0[2], i0[3], i0[4], i0[5], i0[6], i0[7],
                        o0[0], o0[1], o0[2], o0[3], o0[4], o0[5], o0[6], o0[7], 128);
        }

        for (int i=0; i<8; ++i) {
            float* i0 = quant_local + i;
            float* o0 = quant_local + i;
            dct_1d_8_fast<float, float>(i0[0], i0[1*8], i0[2*8], i0[3*8], i0[4*8], i0[5*8], i0[6*8], i0[7*8],
                        o0[0], o0[1*8], o0[2*8], o0[3*8], o0[4*8], o0[5*8], o0[6*8], o0[7*8], 0);
        }

        for (int i=0; i<64; ++i) {
            float v = quant_local[i]*tbl[i];
            if (v < 0.0f) {
                v-=0.5f;
            } else {
                v+=0.5f;
            }
            quant_val[ZIGZAG_TABLE[i]] = (short)v;
        }
    }

}

__device__ BitString get_bit_code(int value) {
	int v = (value > 0) ? value : -value;
	int length = 0;
	for(length = 0; v>0; v >>= 1) {
        length++;
    }

    BitString ret;
	ret.value = value > 0 ? value : (1 << length) + value - 1;
	ret.length = length;
	return ret;
}


__shared__ BitString _S_huffman_table_Y_DC[12];
__shared__ BitString _S_huffman_table_Y_AC[256];
__shared__ BitString _S_huffman_table_CbCr_DC[12];
__shared__ BitString _S_huffman_table_CbCr_AC[256];

__global__ void kernel_huffman_encoding(const BlockUnit dct_result, const BlockUnit huffman_code, int *d_huffman_code_count, const ImageInfo img_info, const HuffmanTable huffman_table) {
    unsigned int mcu_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int mcu_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (mcu_x > img_info.mcu_w-1 || mcu_y > img_info.mcu_h-1) {
        return;
    }
    // if (threadIdx.x == 0) {
    //     for (int i=0; i<12; ++i) {
    //         _S_huffman_table_Y_DC[i] = huffman_table.d_huffman_table_Y_DC[i];
    //         _S_huffman_table_CbCr_DC[i] = huffman_table.d_huffman_table_CbCr_DC[i];
    //     }
    //     for (int i=0; i<256; ++i) {
    //         _S_huffman_table_Y_AC[i] = huffman_table.d_huffman_table_Y_AC[i];
    //         _S_huffman_table_CbCr_AC[i] = huffman_table.d_huffman_table_CbCr_AC[i];
    //     }
        
    // }
    // __syncthreads();

    int mcu_id = mcu_y*img_info.mcu_w + mcu_x;

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


    BitString* HTDCs[3] = {huffman_table.d_huffman_table_Y_DC, huffman_table.d_huffman_table_CbCr_DC, huffman_table.d_huffman_table_CbCr_DC};
    BitString* HTACs[3] = {huffman_table.d_huffman_table_Y_AC, huffman_table.d_huffman_table_CbCr_AC, huffman_table.d_huffman_table_CbCr_AC};
    // BitString* HTDCs[3] = {_S_huffman_table_Y_DC, _S_huffman_table_CbCr_DC, _S_huffman_table_CbCr_DC};
    // BitString* HTACs[3] = {_S_huffman_table_Y_AC, _S_huffman_table_CbCr_AC, _S_huffman_table_CbCr_AC};

    short *quant_base = (short*)dct_result.d_buffer + mcu_id*64*3;
    BitString* output_base = (BitString*)huffman_code.d_buffer + mcu_id*128*3;
    int* output_count = d_huffman_code_count + mcu_id*3;
    short preDC[3] = {0,0,0};
    if (mcu_id != 0) {
        preDC[0] = *(quant_base-64*3);
        preDC[1] = *(quant_base-64*2);
        preDC[2] = *(quant_base-64);
    }

    for (int j=0; j<3; ++j) {
        short *quant = quant_base + 64*j;
        BitString *output = output_base + 128*j;
        BitString *HTDC = HTDCs[j];
        BitString *HTAC = HTACs[j];

        int index = 0;
        //encode DC
        const int diffDC = quant[0] - preDC[j];
        
        if (0 == diffDC) {
            output[index++] = HTDC[0];
        } else {
            BitString bs = get_bit_code(diffDC);
            output[index++] = HTDC[bs.length];
            output[index++] = bs;
        }

        //encode AC
        BitString EOB = HTAC[0x00];
        BitString SIXTEEN_ZEROS = HTAC[0xF0];

        int end_pos = 63;
        while (end_pos > 0 && quant[end_pos] == 0 ) {
            --end_pos;
        }

        for (int i=1; i<=end_pos; ) {
            int start_pos = i;
            while(quant[i] == 0 && i <= end_pos) {
                ++i;
            }

            int zero_counts = i - start_pos;
            if (zero_counts >= 16) {
                for (int j=0; j < zero_counts/16; ++j)
                    output[index++] = SIXTEEN_ZEROS;
                zero_counts = zero_counts%16;
            }

            BitString bs = get_bit_code(quant[i]);

            output[index++] = HTAC[(zero_counts << 4) | bs.length];
            output[index++] = bs;
            i++;
        }

        if (end_pos != 63) {
            output[index++] = EOB;
        }

        output_count[j] = index;
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

extern "C"
cudaError_t rgb_2_yuv_2_dct(const BlockUnit& rgb, const BlockUnit& dct_result, const ImageInfo& img_info, const DCTTable& dct_table) {
    const int BLOCK_SIZEX = 4;
    const int BLOCK_SIZEY = 4;
    dim3 block(BLOCK_SIZEX, BLOCK_SIZEY, 1);
    dim3 grid(img_info.mcu_w / BLOCK_SIZEX, img_info.mcu_h / BLOCK_SIZEY);
    if (grid.x * BLOCK_SIZEX != img_info.mcu_w) {
        grid.x += 1;
    }
    if (grid.y * BLOCK_SIZEY != img_info.mcu_h) {
        grid.y += 1;
    }

    kernel_rgb_2_yuv_2_dct << <grid, block >> >(rgb, dct_result, img_info, dct_table);
    
    return cudaDeviceSynchronize();
}

extern "C"
cudaError_t huffman_encoding(const BlockUnit& dct_result, const BlockUnit& huffman_code, int *d_huffman_code_count, const ImageInfo& img_info, const HuffmanTable& huffman_table) {
    const int BLOCK_SIZEX = 4;
    const int BLOCK_SIZEY = 4;
    dim3 block(BLOCK_SIZEX, BLOCK_SIZEY, 1);
    dim3 grid(img_info.mcu_w / BLOCK_SIZEX, img_info.mcu_h / BLOCK_SIZEY);
    if (grid.x * BLOCK_SIZEX != img_info.mcu_w) {
        grid.x += 1;
    }
    if (grid.y * BLOCK_SIZEY != img_info.mcu_h) {
        grid.y += 1;
    }

    kernel_huffman_encoding << <grid, block >> >(dct_result, huffman_code, d_huffman_code_count, img_info, huffman_table);
    
    return cudaDeviceSynchronize();
}

