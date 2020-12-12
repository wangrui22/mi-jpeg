#include <vector_types.h>
#include <cuda_runtime.h>
#include "mi_gpu_jpeg_define.h"

#define gpujpeg_div_and_round_up(value, div) \
    ((((value) % (div)) != 0) ? ((value) / (div) + 1) : ((value) / (div)))

template <typename T>
__device__ static inline void
gpujpeg_dct_gpu(const T in0, const T in1, const T in2, const T in3, const T in4, const T in5, const T in6, const T in7,
                T & out0, T & out1, T & out2, T & out3, T & out4, T & out5, T & out6, T & out7,
                const float level_shift_8 = 0.0f)
{
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

    out0 = even0 + even1 + level_shift_8;
    out1 = odd_diff1 + odd_diff4;
    out2 = even3 + even_diff * 0.707106781f;
    out3 = odd_diff3 - odd_diff2;
    out4 = even0 - even1;
    out5 = odd_diff3 + odd_diff2;
    out6 = even3 - even_diff * 0.707106781f;
    out7 = odd_diff1 - odd_diff4;
}

template <int WARP_COUNT>
__global__ void gpujpeg_dct_gpu_kernel(int block_count_x, int block_count_y, unsigned char* source, const unsigned int source_stride,
                       int16_t* output, int output_stride, const float * const quant_table)
{
    // each warp processes 4 8x8 blocks (horizontally neighboring)
    const int block_idx_x = threadIdx.x >> 3;
    const int block_idx_y = threadIdx.y;

    // offset of threadblocks's blocks in the image (along both axes)
    const int block_offset_x = blockIdx.x * 4;
    const int block_offset_y = blockIdx.y * WARP_COUNT;

    // stop if thread's block is out of image
    const bool processing = block_offset_x + block_idx_x < block_count_x
                         && block_offset_y + block_idx_y < block_count_y;
    if(!processing) {
        return;
    }

    // index of row/column processed by this thread within its 8x8 block
    const int dct_idx = threadIdx.x & 7;

    // data type of transformed coefficients
    typedef float dct_t;

    // dimensions of shared buffer (compile time constants)
    enum {
        // 4 8x8 blocks, padded to odd number of 4byte banks
        SHARED_STRIDE = ((32 * sizeof(dct_t)) | 4) / sizeof(dct_t),

        // number of shared buffer items needed for 1 warp
        SHARED_SIZE_WARP = SHARED_STRIDE * 8,

        // total number of items in shared buffer
        SHARED_SIZE_TOTAL = SHARED_SIZE_WARP * WARP_COUNT
    };

    // buffer for transpositions of all blocks
    __shared__ dct_t s_transposition_all[SHARED_SIZE_TOTAL];

    // pointer to begin of transposition buffer for thread's block
    dct_t * const s_transposition = s_transposition_all + block_idx_y * SHARED_SIZE_WARP + block_idx_x * 8;

    // input coefficients pointer (each thread loads 1 column of 8 coefficients from its 8x8 block)
    const int in_x = (block_offset_x + block_idx_x) * 8 + dct_idx;
    const int in_y = (block_offset_y + block_idx_y) * 8;
    const int in_offset = in_x + in_y * source_stride;
    const unsigned char * in = source + in_offset*3;

    // load all 8 coefficients of thread's column, but do NOT apply level shift now - will be applied as part of DCT
    dct_t src0 = *in;
    in += source_stride*3;
    dct_t src1 = *in;
    in += source_stride*3;
    dct_t src2 = *in;
    in += source_stride*3;
    dct_t src3 = *in;
    in += source_stride*3;
    dct_t src4 = *in;
    in += source_stride*3;
    dct_t src5 = *in;
    in += source_stride*3;
    dct_t src6 = *in;
    in += source_stride*3;
    dct_t src7 = *in;

    // destination pointer into shared transpose buffer (each thread saves one column)
    dct_t * const s_dest = s_transposition + dct_idx;

    // transform the column (vertically) and save it into the transpose buffer
    gpujpeg_dct_gpu(src0, src1, src2, src3, src4, src5, src6, src7,
                    s_dest[SHARED_STRIDE * 0],
                    s_dest[SHARED_STRIDE * 1],
                    s_dest[SHARED_STRIDE * 2],
                    s_dest[SHARED_STRIDE * 3],
                    s_dest[SHARED_STRIDE * 4],
                    s_dest[SHARED_STRIDE * 5],
                    s_dest[SHARED_STRIDE * 6],
                    s_dest[SHARED_STRIDE * 7],
                    -1024.0f  // = 8 * -128 ... level shift sum for all 8 coefficients
    );

    //TODO 这里感觉应该是要同步的
    __syncthreads();
    
    // read coefficients back - each thread reads one row (no need to sync - only threads within same warp work on each block)
    // ... and transform the row horizontally
    volatile dct_t * s_src = s_transposition + SHARED_STRIDE * dct_idx;
    dct_t dct0, dct1, dct2, dct3, dct4, dct5, dct6, dct7;
    gpujpeg_dct_gpu(s_src[0], s_src[1], s_src[2], s_src[3], s_src[4], s_src[5], s_src[6], s_src[7],
                    dct0, dct1, dct2, dct3, dct4, dct5, dct6, dct7);

    // apply quantization to the row of coefficients (quantization table is actually transposed in global memory for coalesced memory acceses)
    const float * const quantization_row = quant_table + dct_idx; // Cached global memory reads for CCs >= 2.0
    const int out0 = rintf(dct0 * quantization_row[0 * 8]);
    const int out1 = rintf(dct1 * quantization_row[1 * 8]);
    const int out2 = rintf(dct2 * quantization_row[2 * 8]);
    const int out3 = rintf(dct3 * quantization_row[3 * 8]);
    const int out4 = rintf(dct4 * quantization_row[4 * 8]);
    const int out5 = rintf(dct5 * quantization_row[5 * 8]);
    const int out6 = rintf(dct6 * quantization_row[6 * 8]);
    const int out7 = rintf(dct7 * quantization_row[7 * 8]);

    // using single write, save output row packed into 16 bytes
    const int out_x = (block_offset_x + block_idx_x) * 64; // 64 coefficients per one transformed and quantized block
    const int out_y = (block_offset_y + block_idx_y) * output_stride;
    ((uint4*)(output + out_x + out_y))[dct_idx] = make_uint4(
        (out0 & 0xFFFF) + (out1 << 16),
        (out2 & 0xFFFF) + (out3 << 16),
        (out4 & 0xFFFF) + (out5 << 16),  // ... & 0xFFFF keeps only lower 16 bits - useful for negative numbers, which have 1s in upper bits
        (out6 & 0xFFFF) + (out7 << 16)
    );
}

__device__ void jpeg_fdct_8x8(float* data, unsigned char* sample_data) {
    float tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    float tmp10, tmp11, tmp12, tmp13;
    float z1, z2, z3, z4, z5, z11, z13;
    float *dataptr;
    unsigned char* elemptr;
    int ctr;
    const int DCTSIZE = 8; 
    const float CENTERJSAMPLE = 128.0f;

    /* Pass 1: process rows. */

    dataptr = data;
    for (ctr = 0; ctr < DCTSIZE; ctr++) {
        elemptr = sample_data + ctr*DCTSIZE;

        /* Load data into workspace */
        tmp0 = (float)(elemptr[0]) + (float)(elemptr[7]);
        tmp7 = (float)(elemptr[0]) - (float)(elemptr[7]);
        tmp1 = (float)(elemptr[1]) + (float)(elemptr[6]);
        tmp6 = (float)(elemptr[1]) - (float)(elemptr[6]);
        tmp2 = (float)(elemptr[2]) + (float)(elemptr[5]);
        tmp5 = (float)(elemptr[2]) - (float)(elemptr[5]);
        tmp3 = (float)(elemptr[3]) + (float)(elemptr[4]);
        tmp4 = (float)(elemptr[3]) - (float)(elemptr[4]);

        /* Even part */

        tmp10 = tmp0 + tmp3;	/* phase 2 */
        tmp13 = tmp0 - tmp3;
        tmp11 = tmp1 + tmp2;
        tmp12 = tmp1 - tmp2;

        /* Apply unsigned->signed conversion. */
        dataptr[0] = tmp10 + tmp11 - 8 * CENTERJSAMPLE; /* phase 3 */
        dataptr[4] = tmp10 - tmp11;

        z1 = (tmp12 + tmp13) * ((float) 0.707106781); /* c4 */
        dataptr[2] = tmp13 + z1;	/* phase 5 */
        dataptr[6] = tmp13 - z1;

        /* Odd part */

        tmp10 = tmp4 + tmp5;	/* phase 2 */
        tmp11 = tmp5 + tmp6;
        tmp12 = tmp6 + tmp7;

        /* The rotator is modified from fig 4-8 to avoid extra negations. */
        z5 = (tmp10 - tmp12) * ((float) 0.382683433); /* c6 */
        z2 = ((float) 0.541196100) * tmp10 + z5; /* c2-c6 */
        z4 = ((float) 1.306562965) * tmp12 + z5; /* c2+c6 */
        z3 = tmp11 * ((float) 0.707106781); /* c4 */

        z11 = tmp7 + z3;		/* phase 5 */
        z13 = tmp7 - z3;

        dataptr[5] = z13 + z2;	/* phase 6 */
        dataptr[3] = z13 - z2;
        dataptr[1] = z11 + z4;
        dataptr[7] = z11 - z4;

        dataptr += DCTSIZE;		/* advance pointer to next row */
    }

    /* Pass 2: process columns. */

    dataptr = data;
    for (ctr = DCTSIZE-1; ctr >= 0; ctr--) {
        tmp0 = dataptr[DCTSIZE*0] + dataptr[DCTSIZE*7];
        tmp7 = dataptr[DCTSIZE*0] - dataptr[DCTSIZE*7];
        tmp1 = dataptr[DCTSIZE*1] + dataptr[DCTSIZE*6];
        tmp6 = dataptr[DCTSIZE*1] - dataptr[DCTSIZE*6];
        tmp2 = dataptr[DCTSIZE*2] + dataptr[DCTSIZE*5];
        tmp5 = dataptr[DCTSIZE*2] - dataptr[DCTSIZE*5];
        tmp3 = dataptr[DCTSIZE*3] + dataptr[DCTSIZE*4];
        tmp4 = dataptr[DCTSIZE*3] - dataptr[DCTSIZE*4];

        /* Even part */

        tmp10 = tmp0 + tmp3;	/* phase 2 */
        tmp13 = tmp0 - tmp3;
        tmp11 = tmp1 + tmp2;
        tmp12 = tmp1 - tmp2;

        dataptr[DCTSIZE*0] = tmp10 + tmp11; /* phase 3 */
        dataptr[DCTSIZE*4] = tmp10 - tmp11;

        z1 = (tmp12 + tmp13) * ((float) 0.707106781); /* c4 */
        dataptr[DCTSIZE*2] = tmp13 + z1; /* phase 5 */
        dataptr[DCTSIZE*6] = tmp13 - z1;

        /* Odd part */

        tmp10 = tmp4 + tmp5;	/* phase 2 */
        tmp11 = tmp5 + tmp6;
        tmp12 = tmp6 + tmp7;

        /* The rotator is modified from fig 4-8 to avoid extra negations. */
        z5 = (tmp10 - tmp12) * ((float) 0.382683433); /* c6 */
        z2 = ((float) 0.541196100) * tmp10 + z5; /* c2-c6 */
        z4 = ((float) 1.306562965) * tmp12 + z5; /* c2+c6 */
        z3 = tmp11 * ((float) 0.707106781); /* c4 */

        z11 = tmp7 + z3;		/* phase 5 */
        z13 = tmp7 - z3;

        dataptr[DCTSIZE*5] = z13 + z2; /* phase 6 */
        dataptr[DCTSIZE*3] = z13 - z2;
        dataptr[DCTSIZE*1] = z11 + z4;
        dataptr[DCTSIZE*7] = z11 - z4;

        dataptr++;			/* advance pointer to next column */
    }
}

template<typename T0, typename T>
inline __device__ void dct_1d_8_fast(const T0 in0, const T0 in1, const T0 in2, const T0 in3, const T0 in4, const T0 in5, const T0 in6, const T0 in7,
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

inline __device__ unsigned char gpujpeg_clamp(int value) {
    value = (value >= 0) ? value : 0;
    value = (value <= 255) ? value : 255;
    return (unsigned char)value;
}

template<int bit_depth> inline __device__ void
gpujpeg_color_transform_to(unsigned char & c1, unsigned char & c2, unsigned char & c3, const int matrix[9], int base1, int base2, int base3)
{
    // Prepare integer constants
    const int middle = 1 << (bit_depth - 1);

    // Perform color transform
    int r1 = (int)c1 * 256 / 255;
    int r2 = (int)c2 * 256 / 255;
    int r3 = (int)c3 * 256 / 255;
    c1 = gpujpeg_clamp(((matrix[0] * r1 + matrix[1] * r2 + matrix[2] * r3 + middle) >> bit_depth) + base1);
    c2 = gpujpeg_clamp(((matrix[3] * r1 + matrix[4] * r2 + matrix[5] * r3 + middle) >> bit_depth) + base2);
    c3 = gpujpeg_clamp(((matrix[6] * r1 + matrix[7] * r2 + matrix[8] * r3 + middle) >> bit_depth) + base3);
}

__device__ void rgb_2_yuv_unit(unsigned char & c1, unsigned char & c2, unsigned char & c3) {
    /*const double matrix[] = {
          0.299000,  0.587000,  0.114000,
         -0.147400, -0.289500,  0.436900,
          0.615000, -0.515000, -0.100000
    };*/
    const int matrix[] = {77, 150, 29, -38, -74, 112, 157, -132, -26};
    gpujpeg_color_transform_to<8>(c1, c2, c3, matrix, 0, 128, 128);
}


__global__ void kernel_rgb_2_yuv_2_dct(const BlockUnit rgb, const BlockUnit dct_result, const ImageInfo img_info, const DCTTable dct_table) {
    if (threadIdx.x == 0) {
        for (int i=0; i<64; ++i) {
            _S_ZIG_ZAG[i] = dct_table.d_zig_zag[i];
            _S_DCT_TABLE[i] = dct_table.d_quant_tbl_luminance[i];
            _S_DCT_TABLE[i+64] = dct_table.d_quant_tbl_chrominance[i];
        }
    }
    __syncthreads();

    unsigned int mcu_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int mcu_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (mcu_x > img_info.mcu_w-1 || mcu_y > img_info.mcu_h-1) {
        return;
    }

    int mcu_id = mcu_y*img_info.mcu_w + mcu_x;

    int width = img_info.width;
    int height = img_info.height;
    //int width_ext = img_info.width_ext;
    //int height_ext = img_info.height_ext;

    int x0 = mcu_x * 8;
    int y0 = mcu_y * 8;
    int x1 = x0 + 8;
    int y1 = y0 + 8;
    y1 = y1 < height+1 ? y1 : height;
    x1 = x1 < width+1 ? x1 : width;

    int sidx = 0;
    int idx = 0;
    float r,g,b,y,u,v;
    //unsigned char r,g,b;
    unsigned char yuv[64*3];
    for (int i=0; i<64*3; ++i) {
        yuv[i] = 128;
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
            
            //和调用rgb_2_yuv_unit性能上差别不大
            // r = rgb.d_buffer[idx*3];
            // g = rgb.d_buffer[idx*3+1];
            // b = rgb.d_buffer[idx*3+2];
            // rgb_2_yuv_unit(r,g,b);
            // yuv[sidx] = r;
            // yuv[sidx+64] = g;
            // yuv[sidx+128] = b;

        }
    }

    float quant_local[64];
    short *quant_base = (short*)dct_result.d_buffer + mcu_id*64*3;
    // float *tbls[3] = {dct_table.d_quant_tbl_luminance, dct_table.d_quant_tbl_chrominance, dct_table.d_quant_tbl_chrominance};
    // unsigned char* ZIGZAG_TABLE = dct_table.d_zig_zag;
    float *tbls[3] = {_S_DCT_TABLE, _S_DCT_TABLE+64, _S_DCT_TABLE+64};
    //unsigned char* ZIGZAG_TABLE = _S_ZIG_ZAG;

    for (int j=0; j<3; ++j) {
        short *quant_val = quant_base + 64*j;
        unsigned char *val = yuv + 64*j;
        float* tbl = tbls[j];

        for (int i=0; i<8; ++i) {
            unsigned char* i0 = val + 8*i;
            float* o0 = quant_local + 8*i;
            dct_1d_8_fast<unsigned char, float>((float)i0[0], (float)i0[1], (float)i0[2], (float)i0[3], (float)i0[4], (float)i0[5], (float)i0[6], (float)i0[7],
                        o0[0], o0[1], o0[2], o0[3], o0[4], o0[5], o0[6], o0[7], 128);
        }

        for (int i=0; i<8; ++i) {
            float* i0 = quant_local + i;
            float* o0 = quant_local + i;
            dct_1d_8_fast<float, float>(i0[0], i0[1*8], i0[2*8], i0[3*8], i0[4*8], i0[5*8], i0[6*8], i0[7*8],
                        o0[0], o0[1*8], o0[2*8], o0[3*8], o0[4*8], o0[5*8], o0[6*8], o0[7*8], 0);
        }

        for (int i=0; i<64; ++i) {
            // float v = quant_local[i]*tbl[i];
            // if (v < 0.0f) {
            //     //v-=0.5f;
            //     printf("fu val %d\n", v);
            // } else {
            //     v+=0.5f;
            // }
            // quant_val[ZIGZAG_TABLE[i]] = (short)v;
            
            //quant_val[ZIGZAG_TABLE[i]] = (short)rintf(quant_local[i]*tbl[i]);

            //这里不要zigzag
            quant_val[i] = (short)rintf(quant_local[i]*tbl[i]);
        }
    }

}

__global__ void kernel_r_2_dct(const BlockUnit rgb, const BlockUnit dct_result, const ImageInfo img_info, const DCTTable dct_table) {
    const int component = img_info.component;
    if (threadIdx.x == 0) {
        for (int i=0; i<64; ++i) {
            _S_ZIG_ZAG[i] = dct_table.d_zig_zag[i];
            _S_DCT_TABLE[i] = dct_table.d_quant_tbl_luminance[i];
            //_S_DCT_TABLE[i+64] = dct_table.d_quant_tbl_chrominance[i];
        }
    }
    __syncthreads();

    unsigned int mcu_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int mcu_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (mcu_x > img_info.mcu_w-1 || mcu_y > img_info.mcu_h-1) {
        return;
    }

    int mcu_id = mcu_y*img_info.mcu_w + mcu_x;

    int width = img_info.width;
    int height = img_info.height;
    //int width_ext = img_info.width_ext;
    //int height_ext = img_info.height_ext;

    int x0 = mcu_x * 8;
    int y0 = mcu_y * 8;
    int x1 = x0 + 8;
    int y1 = y0 + 8;
    y1 = y1 < height+1 ? y1 : height;
    x1 = x1 < width+1 ? x1 : width;

    int sidx = 0;
    int idx = 0;
    unsigned char yuv[64];
    for (int i=0; i<64; ++i) {
        yuv[i] = 0;
    }
    for (int iy=y0; iy<y1; ++iy) {
        for (int ix=x0; ix<x1; ++ix) {
            idx = iy*width + ix;
            sidx = (iy-y0)*8 + (ix-x0);
            yuv[sidx] =rgb.d_buffer[idx*3];
        }
    }

    float quant_local[64];
    short *quant_base = (short*)dct_result.d_buffer + mcu_id*64*component;
    //unsigned char* ZIGZAG_TABLE = _S_ZIG_ZAG;

    //short *quant_val = quant_base;
    unsigned char *val = yuv;
    float* tbl = _S_DCT_TABLE;

    jpeg_fdct_8x8(quant_local, val);

    // for (int i=0; i<8; ++i) {
    //     unsigned char* i0 = val + 8*i;
    //     float* o0 = quant_local + 8*i;
    //     dct_1d_8_fast<unsigned char, float>((float)i0[0], (float)i0[1], (float)i0[2], (float)i0[3], (float)i0[4], (float)i0[5], (float)i0[6], (float)i0[7],
    //                 o0[0], o0[1], o0[2], o0[3], o0[4], o0[5], o0[6], o0[7], 128);
    // }

    // for (int i=0; i<8; ++i) {
    //     float* i0 = quant_local + i;
    //     float* o0 = quant_local + i;
    //     dct_1d_8_fast<float, float>(i0[0], i0[1*8], i0[2*8], i0[3*8], i0[4*8], i0[5*8], i0[6*8], i0[7*8],
    //                 o0[0], o0[1*8], o0[2*8], o0[3*8], o0[4*8], o0[5*8], o0[6*8], o0[7*8], 0);
    // }

    // for (int i=0; i<64; ++i) {
    //     // float v = quant_local[i]*tbl[i];
    //     // if (v < 0.0f) {
    //     //     v-=0.5f;
    //     // } else {
    //     //     v+=0.5f;
    //     // }
    //     // quant_val[ZIGZAG_TABLE[i]] = (short)v;

    //     //quant_val[ZIGZAG_TABLE[i]] = (short)rintf(quant_local[i]*tbl[i]);
    //     //这里不要zigzag
    //     quant_val[i] = (short)rintf(quant_local[i]*tbl[i]);
    // }

    int out0,out1,out2,out3,out4,out5,out6,out7;
    for (int i=0; i<8; ++i) {
        int id = i*8;
        out0 = rintf(quant_local[id]*tbl[id]);
        out1 = rintf(quant_local[id+1]*tbl[id+1]);
        out2 = rintf(quant_local[id+2]*tbl[id+2]);
        out3 = rintf(quant_local[id+3]*tbl[id+3]);
        out4 = rintf(quant_local[id+4]*tbl[id+4]);
        out5 = rintf(quant_local[id+5]*tbl[id+5]);
        out6 = rintf(quant_local[id+6]*tbl[id+6]);
        out7 = rintf(quant_local[id+7]*tbl[id+7]);

        ((uint4*)(quant_base))[i] = make_uint4(
            (out0 & 0xFFFF) + (out1 << 16),
            (out2 & 0xFFFF) + (out3 << 16),
            (out4 & 0xFFFF) + (out5 << 16),
            (out6 & 0xFFFF) + (out7 << 16)
        );   
    }
}

__global__ void kernel_r_2_dct_ext(const BlockUnit rgb, const BlockUnit dct_result, const ImageInfo img_info, const DCTTable dct_table, int CAL_UNIT) {
    unsigned int mcu_x0 = (blockIdx.x * blockDim.x + threadIdx.x)*CAL_UNIT;
    unsigned int mcu_y0 = (blockIdx.y * blockDim.y + threadIdx.y)*CAL_UNIT;

    const int component = img_info.component;
    if (threadIdx.x == 0) {
        for (int i=0; i<64; ++i) {
            //_S_ZIG_ZAG[i] = dct_table.d_zig_zag[i];
            _S_DCT_TABLE[i] = dct_table.d_quant_tbl_luminance[i];
            //_S_DCT_TABLE[i+64] = dct_table.d_quant_tbl_chrominance[i];
        }
    }
    __syncthreads();

    if (mcu_x0 > img_info.mcu_w-1 || mcu_y0 > img_info.mcu_h-1) {
        return;
    }

    unsigned int mcu_x1 = mcu_x0+CAL_UNIT;
    if (mcu_x1 > img_info.mcu_w-1) {
        mcu_x1 = img_info.mcu_w-1;
    }
    unsigned int mcu_y1 = mcu_y0+CAL_UNIT;
    if (mcu_y1 > img_info.mcu_h-1) {
        mcu_y1 = img_info.mcu_h-1;
    }

    for (int mcu_y = mcu_y0; mcu_y < mcu_y1; ++mcu_y) {
        for (int mcu_x = mcu_x0; mcu_x < mcu_x1; ++mcu_x) {
            int mcu_id = mcu_y*img_info.mcu_w + mcu_x;

            int width = img_info.width;
            int height = img_info.height;
            //int width_ext = img_info.width_ext;
            //int height_ext = img_info.height_ext;

            int x0 = mcu_x * 8;
            int y0 = mcu_y * 8;
            int x1 = x0 + 8;
            int y1 = y0 + 8;
            y1 = y1 < height+1 ? y1 : height;
            x1 = x1 < width+1 ? x1 : width;

            int sidx = 0;
            int idx = 0;
            unsigned char yuv[64];
            for (int i=0; i<64; ++i) {
                yuv[i] = 0;
            }
            for (int iy=y0; iy<y1; ++iy) {
                for (int ix=x0; ix<x1; ++ix) {
                    idx = iy*width + ix;
                    sidx = (iy-y0)*8 + (ix-x0);
                    yuv[sidx] =rgb.d_buffer[idx*3];
                }
            }

            float quant_local[64];
            short *quant_base = (short*)dct_result.d_buffer + mcu_id*64*component;
            //unsigned char* ZIGZAG_TABLE = _S_ZIG_ZAG;

            //short *quant_val = quant_base;
            unsigned char *val = yuv;
            float* tbl = _S_DCT_TABLE;

            jpeg_fdct_8x8(quant_local, val);

            int out0,out1,out2,out3,out4,out5,out6,out7;
            for (int i=0; i<8; ++i) {
                int id = i*8;
                out0 = rintf(quant_local[id]*tbl[id]);
                out1 = rintf(quant_local[id+1]*tbl[id+1]);
                out2 = rintf(quant_local[id+2]*tbl[id+2]);
                out3 = rintf(quant_local[id+3]*tbl[id+3]);
                out4 = rintf(quant_local[id+4]*tbl[id+4]);
                out5 = rintf(quant_local[id+5]*tbl[id+5]);
                out6 = rintf(quant_local[id+6]*tbl[id+6]);
                out7 = rintf(quant_local[id+7]*tbl[id+7]);

                ((uint4*)(quant_base))[i] = make_uint4(
                    (out0 & 0xFFFF) + (out1 << 16),
                    (out2 & 0xFFFF) + (out3 << 16),
                    (out4 & 0xFFFF) + (out5 << 16),
                    (out6 & 0xFFFF) + (out7 << 16)
                );   
            }
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
__shared__ int _S_ORDER_NATURAL[80];

__global__ void kernel_huffman_encoding(const BlockUnit dct_result, const BlockUnit huffman_code, int *d_huffman_code_count, const ImageInfo img_info, const HuffmanTable huffman_table) {
    unsigned int mcu_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int mcu_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int component = img_info.component;
    int* order_natural_ori = (int*)huffman_table.d_order_natural;
    int idxxx = threadIdx.y*blockDim.x + threadIdx.x;
    if (idxxx < 16) {
        _S_ORDER_NATURAL[idxxx*4] = order_natural_ori[idxxx*4];
        _S_ORDER_NATURAL[idxxx*4+1] = order_natural_ori[idxxx*4+1];
        _S_ORDER_NATURAL[idxxx*4+2] = order_natural_ori[idxxx*4+2];
        _S_ORDER_NATURAL[idxxx*4+3] = order_natural_ori[idxxx*4+3];
    }
    __syncthreads();

    if (mcu_x > img_info.mcu_w-1 || mcu_y > img_info.mcu_h-1) {
        return;
    }

    int mcu_id = mcu_y*img_info.mcu_w + mcu_x;

    BitString* HTDCs[3] = {huffman_table.d_huffman_table_Y_DC, huffman_table.d_huffman_table_CbCr_DC, huffman_table.d_huffman_table_CbCr_DC};
    BitString* HTACs[3] = {huffman_table.d_huffman_table_Y_AC, huffman_table.d_huffman_table_CbCr_AC, huffman_table.d_huffman_table_CbCr_AC};
    //BitString* HTDCs[3] = {_S_huffman_table_Y_DC, _S_huffman_table_CbCr_DC, _S_huffman_table_CbCr_DC};
    //BitString* HTACs[3] = {_S_huffman_table_Y_AC, _S_huffman_table_CbCr_AC, _S_huffman_table_CbCr_AC};

    short *quant_base = (short*)dct_result.d_buffer + mcu_id*64*component;
    BitString* output_base = (BitString*)huffman_code.d_buffer + mcu_id*256*component;
    int* output_count = d_huffman_code_count + mcu_id*component;
    //int* order_natural = (int*)huffman_table.d_order_natural; 
    int* order_natural = _S_ORDER_NATURAL; 
    
    int segment_id = mcu_id/img_info.segment_mcu_count;
    int mcu_id_in_seg = mcu_id - segment_id*img_info.segment_mcu_count;
    short preDC[3] = {0,0,0};
    if (component == 3) {
        if (mcu_id_in_seg != 0) {
            preDC[0] = *(quant_base-64*3);
            preDC[1] = *(quant_base-64*2);
            preDC[2] = *(quant_base-64);
        }
    } else {
        if (mcu_id_in_seg != 0) {
            preDC[0] = *(quant_base-64);
        }
    }
    

    for (int j=0; j<component; ++j) {
        short *quant = quant_base + 64*j;
        BitString *output = output_base + 256*j;
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
        while (end_pos > 0 && quant[order_natural[end_pos]] == 0 ) {
            --end_pos;
        }

        for (int i=1; i<=end_pos; ) {
            int start_pos = i;
            while(quant[order_natural[i]] == 0 && i <= end_pos) {
                ++i;
            }

            int zero_counts = i - start_pos;
            if (zero_counts >= 16) {
                for (int j=0; j < zero_counts/16; ++j)
                    output[index++] = SIXTEEN_ZEROS;
                zero_counts = zero_counts%16;
            }

            BitString bs = get_bit_code(quant[order_natural[i]]);

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

__device__ void write_byte(unsigned char val, unsigned char* buffer, int& byte) {
    *buffer = val;
    byte += 1;
}

__device__ void write_bitstring(const BitString* bs, int counts, int& new_byte, int& new_byte_pos, unsigned char* buffer, int& byte) {
    const unsigned short mask[] = {1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768};
	for(int i=0; i<counts; ++i) {
		int value = bs[i].value;
		int posval = bs[i].length - 1;
		while (posval >= 0) {
			if ((value & mask[posval]) != 0) {
				new_byte = new_byte  | mask[new_byte_pos];
			}
			posval--;
			new_byte_pos--;
			if (new_byte_pos < 0) {
				write_byte((unsigned char)(new_byte), buffer++, byte);
				if (new_byte == 0xFF){
					//special case
					write_byte((unsigned char)(0x00), buffer++, byte);
				}
				new_byte_pos = 7;
				new_byte = 0;
			}
		}
	}
}

__global__ void kernel_huffman_writebits(const BlockUnit huffman_code, int *d_huffman_code_count, const ImageInfo img_info, BlockUnit segment_compressed, int *d_segment_compressed_byte) {
    unsigned int segid = blockIdx.x * blockDim.x + threadIdx.x;
    if (segid > img_info.segment_count-1) {
        return;
    }
    const int component = img_info.component;

    const unsigned int MCU_HUFFMAN_CAPACITY = 256;
    const int MAX_SEGMENT_BYTE = 4096;

    const int mcu0 = segid*img_info.segment_mcu_count;
    int mcu1 = mcu0 + img_info.segment_mcu_count;
    if (mcu1 > img_info.mcu_count) {
        mcu1 = img_info.mcu_count;
    }

    unsigned char* buffer = segment_compressed.d_buffer + segid*MAX_SEGMENT_BYTE;
    int segment_compressed_byte = 0;

    
    int new_byte=0, new_byte_pos=7;
    for (int m=mcu0; m<mcu1; ++m) {
        BitString* huffman_code_seg = (BitString*)huffman_code.d_buffer+MCU_HUFFMAN_CAPACITY*component*m;
        int* huffman_code_count_seg = d_huffman_code_count+component*m;
        for (int i=0; i<component; ++i) {
            write_bitstring(huffman_code_seg, *huffman_code_count_seg, new_byte, new_byte_pos, buffer+segment_compressed_byte, segment_compressed_byte);
            huffman_code_seg += MCU_HUFFMAN_CAPACITY;
            huffman_code_count_seg += 1;    
        }
    }
    if (new_byte_pos != 7) {
        int bp = new_byte_pos;
        int b = new_byte; 
        int mask[8] = {1,2,4,8,16,32,64,128};
        while (bp>=0) {
            b = b | mask[bp];                
            --bp;
        }
        write_byte((unsigned char)b, buffer+segment_compressed_byte, segment_compressed_byte);
        new_byte_pos = 7;
        new_byte = 0;
    }

    write_byte(0xFF, buffer+segment_compressed_byte, segment_compressed_byte);

    //补充编码让其是32的整数倍，方便后面的warp来批量赋值
    const int rest = segment_compressed_byte%32;
    for (int i=0; i<31-rest;++i) {
        write_byte(0xFF, buffer+segment_compressed_byte, segment_compressed_byte);
    }
    write_byte(0xD0+segid%8, buffer+segment_compressed_byte, segment_compressed_byte);

    // if (segment_compressed_byte > 4095) {
    //     printf("segment byte error: %d\n", segment_compressed_byte);   
    // }

    d_segment_compressed_byte[segid] = segment_compressed_byte;
}

__global__ void kernel_segment_offset(const ImageInfo img_info, int *d_segment_compressed_byte, int *d_segment_compressed_offset)  {
    unsigned int segid = blockIdx.x * blockDim.x + threadIdx.x;
    if (segid > img_info.segment_count-1) {
        return;
    }
    int val = 0;
    for (int i=0; i<segid; ++i) {
        if (d_segment_compressed_byte[i] < 0) {
            //printf("segment byte error: %d\n", d_segment_compressed_byte[i]);    
        }
        val += d_segment_compressed_byte[i];
    }
    d_segment_compressed_offset[segid] = val;
    // if (segid == img_info.segment_count-1) {
    //     printf("last segment offset: %d\n", val);
    // }
}

__global__ void kernel_segment_compact(const BlockUnit segment_compressed, const ImageInfo img_info, int *d_segment_compressed_byte, int *d_segment_compressed_offset, const BlockUnit segment_compressed_compact) {
    unsigned int segid = blockIdx.x * blockDim.x + threadIdx.x;
    if (segid > img_info.segment_count-1) {
        return;
    }  
    const int MAX_SEGMENT_BYTE = 4096;
    const int src_offset = segid*MAX_SEGMENT_BYTE;
    const int dst_offset = d_segment_compressed_offset[segid];
    const int len = d_segment_compressed_byte[segid];
    unsigned char* src = segment_compressed.d_buffer + src_offset;
    unsigned char* dst = segment_compressed_compact.d_buffer + dst_offset;
    for (int i=0; i<len; ++i) {
        dst[i] = src[i];
    }
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
cudaError_t r_2_dct(const BlockUnit& rgb, const BlockUnit& dct_result, const ImageInfo& img_info, const DCTTable& dct_table) {
    const int BLOCK_SIZEX = 8;
    const int BLOCK_SIZEY = 8;
    const int CAL_UNIT = 2;
    int w = img_info.mcu_w / CAL_UNIT;
    if (CAL_UNIT*w != img_info.mcu_w) {
        w += 1;
    }
    int h = img_info.mcu_h / CAL_UNIT;
    if (CAL_UNIT*h != img_info.mcu_h) {
        h += 1;
    }

    dim3 block(BLOCK_SIZEX, BLOCK_SIZEY, 1);
    dim3 grid(w / BLOCK_SIZEX, h / BLOCK_SIZEY);
    if (grid.x * BLOCK_SIZEX != w) {
        grid.x += 1;
    }
    if (grid.y * BLOCK_SIZEY != h) {
        grid.y += 1;
    }

    kernel_r_2_dct_ext << <grid, block >> >(rgb, dct_result, img_info, dct_table, CAL_UNIT);

    // const int BLOCK_SIZEX = 8;
    // const int BLOCK_SIZEY = 8;
    // dim3 block(BLOCK_SIZEX, BLOCK_SIZEY, 1);
    // dim3 grid(img_info.mcu_w / BLOCK_SIZEX, img_info.mcu_h / BLOCK_SIZEY);
    // if (grid.x * BLOCK_SIZEX != img_info.mcu_w) {
    //     grid.x += 1;
    // }
    // if (grid.y * BLOCK_SIZEY != img_info.mcu_h) {
    //     grid.y += 1;
    // }

    // kernel_r_2_dct << <grid, block >> >(rgb, dct_result, img_info, dct_table);
    
    return cudaDeviceSynchronize();
}


extern "C" 
cudaError_t gpujpeg_r_dct(const BlockUnit& rgb, const BlockUnit& dct_result, const ImageInfo& img_info, const DCTTable& dct_table) {
    int roi_width = img_info.width;
    int roi_height = img_info.height;
    int GPUJPEG_BLOCK_SIZE = 8;

    int block_count_x = roi_width / GPUJPEG_BLOCK_SIZE;
    int block_count_y = roi_height / GPUJPEG_BLOCK_SIZE;

    enum { WARP_COUNT = 4 };

    dim3 dct_grid(
        gpujpeg_div_and_round_up(block_count_x, 4),
        gpujpeg_div_and_round_up(block_count_y, WARP_COUNT),
        1
    );

    dim3 dct_block(4 * 8, WARP_COUNT);

    gpujpeg_dct_gpu_kernel<WARP_COUNT> << <dct_grid, dct_block, 0 >> >(
        block_count_x,
        block_count_y,
        rgb.d_buffer,
        roi_width,
        (int16_t*)dct_result.d_buffer,
        roi_width * GPUJPEG_BLOCK_SIZE,
        (float*)dct_table.d_quant_tbl_luminance);
    
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

extern "C"
cudaError_t huffman_writebits(const BlockUnit& huffman_code, int *d_huffman_code_count, const ImageInfo& img_info, const BlockUnit& segment_compressed, int *d_segment_compressed_byte) {
    const int BLOCK_SIZE = 8;
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid(img_info.segment_count / BLOCK_SIZE, 1, 1);
    if (grid.x * BLOCK_SIZE != img_info.segment_count) {
        grid.x += 1;
    }

    kernel_huffman_writebits << <grid, block >> >(huffman_code, d_huffman_code_count, img_info, segment_compressed, d_segment_compressed_byte);
    
    return cudaDeviceSynchronize();
}

extern "C"
cudaError_t segment_offset(const ImageInfo& img_info, int *d_segment_compressed_byte, int *d_segment_compressed_offset) {
    const int BLOCK_SIZE = 8;
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid(img_info.segment_count / BLOCK_SIZE, 1, 1);
    if (grid.x * BLOCK_SIZE != img_info.segment_count) {
        grid.x += 1;
    }

    kernel_segment_offset << <grid, block >> >(img_info, d_segment_compressed_byte, d_segment_compressed_offset);
    
    return cudaDeviceSynchronize();
}

extern "C"
cudaError_t segment_compact(const BlockUnit& segment_compressed, const ImageInfo& img_info, int *d_segment_compressed_byte, int *d_segment_compressed_offset, const BlockUnit& segment_compressed_compact) {
    const int BLOCK_SIZE = 8;
    dim3 block(BLOCK_SIZE, 1, 1);
    dim3 grid(img_info.segment_count / BLOCK_SIZE, 1, 1);
    if (grid.x * BLOCK_SIZE != img_info.segment_count) {
        grid.x += 1;
    }

    kernel_segment_compact << <grid, block >> >(segment_compressed, img_info, d_segment_compressed_byte, d_segment_compressed_offset, segment_compressed_compact);
    
    return cudaDeviceSynchronize();
}