#include "mi_gpu_jpeg_encoder.h"

extern "C" 
cudaError_t rgb_2_yuv(const BlockUnit& rgb, const BlockUnit& yuv, const ImageInfo& img_info);

extern "C" 
cudaError_t rgb_2_yuv_2_dct(const BlockUnit& rgb, const BlockUnit& dct_result, const ImageInfo& img_info, const DCTTable& dct_table);

extern "C"
cudaError_t huffman_encode(const BlockUnit& dct_result, const BlockUnit& huffman_code, int *d_huffman_code_count, const ImageInfo& img_info, const HuffmanTable& huffman_table);

namespace {
/** Default Quantization Table for Y component (zig-zag order)*/
const unsigned char DEFAULT_QUANTIZATION_LUMINANCE[64] = { 
  16,  11,  12,  14,  12,  10,  16,  14,
  13,  14,  18,  17,  16,  19,  24,  40,
  26,  24,  22,  22,  24,  49,  35,  37,
  29,  40,  58,  51,  61,  60,  57,  51,
  56,  55,  64,  72,  92,  78,  64,  68,
  87,  69,  55,  56,  80, 109,  81,  87,
  95,  98, 103, 104, 103,  62,  77, 113,
  121, 112, 100, 120,  92, 101, 103,  99
};

/** Default Quantization Table for Cb or Cr component (zig-zag order) */
const unsigned char DEFAULT_QUANTIZATION_CHROMINACE[64] = { 
  17,  18,  18,  24,  21,  24,  47,  26,
  26,  47,  99,  66,  56,  66,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99
};

const unsigned char ZIGZAG_TABLE[64] = { 
	0, 1, 5, 6,14,15,27,28,
	2, 4, 7,13,16,26,29,42,
	3, 8,12,17,25,30,41,43,
	9,11,18,24,31,40,44,53,
	10,19,23,32,39,45,52,54,
	20,22,33,38,46,51,55,60,
	21,34,37,47,50,56,59,61,
	35,36,48,49,57,58,62,63 
};

const unsigned char BITS_DC_LUMINANCE[] =
    { 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 };

const unsigned char VAL_DC_LUMINANCE[] =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
  
const unsigned char BITS_DC_CHROMINANCE[] =
    { 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 };
const unsigned char VAL_DC_CHROMINANCE[] =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
  
const unsigned char BITS_AC_LUMINANCE[] =
    { 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d };
const unsigned char VAL_AC_LUMINANCE[] =
    { 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
      0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
      0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
      0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
      0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
      0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
      0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
      0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
      0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
      0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
      0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
      0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
      0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
      0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
      0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
      0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
      0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
      0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
      0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
      0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
      0xf9, 0xfa };

const unsigned char BITS_AC_CHROMINANCE[] =
    {0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77 };
const unsigned char VAL_AC_CHROMINANCE[] =
    { 0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
      0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
      0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
      0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
      0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
      0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
      0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
      0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
      0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
      0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
      0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
      0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
      0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
      0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
      0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
      0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
      0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
      0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
      0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
      0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
      0xf9, 0xfa };


inline int div_and_round_up(const int val, const int div) {
    return (val % div == 0) ? val : (val/div+1)*div;
}

inline void dct_table_apply_quality(unsigned char* table_raw, int quality) {
	int s = (quality < 50) ? (5000 / quality) : (200 - (2 * quality));
    for ( int i = 0; i < 64; i++ ) {
        int value = (s * (int)table_raw[i] + 50) / 100;
        if ( value == 0 ) {
            value = 1;
        }
        if ( value > 255 ) {
            value = 255;
        }
        table_raw[i] = (unsigned char)value;
    }
}

inline void init_qtable(unsigned char (&qt_raw)[64], float (&qt)[64]) {
    static const double aanscalefactor[8] = {
	  1.0, 1.387039845, 1.306562965, 1.175875602,
	  1.0, 0.785694958, 0.541196100, 0.275899379
	};
    int i = 0;
    for (int row = 0; row<8; ++row) {
        for (int col = 0; col<8; ++col) {
            qt[i] = 1.0 / (qt_raw[i]*aanscalefactor[row]*aanscalefactor[col]*8.0);
            ++i;
        }
    }
}

inline void compute_huffman_table(const unsigned char* bit_val_count_array, const unsigned char* val_array, BitString* huffman_table) {
	int pos_in_table = 0;
	unsigned short code_value = 0;
	for(int bit = 1; bit <= 16; ++bit) {
		for(int val_count = 0; val_count < bit_val_count_array[bit-1]; ++val_count){
			huffman_table[val_array[pos_in_table]].value = code_value;
			huffman_table[val_array[pos_in_table]].length = bit;
			pos_in_table++;
			code_value++;
		}
		code_value <<= 1;
	}
}

}

CudaTimeQuery::CudaTimeQuery() {
}

CudaTimeQuery::~CudaTimeQuery() {
}

void CudaTimeQuery::begin() {
    cudaError err = cudaEventCreate(&_start);
    CHECK_CUDA_ERROR(err);
    err = cudaEventRecord(_start, 0);
    CHECK_CUDA_ERROR(err);
}

float CudaTimeQuery::end() {
    cudaError err = cudaEventCreate(&_end);
    CHECK_CUDA_ERROR(err);
    err = cudaEventRecord(_end, 0);
    CHECK_CUDA_ERROR(err);
    err = cudaEventSynchronize(_end);
    CHECK_CUDA_ERROR(err);
    err = cudaEventElapsedTime(&_time_elapsed, _start, _end);
    CHECK_CUDA_ERROR(err);
    err = cudaEventDestroy(_start);
    CHECK_CUDA_ERROR(err);
    err = cudaEventDestroy(_end);
    CHECK_CUDA_ERROR(err);
    return _time_elapsed;
}

GPUJpegEncoder::GPUJpegEncoder() {

}

GPUJpegEncoder::~GPUJpegEncoder() {

}

int GPUJpegEncoder::init(std::vector<int> qualitys) {
    cudaError_t err = cudaSuccess;
    for (size_t i=0; i<qualitys.size(); ++i) {
        const int quality = qualitys[i];
        if (quality < 10 || quality>100) {
            std::cerr << "invalid quality: " << quality << std::endl;
            continue;
        }

        DCTTable dct_table;
        memcpy(dct_table.quant_tbl_luminance, DEFAULT_QUANTIZATION_LUMINANCE, 64);
        dct_table_apply_quality(dct_table.quant_tbl_luminance, quality);
        float tbl_luminance[64];
        init_qtable(dct_table.quant_tbl_luminance, tbl_luminance);

        memcpy(dct_table.quant_tbl_chrominance, DEFAULT_QUANTIZATION_CHROMINACE, 64);
        dct_table_apply_quality(dct_table.quant_tbl_chrominance, quality);
        float tbl_chrominance[64];
        init_qtable(dct_table.quant_tbl_chrominance, tbl_chrominance);

        err = cudaMalloc(&dct_table.d_quant_tbl_luminance, 64*sizeof(float));
        CHECK_CUDA_ERROR(err)
        err = cudaMemcpy(dct_table.d_quant_tbl_luminance, tbl_luminance ,64*sizeof(float), cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err)
        err = cudaMalloc(&dct_table.d_quant_tbl_chrominance, 64*sizeof(float));
        CHECK_CUDA_ERROR(err)
        err = cudaMemcpy(dct_table.d_quant_tbl_chrominance, tbl_luminance ,64*sizeof(float), cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err)

        err = cudaMalloc(&dct_table.d_zig_zag, 64);
        CHECK_CUDA_ERROR(err)
        err = cudaMemcpy(dct_table.d_zig_zag, ZIGZAG_TABLE ,64, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err)

        _dct_table[quality] = dct_table;
    }

    compute_huffman_table(BITS_DC_LUMINANCE, VAL_DC_LUMINANCE, _huffman_table_Y_DC);
    compute_huffman_table(BITS_AC_LUMINANCE, VAL_AC_LUMINANCE, _huffman_table_Y_AC);
    compute_huffman_table(BITS_DC_CHROMINANCE, VAL_DC_CHROMINANCE, _huffman_table_CbCr_DC);
    compute_huffman_table(BITS_AC_CHROMINANCE, VAL_AC_CHROMINANCE, _huffman_table_CbCr_AC);

    err = cudaMalloc(&_huffman_table.d_huffman_table_Y_DC, sizeof(_huffman_table_Y_DC));
    CHECK_CUDA_ERROR(err)
    err = cudaMemcpy(_huffman_table.d_huffman_table_Y_DC, _huffman_table_Y_DC, sizeof(_huffman_table_Y_DC), cudaMemcpyDefault);
    CHECK_CUDA_ERROR(err)

    err = cudaMalloc(&_huffman_table.d_huffman_table_Y_AC, sizeof(_huffman_table_Y_AC));
    CHECK_CUDA_ERROR(err)
    err = cudaMemcpy(_huffman_table.d_huffman_table_Y_AC, _huffman_table_Y_AC, sizeof(_huffman_table_Y_AC), cudaMemcpyDefault);
    CHECK_CUDA_ERROR(err)

    err = cudaMalloc(&_huffman_table.d_huffman_table_CbCr_DC, sizeof(_huffman_table_CbCr_DC));
    CHECK_CUDA_ERROR(err)
    err = cudaMemcpy(_huffman_table.d_huffman_table_CbCr_DC, _huffman_table_CbCr_DC, sizeof(_huffman_table_CbCr_DC), cudaMemcpyDefault);
    CHECK_CUDA_ERROR(err)

    err = cudaMalloc(&_huffman_table.d_huffman_table_CbCr_AC, sizeof(_huffman_table_CbCr_AC));
    CHECK_CUDA_ERROR(err)
    err = cudaMemcpy(_huffman_table.d_huffman_table_CbCr_AC, _huffman_table_CbCr_AC, sizeof(_huffman_table_CbCr_AC), cudaMemcpyDefault);
    CHECK_CUDA_ERROR(err)

    return 0;
}

int GPUJpegEncoder::compress(std::shared_ptr<Image> rgb, int quality, unsigned char*& compress_buffer, unsigned int& buffer_len) {
    _img_info.width = rgb->width;
    _img_info.height = rgb->height;
    _img_info.width_ext = div_and_round_up(rgb->width, 8);
    _img_info.height_ext = div_and_round_up(rgb->height, 8);
    _img_info.mcu_w = _img_info.width_ext/8;
    _img_info.mcu_h = _img_info.height_ext/8;
    _img_info.segment_count = _img_info.mcu_w*_img_info.mcu_h;

    _raw_rgb.length = _img_info.width*_img_info.height*3;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&_raw_rgb.d_buffer, _raw_rgb.length);
    CHECK_CUDA_ERROR(err)
    err = cudaMemcpy(_raw_rgb.d_buffer, rgb->buffer, _raw_rgb.length, cudaMemcpyDefault);
    CHECK_CUDA_ERROR(err)

    _yuv_ext.length = _img_info.width_ext*_img_info.height_ext*3;
    err = cudaMalloc(&_yuv_ext.d_buffer, _yuv_ext.length);
    CHECK_CUDA_ERROR(err)
    err = cudaMemset(_yuv_ext.d_buffer, 0, _yuv_ext.length);
    CHECK_CUDA_ERROR(err)


    _dct_result.length =_img_info.mcu_w*_img_info.mcu_h*64*3*sizeof(short);
    err = cudaMalloc(&_dct_result.d_buffer, _dct_result.length);
    CHECK_CUDA_ERROR(err)
    err = cudaMemset(_dct_result.d_buffer, 0, _dct_result.length);
    CHECK_CUDA_ERROR(err)

    _huffman_result.length = _img_info.mcu_w*_img_info.mcu_h*128*sizeof(BitString)*3;
    err = cudaMalloc(&_huffman_result.d_buffer, _huffman_result.length);
    CHECK_CUDA_ERROR(err)
    err = cudaMemset(_huffman_result.d_buffer, 0, _huffman_result.length);
    CHECK_CUDA_ERROR(err)

    err = cudaMalloc(&_d_huffman_code_count, _img_info.mcu_w*_img_info.mcu_h*sizeof(int));
    CHECK_CUDA_ERROR(err)
    

    // {
    //     CudaTimeQuery t0;
    //     t0.begin();
    //     err = rgb_2_yuv(_raw_rgb, _yuv_ext, _img_info);
    //     CHECK_CUDA_ERROR(err)
        
    //     std::cout << "rgb_2_yuv cost: " << t0.end() << " ms\n";
    // }

    {
        CudaTimeQuery t0;
        t0.begin();
        err = rgb_2_yuv_2_dct(_raw_rgb, _dct_result, _img_info, _dct_table[quality]);
        CHECK_CUDA_ERROR(err)
        
        std::cout << "rgb_2_yuv_2_dct cost: " << t0.end() << " ms\n";
    }

    {
        CudaTimeQuery t0;
        t0.begin();
        err = huffman_encode(_dct_result, _huffman_result, _d_huffman_code_count, _img_info, _huffman_table);
        CHECK_CUDA_ERROR(err)
        
        std::cout << "huffman_encode cost: " << t0.end() << " ms\n";
    }
    

    return 0;
}