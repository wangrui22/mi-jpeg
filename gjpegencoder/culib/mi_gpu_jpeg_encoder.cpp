#include "mi_gpu_jpeg_encoder.h"
#include <chrono>
#include <fstream>

extern "C"
cudaError_t r_2_dct(const BlockUnit& rgb, const BlockUnit& dct_result, const ImageInfo& img_info, const DCTTable& dct_table);

extern "C" 
cudaError_t gpujpeg_r_dct(const BlockUnit& rgb, const BlockUnit& dct_result, const ImageInfo& img_info, const DCTTable& dct_table);

extern "C" 
cudaError_t rgb_2_yuv_2_dct(const BlockUnit& rgb, const BlockUnit& dct_result, const ImageInfo& img_info, const DCTTable& dct_table);

extern "C"
cudaError_t huffman_encoding(const BlockUnit& dct_result, const BlockUnit& huffman_code, int *d_huffman_code_count, const ImageInfo& img_info, const HuffmanTable& huffman_table);

extern "C"
cudaError_t huffman_writebits(const BlockUnit& huffman_code, int *d_huffman_code_count, const ImageInfo& img_info, const BlockUnit& segment_compressed, int *d_segment_compressed_byte);

extern "C"
cudaError_t segment_offset(const ImageInfo& img_info, int *d_segment_compressed_byte, int *d_segment_compressed_offset);

extern "C"
cudaError_t segment_compact(const BlockUnit& segment_compressed, const ImageInfo& img_info, int *d_segment_compressed_byte, int *d_segment_compressed_offset, const BlockUnit& segment_compressed_compact);


extern "C"
cudaError_t r_2_dct_op(const BlockUnit& rgb, const BlockUnit& dct_result, const ImageInfo& img_info, const DCTTable& dct_table);

extern "C"
cudaError_t segment_compact_op(const BlockUnit& segment_compressed, const ImageInfo& img_info, int *d_segment_compressed_byte, int *d_segment_compressed_offset, unsigned int* d_segment_compressed_byte_sum,  const BlockUnit& segment_compressed_compact);

extern "C"
cudaError_t huffman_writebits_op(const BlockUnit& huffman_code, int *d_huffman_code_count, const ImageInfo& img_info, const BlockUnit& segment_compressed, int *d_segment_compressed_byte);

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

const int ORDER_NATURAL[] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
    63, 63, 63, 63, 63, 63, 63, 63, // Extra entries for safety in decoder
    63, 63, 63, 63, 63, 63, 63, 63
};

inline void init_qtable(unsigned char (&qt_raw)[64], float (&qt)[64]) {
    static const double aanscalefactor[8] = {
	  1.0, 1.387039845, 1.306562965, 1.175875602,
	  1.0, 0.785694958, 0.541196100, 0.275899379
	};
    //引用libjpeg的方法
    // int i = 0;
    // for (int row = 0; row<8; ++row) {
    //     for (int col = 0; col<8; ++col) {
    //         qt[i] = 1.0 / (qt_raw[i]*aanscalefactor[row]*aanscalefactor[col]*8.0);
    //         ++i;
    //     }
    // }


    //这里引用gpujpeg的方法
    // for( unsigned int i = 0; i < 64; i++ ) {
    //     const unsigned int x = ORDER_NATURAL[i] % 8;
    //     const unsigned int y = ORDER_NATURAL[i] / 8;
    //     qt[x * 8 + y] = 1.0 / (qt_raw[i] * aanscalefactor[x] * aanscalefactor[y] * 8); // 8 is the gain of 2D DCT
    // }

    //这里引用gpujpeg的方法, 但是 x 和 y是反的
    for( unsigned int i = 0; i < 64; i++ ) {
        const unsigned int y = ORDER_NATURAL[i] % 8;
        const unsigned int x = ORDER_NATURAL[i] / 8;
        qt[x * 8 + y] = 1.0 / (qt_raw[i] * aanscalefactor[x] * aanscalefactor[y] * 8); // 8 is the gain of 2D DCT
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

const unsigned int BASIC_BYTE = 1024*1024*10;
const unsigned int MCU_HUFFMAN_CAPACITY = 256;

inline void write_word(unsigned short val, unsigned char* buffer, unsigned int& byte) {
    unsigned short val0 = ((val>>8)&0xFF) | ((val&0xFF)<<8);
    *((unsigned short*)buffer) = val0;
    byte += 2;
}

inline void write_byte(unsigned char val, unsigned char* buffer, unsigned int& byte) {
    *buffer = val;
    byte += 1;
}

inline void write_byte_array(const unsigned char* buf, unsigned int buf_len, unsigned char* buffer, unsigned int& byte) {
    memcpy(buffer, buf, buf_len);
    byte += buf_len;
}

inline void write_bitstring(const BitString* bs, int counts, int& new_byte, int& new_byte_pos, unsigned char* buffer, unsigned int& byte) {
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
				::write_byte((unsigned char)(new_byte), buffer++, byte);
				if (new_byte == 0xFF) {
					//special case
					::write_byte((unsigned char)(0x00), buffer++, byte);
				}
				new_byte_pos = 7;
				new_byte = 0;
			}
		}
    }    
}

const int MAX_SEGMENT_BYTE = 4096;

}

using namespace std::chrono;

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

int GPUJpegEncoder::init(std::vector<int> qualitys, int restart_interval, std::shared_ptr<Image> rgb, bool gray) {
    _raw_image = rgb;

    _img_info.width = rgb->width;
    _img_info.height = rgb->height;
    _img_info.width_ext = div_and_round_up(rgb->width, 8);
    _img_info.height_ext = div_and_round_up(rgb->height, 8);
    _img_info.mcu_w = _img_info.width_ext/8;
    _img_info.mcu_h = _img_info.height_ext/8;
    _img_info.mcu_count = _img_info.mcu_w*_img_info.mcu_h;
    _img_info.segment_mcu_count = restart_interval;
    _img_info.segment_count = _img_info.mcu_count/_img_info.segment_mcu_count;
    if (_img_info.segment_mcu_count * _img_info.segment_count != _img_info.mcu_count) {
        _img_info.segment_count += 1; 
    }
    if (gray) {
        _img_info.component = 1;
    } else {
        _img_info.component = 3;
    }

    _raw_rgb.length = _img_info.width*_img_info.height*3;


    cudaError_t err = cudaSuccess;

    //init DCT table
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
        err = cudaMemcpy(dct_table.d_quant_tbl_chrominance, tbl_chrominance ,64*sizeof(float), cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err)

        err = cudaMalloc(&dct_table.d_zig_zag, 64);
        CHECK_CUDA_ERROR(err)
        err = cudaMemcpy(dct_table.d_zig_zag, ZIGZAG_TABLE ,64, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err)

        _dct_table[quality] = dct_table;
    }

    //init huffman table
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

    err = cudaMalloc(&_huffman_table.d_order_natural, sizeof(ORDER_NATURAL));
    CHECK_CUDA_ERROR(err)
    err = cudaMemcpy(_huffman_table.d_order_natural, ORDER_NATURAL, sizeof(ORDER_NATURAL), cudaMemcpyDefault);
    CHECK_CUDA_ERROR(err)

    //init segment
    err = cudaMalloc(&_raw_rgb.d_buffer, _raw_rgb.length);
    CHECK_CUDA_ERROR(err)
    err = cudaMemcpy(_raw_rgb.d_buffer, rgb->buffer, _raw_rgb.length, cudaMemcpyDefault);
    CHECK_CUDA_ERROR(err)

    _dct_result.length =_img_info.mcu_w*_img_info.mcu_h*64*3*sizeof(short);
    err = cudaMalloc(&_dct_result.d_buffer, _dct_result.length);
    CHECK_CUDA_ERROR(err)
    err = cudaMemset(_dct_result.d_buffer, 0, _dct_result.length);
    CHECK_CUDA_ERROR(err)

    _huffman_result.length = _img_info.mcu_w*_img_info.mcu_h*MCU_HUFFMAN_CAPACITY*sizeof(BitString)*3;
    err = cudaMalloc(&_huffman_result.d_buffer, _huffman_result.length);
    CHECK_CUDA_ERROR(err)
    err = cudaMemset(_huffman_result.d_buffer, 0, _huffman_result.length);
    CHECK_CUDA_ERROR(err)

    err = cudaMalloc(&_d_huffman_code_count, _img_info.mcu_w*_img_info.mcu_h*3*sizeof(int));
    CHECK_CUDA_ERROR(err)


    _segment_compressed.length = _img_info.segment_count*MAX_SEGMENT_BYTE;
    err = cudaMalloc(&_segment_compressed.d_buffer, _segment_compressed.length);
    CHECK_CUDA_ERROR(err)
    err = cudaMemset(_segment_compressed.d_buffer, 0, _segment_compressed.length);
    CHECK_CUDA_ERROR(err)

    _segment_compressed_compact.length = 2*BASIC_BYTE;
    err = cudaMalloc(&_segment_compressed_compact.d_buffer, _segment_compressed_compact.length);
    CHECK_CUDA_ERROR(err)
    err = cudaMemset(_segment_compressed_compact.d_buffer, 0, _segment_compressed_compact.length);
    CHECK_CUDA_ERROR(err)

    err = cudaMalloc(&_d_segment_compressed_byte, _img_info.segment_count*sizeof(int));
    CHECK_CUDA_ERROR(err)
    err = cudaMemset(_d_segment_compressed_byte, 0, _img_info.segment_count*sizeof(int));
    CHECK_CUDA_ERROR(err)

    err = cudaMalloc(&_d_segment_compressed_offset, _img_info.segment_count*sizeof(int));
    CHECK_CUDA_ERROR(err)
    err = cudaMemset(_d_segment_compressed_offset, 0, _img_info.segment_count*sizeof(int));
    CHECK_CUDA_ERROR(err)

    err = cudaMalloc(&_d_segment_compressed_byte_sum, sizeof(unsigned int));
    CHECK_CUDA_ERROR(err)
    err = cudaMemset(_d_segment_compressed_byte_sum, 0, sizeof(unsigned int));
    CHECK_CUDA_ERROR(err)

    _compress_capacity = BASIC_BYTE*2;
    _compress_buffer = new unsigned char[_compress_capacity];
    _h_segment_compressed_compact = new unsigned char[_compress_capacity];

    _h_segment_compressed_offset = new int[_img_info.segment_count];

    _h_segment_compressed_byte = new int[_img_info.segment_count];

    return 0;
}

int GPUJpegEncoder::compress(int quality, unsigned char*& compress_buffer, unsigned int& buffer_len,bool is_op) {
    steady_clock::time_point _start = steady_clock::now();

    _is_op = is_op;
    cudaError_t err = cudaSuccess;

    // err = cudaMemset(_dct_result.d_buffer, 0, _dct_result.length);
    // CHECK_CUDA_ERROR(err)

    // err = cudaMemset(_huffman_result.d_buffer, 0, _huffman_result.length);
    // CHECK_CUDA_ERROR(err)

    // err = cudaMemset(_segment_compressed.d_buffer, 0, _segment_compressed.length);
    // CHECK_CUDA_ERROR(err)

    // err = cudaMemset(_segment_compressed_compact.d_buffer, 0, _segment_compressed_compact.length);
    // CHECK_CUDA_ERROR(err)

    err = cudaMemset(_d_segment_compressed_byte, 0, _img_info.segment_count*sizeof(int));
    CHECK_CUDA_ERROR(err)
    err = cudaMemset(_d_segment_compressed_offset, 0, _img_info.segment_count*sizeof(int));
    CHECK_CUDA_ERROR(err)
    err = cudaMemset(_d_segment_compressed_byte_sum, 0, sizeof(unsigned int));
    CHECK_CUDA_ERROR(err)

    std::cout << "memset cost: " << duration_cast<duration<double>>(steady_clock::now()-_start).count()*1000 << " ms\n";

    write_jpeg_header(quality);

    {
        CudaTimeQuery t0;
        t0.begin();
        if (3 == _img_info.component) {
            err = rgb_2_yuv_2_dct(_raw_rgb, _dct_result, _img_info, _dct_table[quality]);
            std::cout << "rgb_2_yuv_2_dct cost: " << t0.end() << " ms\n";
        } else {
            //err = gpujpeg_r_dct(_raw_rgb, _dct_result, _img_info, _dct_table[quality]);
            if (_is_op) {
                err = r_2_dct_op(_raw_rgb, _dct_result, _img_info, _dct_table[quality]);
            } else {
                err = r_2_dct(_raw_rgb, _dct_result, _img_info, _dct_table[quality]);
            }
            
            std::cout << "r_2_dct cost: " << t0.end() << " ms\n";
        }
        CHECK_CUDA_ERROR(err)
        
        
    }

    {
        CudaTimeQuery t0;
        t0.begin();
        err = huffman_encoding(_dct_result, _huffman_result, _d_huffman_code_count, _img_info, _huffman_table);
        CHECK_CUDA_ERROR(err)
        
        std::cout << "huffman_encode cost: " << t0.end() << " ms\n";
    }

    if (_is_op) {
        CudaTimeQuery t0;
        t0.begin();

        std::cout << "\nhuffman_writebits <><><><>\n";

        err = cudaMemset(_segment_compressed.d_buffer, 0, _segment_compressed.length);
        CHECK_CUDA_ERROR(err)

        err = huffman_writebits(_huffman_result, _d_huffman_code_count, _img_info, _segment_compressed, _d_segment_compressed_byte);
        CHECK_CUDA_ERROR(err)       

        unsigned char* tmp = new unsigned char[_segment_compressed.length];
        err = cudaMemcpy(tmp, _segment_compressed.d_buffer, _segment_compressed.length, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err)   


        std::cout << "\nhuffman_writebits_op <><><><>\n";


        err = cudaMemset(_segment_compressed.d_buffer, 0, _segment_compressed.length);
        CHECK_CUDA_ERROR(err)

        err = huffman_writebits_op(_huffman_result, _d_huffman_code_count, _img_info, _segment_compressed, _d_segment_compressed_byte);
        CHECK_CUDA_ERROR(err)       

        unsigned char* tmp_op = new unsigned char[_segment_compressed.length];
        err = cudaMemcpy(tmp_op, _segment_compressed.d_buffer, _segment_compressed.length, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err)          

        {
            // std::ofstream out("/data/tmp.raw", std::ios::binary|std::ios::out);
            // if (out.is_open()) {
            //     out.write(tmp,_segment_compressed.length);
            //     out.close();
            // }

            for (unsigned int i=0; i<_segment_compressed.length; ++i) {
                if (tmp[i] != tmp_op[i]) {
                    int t = tmp[i];
                    int top = tmp_op[i];
                    std::cerr << "err: " << i << ", " << (int)t << " / " << (int)top << std::endl;
                    break;
                }
            }
        }
        



        
        std::cout << "huffman_writebits_op cost: " << t0.end() << " ms\n";
    } else {
        CudaTimeQuery t0;
        t0.begin();
        err = huffman_writebits(_huffman_result, _d_huffman_code_count, _img_info, _segment_compressed, _d_segment_compressed_byte);
        CHECK_CUDA_ERROR(err)
        
        std::cout << "huffman_writebits cost: " << t0.end() << " ms\n";
    }

    int segment_byte = 0;

    if (_is_op) {
        CudaTimeQuery t0;
        t0.begin();
        err = segment_compact_op(_segment_compressed, _img_info, _d_segment_compressed_byte, _d_segment_compressed_offset, _d_segment_compressed_byte_sum, _segment_compressed_compact);
        CHECK_CUDA_ERROR(err)

        unsigned int segsum = 0;
        err = cudaMemcpy(&segsum, _d_segment_compressed_byte_sum, sizeof(unsigned int), cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err)
        segment_byte = segsum;

        std::cout << "segment_compact_op cost: " << t0.end() << " ms. segment byte: " << segment_byte << "\n";
    } else {
        {
            CudaTimeQuery t0;
            t0.begin();
            err = segment_offset(_img_info, _d_segment_compressed_byte, _d_segment_compressed_offset);
            CHECK_CUDA_ERROR(err)

            int last_segment_offset = 0;
            err = cudaMemcpy(&last_segment_offset, _d_segment_compressed_offset+_img_info.segment_count-1, sizeof(int), cudaMemcpyDefault);
            CHECK_CUDA_ERROR(err)
            int last_segment_byte = 0;
            err = cudaMemcpy(&last_segment_byte, _d_segment_compressed_byte+_img_info.segment_count-1, sizeof(int), cudaMemcpyDefault);
            CHECK_CUDA_ERROR(err)
            segment_byte = last_segment_byte + last_segment_offset;

            
            std::cout << "segment_offset cost: " << t0.end() << " ms. segment byte: " << segment_byte << "\n";

        }

        {
            CudaTimeQuery t0;
            t0.begin();
            err = segment_compact(_segment_compressed, _img_info, _d_segment_compressed_byte, _d_segment_compressed_offset, _segment_compressed_compact);
            CHECK_CUDA_ERROR(err)
            
            std::cout << "segment_compact cost: " << t0.end() << " ms\n";
            
        }
    }
    
    
    write_jpeg_segment_gpu(segment_byte);

    //write_jpeg_segment();

    write_word(0xFFD9); //Write End of Image Marker  

    buffer_len = _compress_byte;
    compress_buffer = _compress_buffer;
    std::cout << "gpu jpeg compress cost " << duration_cast<duration<double>>(steady_clock::now()-_start).count()*1000 << " ms\n\n";

    return 0;
}

void GPUJpegEncoder::write_word(unsigned short val) {
    ::write_word(val, _compress_buffer+_compress_byte, _compress_byte);
}

void GPUJpegEncoder::write_byte(unsigned char val) {
    ::write_byte(val, _compress_buffer+_compress_byte, _compress_byte);
}

void GPUJpegEncoder::write_byte_array(const unsigned char* buf, unsigned int buf_len) {
    ::write_byte_array(buf, buf_len, _compress_buffer+_compress_byte, _compress_byte);
}

void GPUJpegEncoder::write_bitstring(const BitString* bs, int counts, int& new_byte, int& new_byte_pos) {
    ::write_bitstring(bs, counts, new_byte, new_byte_pos, _compress_buffer+_compress_byte, _compress_byte);
}

void GPUJpegEncoder::write_jpeg_header(int quality) {

    steady_clock::time_point _start = steady_clock::now();

    _compress_byte = 0;
    
    //SOI
	write_word(0xFFD8);		// marker = 0xFFD8

	//APPO
	write_word(0xFFE0);		// marker = 0xFFE0
	write_word(16);			// length = 16 for usual JPEG, no thumbnail
    unsigned char JFIF[5];
    JFIF[0] = 'J';
    JFIF[1] = 'F';
    JFIF[2] = 'I';
    JFIF[3] = 'F';
    JFIF[4] = '\0';
	write_byte_array(JFIF, 5);			// 'JFIF\0'
	write_byte(1);			// version_hi
	write_byte(1);			// version_low
	write_byte(0);			// xyunits = 0 no units, normal density
	write_word(1);			// xdensity
	write_word(1);			// ydensity
	write_byte(0);			// thumbWidth
	write_byte(0);			// thumbHeight

    if (3 == _img_info.component) {
        //DQT
        write_word(0xFFDB);		//marker = 0xFFDB
        write_word(132);			//size=132
        write_byte(0);			//QTYinfo== 0:  bit 0..3: number of QT = 0 (table for Y) 
                                        //				bit 4..7: precision of QT
                                        //				bit 8	: 0

        write_byte_array(_dct_table[quality].quant_tbl_luminance, 64);		//YTable
        write_byte(1);			//QTCbinfo = 1 (quantization table for Cb,Cr)
        write_byte_array(_dct_table[quality].quant_tbl_chrominance, 64);	//CbCrTable

        //SOFO
        write_word(0xFFC0);			//marker = 0xFFC0
        write_word(17);				//length = 17 for a truecolor YCbCr JPG
        write_byte(8);				//precision = 8: 8 bits/sample 
        write_word(_img_info.height&0xFFFF);	//height
        write_word(_img_info.width&0xFFFF);	//width

        write_byte(3);				//nrofcomponents = 3: We encode a truecolor JPG

        write_byte(1);				//IdY = 1
        write_byte(0x11);				//HVY sampling factors for Y (bit 0-3 vert., 4-7 hor.)(SubSamp 1x1)
        write_byte(0);				//QTY  Quantization Table number for Y = 0

        write_byte(2);				//IdCb = 2
        write_byte(0x11);				//HVCb = 0x11(SubSamp 1x1)
        write_byte(1);				//QTCb = 1

        write_byte(3);				//IdCr = 3
        write_byte(0x11);				//HVCr = 0x11 (SubSamp 1x1)
        write_byte(1);				//QTCr Normally equal to QTCb = 1    

        //DHT
        int hdt_len = (int)(
        sizeof(BITS_DC_LUMINANCE) + 
        sizeof(VAL_DC_LUMINANCE) +
        sizeof(BITS_AC_LUMINANCE) + 
        sizeof(VAL_AC_LUMINANCE) +
        sizeof(BITS_DC_CHROMINANCE) + 
        sizeof(VAL_DC_CHROMINANCE) +
        sizeof(BITS_AC_CHROMINANCE) + 
        sizeof(VAL_AC_CHROMINANCE) + 6);

        write_word(0xFFC4);		//marker = 0xFFC4
        //write_word(0x01A2);		//length = 0x01A2
        write_word(hdt_len);		//length = 0x01A2
        write_byte(0);			//HTYDCinfo bit 0..3	: number of HT (0..3), for Y =0
                                        //			bit 4		: type of HT, 0 = DC table,1 = AC table
                                        //			bit 5..7	: not used, must be 0
        write_byte_array(BITS_DC_LUMINANCE, sizeof(BITS_DC_LUMINANCE));	//DC_L_NRC
        write_byte_array(VAL_DC_LUMINANCE, sizeof(VAL_DC_LUMINANCE));		//DC_L_VALUE
        write_byte(0x10);			//HTYACinfo
        write_byte_array(BITS_AC_LUMINANCE, sizeof(BITS_AC_LUMINANCE));
        write_byte_array(VAL_AC_LUMINANCE, sizeof(VAL_AC_LUMINANCE)); //we'll use the standard Huffman tables
        write_byte(0x01);			//HTCbDCinfo
        write_byte_array(BITS_DC_CHROMINANCE, sizeof(BITS_DC_CHROMINANCE));
        write_byte_array(VAL_DC_CHROMINANCE, sizeof(VAL_DC_CHROMINANCE));
        write_byte(0x11);			//HTCbACinfo
        write_byte_array(BITS_AC_CHROMINANCE, sizeof(BITS_AC_CHROMINANCE));
        write_byte_array(VAL_AC_CHROMINANCE, sizeof(VAL_AC_CHROMINANCE));

        //DRI
        write_word(0xFFDD);
        write_word(4);
        write_word(_img_info.segment_mcu_count);

        //SOS
        write_word(0xFFDA);		//marker = 0xFFC4
        write_word(12);			//length = 12
        write_byte(3);			//nrofcomponents, Should be 3: truecolor JPG

        write_byte(1);			//Idy=1
        write_byte(0);			//HTY	bits 0..3: AC table (0..3)
                                        //		bits 4..7: DC table (0..3)
        write_byte(2);			//IdCb
        write_byte(0x11);			//HTCb

        write_byte(3);			//IdCr
        write_byte(0x11);			//HTCr

        write_byte(0);			//Ss not interesting, they should be 0,63,0
        write_byte(0x3F);			//Se
        write_byte(0);			//Bf


    } else {
        //DQT
        write_word(0xFFDB);		//marker = 0xFFDB
        write_word(67);			//size=132
        write_byte(0);			//QTYinfo== 0:  bit 0..3: number of QT = 0 (table for Y) 
                                        //				bit 4..7: precision of QT
                                        //				bit 8	: 0
        write_byte_array(_dct_table[quality].quant_tbl_luminance, 64);		//YTable

        //SOFO
        write_word(0xFFC0);			//marker = 0xFFC0
        write_word(11);				//length = 17 for a truecolor YCbCr JPG
        write_byte(8);				//precision = 8: 8 bits/sample 
        write_word(_img_info.height&0xFFFF);	//height
        write_word(_img_info.width&0xFFFF);	//width

        write_byte(1);				//nrofcomponents = 3: We encode a truecolor JPG

        write_byte(1);				//IdY = 1
        write_byte(0x11);				//HVY sampling factors for Y (bit 0-3 vert., 4-7 hor.)(SubSamp 1x1)
        write_byte(0);				//QTY  Quantization Table number for Y = 0

        //DHT
        int hdt_len = (int)(
        sizeof(BITS_DC_LUMINANCE) + 
        sizeof(VAL_DC_LUMINANCE) +
        sizeof(BITS_AC_LUMINANCE) + 
        sizeof(VAL_AC_LUMINANCE) + 4);

        write_word(0xFFC4);		//marker = 0xFFC4
        //write_word(0x01A2);		//length = 0x01A2
        write_word(hdt_len);		//length = 0x01A2
        write_byte(0);			//HTYDCinfo bit 0..3	: number of HT (0..3), for Y =0
                                        //			bit 4		: type of HT, 0 = DC table,1 = AC table
                                        //			bit 5..7	: not used, must be 0
        write_byte_array(BITS_DC_LUMINANCE, sizeof(BITS_DC_LUMINANCE));	//DC_L_NRC
        write_byte_array(VAL_DC_LUMINANCE, sizeof(VAL_DC_LUMINANCE));		//DC_L_VALUE
        write_byte(0x10);			//HTYACinfo
        write_byte_array(BITS_AC_LUMINANCE, sizeof(BITS_AC_LUMINANCE));
        write_byte_array(VAL_AC_LUMINANCE, sizeof(VAL_AC_LUMINANCE)); //we'll use the standard Huffman tables

        //DRI
        write_word(0xFFDD);
        write_word(4);
        write_word(_img_info.segment_mcu_count);

        //SOS
        write_word(0xFFDA);		//marker = 0xFFC4
        write_word(8);			//length = 12
        write_byte(1);			//nrofcomponents, Should be 3: truecolor JPG

        write_byte(1);			//Idy=1
        write_byte(0);			//HTY	bits 0..3: AC table (0..3)
                                        //		bits 4..7: DC table (0..3)

        write_byte(0);			//Ss not interesting, they should be 0,63,0
        write_byte(0x3F);			//Se
        write_byte(0);			//Bf
    }	

    std::cout << "gpu jpeg write header cost " << duration_cast<duration<double>>(steady_clock::now()-_start).count()*1000 << " ms\n";
}

void GPUJpegEncoder::write_jpeg_segment() {

    steady_clock::time_point _start = steady_clock::now();

    BitString* huffman_code = new BitString[_img_info.mcu_w*_img_info.mcu_h*MCU_HUFFMAN_CAPACITY*3];
    int* huffman_code_count = new int[_img_info.mcu_w*_img_info.mcu_h*3];
    cudaError_t err = cudaSuccess;

    err = cudaMemcpy(huffman_code, _huffman_result.d_buffer, _img_info.mcu_w*_img_info.mcu_h*MCU_HUFFMAN_CAPACITY*3*sizeof(BitString), cudaMemcpyDefault);
    CHECK_CUDA_ERROR(err)
    err = cudaMemcpy(huffman_code_count, _d_huffman_code_count, _img_info.mcu_w*_img_info.mcu_h*3*sizeof(int), cudaMemcpyDefault);
    CHECK_CUDA_ERROR(err)


    //连续没有restart interval
    //int new_byte=0, new_byte_pos=7;
    // for (int i=0; i<_img_info.segment_count; ++i) {
    //     BitString* huffman_code_seg = huffman_code+MCU_HUFFMAN_CAPACITY*3*i;
    //     int* huffman_code_count_seg = huffman_code_count+3*i;
    //     write_bitstring(huffman_code_seg, *huffman_code_count_seg, new_byte, new_byte_pos);
    //     huffman_code_seg += MCU_HUFFMAN_CAPACITY;
    //     huffman_code_count_seg += 1;
    //     write_bitstring(huffman_code_seg, *huffman_code_count_seg, new_byte, new_byte_pos);
    //     huffman_code_seg += MCU_HUFFMAN_CAPACITY;
    //     huffman_code_count_seg += 1;
    //     write_bitstring(huffman_code_seg, *huffman_code_count_seg, new_byte, new_byte_pos);
    // }


    //有restart interval的segment
    for (int i=0; i<_img_info.segment_count; ++i) {
        const int mcu0 = i*_img_info.segment_mcu_count;
        int mcu1 = mcu0 + _img_info.segment_mcu_count;
        if (mcu1 > _img_info.mcu_count-1) {
            mcu1 = _img_info.mcu_count-1;
        }
        int new_byte=0, new_byte_pos=7;
        for (int m=mcu0; m<mcu1; ++m) {
            BitString* huffman_code_seg = huffman_code+MCU_HUFFMAN_CAPACITY*3*m;
            int* huffman_code_count_seg = huffman_code_count+3*m;
            write_bitstring(huffman_code_seg, *huffman_code_count_seg, new_byte, new_byte_pos);
            huffman_code_seg += MCU_HUFFMAN_CAPACITY;
            huffman_code_count_seg += 1;
            write_bitstring(huffman_code_seg, *huffman_code_count_seg, new_byte, new_byte_pos);
            huffman_code_seg += MCU_HUFFMAN_CAPACITY;
            huffman_code_count_seg += 1;
            write_bitstring(huffman_code_seg, *huffman_code_count_seg, new_byte, new_byte_pos);
            
        }
        if (new_byte_pos != 7) {
            int bp = new_byte_pos;
            int b = new_byte; 
            int mask[8] = {1,2,4,8,16,32,64,128};
            while (bp>=0) {
                b = b | mask[bp];                
                --bp;
            }
            write_byte((unsigned char)b);
            new_byte_pos = 7;
            new_byte = 0;
        }

        write_word(0xFFD0+i%8);
    }

    std::cout << "gpu jpeg write seg cost " << duration_cast<duration<double>>(steady_clock::now()-_start).count()*1000 << " ms\n";
}

void GPUJpegEncoder::write_jpeg_segment_gpu(int segment_compressed_byte) {
    steady_clock::time_point _start = steady_clock::now();
    if (!_is_op) {
        //1 直接拷贝
        cudaError_t err = cudaMemcpy(_compress_buffer+_compress_byte, _segment_compressed_compact.d_buffer, segment_compressed_byte, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err)

        _compress_byte += segment_compressed_byte;
    } else {
        //2 拷贝后segment排序
        cudaError_t err = cudaMemcpy(_h_segment_compressed_compact, _segment_compressed_compact.d_buffer, segment_compressed_byte, cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err)

        err = cudaMemcpy(_h_segment_compressed_offset, _d_segment_compressed_offset, _img_info.segment_count*sizeof(int), cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err)

        err = cudaMemcpy(_h_segment_compressed_byte, _d_segment_compressed_byte, _img_info.segment_count*sizeof(int), cudaMemcpyDefault);
        CHECK_CUDA_ERROR(err)

        for (int i=0; i<_img_info.segment_count; ++i) {
            const int offset = _h_segment_compressed_offset[i];
            const int byte = _h_segment_compressed_byte[i];

            memcpy(_compress_buffer+_compress_byte, _h_segment_compressed_compact+offset, byte);
            _compress_byte += byte;
        }
    }

    std::cout << "gpu jpeg write seg(2) cost " << duration_cast<duration<double>>(steady_clock::now()-_start).count()*1000 << " ms\n";
}