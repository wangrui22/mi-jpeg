#include "mi_jpeg_encoder.h"
#include <cmath>
#include <fstream>
#include <sstream>

namespace {

// /** Default Quantization Table for Y component (zig-zag order)*/
static unsigned char default_quantization_luminance[64] = { 
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
static unsigned char default_quantization_chrominance[64] = { 
  17,  18,  18,  24,  21,  24,  47,  26,
  26,  47,  99,  66,  56,  66,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99
};

// static unsigned char default_quantization_luminance[64] = { 
//   16,  11,  10,  16,  24,  40,  51,  61,
//   12,  12,  14,  19,  26,  58,  60,  55,
//   14,  13,  16,  24,  40,  57,  69,  56,
//   14,  17,  22,  29,  51,  87,  80,  62,
//   18,  22,  37,  56,  68, 109, 103,  77,
//   24,  35,  55,  64,  81, 104, 113,  92,
//   49,  64,  78,  87, 103, 121, 120, 101,
//   72,  92,  95,  98, 112, 100, 103,  99
// };

// static unsigned char default_quantization_chrominance[64] = { 
//   17,  18,  24,  47,  99,  99,  99,  99,
//   18,  21,  26,  66,  99,  99,  99,  99,
//   24,  26,  56,  99,  99,  99,  99,  99,
//   47,  66,  99,  99,  99,  99,  99,  99,
//   99,  99,  99,  99,  99,  99,  99,  99,
//   99,  99,  99,  99,  99,  99,  99,  99,
//   99,  99,  99,  99,  99,  99,  99,  99,
//   99,  99,  99,  99,  99,  99,  99,  99
// };

static const unsigned char ZIGZAG_TABLE[64] = { 
	0, 1, 5, 6,14,15,27,28,
	2, 4, 7,13,16,26,29,42,
	3, 8,12,17,25,30,41,43,
	9,11,18,24,31,40,44,53,
	10,19,23,32,39,45,52,54,
	20,22,33,38,46,51,55,60,
	21,34,37,47,50,56,59,61,
	35,36,48,49,57,58,62,63 
};

static const unsigned char bits_dc_luminance[] =
    { 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 };

static const unsigned char val_dc_luminance[] =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
  
static const unsigned char bits_dc_chrominance[] =
    { 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 };
static const unsigned char val_dc_chrominance[] =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
  
static const unsigned char bits_ac_luminance[] =
    { 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d };
static const unsigned char val_ac_luminance[] =
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

static const unsigned char bits_ac_chrominance[] =
    {0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77 };
static const unsigned char val_ac_chrominance[] =
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

template<typename T0, typename T>
inline void dct_1d_8_fast(const T0 in0, const T0 in1, const T0 in2, const T0 in3, const T0 in4, const T0 in5, const T0 in6, const T0 in7,
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

// inline void init_qtable(unsigned char (&qt_raw)[64], float (&qt)[64]) {
//     static const double aanscalefactor[8] = {
// 	  1.0, 1.387039845, 1.306562965, 1.175875602,
// 	  1.0, 0.785694958, 0.541196100, 0.275899379
// 	};
//     int i = 0;
//     for (int row = 0; row<8; ++row) {
//         for (int col = 0; col<8; ++col) {
//             qt[i] = 1.0 / (qt_raw[i]*aanscalefactor[row]*aanscalefactor[col]*8.0);
//             ++i;
//         }
//     }
// }

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
    /* For AA&N IDCT method, divisors are equal to quantization
	 * coefficients scaled by scalefactor[row]*scalefactor[col], where
	 *   scalefactor[0] = 1
	 *   scalefactor[k] = cos(k*PI/16) * sqrt(2)    for k=1..7
	 * We apply a further scale factor of 8.
	 */

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

    //这里引用gpujpeg的方法, 但是x 和 y是反的
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

inline BitString get_bit_code(int value) {
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

const static unsigned int BASIC_BYTE = 1024*1024*2;

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

inline void write_bitstring(const BitString* bs, int counts, int& new_byte, int& new_byte_pos, unsigned char* buffer, unsigned int& byte)
{
	const unsigned short mask[] = {1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768};
	
	for(int i=0; i<counts; ++i)
	{
		int value = bs[i].value;
		int posval = bs[i].length - 1;

		while (posval >= 0)
		{
			if ((value & mask[posval]) != 0)
			{
				new_byte = new_byte  | mask[new_byte_pos];
			}
			posval--;
			new_byte_pos--;
			if (new_byte_pos < 0)
			{
				// Write to stream
				::write_byte((unsigned char)(new_byte), buffer++, byte);
				if (new_byte == 0xFF)
				{
					// Handle special case
					::write_byte((unsigned char)(0x00), buffer++, byte);
				}

				// Reinitialize
				new_byte_pos = 7;
				new_byte = 0;
			}
		}
	}
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

}


JpegEncoder::JpegEncoder() {
}

JpegEncoder::~JpegEncoder() {

}

void JpegEncoder::init(int quality) {
    //DCT table
	memcpy(_quality_quantization_table_luminance, default_quantization_luminance, 64);
	dct_table_apply_quality(_quality_quantization_table_luminance, quality);
	init_qtable(_quality_quantization_table_luminance, _quantization_table_luminance);

	memcpy(_quality_quantization_table_chrominance, default_quantization_chrominance, 64);
	dct_table_apply_quality(_quality_quantization_table_chrominance, quality);
	init_qtable(_quality_quantization_table_chrominance, _quantization_table_chrominance);

    //Huffman Table
    compute_huffman_table(bits_dc_luminance, val_dc_luminance, _huffman_table_Y_DC);
    compute_huffman_table(bits_ac_luminance, val_ac_luminance, _huffman_table_Y_AC);

    compute_huffman_table(bits_dc_chrominance, val_dc_chrominance, _huffman_table_CbCr_DC);
    compute_huffman_table(bits_ac_chrominance, val_ac_chrominance, _huffman_table_CbCr_AC);
}

int JpegEncoder::compress(std::shared_ptr<Image> rgb, int quality, unsigned char*& compress_buffer, unsigned int& buffer_len) {  
    _img_info.width = rgb->width;
    _img_info.height = rgb->height;
    _img_info.width_ext = div_and_round_up(rgb->width, 8);
    _img_info.height_ext = div_and_round_up(rgb->height, 8);
    _img_info.mcu_w = _img_info.width_ext/8;
    _img_info.mcu_h = _img_info.height_ext/8;
    _img_info.mcu_count = _img_info.mcu_w*_img_info.mcu_h;
    _img_info.segment_mcu_count = 8;
    _img_info.segment_count = _img_info.mcu_count/_img_info.segment_mcu_count;
    if (_img_info.segment_mcu_count * _img_info.segment_count != _img_info.mcu_count) {
        _img_info.segment_count += 1; 
    }

	init(quality);

    write_jpeg_header(rgb);

    std::vector<MCU> mcus;
    rgb_2_yuv(rgb, mcus);

    float* quant_table = _quantization_table_luminance;
    printf("quat table:\n");
    for (int i=0; i<8; ++i) {
        printf("%f %f %f %f %f %f %f %f\n", quant_table[i*8],quant_table[i*8+1],
        quant_table[i*8+2],quant_table[i*8+3],quant_table[i*8+4],quant_table[i*8+5],quant_table[i*8+6],quant_table[i*8+7]);
    }

    const int mcu_count = (int)mcus.size();
    for (int i=0; i<mcu_count; ++i) {
        MCU& mcu = mcus[i];

        dct_8x8(mcu.y, mcu.quat_y, _quantization_table_luminance);
        dct_8x8(mcu.u, mcu.quat_u, _quantization_table_chrominance);
        dct_8x8(mcu.v, mcu.quat_v, _quantization_table_chrominance);
    }

    //FOR test
    // {   
    //     short *quat0 = new short[mcu_count*64];
    //     short *quat1 = new short[mcu_count*64];
    //     short *quat2 = new short[mcu_count*64];
    //     for (int i=0; i<mcu_count; ++i) {
    //         MCU& mcu = mcus[i];
    //         memcpy(quat0+i*64, mcu.quat_y, 64*2);
    //         memcpy(quat1+i*64, mcu.quat_u, 64*2);
    //         memcpy(quat2+i*64, mcu.quat_v, 64*2);
    //     }
        
    //     short * quat_host[3] = {quat0, quat1, quat2};
    //     for (int comp=0; comp<3; ++comp) {
    //         std::stringstream ss;
    //         ss << "/home/wangrui22/projects/mi-jpeg/data/cjpegencoder-quat-" << mcu_count << "-" << comp << ".raw";
    //         std::ofstream out(ss.str().c_str(), std::ios::binary|std::ios::out);
    //         if(out.is_open()) {
    //            out.write((char*)(quat_host[comp]), mcu_count*sizeof(short)*64);
    //            out.close();
    //         }
    //     }
    // }

    //FOR test
    // {
    //     std::string quat_gpujpeg_f = "/home/wangrui22/projects/mi-jpeg/data/gpujpeg-quat-16384-0.raw";

    //     const int mcu_count = 16384;
    //     unsigned int len = mcu_count*64;
    //     short* quat_gpujpeg_raw = new short[len];
    //     std::ifstream in(quat_gpujpeg_f, std::ios::binary | std::ios::in);
    //     if (!in.is_open()) {
    //         return -1;
    //     }
    //     in.read((char*)quat_gpujpeg_raw, len*2);
    //     in.close();

    //     int idx = 0;
    //     for (int i=0; i<mcu_count; ++i) {
    //         MCU& mcu = mcus[i];
    //         int sidx = 0;
    //         for (int i=0; i<64; ++i) {
    //             mcu.quat_y[sidx++] = quat_gpujpeg_raw[idx++];
    //         }
            
    //     }

    // }

    // {
    //     unsigned char *data0 = new unsigned char[mcu_count*64];
    //     unsigned char *data1 = new unsigned char[mcu_count*64];
    //     unsigned char *data2 = new unsigned char[mcu_count*64];
    //     for (int i=0; i<mcu_count; ++i) {
    //         MCU& mcu = mcus[i];
    //         memcpy(data0+i*64, mcu.y, 64);
    //         memcpy(data1+i*64, mcu.u, 64);
    //         memcpy(data2+i*64, mcu.v, 64);
    //     }
        
    //     unsigned char * data_host[3] = {data0, data1, data2};
    //     for (int comp=0; comp<3; ++comp) {
    //         std::stringstream ss;
    //         ss << "/home/wangrui22/projects/mi-jpeg/data/cjpegencoder-data-" << mcu_count << "-" << comp << ".raw";
    //         std::ofstream out(ss.str().c_str(), std::ios::binary|std::ios::out);
    //         if(out.is_open()) {
    //            out.write((char*)(data_host[comp]), mcu_count*64);
    //            out.close();
    //         }
    //     }
    // }
    
    for (int i=0; i<_img_info.segment_count; ++i) {
        const int mcu0 = i*_img_info.segment_mcu_count;
        int mcu1 = mcu0 + _img_info.segment_mcu_count;
        if (mcu1 > _img_info.mcu_count) {
            mcu1 = _img_info.mcu_count;
        }
        for (int m=mcu0; m<mcu1; ++m) {
            MCU& mcu = mcus[m];
            short y_preDC = m == mcu0 ? 0 : mcus[m-1].quat_y[0];
            short u_preDC = m == mcu0 ? 0 : mcus[m-1].quat_u[0];
            short v_preDC = m == mcu0 ? 0 : mcus[m-1].quat_v[0];

            huffman_encode_8x8(mcu.quat_y, y_preDC, _huffman_table_Y_DC, _huffman_table_Y_AC, mcu.huffman_code_y, mcu.huffman_code_y_count);
            huffman_encode_8x8(mcu.quat_u, u_preDC, _huffman_table_CbCr_DC, _huffman_table_CbCr_AC, mcu.huffman_code_u, mcu.huffman_code_u_count);
            huffman_encode_8x8(mcu.quat_v, v_preDC, _huffman_table_CbCr_DC, _huffman_table_CbCr_AC, mcu.huffman_code_v, mcu.huffman_code_v_count);
        }
    }

    for (int i=0; i<_img_info.segment_count; ++i) {
    //for (int i=_img_info.segment_count-1; i>=0; --i) {
        const int mcu0 = i*_img_info.segment_mcu_count;
        int mcu1 = mcu0 + _img_info.segment_mcu_count;
        if (mcu1 > _img_info.mcu_count) {
            mcu1 = _img_info.mcu_count;
        }
        int new_byte=0, new_byte_pos=7;
        for (int m=mcu0; m<mcu1; ++m) {
            MCU& mcu = mcus[m];
            write_bitstring(mcu.huffman_code_y, mcu.huffman_code_y_count, new_byte, new_byte_pos);
            write_bitstring(mcu.huffman_code_u, mcu.huffman_code_u_count, new_byte, new_byte_pos);
            write_bitstring(mcu.huffman_code_v, mcu.huffman_code_v_count, new_byte, new_byte_pos);
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


    // for (int i=0; i<mcu_count; ++i) {
    //     MCU& mcu = mcus[i];
    //     short y_preDC = i == 0 ? 0 : mcus[i-1].quat_y[0];
    //     short u_preDC = i == 0 ? 0 : mcus[i-1].quat_u[0];
    //     short v_preDC = i == 0 ? 0 : mcus[i-1].quat_v[0];

    //     huffman_encode_8x8(mcu.quat_y, y_preDC, _huffman_table_Y_DC, _huffman_table_Y_AC, mcu.huffman_code_y, mcu.huffman_code_y_count);
    //     huffman_encode_8x8(mcu.quat_u, u_preDC, _huffman_table_CbCr_DC, _huffman_table_CbCr_AC, mcu.huffman_code_u, mcu.huffman_code_u_count);
    //     huffman_encode_8x8(mcu.quat_v, v_preDC, _huffman_table_CbCr_DC, _huffman_table_CbCr_AC, mcu.huffman_code_v, mcu.huffman_code_v_count);
    // }

    // int new_byte=0, new_byte_pos=7;
    // for (int i=0; i<mcu_count; ++i) {
    //     MCU& mcu = mcus[i];
    //     write_bitstring(mcu.huffman_code_y, mcu.huffman_code_y_count, new_byte, new_byte_pos);
    //     write_bitstring(mcu.huffman_code_u, mcu.huffman_code_u_count, new_byte, new_byte_pos);
    //     write_bitstring(mcu.huffman_code_v, mcu.huffman_code_v_count, new_byte, new_byte_pos);
    // }

    write_word(0xFFD9); //Write End of Image Marker  

    buffer_len = _compress_byte;
    compress_buffer = new unsigned char[buffer_len];
    memcpy(compress_buffer, _compress_buffer, _compress_byte);

    return 0;
}

void JpegEncoder::rgb_2_yuv(std::shared_ptr<Image> rgb, std::vector<MCU>& mcus) {
    const int width = _img_info.width;
    const int height = _img_info.height;
    const int width_ext = _img_info.width_ext;
    const int height_ext = _img_info.height_ext;
    const int mcu_w = _img_info.mcu_w;
    const int mcu_h = _img_info.mcu_h;
    const int mcu_count = _img_info.mcu_count;

    mcus.resize(mcu_count);
    for (int i=0; i<mcu_count; ++i) {
        mcus[i].huffman_code_y = new BitString[256];
        mcus[i].huffman_code_u = new BitString[256];
        mcus[i].huffman_code_v = new BitString[256];
        memset(mcus[i].huffman_code_y, 0, sizeof(BitString)*256);
        memset(mcus[i].huffman_code_u, 0, sizeof(BitString)*256);
        memset(mcus[i].huffman_code_v, 0, sizeof(BitString)*256);
    }

    for (int i=0; i<mcu_count; ++i) {
        int y0 = i/mcu_w;
        int x0 = i-y0*mcu_w;
        y0*=8;
        x0*=8;
        int y1 = y0+8;
        int x1 = x0+8;
        y1 = y1 < height+1 ? y1 : height;
        x1 = x1 < width+1 ? x1 : width;
        int sidx = 0;
        int idx = 0;
        float r,g,b,y,u,v;
        for (int iy=y0; iy<y1; ++iy) {
            for (int ix=x0; ix<x1; ++ix) {

                idx = iy*width + ix;
                r = (float)rgb->buffer[idx*3];
                g = (float)rgb->buffer[idx*3+1];
                b = (float)rgb->buffer[idx*3+2];
                y =  0.2990f*r + 0.5870f*g + 0.1140f*b ;
                u = -0.1687f*r - 0.3313f*g + 0.5000f*b + 128.0f;
                v =  0.5000f*r - 0.4187f*g - 0.0813f*b + 128.0f;
                y = y < 0.0f ? 0.0f : y;
                y = y > 255.0f ? 255.0f : y;
                u = u < 0.0f ? 0.0f : u;
                u = u > 255.0f ? 255.0f : u;
                v = v < 0.0f ? 0.0f : v;
                v = v > 255.0f ? 255.0f : v;

                mcus[i].rgb[sidx*3]   = rgb->buffer[idx*3];
                mcus[i].rgb[sidx*3+1] = rgb->buffer[idx*3+1];
                mcus[i].rgb[sidx*3+2] = rgb->buffer[idx*3+2];
                mcus[i].y[sidx] = (unsigned char)y;
                mcus[i].u[sidx] = (unsigned char)u;
                mcus[i].v[sidx] = (unsigned char)v;
                ++sidx;

            } 
        }

        // std::cout << "\n<><><><><><><><>mcu: " << i << "<><><><><><><><>\n";
        // int idx0 = 0;
        // for (int y00=0; y00<8; ++y00) {
        //     for (int x00=0; x00<8; ++x00) {
        //         std::cout << (int)mcus[i].y[idx0++] << " ";
        //     }
        //     std::cout << "\n";
        // }

        
    }
}


static void jpeg_fdct_float (float* data, unsigned char* sample_data)
{
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

void JpegEncoder::dct_8x8(unsigned char* val, short* output, float* quant_table) {
    float quat[64];
    jpeg_fdct_float(quat, val);

    //quantization
    for (int i=0; i<64; ++i) {
		float v = quat[i]*quant_table[i];
		// if (v < 0.0f) {
		// 	v-=0.5f;
		// } else {
		// 	v+=0.5f;
		// }
        // //output[ZIGZAG_TABLE[i]] = (short)v;
        // output[i] = (short)v;
        output[i] = rintf(v);
    }
}

void JpegEncoder::huffman_encode_8x8(short* quant, short preDC, 
    const BitString* HTDC, const BitString* HTAC, BitString* output, int& output_count ) {

    int index = 0;
    //encode DC
    const int diffDC = quant[0] - preDC;
    //preDC = quant[0];
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
    while (end_pos > 0 && quant[ORDER_NATURAL[end_pos]] == 0 ) {
        --end_pos;
    }

    for (int i=1; i<=end_pos; ) {
        int start_pos = i;
		while(quant[ORDER_NATURAL[i]] == 0 && i <= end_pos) {
            ++i;
        }

		int zero_counts = i - start_pos;
		if (zero_counts >= 16) {
			for (int j=0; j < zero_counts/16; ++j)
				output[index++] = SIXTEEN_ZEROS;
			zero_counts = zero_counts%16;
		}

		BitString bs = get_bit_code(quant[ORDER_NATURAL[i]]);

		output[index++] = HTAC[(zero_counts << 4) | bs.length];
		output[index++] = bs;
		i++;
    }

    if (end_pos != 63) {
        output[index++] = EOB;
    }

	output_count = index;
}

void JpegEncoder::write_word(unsigned short val) {
    ::write_word(val, _compress_buffer+_compress_byte, _compress_byte);
}

void JpegEncoder::write_byte(unsigned char val) {
    ::write_byte(val, _compress_buffer+_compress_byte, _compress_byte);
}

void JpegEncoder::write_byte_array(const unsigned char* buf, unsigned int buf_len) {
    ::write_byte_array(buf, buf_len, _compress_buffer+_compress_byte, _compress_byte);
}

void JpegEncoder::write_bitstring(const BitString* bs, int counts, int& new_byte, int& new_byte_pos) {
    ::write_bitstring(bs, counts, new_byte, new_byte_pos, _compress_buffer+_compress_byte, _compress_byte);
}

void JpegEncoder::write_jpeg_header(std::shared_ptr<Image> rgb) {

    _compress_capacity = BASIC_BYTE;
    _compress_byte = 0;
    _compress_buffer = new unsigned char[_compress_capacity];
    
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

	//DQT
	write_word(0xFFDB);		//marker = 0xFFDB
	write_word(132);			//size=132
	write_byte(0);			//QTYinfo== 0:  bit 0..3: number of QT = 0 (table for Y) 
									//				bit 4..7: precision of QT
									//				bit 8	: 0

	write_byte_array(_quality_quantization_table_luminance, 64);		//YTable
	write_byte(1);			//QTCbinfo = 1 (quantization table for Cb,Cr)
	write_byte_array(_quality_quantization_table_chrominance, 64);	//CbCrTable

	//SOFO
	write_word(0xFFC0);			//marker = 0xFFC0
	write_word(17);				//length = 17 for a truecolor YCbCr JPG
	write_byte(8);				//precision = 8: 8 bits/sample 
	write_word(rgb->height&0xFFFF);	//height
	write_word(rgb->width&0xFFFF);	//width
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
	sizeof(bits_dc_luminance) + 
	sizeof(val_dc_luminance) +
	sizeof(bits_ac_luminance) + 
	sizeof(val_ac_luminance) +
	sizeof(bits_dc_chrominance) +
	sizeof(val_dc_chrominance) +
	sizeof(bits_ac_chrominance) +
	sizeof(val_ac_chrominance)) + 6;

	write_word(0xFFC4);		//marker = 0xFFC4
	//write_word(0x01A2);		//length = 0x01A2
    write_word(hdt_len);		//length = 0x01A2
	write_byte(0);			//HTYDCinfo bit 0..3	: number of HT (0..3), for Y =0
									//			bit 4		: type of HT, 0 = DC table,1 = AC table
									//			bit 5..7	: not used, must be 0
	write_byte_array(bits_dc_luminance, sizeof(bits_dc_luminance));	//DC_L_NRC
	write_byte_array(val_dc_luminance, sizeof(val_dc_luminance));		//DC_L_VALUE
	write_byte(0x10);			//HTYACinfo
	write_byte_array(bits_ac_luminance, sizeof(bits_ac_luminance));
	write_byte_array(val_ac_luminance, sizeof(val_ac_luminance)); //we'll use the standard Huffman tables
	write_byte(0x01);			//HTCbDCinfo
	write_byte_array(bits_dc_chrominance, sizeof(bits_dc_chrominance));
	write_byte_array(val_dc_chrominance, sizeof(val_dc_chrominance));
	write_byte(0x11);			//HTCbACinfo
	write_byte_array(bits_ac_chrominance, sizeof(bits_ac_chrominance));
	write_byte_array(val_ac_chrominance, sizeof(val_ac_chrominance));

    //DRI
    write_word(0xFFDD);
    write_word(4);
    write_word(8);

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

}