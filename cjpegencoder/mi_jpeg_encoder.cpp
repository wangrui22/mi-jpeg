#include "mi_jpeg_encoder.h"

namespace {

static int gpujpeg_table_default_quantization_luminance[64] = { 
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
static int gpujpeg_table_default_quantization_chrominance[64] = { 
  17,  18,  18,  24,  21,  24,  47,  26,
  26,  47,  99,  66,  56,  66,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99
};

static const char ZIGZAG_TABLE[64] = { 
	0, 1, 5, 6,14,15,27,28,
	2, 4, 7,13,16,26,29,42,
	3, 8,12,17,25,30,41,43,
	9,11,18,24,31,40,44,53,
	10,19,23,32,39,45,52,54,
	20,22,33,38,46,51,55,60,
	21,34,37,47,50,56,59,61,
	35,36,48,49,57,58,62,63 
};

static const unsigned char bits_dc_luminance[17] =
    { /* 0-base */ 0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 };

static const unsigned char val_dc_luminance[] =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
  
static const unsigned char bits_dc_chrominance[17] =
    { /* 0-base */ 0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 };
static const unsigned char val_dc_chrominance[] =
    { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
  
static const unsigned char bits_ac_luminance[17] =
    { /* 0-base */ 0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d };
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
  
static const unsigned char bits_ac_chrominance[17] =
    { /* 0-base */ 0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77 };
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

inline void init_qtable(int (&qt_raw)[64], float (&qt)[64]) {
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
};


}

JpegEncoder::JpegEncoder() {
    init();
}

JpegEncoder::~JpegEncoder() {

}

void JpegEncoder::init() {
    //DCT table
    init_qtable(gpujpeg_table_default_quantization_luminance, _quantization_table_luminance);
    init_qtable(gpujpeg_table_default_quantization_chrominance, _quantization_table_chrominance);

    //Huffman Table
    compute_huffman_table(bits_dc_luminance, val_dc_luminance, _huffman_table_Y_DC);
    compute_huffman_table(bits_ac_luminance, val_ac_luminance, _huffman_table_Y_AC);

    compute_huffman_table(bits_dc_chrominance, val_dc_chrominance, _huffman_table_CbCr_DC);
    compute_huffman_table(bits_ac_chrominance, val_ac_chrominance, _huffman_table_CbCr_AC);
}

int JpegEncoder::compress(std::shared_ptr<Image> rgb, unsigned char*& compress_buffer, unsigned int& buffer_len) {  

    std::vector<Segment> segments;
    rgb_2_yuv_segment(rgb, segments);

    const int seg_count = (int)segments.size();
    for (int i=0; i<seg_count; ++i) {
        Segment& segment = segments[i];

        dct_8x8(segment.y, segment.quat_y, _quantization_table_luminance);
        dct_8x8(segment.u, segment.quat_u, _quantization_table_chrominance);
        dct_8x8(segment.v, segment.quat_v, _quantization_table_chrominance);
    }

    for (int i=0; i<seg_count; ++i) {
        Segment& segment = segments[i];
        short y_preDC = i == 0 ? 0 : segments[i-1].quat_y[0];
        short u_preDC = i == 0 ? 0 : segments[i-1].quat_u[0];
        short v_preDC = i == 0 ? 0 : segments[i-1].quat_v[0];

        huffman_encode_8x8(segment.quat_y, y_preDC, _huffman_table_Y_DC, _huffman_table_Y_AC, segment.huffman_code_y, segment.huffman_code_y_count);
        huffman_encode_8x8(segment.quat_u, u_preDC, _huffman_table_CbCr_DC, _huffman_table_CbCr_AC, segment.huffman_code_u, segment.huffman_code_u_count);
        huffman_encode_8x8(segment.quat_v, v_preDC, _huffman_table_CbCr_DC, _huffman_table_CbCr_AC, segment.huffman_code_v, segment.huffman_code_v_count);
    }

    return 0;
}

void JpegEncoder::rgb_2_yuv_segment(std::shared_ptr<Image> rgb, std::vector<Segment>& segments) {
    const int width = rgb->width;
    const int height = rgb->height;
    const int width_ext = div_and_round_up(width, 8);
    const int height_ext = div_and_round_up(height, 8);
    const int mcu_w = width_ext/8;
    const int mcu_h = height_ext/8;
    const int seg_count = mcu_w*mcu_h;

    segments.resize(seg_count);
    for (int i=0; i<seg_count; ++i) {
        segments[i].rgb = new unsigned char[8*8*3];
        segments[i].y = new unsigned char[8*8];
        segments[i].u = new unsigned char[8*8];
        segments[i].v = new unsigned char[8*8];
        segments[i].quat_y = new short[8*8];
        segments[i].quat_u = new short[8*8];
        segments[i].quat_v = new short[8*8];
        segments[i].huffman_code_y = new BitString[256];
        segments[i].huffman_code_u = new BitString[256];
        segments[i].huffman_code_v = new BitString[256];
    }

    for (int i=0; i<seg_count; ++i) {
        int y0 = i/mcu_w;
        int x0 = i-y0*mcu_w;
        y0*=8;
        x0*=8;
        int y1 = y0+8;
        int x1 = x1+8;
        y1 = y1 < height;
        x1 = x1 < width;
        int sidx = 0;
        int idx = 0;
        float r,g,b,y,u,v;
        for (int iy=y0; iy<y1; ++iy) {
            for (int ix=x0; ix<x1; ++ix) {

                idx = iy*height + ix;
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

                segments[i].rgb[sidx*3]   = rgb->buffer[idx*3];
                segments[i].rgb[sidx*3+1] = rgb->buffer[idx*3+1];
                segments[i].rgb[sidx*3+2] = rgb->buffer[idx*3+2];
                segments[i].y[sidx] = (unsigned char)y;
                segments[i].u[sidx] = (unsigned char)u;
                segments[i].v[sidx] = (unsigned char)v;
                ++sidx;

            } 
        }
    }
}

void JpegEncoder::dct_8x8(unsigned char* val, short* output, float* quant_table) {
    //row 1d-dct
    for (int i=0; i<8; ++i) {
        unsigned char* i0 = val + 8*i;
        short* o0 = output + 8*i;
        dct_1d_8_fast<unsigned char, short>(i0[0], i0[1], i0[2], i0[3], i0[4], i0[5], i0[6], i0[7],
                      o0[0], o0[1], o0[2], o0[3], o0[4], o0[5], o0[6], o0[7], 128);
    }

    //collum 1d-dct
    for (int i=0; i<8; ++i) {
        short* i0 = output + i;
        short* o0 = output + i;
        dct_1d_8_fast<short, short>(i0[0], i0[1*8], i0[2*8], i0[3*8], i0[4*8], i0[5*8], i0[6*8], i0[7*8],
                      o0[0], o0[1*8], o0[2*8], o0[3*8], o0[4*8], o0[5*8], o0[6*8], o0[7*8], 0);
    }

    //quantization
    for (int i=0; i<64; ++i) {
        output[ZIGZAG_TABLE[i]] = (int)(output[i]*quant_table[ZIGZAG_TABLE[i]] + 0.5f);
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
    while (end_pos > 0 && quant[end_pos] == 0 ) {
        --end_pos;
    }

    for (int i=0; i<end_pos; ) {
        int start_pos = i;
		while(quant[i] == 0 && i <= start_pos) {
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

    if (end_pos != 63)
		output[index++] = EOB;

	output_count = index;
}

void JpegEncoder::write_jpeg_header(std::shared_ptr<Image> rgb) {
    
}

void JpegEncoder::write_jpeg_segment(const std::vector<Segment>& segment) {

}
