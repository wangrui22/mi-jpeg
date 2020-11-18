#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <iostream>

#include "jpeg_encoder.h"

namespace {

const unsigned char Luminance_Quantization_Table[64] = 
{
	16,  11,  10,  16,  24,  40,  51,  61,
	12,  12,  14,  19,  26,  58,  60,  55,
	14,  13,  16,  24,  40,  57,  69,  56,
	14,  17,  22,  29,  51,  87,  80,  62,
	18,  22,  37,  56,  68, 109, 103,  77,
	24,  35,  55,  64,  81, 104, 113,  92,
	49,  64,  78,  87, 103, 121, 120, 101,
	72,  92,  95,  98, 112, 100, 103,  99
};


const unsigned char Chrominance_Quantization_Table[64] = 
{
	17,  18,  24,  47,  99,  99,  99,  99,
	18,  21,  26,  66,  99,  99,  99,  99,
	24,  26,  56,  99,  99,  99,  99,  99,
	47,  66,  99,  99,  99,  99,  99,  99,
	99,  99,  99,  99,  99,  99,  99,  99,
	99,  99,  99,  99,  99,  99,  99,  99,
	99,  99,  99,  99,  99,  99,  99,  99,
	99,  99,  99,  99,  99,  99,  99,  99
};


const char ZigZag[64] =
{ 
	0, 1, 5, 6,14,15,27,28,
	2, 4, 7,13,16,26,29,42,
	3, 8,12,17,25,30,41,43,
	9,11,18,24,31,40,44,53,
	10,19,23,32,39,45,52,54,
	20,22,33,38,46,51,55,60,
	21,34,37,47,50,56,59,61,
	35,36,48,49,57,58,62,63 
};     

// static unsigned char gpujpeg_table_huffman_y_dc_bits[17] = {
//     0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 
// };
// static unsigned char gpujpeg_table_huffman_y_dc_value[] = { 
//     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 
// };
// /** Huffman Table DC for Cb or Cr component */
// static unsigned char gpujpeg_table_huffman_cbcr_dc_bits[17] = { 
//     0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 
// };
// static unsigned char gpujpeg_table_huffman_cbcr_dc_value[] = { 
//     0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 
// };

const char Standard_DC_Luminance_NRCodes[] = { 0, 0, 7, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
const unsigned char Standard_DC_Luminance_Values[] = { 4, 5, 3, 2, 6, 1, 0, 7, 8, 9, 10, 11 };
// const char Standard_DC_Luminance_NRCodes[] = { 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 };
// const unsigned char Standard_DC_Luminance_Values[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};


const char Standard_DC_Chrominance_NRCodes[] = { 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 };
const unsigned char Standard_DC_Chrominance_Values[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };


const char Standard_AC_Luminance_NRCodes[] = { 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d };
const unsigned char Standard_AC_Luminance_Values[] = 
{
	0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
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
	0xf9, 0xfa
};


const char Standard_AC_Chrominance_NRCodes[] = { 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77 };
const unsigned char Standard_AC_Chrominance_Values[] =
{
	0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
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
	0xf9, 0xfa
};

}

JpegEncoder::JpegEncoder()
	: _width(0)
	, _height(0)
	, _rgb_buffer(0) {
	//初始化静态表格
	init_huffman_tables();
}

JpegEncoder::~JpegEncoder() {
	clean();
}

void JpegEncoder::clean(void) {
	if(_rgb_buffer) {
        delete[] _rgb_buffer;
    }
	_rgb_buffer=0;
	_width=0;
	_height=0;
}

bool JpegEncoder::encode_to_jpeg(unsigned char* rgb, int width, int height, int quality_scale , const std::string& file_name) {
    _rgb_buffer = rgb;
    _width = width;
    _height = height;

	//输出文件
	FILE* fp = fopen(file_name.c_str(), "wb");
	if(fp==0) {
        return false;
    }

	//初始化量化表
	init_quality_tables(quality_scale);

	//文件头
	write_jpeg_header(fp);

	short prev_DC_Y = 0, prev_DC_Cb = 0, prev_DC_Cr = 0;
	int newByte=0, newBytePos=7;

	for(int y_pos=0; y_pos<_height; y_pos+=8) {
		for (int x_pos=0; x_pos<_width; x_pos+=8) {
			char y_data[64], cb_data[64], cr_data[64];
			short yQuant[64], cbQuant[64], crQuant[64];

			//转换颜色空间
			convert_color_space(x_pos, y_pos, y_data, cb_data, cr_data);

			BitString outputBitString[128];
			int bitStringCounts;

			//Y通道压缩
			foword_fdc(y_data, yQuant);
			do_huffman_encoding(yQuant, prev_DC_Y, m_Y_DC_Huffman_Table, m_Y_AC_Huffman_Table, outputBitString, bitStringCounts); 
			write_bitstring(outputBitString, bitStringCounts, newByte, newBytePos, fp);

			//Cb通道压缩
			foword_fdc(cb_data, cbQuant);			
			do_huffman_encoding(cbQuant, prev_DC_Cb, m_CbCr_DC_Huffman_Table, m_CbCr_AC_Huffman_Table, outputBitString, bitStringCounts);
			write_bitstring(outputBitString, bitStringCounts, newByte, newBytePos, fp);

			//Cr通道压缩
			foword_fdc(cr_data, crQuant);			
			do_huffman_encoding(crQuant, prev_DC_Cr, m_CbCr_DC_Huffman_Table, m_CbCr_AC_Huffman_Table, outputBitString, bitStringCounts);
			write_bitstring(outputBitString, bitStringCounts, newByte, newBytePos, fp);
		}
	}

	write_word(0xFFD9, fp); //Write End of Image Marker   
	
	fclose(fp);

	return true;
}


void JpegEncoder::init_huffman_tables(void) {
	memset(&m_Y_DC_Huffman_Table, 0, sizeof(m_Y_DC_Huffman_Table));
	compute_huffman_table(Standard_DC_Luminance_NRCodes, Standard_DC_Luminance_Values, m_Y_DC_Huffman_Table);
	for (int i=0; i<12; ++i) {
		std::cout << m_Y_DC_Huffman_Table[i].length << ", " << m_Y_DC_Huffman_Table[i].value << std::endl;
	}

	memset(&m_Y_AC_Huffman_Table, 0, sizeof(m_Y_AC_Huffman_Table));
	compute_huffman_table(Standard_AC_Luminance_NRCodes, Standard_AC_Luminance_Values, m_Y_AC_Huffman_Table);

	memset(&m_CbCr_DC_Huffman_Table, 0, sizeof(m_CbCr_DC_Huffman_Table));
	compute_huffman_table(Standard_DC_Chrominance_NRCodes, Standard_DC_Chrominance_Values, m_CbCr_DC_Huffman_Table);

	memset(&m_CbCr_AC_Huffman_Table, 0, sizeof(m_CbCr_AC_Huffman_Table));
	compute_huffman_table(Standard_AC_Chrominance_NRCodes, Standard_AC_Chrominance_Values, m_CbCr_AC_Huffman_Table);
}

JpegEncoder::BitString JpegEncoder::get_bit_code(int value) {
	BitString ret;
	int v = (value>0) ? value : -value;
	
	//bit 的长度
	int length = 0;
	for(length=0; v; v>>=1) length++;

	ret.value = value>0 ? value : ((1<<length)+value-1);
	ret.length = length;

	return ret;
};

void JpegEncoder::init_quality_tables(int quality_scale) {
	if(quality_scale<=0) quality_scale=1;
	if(quality_scale>=100) quality_scale=99;

	for(int i=0; i<64; i++) {
		int temp = ((int)(Luminance_Quantization_Table[i] * quality_scale + 50) / 100);
		if (temp<=0) temp = 1;
		if (temp>0xFF) temp = 0xFF;
		_y_table[ZigZag[i]] = (unsigned char)temp;

		temp = ((int)(Chrominance_Quantization_Table[i] * quality_scale + 50) / 100);
		if (temp<=0) 	temp = 1;
		if (temp>0xFF) temp = 0xFF;
		_cb_cr_table[ZigZag[i]] = (unsigned char)temp;
	}
}


void JpegEncoder::compute_huffman_table(const char* nr_codes, const unsigned char* std_table, BitString* huffman_table) {
	unsigned char pos_in_table = 0;
	unsigned short code_value = 0;

	for(int k = 1; k <= 16; k++)
	{
		for(int j = 1; j <= nr_codes[k-1]; j++)
		{
			huffman_table[std_table[pos_in_table]].value = code_value;
			huffman_table[std_table[pos_in_table]].length = k;
			pos_in_table++;
			code_value++;
		}
		code_value <<= 1;
	}  
}


void JpegEncoder::write_byte(unsigned char value, FILE* fp)
{
	f_write(&value, 1, fp);
}


void JpegEncoder::write_word(unsigned short value, FILE* fp)
{
	unsigned short _value = ((value>>8)&0xFF) | ((value&0xFF)<<8);
	f_write(&_value, 2, fp);
}


void JpegEncoder::f_write(const void* p, int byte_size, FILE* fp)
{
	fwrite(p, 1, byte_size, fp);
}


void JpegEncoder::do_huffman_encoding(const short* DU, short& prevDC, const BitString* HTDC, const BitString* HTAC, 
	BitString* outputBitString, int& bitStringCounts)
{
	BitString EOB = HTAC[0x00];
	BitString SIXTEEN_ZEROS = HTAC[0xF0];

	int index=0;

	// encode DC
	int dcDiff = (int)(DU[0] - prevDC);
	prevDC = DU[0];

	if (dcDiff == 0) 
		outputBitString[index++] = HTDC[0];
	else
	{
		BitString bs = get_bit_code(dcDiff);

		outputBitString[index++] = HTDC[bs.length];
		outputBitString[index++] = bs;
	}

	// encode ACs
	int endPos=63; //end0pos = first element in reverse order != 0
	while((endPos > 0) && (DU[endPos] == 0)) endPos--;

	for(int i=1; i<=endPos; )
	{
		int startPos = i;
		while((DU[i] == 0) && (i <= endPos)) i++;

		int zeroCounts = i - startPos;
		if (zeroCounts >= 16)
		{
			for (int j=1; j<=zeroCounts/16; j++)
				outputBitString[index++] = SIXTEEN_ZEROS;
			zeroCounts = zeroCounts%16;
		}

		BitString bs = get_bit_code(DU[i]);

		outputBitString[index++] = HTAC[(zeroCounts << 4) | bs.length];
		outputBitString[index++] = bs;
		i++;
	}

	if (endPos != 63)
		outputBitString[index++] = EOB;

	bitStringCounts = index;
}


void JpegEncoder::write_bitstring(const BitString* bs, int counts, int& newByte, int& newBytePos, FILE* fp)
{
	unsigned short mask[] = {1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768};
	
	for(int i=0; i<counts; i++)
	{
		int value = bs[i].value;
		int posval = bs[i].length - 1;

		while (posval >= 0)
		{
			if ((value & mask[posval]) != 0)
			{
				newByte = newByte  | mask[newBytePos];
			}
			posval--;
			newBytePos--;
			if (newBytePos < 0)
			{
				// Write to stream
				write_byte((unsigned char)(newByte), fp);
				if (newByte == 0xFF)
				{
					// Handle special case
					write_byte((unsigned char)(0x00), fp);
				}

				// Reinitialize
				newBytePos = 7;
				newByte = 0;
			}
		}
	}
}


void JpegEncoder::convert_color_space(int x_pos, int y_pos, char* y_data, char* cb_data, char* cr_data)
{
	for (int y=0; y<8; y++)
	{
		unsigned char* p = _rgb_buffer + (y+y_pos)*_width*3 + x_pos*3;
		for (int x=0; x<8; x++)
		{
			unsigned char B = *p++;
			unsigned char G = *p++;
			unsigned char R = *p++;

			y_data[y*8+x] = (char)(0.299f * R + 0.587f * G + 0.114f * B - 128);
			cb_data[y*8+x] = (char)(-0.1687f * R - 0.3313f * G + 0.5f * B );
			cr_data[y*8+x] = (char)(0.5f * R - 0.4187f * G - 0.0813f * B);
		}
	}
}


void JpegEncoder::foword_fdc(const char* channel_data, short* fdc_data)
{
	const float PI = 3.1415926f;
	for(int v=0; v<8; v++)
	{
		for(int u=0; u<8; u++)
		{
			float alpha_u = (u==0) ? 1/sqrt(8.0f) : 0.5f;
			float alpha_v = (v==0) ? 1/sqrt(8.0f) : 0.5f;

			float temp = 0.f;
			for(int x=0; x<8; x++)
			{
				for(int y=0; y<8; y++)
				{
					float data = channel_data[y*8+x];

					data *= cos((2*x+1)*u*PI/16.0f);
					data *= cos((2*y+1)*v*PI/16.0f);

					temp += data;
				}
			}

			temp *= alpha_u*alpha_v/_y_table[ZigZag[v*8+u]];
			fdc_data[ZigZag[v*8+u]] = (short) ((short)(temp + 16384.5) - 16384);
		}
	}
}


void JpegEncoder::write_jpeg_header(FILE* fp)
{
	//SOI
	write_word(0xFFD8, fp);		// marker = 0xFFD8

	//APPO
	write_word(0xFFE0,fp);		// marker = 0xFFE0
	write_word(16, fp);			// length = 16 for usual JPEG, no thumbnail
	f_write("JFIF", 5, fp);			// 'JFIF\0'
	write_byte(1, fp);			// version_hi
	write_byte(1, fp);			// version_low
	write_byte(0, fp);			// xyunits = 0 no units, normal density
	write_word(1, fp);			// xdensity
	write_word(1, fp);			// ydensity
	write_byte(0, fp);			// thumbWidth
	write_byte(0, fp);			// thumbHeight

	//DQT
	write_word(0xFFDB, fp);		//marker = 0xFFDB
	write_word(132, fp);			//size=132
	write_byte(0, fp);			//QTYinfo== 0:  bit 0..3: number of QT = 0 (table for Y) 
									//				bit 4..7: precision of QT
									//				bit 8	: 0
	f_write(_y_table, 64, fp);		//YTable
	write_byte(1, fp);			//QTCbinfo = 1 (quantization table for Cb,Cr)
	f_write(_cb_cr_table, 64, fp);	//CbCrTable

	//SOFO
	write_word(0xFFC0, fp);			//marker = 0xFFC0
	write_word(17, fp);				//length = 17 for a truecolor YCbCr JPG
	write_byte(8, fp);				//precision = 8: 8 bits/sample 
	write_word(_height&0xFFFF, fp);	//height
	write_word(_width&0xFFFF, fp);	//width
	write_byte(3, fp);				//nrofcomponents = 3: We encode a truecolor JPG

	write_byte(1, fp);				//IdY = 1
	write_byte(0x11, fp);				//HVY sampling factors for Y (bit 0-3 vert., 4-7 hor.)(SubSamp 1x1)
	write_byte(0, fp);				//QTY  Quantization Table number for Y = 0

	write_byte(2, fp);				//IdCb = 2
	write_byte(0x11, fp);				//HVCb = 0x11(SubSamp 1x1)
	write_byte(1, fp);				//QTCb = 1

	write_byte(3, fp);				//IdCr = 3
	write_byte(0x11, fp);				//HVCr = 0x11 (SubSamp 1x1)
	write_byte(1, fp);				//QTCr Normally equal to QTCb = 1
	
	//DHT
	write_word(0xFFC4, fp);		//marker = 0xFFC4
	write_word(0x01A2, fp);		//length = 0x01A2
	write_byte(0, fp);			//HTYDCinfo bit 0..3	: number of HT (0..3), for Y =0
									//			bit 4		: type of HT, 0 = DC table,1 = AC table
									//			bit 5..7	: not used, must be 0
	f_write(Standard_DC_Luminance_NRCodes, sizeof(Standard_DC_Luminance_NRCodes), fp);	//DC_L_NRC
	f_write(Standard_DC_Luminance_Values, sizeof(Standard_DC_Luminance_Values), fp);		//DC_L_VALUE
	write_byte(0x10, fp);			//HTYACinfo
	f_write(Standard_AC_Luminance_NRCodes, sizeof(Standard_AC_Luminance_NRCodes), fp);
	f_write(Standard_AC_Luminance_Values, sizeof(Standard_AC_Luminance_Values), fp); //we'll use the standard Huffman tables
	write_byte(0x01, fp);			//HTCbDCinfo
	f_write(Standard_DC_Chrominance_NRCodes, sizeof(Standard_DC_Chrominance_NRCodes), fp);
	f_write(Standard_DC_Chrominance_Values, sizeof(Standard_DC_Chrominance_Values), fp);
	write_byte(0x11, fp);			//HTCbACinfo
	f_write(Standard_AC_Chrominance_NRCodes, sizeof(Standard_AC_Chrominance_NRCodes), fp);
	f_write(Standard_AC_Chrominance_Values, sizeof(Standard_AC_Chrominance_Values), fp);

	//SOS
	write_word(0xFFDA, fp);		//marker = 0xFFC4
	write_word(12, fp);			//length = 12
	write_byte(3, fp);			//nrofcomponents, Should be 3: truecolor JPG

	write_byte(1, fp);			//Idy=1
	write_byte(0, fp);			//HTY	bits 0..3: AC table (0..3)
									//		bits 4..7: DC table (0..3)
	write_byte(2, fp);			//IdCb
	write_byte(0x11, fp);			//HTCb

	write_byte(3, fp);			//IdCr
	write_byte(0x11, fp);			//HTCr

	write_byte(0, fp);			//Ss not interesting, they should be 0,63,0
	write_byte(0x3F, fp);			//Se
	write_byte(0, fp);			//Bf
}
