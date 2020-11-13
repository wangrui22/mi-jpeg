#ifndef JPEG_ENCODER_HE
#define JPEG_ENCODER_HE

#include <string>

class JpegEncoder {
public:
	JpegEncoder();
	~JpegEncoder();
	void clean(void);
	bool encode_to_jpeg(unsigned char* rgb, int width, int height, int quality_scale , const std::string& file_name);

private:
    struct BitString {
		int length;	
		int value;
	};
	void init_huffman_tables(void);
	void init_quality_tables(int quality);
	void compute_huffman_table(const char* nr_codes, const unsigned char* std_table, BitString* huffman_table);
	BitString get_bit_code(int value);

	void convert_color_space(int x_pos, int y_pos, char* y_data, char* cb_data, char* cr_data);
	void foword_fdc(const char* channel_data, short* fdc_data);
	void do_huffman_encoding(const short* DU, short& prevDC, const BitString* HTDC, const BitString* HTAC, 
		BitString* outputBitString, int& bitStringCounts);

	void write_jpeg_header(FILE* fp);
	void write_byte(unsigned char value, FILE* fp);
	void write_word(unsigned short value, FILE* fp);
	void write_bitstring(const BitString* bs, int counts, int& newByte, int& newBytePos, FILE* fp);
	void f_write(const void* p, int byte_size, FILE* fp);

private:
	int				_width;
	int				_height;
	unsigned char*	_rgb_buffer;
	
	unsigned char	_y_table[64];
	unsigned char	_cb_cr_table[64];

	BitString m_Y_DC_Huffman_Table[12];
	BitString m_Y_AC_Huffman_Table[256];

	BitString m_CbCr_DC_Huffman_Table[12];
	BitString m_CbCr_AC_Huffman_Table[256];

};

#endif
