#ifndef MI_JPEG_QUANTIZE_H
#define MI_JPEG_QUANTIZE_H

/** Default Quantization Table for Y component (zig-zag order)*/
static int gpujpeg_table_default_quantization_luminance[] = { 
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
static int gpujpeg_table_default_quantization_chrominance[] = { 
  17,  18,  18,  24,  21,  24,  47,  26,
  26,  47,  99,  66,  56,  66,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99,
  99,  99,  99,  99,  99,  99,  99,  99
};

inline void quantize8(int* inout, float* q_table) {
    for (int i=0; i<64; ++i) {
        inout[i] = (int)(inout[i]*q_table[i] + 0.5f);
    }
}

inline void init_qtable(float (&qt)[64]) {
    int* qt_raw = gpujpeg_table_default_quantization_luminance;
    for (int i=0; i<64; ++i) {
        qt[i] = 1.0f/qt_raw[i];
    }
}


///////////////////////////////////////////////////////////////////////
//quantization with fast DCT

////////////////
//libjpeg
////////////////

// #ifdef DCT_FLOAT_SUPPORTED
//     case JDCT_FLOAT:
//       {
// 	/* For float AA&N IDCT method, divisors are equal to quantization
// 	 * coefficients scaled by scalefactor[row]*scalefactor[col], where
// 	 *   scalefactor[0] = 1
// 	 *   scalefactor[k] = cos(k*PI/16) * sqrt(2)    for k=1..7
// 	 * We apply a further scale factor of 8.
// 	 * What's actually stored is 1/divisor so that the inner loop can
// 	 * use a multiplication rather than a division.
// 	 */
// 	FAST_FLOAT * fdtbl = (FAST_FLOAT *) compptr->dct_table;
// 	int row, col;
// 	static const double aanscalefactor[DCTSIZE] = {
// 	  1.0, 1.387039845, 1.306562965, 1.175875602,
// 	  1.0, 0.785694958, 0.541196100, 0.275899379
// 	};

// 	i = 0;
// 	for (row = 0; row < DCTSIZE; row++) {
// 	  for (col = 0; col < DCTSIZE; col++) {
// 	    fdtbl[i] = (FAST_FLOAT)
// 	      (1.0 / ((double) qtbl->quantval[i] *
// 		      aanscalefactor[row] * aanscalefactor[col] *
// 		      (compptr->component_needed ? 16.0 : 8.0)));
// 	    i++;
// 	  }
// 	}
//       }
//       fdct->pub.forward_DCT[ci] = forward_DCT_float;
//       break;


////////////////
//gpujpeg
////////////////

 // Scales of outputs of 1D DCT.
    // const double dct_scales[8] = {1.0, 1.387039845, 1.306562965, 1.175875602, 1.0, 0.785694958, 0.541196100, 0.275899379};
    
    // // Prepare transposed float quantization table, pre-divided by output DCT weights
    // float h_quantization_table[64];
    // for( unsigned int i = 0; i < 64; i++ ) {
    //     const unsigned int x = gpujpeg_order_natural[i] % 8;
    //     const unsigned int y = gpujpeg_order_natural[i] / 8;
    //     h_quantization_table[x * 8 + y] = 1.0 / (table->table_raw[i] * dct_scales[x] * dct_scales[y] * 8); // 8 is the gain of 2D DCT
    // }



inline void init_qtable_fast(float (&qt)[64]) {
    int* qt_raw = gpujpeg_table_default_quantization_luminance;
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


#endif