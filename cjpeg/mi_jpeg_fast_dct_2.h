#ifndef MI_JPEG_FAST_DCT_2_H
#define MI_JPEG_FAST_DCT_2_H

//WR TODO
//比libjpeg和gpujpeg的fast要慢，直接输出最终的DCT

static const double S[] = {
	0.353553390593273762200422,
	0.254897789552079584470970,
	0.270598050073098492199862,
	0.300672443467522640271861,
	0.353553390593273762200422,
	0.449988111568207852319255,
	0.653281482438188263928322,
	1.281457723870753089398043,
};

static const double A[] = {
	0,
	0.707106781186547524400844,
	0.541196100146196984399723,
	0.707106781186547524400844,
	1.306562964876376527856643,
	0.382683432365089771728460,
};


// DCT type II, scaled. Algorithm by Arai, Agui, Nakajima, 1988.
// See: https://web.stanford.edu/class/ee398a/handouts/lectures/07-TransformCoding.pdf#page=30
template<typename T>
inline void FastDct8_transform(const T in0, const T in1, const T in2, const T in3, const T in4, const T in5, const T in6, const T in7,
                    T & out0, T & out1, T & out2, T & out3, T & out4, T & out5, T & out6, T & out7) {
	const double v0 = (double)in0 + (double)in7;
	const double v1 = (double)in1 + (double)in6;
	const double v2 = (double)in2 + (double)in5;
	const double v3 = (double)in3 + (double)in4;
	const double v4 = (double)in3 - (double)in4;
	const double v5 = (double)in2 - (double)in5;
	const double v6 = (double)in1 - (double)in6;
	const double v7 = (double)in0 - (double)in7;
	
	const double v8 = v0 + v3;
	const double v9 = v1 + v2;
	const double v10 = v1 - v2;
	const double v11 = v0 - v3;
	const double v12 = -v4 - v5;
	const double v13 = (v5 + v6) * A[3];
	const double v14 = v6 + v7;
	
	const double v15 = v8 + v9;
	const double v16 = v8 - v9;
	const double v17 = (v10 + v11) * A[1];
	const double v18 = (v12 + v14) * A[5];
	
	const double v19 = -v12 * A[2] - v18;
	const double v20 = v14 * A[4] - v18;
	
	const double v21 = v17 + v11;
	const double v22 = v11 - v17;
	const double v23 = v13 + v7;
	const double v24 = v7 - v13;
	
	const double v25 = v19 + v24;
	const double v26 = v23 + v20;
	const double v27 = v23 - v20;
	const double v28 = v24 - v19;
	
	out0 = (int)(S[0] * v15);
	out1 = (int)(S[1] * v26);
	out2 = (int)(S[2] * v21);
	out3 = (int)(S[3] * v28);
	out4 = (int)(S[4] * v16);
	out5 = (int)(S[5] * v25);
	out6 = (int)(S[6] * v22);
	out7 = (int)(S[7] * v27);
}

#endif