static
void salsa20_8(uint4 *S, uint4 *Bx)
{
	uint4 x0,x1,x2,x3;
    x0 = S[0]^=Bx[0];
    x3 = S[1]^=Bx[1];
    x2 = S[2]^=Bx[2];
    x1 = S[3]^=Bx[3];
	int i;
	__attribute__((opencl_unroll_hint(8)))
	for (i=0;i<8;i+=2) {
		x1 = x1.s1230;
		x2 = x2.s2301;
		x3 = x3.s3012;
		x1 ^= rotate(x0 + x3, 7);
		x2 ^= rotate(x1 + x0, 9);
		x3 ^= rotate(x2 + x1,13);
		x0 ^= rotate(x3 + x2,18);
		x3 = x3.s1230;
		x2 = x2.s2301;
		x1 = x1.s3012;
		x3 ^= rotate(x0 + x1, 7);
		x2 ^= rotate(x3 + x0, 9);
		x1 ^= rotate(x2 + x3,13);
		x0 ^= rotate(x1 + x2,18);
    }
	S[0] += x0;
	S[1] += x3;
	S[2] += x2;
	S[3] += x1;
}
#define SHUFFLE2(v, w, a,b,c,d) shuffle2((v),(w), (uint4)(a,b,c,d))
static inline void transpose_4x4 (uint4 *s)
{
    uint4 x0 = SHUFFLE2 (s[0], s[1], 0, 1, 4, 5); // 0 1 4 5
    uint4 x1 = SHUFFLE2 (s[0], s[1], 2, 3, 6, 7); // 2 3 6 7
    uint4 x2 = SHUFFLE2 (s[2], s[3], 0, 1, 4, 5); // 8 9 12 13
    uint4 x3 = SHUFFLE2 (s[2], s[3], 2, 3, 6, 7); // 10 11 14 15
    s[0] = SHUFFLE2 (x0, x2, 0, 2, 4, 6);// 0 4  8 12
    s[1] = SHUFFLE2 (x0, x2, 1, 3, 5, 7);// 1 5  9 13
    s[2] = SHUFFLE2 (x1, x3, 0, 2, 4, 6);// 2 6 10 14
    s[3] = SHUFFLE2 (x1, x3, 1, 3, 5, 7);// 3 7 11 15
}
#if 0
static inline void rotate_cw_4x4(uint4 *S, uint4 *s){
	S[0] = s[0];
	S[1] = s[1].s1230;
	S[2] = s[2].s2301;
	S[3] = s[3].s3012;
	transpose_4x4(S);
}
static inline void rotate_ccw_4x4(uint4 *S, uint4 *s){
	transpose_4x4(s);
	S[0] = s[0];
	S[1] = s[1].s3012;
	S[2] = s[2].s2301;
	S[3] = s[3].s1230;
}
#else
static inline void rotate_cw_4x4(uint4 *d, uint4 *s)
{
    uint4 s0 = SHUFFLE2 (s[0], s[1], 0, 1, 5, 6); // 0 1 4 5
    uint4 s1 = SHUFFLE2 (s[0], s[1], 2, 3, 7, 4); // 2 3 6 7
    uint4 s2 = SHUFFLE2 (s[2], s[3], 2, 3, 7, 4); // 8 9 12 13
    uint4 s3 = SHUFFLE2 (s[2], s[3], 0, 1, 5, 6); // 10 11 14 15
    d[0] = SHUFFLE2 (s0, s2, 0, 2, 4, 6);// 0 4  8 12
    d[1] = SHUFFLE2 (s0, s2, 1, 3, 5, 7);// 1 5  9 13
    d[2] = SHUFFLE2 (s1, s3, 0, 2, 4, 6);// 2 6 10 14
    d[3] = SHUFFLE2 (s1, s3, 1, 3, 5, 7);// 3 7 11 15
}

static inline void rotate_ccw_4x4(uint4 *d, uint4 *s)
{
	uint4 x0 = SHUFFLE2 (s[0], s[1], 0, 1, 4, 5); // 0 1 4 5 - lo lo
    uint4 x1 = SHUFFLE2 (s[0], s[1], 2, 3, 6, 7); // 2 3 6 7 - hi hi 
    uint4 x2 = SHUFFLE2 (s[2], s[3], 0, 1, 4, 5); // 8 9 12 13 - lo lo
    uint4 x3 = SHUFFLE2 (s[2], s[3], 2, 3, 6, 7); // 10 11 14 15 - hi hi 
    d[0] = SHUFFLE2 (x0, x2, 0, 2, 4, 6);// 0 4  8 12
    d[1] = SHUFFLE2 (x0, x2, 7, 1, 3, 5);// 1 5  9 13
    d[2] = SHUFFLE2 (x1, x3, 4, 6, 0, 2);// 2 6 10 14
    d[3] = SHUFFLE2 (x1, x3, 3, 5, 7, 1);// 3 7 11 15
}
#endif
__kernel void scrypt_(__global uint4* s, __global uint4 *V)
{
	size_t idx = get_global_id(0);
//	uint4 V[1024*8];
	uint4 X[8];
	for(int i=0; i < 8; i ++) X[i] = s[i];
	rotate_cw_4x4(&X[0], &X[0]);
	rotate_cw_4x4(&X[4], &X[4]);
	for(int i=0; i < 1024; i ++) {
		for (int n=0; n<8; n++) V[1024*idx+i*8+n] = X[n];
		salsa20_8(&X[0], &X[4]);
		salsa20_8(&X[4], &X[0]);
	}
	for(int i=0; i < 1024; i ++) {
		int j = X[4][0] & 1023;
		for (int n=0; n<8; n++) X[n]^=V[j*8+n];
		salsa20_8(&X[0], &X[4]);
		salsa20_8(&X[4], &X[0]);
	}
	rotate_ccw_4x4(&X[0],&X[0]);// можно без этого на uint16
	rotate_ccw_4x4(&X[4],&X[4]);
	for(int i=0; i < 8; i ++) s[idx*8+i] = X[i];
}