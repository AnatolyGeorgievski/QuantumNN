// $ clang -Xclang -finclude-default-header -cl-std=CL2.0 -march=native -DVECTOR_SIZE=32 -O3 -S -o - sha256_vec.cl
// clang -target arm-none-eabi -S foo.c -mcpu=cortex-m3 -mthumb -O2
// $ gcc -march=native -O6 -o sha2_vec8.exe sha2_vec.c

// $ time ./sha2_vec8.exe
// см реализацию тернарной логики vpternlogd в AVX-512 можно сделать любую логическую функцию
// см функцию bitselect в спецификации OpenCL см инструкцию bitselect в NEON
// Тернарные операции присутствуют в Intel Graphics.
// #define ch(x, y, z) (z ^ (x & (y ^ z))) 
// #define ma(x, y, z) ((x & z) | (y & (x | z)))
// #define ch(x, y, z) bitselect(z, y, x)
// #define ma(x, y, z) bitselect(z, y, z ^ x)

#ifndef VECTOR_SIZE
#define VECTOR_SIZE 16
#endif

#if (VECTOR_SIZE==64)
typedef uint16 v4su;
typedef uchar16 v4bu;
#elif (VECTOR_SIZE==32)
typedef uint8 v4su;
typedef uchar8 v4bu;
#elif (VECTOR_SIZE==16)
typedef uint4 v4su;
typedef uchar4 v4bu;
#elif (VECTOR_SIZE==8)
typedef uint2 v4su;
typedef uchar2 v4bu;
#else
typedef uint v4su;
typedef uchar v4bu;
#endif

//#define bitselect(a,b,c) (~(c)&(a) ^ (c)&(b))
#define rotr(v,i) rotate(v, (uint)32-i)//(((v)<<(32-i)) ^ ((v)>>(i)))

/* три операции для тернарной логики: Ch, Maj, Parity */
//    return (x&y) ^ (~x&z);
#define Ch(x, y, z)  bitselect(z, y, x)
#define Maj(x, y, z) /*bitselect(z, y, z ^ x)*/((x&y) ^ (x&z) ^ (y&z))

#define sigma0(x) (rotr(x, 7) ^ rotr(x,18) ^ (x>> 3))
#define sigma1(x) (rotr(x,17) ^ rotr(x,19) ^ (x>>10))

#define Sum0(x) (rotr(x,2) ^ rotr(x,13) ^ rotr(x,22))
#define Sum1(x) (rotr(x,6) ^ rotr(x,11) ^ rotr(x,25))

#define ROUND_(a,b,c,d,e,f,g,h,t,i) ({\
	h += Ch(e,f,g) + Sum1(e) + W[t+i];  \
	d += h; \
	h += Sum0(a) + Maj(a,b,c); \
	})
#define SHA256MSG_(W,i) ({ \
	uint4 T = W[i&3] + shuffle2(W[(i+2)&3],W[(i+3)&3], (uint4)(1,2,3,4)) + sigma0(shuffle2(W[i&3],W[(i+1)&3],(uint4)(1,2,3,4))); \
	T.lo += sigma1(W[(i+3)&3].hi); \
	T.hi += sigma1(T.lo); \
	T; \
	})
#define PALIGNR(a,b,k) shuffle2(b,a,(uint4){1,2,3,4})
#define SHA256MSG1(a,b) (a + sigma0(shuffle2(a,b,(uint4){1,2,3,4})))
#define SHA256MSG2(a,b) ({ uint4 T;\
	T.lo = a.lo + sigma1(b.hi); \
	T.hi = a.hi + sigma1(T.lo); \
	T; \
	})
#define SHA256MSG(W,i) ({\
			uint4 X = SHA256MSG1(W[i&3], W[(i-3)&3])\
			  + PALIGNR(W[(i-1)&3],W[(i-2)&3],4);\
			SHA256MSG2(X, W[(i-1)&3]); \
			})
#define BE32(x) __builtin_bswap32(x)// ((((x)&0xFF)<<24) | (((x)&0xFF000000)>>24) | (((x)&0xFF00)<<8) | (((x)&0xFF0000)>>8))
#define LE32(x) (x)
__constant const uint4 K256[] = {
	{0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5}, {0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5},
	{0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3}, {0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174},
	{0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc}, {0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da},
	{0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7}, {0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967},
	{0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13}, {0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85},
	{0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3}, {0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070},
	{0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5}, {0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3},
	{0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208}, {0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2},
};
#define ROUND(a,b,c,d,e,f,g,h,t,i) ({\
	uint4 x = Sum1(e) + Ch (e,f,g) + W[t+i];  \
	uint4 y = Sum0(a) + Maj(a,b,c); \
	d += x + h; \
	h += x + y; \
	})
__constant const uint H0[8] = {
		(0x6a09e667), (0xbb67ae85), (0x3c6ef372), (0xa54ff53a), 
		(0x510e527f), (0x9b05688c), (0x1f83d9ab), (0x5be0cd19)};
// развертывание сообщения
__kernel void SHA256_W(__global uint4 * WK, __constant uint4 * W0)
{
//	uint4 (^sigma0)(uint4) = ^(uint4 x){return rotr(x, 7) ^ rotr(x,18) ^ (x>> 3);};
//	uint2 (^sigma1)(uint2) = ^(uint2 x){return rotr(x,17) ^ rotr(x,19) ^ (x>>10);};
	uint16 W;
	uint gid = get_global_id(0);
	int i = 0;
	W.s0123 = W0[gid*4 + 0];
	WK[i+0] = W.s0123 + K256[i+0];
	W.s4567 = W0[gid*4 + 1];
	WK[i+1] = W.s4567 + K256[i+1];
	W.s89AB = W0[gid*4 + 2];
	WK[i+2] = W.s89AB + K256[i+2];
	W.sCDEF = W0[gid*4 + 3];
	WK[i+3] = W.sCDEF + K256[i+3];
	//#pragma unroll
	for (i=4; i<16; i+=4){
		W.s0123 += W.s9ABC + sigma0(W.s1234);
		W.s01 += sigma1(W.sEF);
		W.s23 += sigma1(W.s01);
		WK[i+0] = W.s0123 + K256[i+0];

		W.s4567 += W.sDEF0 + sigma0(W.s5678);
		W.s45 += sigma1(W.s23);
		W.s67 += sigma1(W.s45);
		WK[i+1] = W.s4567 + K256[i+1];

		W.s89AB += W.s1234 + sigma0(W.s9ABC);
		W.s89 += sigma1(W.s67);
		W.sAB += sigma1(W.s89);
		WK[i+2] = W.s89AB + K256[i+2];

		W.sCDEF += W.s5678 + sigma0(W.sDEF0);
		W.sCD += sigma1(W.sAB);
		W.sEF += sigma1(W.sCD);
		WK[i+3] = W.sCDEF + K256[i+3];		//W = SHA256MSG()
	}
}
// Оптимизировано под AMD RDNA 2
__kernel void SHA256_W4(__global uint4 * WK, __constant uint * W0)
{
	const uint4 K256[] = {
		{0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5}, {0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5},
		{0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3}, {0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174},
		{0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc}, {0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da},
		{0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7}, {0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967},
		{0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13}, {0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85},
		{0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3}, {0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070},
		{0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5}, {0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3},
		{0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208}, {0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2},
	};
//	uint2 (^Sigma0)(uint2) = ^(uint2 x){return rotr(x, 7) ^ rotr(x,18) ^ (x>> 3);};
//	uint2 (^Sigma1)(uint2) = ^(uint2 x){return rotr(x,17) ^ rotr(x,19) ^ (x>>10);};
	uint4 W[4];
	uint gid = get_global_id(0);
	int i = 0;
	W[0] = W0[0]+gid;
	WK[0] = W[0] + K256[i+0];
	W[1] = W0[1];
	WK[1] = W[1] + K256[i+1];
	W[2] = W0[2];
	WK[2] = W[2] + K256[i+2];
	W[3] = W0[3];
	WK[3] = W[3] + K256[i+3];
	//#pragma unroll
	for (i=4; i<16; i+=4){
		W[0].lo += sigma1(W[3].hi) + (uint2)(W[2].y,W[2].z) + sigma0((uint2)(W[0].y,W[0].z));
		W[0].hi += sigma1(W[0].lo) + (uint2)(W[2].w,W[3].x) + sigma0((uint2)(W[0].w,W[1].x));
		WK[i+0] = W[0] + K256[i+0];

		W[1].lo += sigma1(W[0].hi) + (uint2)(W[3].y,W[3].z) + sigma0((uint2)(W[1].y,W[1].z));
		W[1].hi += sigma1(W[1].lo) + (uint2)(W[3].w,W[0].x) + sigma0((uint2)(W[1].w,W[2].x));
		WK[i+1] = W[1] + K256[i+1];

		W[2].lo += sigma1(W[1].hi) + (uint2)(W[0].y,W[0].z) + sigma0((uint2)(W[2].y,W[2].z));
		W[2].hi += sigma1(W[2].lo) + (uint2)(W[0].w,W[1].x) + sigma0((uint2)(W[2].w,W[3].x));
		WK[i+2] = W[2] + K256[i+2];
 
		W[3].lo += sigma1(W[2].hi) + (uint2)(W[1].y,W[1].z) + sigma0((uint2)(W[3].y,W[3].z));
		W[3].hi += sigma1(W[3].lo) + (uint2)(W[1].w,W[2].x) + sigma0((uint2)(W[3].w,W[0].x));
		WK[i+3] = W[3] + K256[i+3];
	}
}
// генерация мидстейтов
#if 1
__kernel void SHA256_8(__global uint4 * H, __constant uint4 * W)
{
	uint gid = 0;//get_global_id(0);
	uint mid = get_global_id(2);
	uint4 a,b,e,f, c,d,g,h;
    a = H0[0],b = H0[1],c = H0[2],d = H0[3],
	e = H0[4],f = H0[5],g = H0[6],h = H0[7];
	int t;
	#pragma unroll
    for (t=0; t<64; t+=16)
    {
		ROUND(a,b,c,d,e,f,g,h,t,0);
        ROUND(h,a,b,c,d,e,f,g,t,1);
        ROUND(g,h,a,b,c,d,e,f,t,2);
        ROUND(f,g,h,a,b,c,d,e,t,3);
        ROUND(e,f,g,h,a,b,c,d,t,4);
        ROUND(d,e,f,g,h,a,b,c,t,5);
        ROUND(c,d,e,f,g,h,a,b,t,6);
        ROUND(b,c,d,e,f,g,h,a,t,7);
		
        ROUND(a,b,c,d,e,f,g,h,t,8);
        ROUND(h,a,b,c,d,e,f,g,t,9);
        ROUND(g,h,a,b,c,d,e,f,t,10);
        ROUND(f,g,h,a,b,c,d,e,t,11);
        ROUND(e,f,g,h,a,b,c,d,t,12);
        ROUND(d,e,f,g,h,a,b,c,t,13);
        ROUND(c,d,e,f,g,h,a,b,t,14);
        ROUND(b,c,d,e,f,g,h,a,t,15);
    }
	int of = get_global_id(0)*get_global_size(0) + get_global_id(1);
    H[of+0]=a+H0[0], H[of+1]=b+H0[1], H[of+2]=c+H0[2], H[of+3]=d+H0[3], 
	H[of+4]=e+H0[4], H[of+5]=f+H0[5], H[of+6]=g+H0[6], H[of+7]=h+H0[7];
}
#endif
#undef  ROUND_
#undef  ROUND
#define ROUND(a,b,c,d,e,f,g,h,t,i) ({\
	v4su x = Sum1(e) + Ch (e,f,g) + WK[i&3];  \
	v4su y = Sum0(a) + Maj(a,b,c); \
	d += x + h; \
	h += x + y; \
	})
#define ROUND_(a,b,c,d,e,f,g,h,t,i) ({\
	h += Sum1(e) + Ch(e,f,g) + WK[i&3];  \
	d += h; \
	h += Sum0(a) + Maj(a,b,c); \
	})
/*! параллельно для нескольких версий и общего сообщения, несколько мидстейтов */
__kernel void SHA256_v4su(__constant v4su * H, __constant uint * M, __global v4bu * T)
{
	uint gid = get_global_id(0)*get_global_size(1)+get_global_id(1);
	v4su a,b,e,f, c,d,g,h;

    a = H[0],b = H[1],c = H[2],d = H[3],
	e = H[4],f = H[5],g = H[6],h = H[7];
  {
    uint16 W;
	uint4 WK;
	{
		const int t=0;
//		W[0] = vload4(gid+0, M);
		W.s0123 = vload4(0, M);
		W.s3 += gid;
		WK = W.s0123 + K256[0];
        ROUND(a,b,c,d,e,f,g,h,t,0);
        ROUND(h,a,b,c,d,e,f,g,t,1);
        ROUND(g,h,a,b,c,d,e,f,t,2);
        ROUND(f,g,h,a,b,c,d,e,t,3);
//		W[1] = (uint4)(BE32(0x80),0,0,0);
		W.s4567 = (uint4)(BE32(0x80),0,0,0);
		WK = W.s4567 + K256[1];
        ROUND(e,f,g,h,a,b,c,d,t,4);
        ROUND(d,e,f,g,h,a,b,c,t,5);
        ROUND(c,d,e,f,g,h,a,b,t,6);
        ROUND(b,c,d,e,f,g,h,a,t,7);
//		W[2] = 0;//vload4(4*gid+2, M);
		W.s89AB = 0;//vload4(4*gid+2, M);
		WK = W.s89AB + K256[2];
        ROUND(a,b,c,d,e,f,g,h,t,8);
        ROUND(h,a,b,c,d,e,f,g,t,9);
        ROUND(g,h,a,b,c,d,e,f,t,10);
        ROUND(f,g,h,a,b,c,d,e,t,11);
//		W[3] = (uint4){0,0,0,LE32(80<<3)};
		W.sCDEF = (uint4){0,0,0,LE32(80<<3)};
		WK = W.sCDEF + K256[3];
        ROUND(e,f,g,h,a,b,c,d,t,12);
        ROUND(d,e,f,g,h,a,b,c,t,13);
        ROUND(c,d,e,f,g,h,a,b,t,14);
        ROUND(b,c,d,e,f,g,h,a,t,15);
	}
	int t;
	//__attribute__((opencl_unroll_hint(4)))
	//#pragma unroll
    for (t=16; t<64; t+=16) 
    {
		const int t=16;
		W.s01 += sigma1(W.sEF) + W.s9A + sigma0(W.s12);
		W.s23 += sigma1(W.s01) + W.sBC + sigma0(W.s34);
		WK = W.s0123 + K256[t/4+0];

		ROUND(a,b,c,d,e,f,g,h,t,0);
        ROUND(h,a,b,c,d,e,f,g,t,1);
        ROUND(g,h,a,b,c,d,e,f,t,2);
        ROUND(f,g,h,a,b,c,d,e,t,3);

		W.s45 += sigma1(W.s23) + W.sDE + sigma0(W.s56);
		W.s67 += sigma1(W.s45) + W.sF0 + sigma0(W.s78);
		WK = W.s4567 + K256[t/4+1];

        ROUND(e,f,g,h,a,b,c,d,t,4);
        ROUND(d,e,f,g,h,a,b,c,t,5);
        ROUND(c,d,e,f,g,h,a,b,t,6);
        ROUND(b,c,d,e,f,g,h,a,t,7);

		W.s89 += sigma1(W.s67) + W.s12 + sigma0(W.s9A);
		W.sAB += sigma1(W.s89) + W.s34 + sigma0(W.sBC);
		WK = W.s89AB + K256[t/4+2];

        ROUND(a,b,c,d,e,f,g,h,t,8);
        ROUND(h,a,b,c,d,e,f,g,t,9);
        ROUND(g,h,a,b,c,d,e,f,t,10);
        ROUND(f,g,h,a,b,c,d,e,t,11);

		W.sCD += sigma1(W.sAB) + W.s56 + sigma0(W.sDE);
		W.sEF += sigma1(W.sCD) + W.s78 + sigma0(W.sF0);
		WK = W.sCDEF + K256[t/4+3];

        ROUND(e,f,g,h,a,b,c,d,t,12);
        ROUND(d,e,f,g,h,a,b,c,t,13);
        ROUND(c,d,e,f,g,h,a,b,t,14);
        ROUND(b,c,d,e,f,g,h,a,t,15);
    }
//    T[0] = H[0]+a, T[1] = H[1]+b, T[2] = H[2]+c, T[3] = H[3]+d, 
//	T[4] = H[4]+e, T[5] = H[5]+f, T[6] = H[6]+g, T[7] = H[7]+h;
  }

#undef ROUND
#define ROUND(a,b,c,d,e,f,g,h,t,i) ({\
	v4su x = Sum1(e) + Ch(e,f,g) + (W[i] + K256[(t+i)/4][i&3]);  \
	v4su y = Sum0(a) + Maj(a,b,c); \
	d += x + h; \
	h += x + y; \
	})

/*
__kernel void SHA256_2(__constant v4su * H, __global v4su* T)
{*/
	
    v4su W[16];
	W[0] = a+H[0]; a = H0[0];
	W[1] = b+H[1]; b = H0[1];
	W[2] = c+H[2]; c = H0[2];
	W[3] = d+H[3]; d = H0[3];
	W[4] = e+H[4]; e = H0[4];
	W[5] = f+H[5]; f = H0[5];
	W[6] = g+H[6]; g = H0[6];
	W[7] = h+H[7]; h = H0[7];
	
	{
		const int t=0;
        ROUND(a,b,c,d,e,f,g,h,t,0);
        ROUND(h,a,b,c,d,e,f,g,t,1);
        ROUND(g,h,a,b,c,d,e,f,t,2);
        ROUND(f,g,h,a,b,c,d,e,t,3);
        ROUND(e,f,g,h,a,b,c,d,t,4);
        ROUND(d,e,f,g,h,a,b,c,t,5);
        ROUND(c,d,e,f,g,h,a,b,t,6);
        ROUND(b,c,d,e,f,g,h,a,t,7);
		W[8] = BE32(0x80); 
        ROUND(a,b,c,d,e,f,g,h,t,8);
		W[9] = 0;
        ROUND(h,a,b,c,d,e,f,g,t,9);
		W[10] = 0;
        ROUND(g,h,a,b,c,d,e,f,t,10);
		W[11] = 0;
        ROUND(f,g,h,a,b,c,d,e,t,11);
		W[12] = 0; 
        ROUND(e,f,g,h,a,b,c,d,t,12);
		W[13] = 0;
        ROUND(d,e,f,g,h,a,b,c,t,13);
		W[14] = 0;
        ROUND(c,d,e,f,g,h,a,b,t,14);
		W[15] = LE32(32<<3);
        ROUND(b,c,d,e,f,g,h,a,t,15);
	}
	int t;
	//__attribute__((opencl_unroll_hint(4)))
	//#pragma unroll
    for (t=16; t<64; t+=16)
    {
		//const int t=16;
		W[ 0] += sigma1(W[14]) + W[ 9] + sigma0(W[ 1]);
		ROUND(a,b,c,d,e,f,g,h,t,0);
		W[ 1] += sigma1(W[15]) + W[10] + sigma0(W[ 2]);
        ROUND(h,a,b,c,d,e,f,g,t,1);
		W[ 2] += sigma1(W[ 0]) + W[11] + sigma0(W[ 3]);
        ROUND(g,h,a,b,c,d,e,f,t,2);
		W[ 3] += sigma1(W[ 1]) + W[12] + sigma0(W[ 4]);
        ROUND(f,g,h,a,b,c,d,e,t,3);
		W[ 4] += sigma1(W[ 2]) + W[13] + sigma0(W[ 5]);
        ROUND(e,f,g,h,a,b,c,d,t,4);
		W[ 5] += sigma1(W[ 3]) + W[14] + sigma0(W[ 6]);
        ROUND(d,e,f,g,h,a,b,c,t,5);
		W[ 6] += sigma1(W[ 4]) + W[15] + sigma0(W[ 7]);
        ROUND(c,d,e,f,g,h,a,b,t,6);
		W[ 7] += sigma1(W[ 5]) + W[ 0] + sigma0(W[ 8]);
        ROUND(b,c,d,e,f,g,h,a,t,7);
		W[ 8] += sigma1(W[ 6]) + W[ 1] + sigma0(W[ 9]);
        ROUND(a,b,c,d,e,f,g,h,t,8);
		W[ 9] += sigma1(W[ 7]) + W[ 2] + sigma0(W[10]);
        ROUND(h,a,b,c,d,e,f,g,t,9);
		W[10] += sigma1(W[ 8]) + W[ 3] + sigma0(W[11]);
        ROUND(g,h,a,b,c,d,e,f,t,10);
		W[11] += sigma1(W[ 9]) + W[ 4] + sigma0(W[12]);
        ROUND(f,g,h,a,b,c,d,e,t,11);
		W[12] += sigma1(W[10]) + W[ 5] + sigma0(W[13]);
        ROUND(e,f,g,h,a,b,c,d,t,12);
		W[13] += sigma1(W[11]) + W[ 6] + sigma0(W[14]);
        ROUND(d,e,f,g,h,a,b,c,t,13);
		W[14] += sigma1(W[12]) + W[ 7] + sigma0(W[15]);
        ROUND(c,d,e,f,g,h,a,b,t,14);
		W[15] += sigma1(W[13]) + W[ 8] + sigma0(W[ 0]);
        ROUND(b,c,d,e,f,g,h,a,t,15);
    }

	T[gid*2+0] = convert_uchar4(clz(h+H0[7]));
/*
	for (int i=0;i<4; i++)
	 if(gid == 0x100000)//h[i]+H0[7]==0) 
		printf ("mid=%d %08X\n", i, gid);
*/		
}



typedef struct _Nonce Nonce_t;
struct _Nonce {
	uint nonce;
	uchar mid;
	uchar diff;
	ushort version;
};
#if 0
#undef ROUND
#define ROUND(a,b,c,d,e,f,g,h,t,i) ({\
	h += Sum1(e) + Ch(e,f,g) + (W[i] + K256[(t+i)/4][i&3]);  \
	d += h; \
	h += Sum0(a) + Maj(a,b,c); \
	})
__kernel void SHA256_0(__constant uint * M, __global uint4* H)
{
	uint gid = get_global_id(0);
    uint4 W[16];
	uint4 a,b,e,f, c,d,g,h;
    a = H0[0],b = H0[1],c = H0[2],d = H0[3],
	e = H0[4],f = H0[5],g = H0[6],h = H0[7];
	{
		const int t=0;
		W[0] = (uint4)(0,1,2,3)+M[0]+gid;
        ROUND(a,b,c,d,e,f,g,h,t,0);
		W[1] = M[1];
        ROUND(h,a,b,c,d,e,f,g,t,1);
		W[2] = M[2];
        ROUND(g,h,a,b,c,d,e,f,t,2);
		W[3] = M[3];
        ROUND(f,g,h,a,b,c,d,e,t,3);
		W[4] = M[4];
        ROUND(e,f,g,h,a,b,c,d,t,4);
		W[5] = M[5];
        ROUND(d,e,f,g,h,a,b,c,t,5);
		W[6] = M[6];
        ROUND(c,d,e,f,g,h,a,b,t,6);
		W[7] = M[7];
        ROUND(b,c,d,e,f,g,h,a,t,7);
		W[8] = M[8];
        ROUND(a,b,c,d,e,f,g,h,t,8);
		W[9] = M[9];
        ROUND(h,a,b,c,d,e,f,g,t,9);
		W[10] = M[10];
        ROUND(g,h,a,b,c,d,e,f,t,10);
		W[11] = M[11];
        ROUND(f,g,h,a,b,c,d,e,t,11);
		W[12] = M[12];
        ROUND(e,f,g,h,a,b,c,d,t,12);
		W[13] = M[13];
        ROUND(d,e,f,g,h,a,b,c,t,13);
		W[14] = M[14];
        ROUND(c,d,e,f,g,h,a,b,t,14);
		W[15] = M[15];
        ROUND(b,c,d,e,f,g,h,a,t,15);
	}
    {
		const int t=16;
		W[ 0] += sigma1(W[14]) + W[ 9] + sigma0(W[ 1]);
		ROUND(a,b,c,d,e,f,g,h,t,0);
		W[ 1] += sigma1(W[15]) + W[10] + sigma0(W[ 2]);
        ROUND(h,a,b,c,d,e,f,g,t,1);
		W[ 2] += sigma1(W[ 0]) + W[11] + sigma0(W[ 3]);
        ROUND(g,h,a,b,c,d,e,f,t,2);
		W[ 3] += sigma1(W[ 1]) + W[12] + sigma0(W[ 4]);
        ROUND(f,g,h,a,b,c,d,e,t,3);
		W[ 4] += sigma1(W[ 2]) + W[13] + sigma0(W[ 5]);
        ROUND(e,f,g,h,a,b,c,d,t,4);
		W[ 5] += sigma1(W[ 3]) + W[14] + sigma0(W[ 6]);
        ROUND(d,e,f,g,h,a,b,c,t,5);
		W[ 6] += sigma1(W[ 4]) + W[15] + sigma0(W[ 7]);
        ROUND(c,d,e,f,g,h,a,b,t,6);
		W[ 7] += sigma1(W[ 5]) + W[ 0] + sigma0(W[ 8]);
        ROUND(b,c,d,e,f,g,h,a,t,7);
		W[ 8] += sigma1(W[ 6]) + W[ 1] + sigma0(W[ 9]);
        ROUND(a,b,c,d,e,f,g,h,t,8);
		W[ 9] += sigma1(W[ 7]) + W[ 2] + sigma0(W[10]);
        ROUND(h,a,b,c,d,e,f,g,t,9);
		W[10] += sigma1(W[ 8]) + W[ 3] + sigma0(W[11]);
        ROUND(g,h,a,b,c,d,e,f,t,10);
		W[11] += sigma1(W[ 9]) + W[ 4] + sigma0(W[12]);
        ROUND(f,g,h,a,b,c,d,e,t,11);
		W[12] += sigma1(W[10]) + W[ 5] + sigma0(W[13]);
        ROUND(e,f,g,h,a,b,c,d,t,12);
		W[13] += sigma1(W[11]) + W[ 6] + sigma0(W[14]);
        ROUND(d,e,f,g,h,a,b,c,t,13);
		W[14] += sigma1(W[12]) + W[ 7] + sigma0(W[15]);
        ROUND(c,d,e,f,g,h,a,b,t,14);
		W[15] += sigma1(W[13]) + W[ 8] + sigma0(W[ 0]);
        ROUND(b,c,d,e,f,g,h,a,t,15);
    }
    {
		const int t=32;
		W[ 0] += sigma1(W[14]) + W[ 9] + sigma0(W[ 1]);
		ROUND(a,b,c,d,e,f,g,h,t,0);
		W[ 1] += sigma1(W[15]) + W[10] + sigma0(W[ 2]);
        ROUND(h,a,b,c,d,e,f,g,t,1);
		W[ 2] += sigma1(W[ 0]) + W[11] + sigma0(W[ 3]);
        ROUND(g,h,a,b,c,d,e,f,t,2);
		W[ 3] += sigma1(W[ 1]) + W[12] + sigma0(W[ 4]);
        ROUND(f,g,h,a,b,c,d,e,t,3);
		W[ 4] += sigma1(W[ 2]) + W[13] + sigma0(W[ 5]);
        ROUND(e,f,g,h,a,b,c,d,t,4);
		W[ 5] += sigma1(W[ 3]) + W[14] + sigma0(W[ 6]);
        ROUND(d,e,f,g,h,a,b,c,t,5);
		W[ 6] += sigma1(W[ 4]) + W[15] + sigma0(W[ 7]);
        ROUND(c,d,e,f,g,h,a,b,t,6);
		W[ 7] += sigma1(W[ 5]) + W[ 0] + sigma0(W[ 8]);
        ROUND(b,c,d,e,f,g,h,a,t,7);
		W[ 8] += sigma1(W[ 6]) + W[ 1] + sigma0(W[ 9]);
        ROUND(a,b,c,d,e,f,g,h,t,8);
		W[ 9] += sigma1(W[ 7]) + W[ 2] + sigma0(W[10]);
        ROUND(h,a,b,c,d,e,f,g,t,9);
		W[10] += sigma1(W[ 8]) + W[ 3] + sigma0(W[11]);
        ROUND(g,h,a,b,c,d,e,f,t,10);
		W[11] += sigma1(W[ 9]) + W[ 4] + sigma0(W[12]);
        ROUND(f,g,h,a,b,c,d,e,t,11);
		W[12] += sigma1(W[10]) + W[ 5] + sigma0(W[13]);
        ROUND(e,f,g,h,a,b,c,d,t,12);
		W[13] += sigma1(W[11]) + W[ 6] + sigma0(W[14]);
        ROUND(d,e,f,g,h,a,b,c,t,13);
		W[14] += sigma1(W[12]) + W[ 7] + sigma0(W[15]);
        ROUND(c,d,e,f,g,h,a,b,t,14);
		W[15] += sigma1(W[13]) + W[ 8] + sigma0(W[ 0]);
        ROUND(b,c,d,e,f,g,h,a,t,15);
    }
    {
		const int t=48;
		W[ 0] += sigma1(W[14]) + W[ 9] + sigma0(W[ 1]);
		ROUND(a,b,c,d,e,f,g,h,t,0);
		W[ 1] += sigma1(W[15]) + W[10] + sigma0(W[ 2]);
        ROUND(h,a,b,c,d,e,f,g,t,1);
		W[ 2] += sigma1(W[ 0]) + W[11] + sigma0(W[ 3]);
        ROUND(g,h,a,b,c,d,e,f,t,2);
		W[ 3] += sigma1(W[ 1]) + W[12] + sigma0(W[ 4]);
        ROUND(f,g,h,a,b,c,d,e,t,3);
		W[ 4] += sigma1(W[ 2]) + W[13] + sigma0(W[ 5]);
        ROUND(e,f,g,h,a,b,c,d,t,4);
		W[ 5] += sigma1(W[ 3]) + W[14] + sigma0(W[ 6]);
        ROUND(d,e,f,g,h,a,b,c,t,5);
		W[ 6] += sigma1(W[ 4]) + W[15] + sigma0(W[ 7]);
        ROUND(c,d,e,f,g,h,a,b,t,6);
		W[ 7] += sigma1(W[ 5]) + W[ 0] + sigma0(W[ 8]);
        ROUND(b,c,d,e,f,g,h,a,t,7);
		W[ 8] += sigma1(W[ 6]) + W[ 1] + sigma0(W[ 9]);
        ROUND(a,b,c,d,e,f,g,h,t,8);
		W[ 9] += sigma1(W[ 7]) + W[ 2] + sigma0(W[10]);
        ROUND(h,a,b,c,d,e,f,g,t,9);
		W[10] += sigma1(W[ 8]) + W[ 3] + sigma0(W[11]);
        ROUND(g,h,a,b,c,d,e,f,t,10);
		W[11] += sigma1(W[ 9]) + W[ 4] + sigma0(W[12]);
        ROUND(f,g,h,a,b,c,d,e,t,11);
		W[12] += sigma1(W[10]) + W[ 5] + sigma0(W[13]);
        ROUND(e,f,g,h,a,b,c,d,t,12);
		W[13] += sigma1(W[11]) + W[ 6] + sigma0(W[14]);
        ROUND(d,e,f,g,h,a,b,c,t,13);
		W[14] += sigma1(W[12]) + W[ 7] + sigma0(W[15]);
        ROUND(c,d,e,f,g,h,a,b,t,14);
		W[15] += sigma1(W[13]) + W[ 8] + sigma0(W[ 0]);
        ROUND(b,c,d,e,f,g,h,a,t,15);
    }
int of = gid*8;
	H[of+0]=a+H0[0], H[of+1]=b+H0[1], H[of+2]=c+H0[2], H[of+3]=d+H0[3], 
	H[of+4]=e+H0[4], H[of+5]=f+H0[5], H[of+6]=g+H0[6], H[of+7]=h+H0[7];
}
#endif