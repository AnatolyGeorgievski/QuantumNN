enum {
	h0 = 0x67452301,
	h1 = 0xEFCDAB89,
	h2 = 0x98BADCFE,
	h3 = 0x10325476,
	h4 = 0xC3D2E1F0,
};
#define R0(a,b,c,d,e, f, j) ({ \
    a = rotate(a + f(b, c, d) + W[r0[j]] + K0[j/16], s0[j]) + e; \
    c = rotate(c, 10); \
	})
#define R1(a,b,c,d,e, f, j) ({ \
    a = rotate(a + f(b, c, d) + W[r1[j]] + K1[j/16], s1[j]) + e; \
    c = rotate(c, 10); \
	})
__kernel void RIPEMD160(__global uint *h, __global uint *W){
	uint (^f1)(uint x,uint y,uint z) = ^(uint x,uint y,uint z){return x^y^z;};
	uint (^f2)(uint x,uint y,uint z) = ^(uint x,uint y,uint z){return (x&y)|(~x&z);};
	uint (^f3)(uint x,uint y,uint z) = ^(uint x,uint y,uint z){return (x|~y)^z;};
	uint (^f4)(uint x,uint y,uint z) = ^(uint x,uint y,uint z){return (x&z)|(y&~z);};
	uint (^f5)(uint x,uint y,uint z) = ^(uint x,uint y,uint z){return x^(y|~z);};
	uint2 a0,b0,c0,d0,e0;
	a0 = h[0]; b0 = h[1]; c0 = h[2]; d0 = h[3]; e0 = h[4];
	const uint  K0[] = {0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E};
	const uint  K1[] = {0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000};
	const uchar r0[] = {
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
		7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
		3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12,
		1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
		4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13
	};
	const uchar r1[] = {
		5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12,
		6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
		15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13,
		8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
		12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11
	};
	const uchar s0[] = {
		11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8,
		7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
		11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5,
		11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
		9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6
	};
	const uchar s1[] = {
		8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6,
		9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
		9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5,
		15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
		8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11
	};
	#pragma unroll
	for (int j=0; j<80; ++j) {
	R0(a,b,c,d,e, f1, j);R1(a,b,c,d,e, f5, j);
	R0(e,a,b,c,d, f1, j);R1(e,a,b,c,d, f5, j);
	R0(d,e,a,b,c, f1, j);R1(d,e,a,b,c, f5, j);
	R0(c,d,e,a,b, f1, j);R1(c,d,e,a,b, f5, j);
	R0(b,c,d,e,a, f1, j);R1(b,c,d,e,a, f5, j);

	R0(a,b,c,d,e, f1, j);R1(a,b,c,d,e, f5, j);
	R0(a,b,c,d,e, f2, j);R1(a,b,c,d,e, f4, j);
	R0(a,b,c,d,e, f3, j);R1(a,b,c,d,e, f3, j);
	R0(a,b,c,d,e, f4, j);R1(a,b,c,d,e, f2, j);
	R0(a,b,c,d,e, f5, j);R1(a,b,c,d,e, f1, j);
	}
	uint t=h[1]+c.x+d.y;
	h[1] = h[2]+d.x+e.y; 
	h[2] = h[3]+e.x+a.y; 
	h[3] = h[4]+a.x+b.y; 
	h[4] = h[0]+b.x+c.y; 
	h[0] = t;
}
/*
RIPEMD-160("The quick brown fox jumps over the lazy dog") =
 37f332f68db77bd9d7edd4969571ad671cf9dd3b
RIPEMD-160("The quick brown fox jumps over the lazy cog") =
 132072df690933835eb8b6ad0b77e7b6f14acad7
RIPEMD-160("") = 
 9c1185a5c5e9fc54612808977ee8f548b2258d31
 */