/*! ChaCha20  OpenCL implementation of ChaCha20 cipher
	[RFC 8439] ChaCha20 & Poly1305, June 2018
	Copyright (c) 2024 Anatoly Georgievskii <anatoly.georgievski@gmail.com>

//clang -Xclang -finclude-default-header -cl-std=CL2.0 -march=rocketlake -S -O3 -o magma.s magma.cl
clang --target=amdgcn -mcpu=gfx1102 -Xclang -finclude-default-header -cl-mad-enable -cl-std=CL2.0 -S -O3 -o magma.s magma.cl
llvm-mca --march=amdgcn --mcpu=gfx1102 -timeline --timeline-max-cycles=210 magma.s | less
 */

static inline
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
		x1 = shuffle(x1, (uint4)(1,2,3,0));
		x2 = shuffle(x2, (uint4)(2,3,0,1));
		x3 = shuffle(x3, (uint4)(3,0,1,2));
		x1 ^= rotate(x0 + x3, 7);
		x2 ^= rotate(x1 + x0, 9);
		x3 ^= rotate(x2 + x1,13);
		x0 ^= rotate(x3 + x2,18);
		x3 = shuffle(x3, (uint4)(1,2,3,0));
		x2 = shuffle(x2, (uint4)(2,3,0,1));
		x1 = shuffle(x1, (uint4)(3,0,1,2));
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
__kernel void salsa20_8_2(__global uint4 *d, __global uint4 *s)
{
	uint4 X[8];
	for (int i=0;i<8; i++)
		X[i] = s[i];

	salsa20_8(&X[0], &X[4]);
	salsa20_8(&X[4], &X[0]);
	for (int i=0;i<8; i++)
		d[i] = X[i];
	
}
