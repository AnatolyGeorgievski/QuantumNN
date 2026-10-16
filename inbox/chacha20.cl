/*! ChaCha20  OpenCL implementation of ChaCha20 cipher
	[RFC 8439] ChaCha20 & Poly1305, June 2018
	Copyright (c) 2024 Anatoly Georgievskii <anatoly.georgievski@gmail.com>

$ clang -Xclang -finclude-default-header -march=native --std=CL2.0 -O3 -S -o sha.s  opencl/chacha20.cl
$ clang --target=amdgcn -march=gfx1102 --std=CL2.0 -O3 -S -o sha.s  opencl/chacha20.cl
//clang -Xclang -finclude-default-header -cl-std=CL2.0 -march=rocketlake -S -O3 -o magma.s magma.cl
clang --target=amdgcn -mcpu=gfx1102 -Xclang -finclude-default-header -cl-mad-enable -cl-std=CL2.0 -S -O3 -o magma.s magma.cl
llvm-mca --march=amdgcn --mcpu=gfx1102 -timeline --timeline-max-cycles=210 magma.s | less
 */

#if 0
#define rotate(x,n) ((x)<<(n) | (x)>>(32-(n)))
#endif

typedef uint4 v4su;
__kernel void ChaCha20(__global v4su *s) 
{
	//uint gid = get_global_id(0);
	uint gid = get_global_id(0);//+get_global_id(1)*get_global_size(0);
	v4su a = s[gid*4+0];
	v4su b = s[gid*4+1];
	v4su c = s[gid*4+2];
	v4su d = s[gid*4+3];
	int i;
//	__attribute__((opencl_unroll_hint(10)))
	for (i=0;i<10;i++) {
		a += b; d = rotate(d^a,16);//16,0
		c += d; b = rotate(b^c,12);//12
		a += b; d = rotate(d^a, 8);//24,
		c += d; b = rotate(b^c, 7);//19
		b = b.s1230;//5674;
		c = c.s2301;//6745;
		d = d.s3012;//7456;
		a += b; d = rotate(d^a,16);// 8
		c += d; b = rotate(b^c,12);//31
		a += b; d = rotate(d^a, 8);//16
		c += d; b = rotate(b^c, 7);//
		b = b.s3012;//7456;
		c = c.s2301;//6745;
		d = d.s1230;//5674;
	}
	s[gid*4+0] += a;
	s[gid*4+1] += b;
	s[gid*4+2] += c;
	s[gid*4+3] += d;
}