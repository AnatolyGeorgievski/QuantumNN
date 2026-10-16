#define H0 0x6a09e667
#define H1 0xbb67ae85
#define H2 0x3c6ef372
#define H3 0xa54ff53a
#define H4 0x510e527f
#define H5 0x9b05688c
#define H6 0x1f83d9ab
#define H7 0x5be0cd19

#define rotr(v,i) rotate(v, (uint)32-i)//(((v)<<(32-i)) ^ ((v)>>(i)))
/* три операции для тернарной логики: Ch, Maj, Parity */
//    return (x&y) ^ (~x&z);
#define Ch(x, y, z)  bitselect(z, y, x)
#define Maj(x, y, z) /*bitselect(z, y, z ^ x)*/((x&y) ^ (x&z) ^ (y&z))

#define sigma0(x) (rotr(x, 7) ^ rotr(x,18) ^ ((x)>> 3))
#define sigma1(x) (rotr(x,17) ^ rotr(x,19) ^ ((x)>>10))

#define Sum0(x) (rotr(x,2) ^ rotr(x,13) ^ rotr(x,22))
#define Sum1(x) (rotr(x,6) ^ rotr(x,11) ^ rotr(x,25))

#define uintV uint
__kernel void SHA256(__global uintV *H,__constant uint *M)
{
	//const uint M[16]={1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
	const uint K[64]={
		0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
		0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
		0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
		0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
		0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
		0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
		0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
		0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
	};
#define ROUND_(a,b,c,d,e,f,g,h,i) ({\
	h += Ch(e,f,g) + Sum1(e) + K[i] + W[i&15];  \
	d += h; \
	h += Sum0(a) + Maj(a,b,c); \
	})
// все операции тернарные
#define ROUND(a,b,c,d,e,f,g,h,i) ({\
	h += K[i] + W[i&15];  \
	uintV x = Sum1(e) + Ch (e,f,g); \
	uintV y = Sum0(a) + Maj(a,b,c); \
	d += x + h; \
	h += x + y; \
	})

    uint W[16];
    int t;
	size_t gid = get_global_id(0);//+get_global_id(1)*get_global_size(0);
    uintV a = H[gid*8+0],b = H[gid*8+1],c = H[gid*8+2],d = H[gid*8+3],
		  e = H[gid*8+4],f = H[gid*8+5],g = H[gid*8+6],h = H[gid*8+7];
    { t=0;
		W[0] = M[0];
        ROUND(a,b,c,d,e,f,g,h,t+0);
		W[1] = M[1];
        ROUND(h,a,b,c,d,e,f,g,t+1);
		W[2] = M[2];
        ROUND(g,h,a,b,c,d,e,f,t+2);
		W[3] = M[3];
        ROUND(f,g,h,a,b,c,d,e,t+3);
		W[4] = M[4];
        ROUND(e,f,g,h,a,b,c,d,t+4);
		W[5] = M[5];
        ROUND(d,e,f,g,h,a,b,c,t+5);
		W[6] = M[6];
        ROUND(c,d,e,f,g,h,a,b,t+6);
		W[7] = M[7];
        ROUND(b,c,d,e,f,g,h,a,t+7);

		W[8] = M[8];
        ROUND(a,b,c,d,e,f,g,h,t+8);
		W[9] = M[9];
        ROUND(h,a,b,c,d,e,f,g,t+9);
		W[10] = M[10];
        ROUND(g,h,a,b,c,d,e,f,t+10);
		W[11] = M[11];
        ROUND(f,g,h,a,b,c,d,e,t+11);
		W[12] = M[12];
        ROUND(e,f,g,h,a,b,c,d,t+12);
		W[13] = M[13];
        ROUND(d,e,f,g,h,a,b,c,t+13);
		W[14] = M[14];
        ROUND(c,d,e,f,g,h,a,b,t+14);
		W[15] = M[15];
        ROUND(b,c,d,e,f,g,h,a,t+15);
    }
	#pragma unroll
    for (t=16; t<64; t+=16)
    {
		W[0] += sigma1(W[14]) + W[ 9] + sigma0(W[1]);
        ROUND(a,b,c,d,e,f,g,h,t+0);
		W[1] += sigma1(W[15]) + W[10] + sigma0(W[2]);
        ROUND(h,a,b,c,d,e,f,g,t+1);
		W[2] += sigma1(W[ 0]) + W[11] + sigma0(W[3]);
        ROUND(g,h,a,b,c,d,e,f,t+2);
		W[3] += sigma1(W[ 1]) + W[12] + sigma0(W[4]);
        ROUND(f,g,h,a,b,c,d,e,t+3);
		W[4] += sigma1(W[ 2]) + W[13] + sigma0(W[5]);
        ROUND(e,f,g,h,a,b,c,d,t+4);
		W[5] += sigma1(W[ 3]) + W[14] + sigma0(W[6]);
        ROUND(d,e,f,g,h,a,b,c,t+5);
		W[6] += sigma1(W[ 4]) + W[15] + sigma0(W[7]);
        ROUND(c,d,e,f,g,h,a,b,t+6);
		W[7] += sigma1(W[ 5]) + W[ 0] + sigma0(W[8]);
        ROUND(b,c,d,e,f,g,h,a,t+7);

		W[ 8] += sigma1(W[ 6]) + W[ 1] + sigma0(W[ 9]);
        ROUND(a,b,c,d,e,f,g,h,t+ 8);
		W[ 9] += sigma1(W[ 7]) + W[ 2] + sigma0(W[10]);
        ROUND(h,a,b,c,d,e,f,g,t+ 9);
		W[10] += sigma1(W[ 8]) + W[ 3] + sigma0(W[11]);
        ROUND(g,h,a,b,c,d,e,f,t+10);
		W[11] += sigma1(W[ 9]) + W[ 4] + sigma0(W[12]);
        ROUND(f,g,h,a,b,c,d,e,t+11);
		W[12] += sigma1(W[10]) + W[ 5] + sigma0(W[13]);
        ROUND(e,f,g,h,a,b,c,d,t+12);
		W[13] += sigma1(W[11]) + W[ 6] + sigma0(W[14]);
        ROUND(d,e,f,g,h,a,b,c,t+13);
		W[14] += sigma1(W[12]) + W[ 7] + sigma0(W[15]);
        ROUND(c,d,e,f,g,h,a,b,t+14);
		W[15] += sigma1(W[13]) + W[ 8] + sigma0(W[ 0]);
        ROUND(b,c,d,e,f,g,h,a,t+15);
    }
    H[gid*8+0]+=a,H[gid*8+1]+=b,H[gid*8+2]+=c,H[gid*8+3]+=d,
	H[gid*8+4]+=e,H[gid*8+5]+=f,H[gid*8+6]+=g,H[gid*8+7]+=h;	
}