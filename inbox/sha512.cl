
#define rotr(v,i) rotate(v, (ulong)32u-i)
#define Ch(x, y, z)  bitselect(z, y, x)
#define Maj(x, y, z) /*bitselect(z, y, z ^ x)*/((x&y) ^ (x&z) ^ (y&z))

#define sigma0(x) (rotr(x, 7) ^ rotr(x,18) ^ ((x)>> 3))
#define sigma1(x) (rotr(x,17) ^ rotr(x,19) ^ ((x)>>10))

#define Sum0(x) (rotr(x,2) ^ rotr(x,13) ^ rotr(x,22))
#define Sum1(x) (rotr(x,6) ^ rotr(x,11) ^ rotr(x,25))


#define ROUND(a,b,c,d,e,f,g,h,t) ({\
        ulong T = h + Sum1(e) + Ch(e,f,g) + K512[t] + W[(t)&15];  \
        d += T; \
        T += Sum0(a) + Maj(a,b,c); \
        h  = T; })
		
__kernel void SHA512(__global ulong * H, __constant ulong * M)
{
	const ulong K512[] = {
	0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
	0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
	0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
	0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
	0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
	0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
	0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
	0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
	0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
	0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
	0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
	0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
	0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
	0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
	0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
	0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
	0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
	0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
	0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
	0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL,
	};

    ulong W[16];
    int t; // Prepare the message schedule
	size_t gid = get_global_id(0)+get_global_id(1)*get_global_size(0);
    ulong a = H[gid*8+0],b = H[gid*8+1],c = H[gid*8+2],d = H[gid*8+3],
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
	//#pragma unroll
    for (t=16; t<80; t+=16)
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