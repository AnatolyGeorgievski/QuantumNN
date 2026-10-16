// Little-endian byte access.
#define B2B_G(a, b, c, d, x, y) {   \
       a = a + b + x;         \
       d = rotate(d ^ a, 64-32); \
       c = c + d;             \
       b = rotate(b ^ c, 64-24); \
       a = a + b + y;         \
       d = rotate(d ^ a, 64-16); \
       c = c + d;             \
       b = rotate(b ^ c, 64-63); }
// Compression function. "last" flag indicates last block.

__kernel void blake2b_compress(__global ulong8 *H, __constant ulong2 *T, __constant ulong8 *M)
{
    const uchar16 sigma[12] =
    {
        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
        { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 },
        { 11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4 },
        { 7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8 },
        { 9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13 },
        { 2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9 },
        { 12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11 },
        { 13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10 },
        { 6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5 },
        { 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 },
        { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
        { 14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3 }
    };
	// Initialization Vector.
	
	const ulong4 blake2b_iv[2] = {
		{0x6A09E667F3BCC908, 0xBB67AE8584CAA73B, 0x3C6EF372FE94F82B, 0xA54FF53A5F1D36F1},
		{0x510E527FADE682D1, 0x9B05688C2B3E6C1F, 0x1F83D9ABFB41BD6B, 0x5BE0CD19137E2179}
	};

    int i;
    ulong4 v[4];
	size_t gid = get_global_id(0);//+get_global_id(1)*get_global_size(0);
	v[0] = H[gid].lo;
	v[1] = H[gid].hi;
	v[2] = blake2b_iv[0];
	v[3] = blake2b_iv[1];
//	v[0].x ^= 0x01010040;
	v[3].lo ^= T[gid];// offset LE
//	v[3].z = ~v[3].z;
#if 0
    if (last)                           // last block flag set ?
        v[14] = ~v[14];
#endif
	ulong8 m[2];
    m[0] = M[0+ 2*gid];
    m[1] = M[1+ 2*gid];
	ulong8 me,mo;
#pragma unroll
    for (i = 0; i < 12; i++)
    {
		me = shuffle2(m[0],m[1], convert_ulong8(sigma[i].even));
		mo = shuffle2(m[0],m[1], convert_ulong8(sigma[i].odd ));
		B2B_G( v[0], v[1], v[2], v[3], me.lo, mo.lo);
		v[1] = v[1].yzwx;
		v[2] = v[2].zwxy;
		v[3] = v[3].wxyz;
		B2B_G( v[0], v[1], v[2], v[3], me.hi, mo.hi);
		v[1] = v[1].wxyz;
		v[2] = v[2].zwxy;
		v[3] = v[3].yzwx;
	}
	H[gid] ^= (ulong8)(v[0] ^ v[2], v[1] ^ v[3]);
}