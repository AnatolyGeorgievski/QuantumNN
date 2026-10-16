#ifndef ROTR32
#define ROTR32(x, y)  (((x) >> (y)) ^ ((x) << (32 - (y))))
#endif
// Compression function. "last" flag indicates last block.
#define B2S_G(a, b, c, d, x, y) {  \
   a = a + b + x;         \
   d = rotate(d ^ a, 32-16); \
   c = c + d;             \
   b = rotate(b ^ c, 32-12); \
   a = a + b + y;         \
   d = rotate(d ^ a, 32-8);  \
   c = c + d;             \
   b = rotate(b ^ c, 32-7); }
__kernel void blake2s_compress(__global uint8 *H, __global uint2 *T, __global uint8 *M)
{
	const int last = 0;
    const uchar16 sigma[10] =
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
        { 10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0 }
    };
	const uint4 blake2s_iv[2] = 
	{
		{0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A},
		{0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19}
	};
    uint4 v[4];

	size_t gid = get_global_id(0);//+get_global_id(1)*get_global_size(0);
	v[0] = H[gid].lo;
	v[1] = H[gid].hi;
	v[2] = blake2s_iv[0];
	v[3] = blake2s_iv[1];
	v[3].lo ^= T[gid];// offset LE
#if 0
    if (last)                           // last block flag set ?
        v[14] = ~v[14];// ~blake2s_iv[1].lo
#endif
    // get little-endian words
	uint8 m[2];
    m[0] = M[0+ 2*gid];
    m[1] = M[1+ 2*gid];
	uint8 me,mo;
//#pragma unroll
    for (int i = 0; i < 10; i++)
    {
		me = shuffle2(m[0],m[1], convert_uint8(sigma[i].even));
		mo = shuffle2(m[0],m[1], convert_uint8(sigma[i].odd ));
		B2S_G( v[0], v[1], v[2], v[3], me.lo, mo.lo);
		v[1] = v[1].yzwx;
		v[2] = v[2].zwxy;
		v[3] = v[3].wxyz;
		B2S_G( v[0], v[1], v[2], v[3], me.hi, mo.hi);
		v[1] = v[1].wxyz;
		v[2] = v[2].zwxy;
		v[3] = v[3].yzwx;
    }
	H[gid] ^= (uint8)(v[0] ^ v[2], v[1] ^ v[3]);
}