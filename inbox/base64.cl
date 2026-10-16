
__constant static const uchar base64_table[64] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";
__kernel void base64enc(__global uchar16* dst, __constant uint3* src)
{
	int idx = get_global_id(0);
	uint3 s = src[idx];
	uchar16 r;
	uint s0 = s.s0;
	r.s0123 = as_uchar4(((s0 & 0x3f)<<0) | ((s0 & 0xFC0)<<2) | ((s0 & 0x3F000)<<4) | ((s0 & 0xFC0000)<<6));
	s0 = (s.s0>>24) | (s.s1<< 8);
	r.s4567 = as_uchar4(((s0 & 0x3f)<<0) | ((s0 & 0xFC0)<<2) | ((s0 & 0x3F000)<<4) | ((s0 & 0xFC0000)<<6));
	s0 = (s.s1>>16) | (s.s2<<16);
	r.s89AB = as_uchar4(((s0 & 0x3f)<<0) | ((s0 & 0xFC0)<<2) | ((s0 & 0x3F000)<<4) | ((s0 & 0xFC0000)<<6));
	s0 = (s.s2>> 8);
	r.sCDEF = as_uchar4(((s0 & 0x3f)<<0) | ((s0 & 0xFC0)<<2) | ((s0 & 0x3F000)<<4) | ((s0 & 0xFC0000)<<6));
//	d = select(d, shuffle(tab[0], r), cond0);
//	d = select(d, shuffle(tab[1], r), cond1);
//	d = select(d, shuffle(tab[2], r), cond2);
//	d = select(d, shuffle(tab[3], r), cond3);
	dst[idx] = r;
}

__kernel void bin2hex(__global uchar16* dst, __constant uchar8* src)
{
	int idx = get_global_id(0);
	uchar8 s = src[idx];
	uchar16 r;
	r.even = s & (uchar)0xF;
	r.odd  = s >>4;
	char16 cond = r<(uchar16)(10u);
	r += select((uchar16)('A'-10), (uchar16)('0'), cond);
	dst[idx] = r;
}