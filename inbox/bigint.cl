// Multiprecision 
#if 0
__kernel void add256(__global ulong *a, __global ulong *b, __global ulong *c)
{
	ulong carry=0;
	for (int i=0; i<4; ++i)
		c[i] = __builtin_addcl (a[i], b[i], carry, &carry);
}

__kernel void mul256(__global ulong *a, __global ulong* b, __global ulong *c)
{
	ulong cy = 0;
	ulong b0 = b[0];
	c[0] = a[0]*b0;
	c[1] = __builtin_addcl (a[1]*b0, mul_hi(a[0],b0), cy, &cy);
	c[2] = __builtin_addcl (a[2]*b0, mul_hi(a[1],b0), cy, &cy);
	c[3] = __builtin_addcl (a[3]*b0, mul_hi(a[2],b0), cy, &cy);
	//return   = mul_hi(a[2],b0);
}
#endif

//ulong bn_mla_256(ulong* r, ulong *a, ulong d)
__kernel void bn_mla_256(__global ulong* r, __global ulong *a, __global ulong *b)
{
	uint gid = get_global_id(1);
    unsigned __int128 ac;
    ulong cy=0;
    ulong d=b[gid];
    int i;
    for (i=0;i<4; i++) {
        ac =  (unsigned __int128)d*a[gid*4+i] + r[gid*4+i] + cy;
        r[gid*4+i] = (ulong)ac;
        cy = ac>>64;
    }
//    return cy;
}
__kernel void bn_mla_256_(__global uint* r, __global uint *a, __global uint *b)
{
	uint gid = get_global_id(0);
    ulong ac;
    uint cy=0;
    uint d=b[gid];
    int i;
    for (i=0;i<8; i++) {
        ac =  (ulong)d*a[gid*8+i] + r[gid*8+i] + cy;
        r[gid*4+i] = (uint)ac;
        cy = ac>>32;
    }
//    return cy;
}
ulong bn_mla_128_256(ulong* r, ulong *a, ulong d)
{
    unsigned __int128 ac=0;
    int i;
    for (i=0;i<2; i++) {
        ac +=  (unsigned __int128)d*a[i] + r[i];
        r[i] = (ulong)ac;
        ac = ac>>64;
    }
    for (;i<4; i++) {
        ac += r[i];
        r[i] = (ulong)ac;
        ac = ac>>64;
    }
    return (ulong)ac;
}
//ulong bn_add_256 (ulong * r, ulong * a, ulong * b)
__kernel void bn_add_256 (__global ulong * r, __global ulong * a, __global ulong * b)
{
    ulong cy = 0;
    int i;
    for (i=0; i<4; i++) {
        unsigned __int128 t = (unsigned __int128)a[i] + b[i] + cy;
        r[i] = t;
        cy = t >> 64;
    }
    //return cy;
}


#define mrg31k3p_M1 2147483647             /* 2^31 - 1 */
#define mrg31k3p_M2 2147462579             /* 2^31 - 21069 */

#define mrg31k3p_MASK12 511                /* 2^9 - 1 */
#define mrg31k3p_MASK13 16777215           /* 2^24 - 1 */
#define mrg31k3p_MASK2 65535               /* 2^16 - 1 */
#define mrg31k3p_MULT2 21069

#define mrg31k3p_NORM_cl_double 4.656612873077392578125e-10  /* 1/2^31 */
#define mrg31k3p_NORM_cl_float  4.6566126e-10

__kernel void clrngMrg31k3pNextState(__global uint3 * state)
{
	uint gid = get_global_id(0);
	uint3 g1 = state[gid*2+0];
	uint3 g2 = state[gid*2+1];
	
	uint y1, y2;

	// first component
	y1 = ((g1.y & mrg31k3p_MASK12) << 22) + (g1.y >>  9)
	   + ((g1.z & mrg31k3p_MASK13) <<  7) + (g1.z >> 24);

	if (y1 >= mrg31k3p_M1) y1 -= mrg31k3p_M1;

	y1 += g1.z;
	if (y1 >= mrg31k3p_M1) y1 -= mrg31k3p_M1;
	
	g1 = (uint3)(y1,g1.xy);

	// second component
	y1 = ((g2.x & mrg31k3p_MASK2) << 15) + (mrg31k3p_MULT2 * (g2.x >> 16));
	if (y1 >= mrg31k3p_M2)	y1 -= mrg31k3p_M2;
	y2 = ((g2.z & mrg31k3p_MASK2) << 15) + (mrg31k3p_MULT2 * (g2.z >> 16));
	if (y2 >= mrg31k3p_M2)	y2 -= mrg31k3p_M2;
	y2 += g2.z;
	if (y2 >= mrg31k3p_M2)	y2 -= mrg31k3p_M2;
	y2 += y1;
	if (y2 >= mrg31k3p_M2)	y2 -= mrg31k3p_M2;
	g2 = (uint3)(y2,g2.xy);

	state[gid*2+0] = g1;
	state[gid*2+1] = g2;

/*
	if (g1[0] <= g2[0])
		return (g1[0] - g2[0] + mrg31k3p_M1);
	else
		return (g1[0] - g2[0]); */
}
uint xoshiro128p(uint *s) {
	const uint result = s[0] + s[3];
	const uint t = s[1] << 9;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;
	s[3] = rotate(s[3], 11u);

	return result;
}
uint xoshiro128pp(uint *s) 
{
	const uint result = rotate(s[0] + s[3], 7u) + s[0];
	const uint t = s[1] << 9;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;
	s[3] = rotate(s[3], 11u);

	return result;
}
uint xoshiro128ss(uint *s) {
	const uint result = rotate(s[1] * 5u, 7u) * 9u;
	const uint t = s[1] << 9;

	s[2] ^= s[0];
	s[3] ^= s[1];
	s[1] ^= s[2];
	s[0] ^= s[3];

	s[2] ^= t;
	s[3] = rotate(s[3], 11u);

	return result;
}
__kernel void k_mwc64x(__global ulong * state, __global float* data)
//uint mwc64x(ulong *state)
{
	uint gid = get_local_id(0);
	ulong x = state[gid];
	data += 256*gid;
	for (int i=0; i<256; i++){
		x =  (uint)x*4294883355UL + (x>>32);
		data[i] = ((uint)((x>>32) ^ (x&0xFFFFFFFFU))>>8)* 0x1.0p-24f;// однородное распрелеление [0,1)
	}
	state[gid] = x;
    // return ((x>>32) ^ (x&0xFFFFFFFFU));
}
#define MWC_A1 0xFFEBB71D94FCDAF9UL
__kernel void k_mwc128(__global ulong* state, __global uint* data) {
	uint gid = get_local_id(0);
	ulong x = state[2*gid+0];
	ulong c = state[2*gid+1];
	data += 256*gid;
	for (int i=0; i<256; i++){
		data[i] = (((x>>32) ^ (x&0xFFFFFFFFU))>>8)* 0x1.0p-24f;
		ulong t;
		t = (MWC_A1* x + c);
		c = mad_hi(MWC_A1, x, c);
		x = t;
	}
	state[2*gid+0] = x;
	state[2*gid+1] = c;
}


enum{ MWC64X_A = 4294883355U };
__kernel void mwc64x_step(__global uint2 *s)
{
	uint gid = get_local_id(0);
	uint x=s[gid].x, c=s[gid].y;
	
	uint xn=MWC64X_A*x+c;
	uint carry=(uint)(xn<c);				// The (Xn<C) will be zero or one for scalar
	uint cn=mad_hi(MWC64X_A,x,carry);  
	
	s[gid]=(uint2)(xn,cn);
}

#if 0
#define BN2(a,b)    ((ulong)(a) | ((ulong)(b)<<32))
static void mp_reduction_sec_p256k1(const MPCtx* const ctx, BNuint* x, BNuint cx)
{
    const BNuint p[256>>BN_BIT_LOG] BN_ALIGN = {BN2(977, 0x1)};// {lo, hi+1,};
    BNuint cy = bn_mla_128_256(x, p, cx);
    if (cy>0)/* cy +=*/ bn_add1(x, p, 8);
}
#endif

uint INVL(uint v)
{
    return ((ulong)(-v)<<32)/v;
}
