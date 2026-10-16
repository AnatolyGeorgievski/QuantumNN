__kernel void color_dim(__global uchar4 *r, __global uchar4 *c, __global float *x) 
{
	uint gid = get_global_id(0);
	r[gid] = convert_uchar4_sat_rte(convert_float4(c[gid])*x[gid]); // округление к ближайшему целому
/*	
	const int r = clamp((int)fma((float)((c>>16)&255), x, 0.5f), 0, 255);
	const int g = clamp((int)fma((float)((c>> 8)&255), x, 0.5f), 0, 255);
	const int b = clamp((int)fma((float)( c     &255), x, 0.5f), 0, 255);
	return (r&255)<<16|(g&255)<<8|(b&255); */
}
__kernel void color_dim_s(__global ushort4 *c, half x) 
{
	uint gid = get_global_id(0);
	c[gid] = convert_ushort4_sat_rte(convert_half4(c[gid])*x); // округление к ближайшему целому
}
__kernel void color_dim_h(__global half4 *c, half x) 
{
	uint gid = get_global_id(0);
	c[gid] = convert_half4(c[gid]*x); 
}

__kernel void color_dim_(__global uint *CC, float x) 
{
	uint gid = get_global_id(0);
	uint c = CC[gid];
	const int r = clamp((int)fma((float)((c>>16)&255), x, 0.5f), 0, 255);
	const int g = clamp((int)fma((float)((c>> 8)&255), x, 0.5f), 0, 255);
	const int b = clamp((int)fma((float)( c     &255), x, 0.5f), 0, 255);
	CC[gid] = (r&255)<<16|(g&255)<<8|(b&255);
}


__kernel
void color_mix(global uchar4 *r, constant uchar4 *c1, constant uchar4 *c2) { 
	const float w=0.358f;
	uint gid = get_global_id(0);
	r[gid] = convert_uchar4_sat_rte(mix(convert_float4(c2[gid]),convert_float4(c1[gid]), w));
}

uchar4 color_mix_3(uchar4 c0, uchar4 c1, uchar4 c2, float w0, float w1, float w2) { // w1*c1+w2*c2+w3*c3, w0+w1+w2 = 1
	return convert_uchar4_sat_rte(convert_float4(c0)*w0 + convert_float4(c1)*w1 + convert_float4(c2)*w2);
}