const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
	CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
__kernel void filter2d(__read_only image2d_t src, __write_only image2d_t dst)
{
	int2 c = (int2)(get_global_id(0), get_global_id(1));
	uint4 p1;
	uint4 p = 0;
	int2 t;
	int2 k;
#pragma unroll
	for (k.y = -1; k.y <= 1; k.y++){
		for(k.x = -1; k.y <= 1; k.x++){
			p1  = read_imageui(src, sampler, c+k);
			p += p1;
		}
	}
	write_imageui(dst, c, p);
}