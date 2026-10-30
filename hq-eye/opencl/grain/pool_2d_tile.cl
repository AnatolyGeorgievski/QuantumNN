/* POOL_2D avg · s==k · pad=0 · one subgroup per WG.
 * Host: local_size[0] == POOL_SG (16). Preferred WG multiple (often 64) — не использовать.
 */
#ifdef cl_khr_subgroups
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#elif defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#endif

#ifndef POOL_SG
#define POOL_SG 16
#endif

#ifdef cl_intel_subgroups
__attribute__((intel_reqd_sub_group_size(POOL_SG)))
#endif
kernel void pool_2D_avg(
    global float *src,
    global float *dst,
    uint ne00,   /* W */
    uint ne01,   /* H (layout; unused in index math when s==k pad=0) */
    uint ne0,    /* out W */
    uint ne1,    /* out H */
    uint kw,
    uint kh
){
    uint pos = get_group_id(0);
    uint row = get_group_id(1);
    uint  im = get_group_id(2);

    dst += im * ne0 * ne1;
    src += im * ne00 * ne01 + row * kh * ne00 + kw * pos;

    uint tid = get_sub_group_local_id();
    const uint stride = get_max_sub_group_size();
    float sum = 0;
    for (uint i = 0; i < kh; i++, src += ne00)
        for (uint j = tid; j < kw; j += stride)
            sum += src[j];
    sum = sub_group_reduce_add(sum);
    if (tid == 0)
        dst[mad24(row, ne0, pos)] = sum / (float)(kh * kw);
}
