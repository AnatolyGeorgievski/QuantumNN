/* FP16 FMA probe — только при cl_khr_fp16 */
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

kernel void half_fma(global float *data) {
    half2 x = (half2)((float)get_global_id(0), (float)get_local_id(0));
    half2 y = (half2)((float)get_local_id(0), (float)get_global_id(0));
    for (uint i = 0u; i < 512u; i++) {
#if defined(FP_FAST_FMA_HALF)
        x = fma(y, x, y);
        y = fma(x, y, x);
#else
        x = y * x + y;
        y = x * y + x;
#endif
    }
    data[get_global_id(0)] = (float)y.x + (float)y.y;
}
#endif /* cl_khr_fp16 */
