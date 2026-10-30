/* своё зерно · FP32 FMA-петля (из линии OpenCL-Benchmark, без ggml) */
kernel void float_fma(global float *data) {
	float x = (float)get_global_id(0);
	float y = (float)get_local_id(0);
	for (uint i = 0u; i < 512u; i++) {
#if defined(FP_FAST_FMAF)
		x = fma(y, x, y);
		y = fma(x, y, x);
#else
		x = y * x + y;
		y = x * y + x;
#endif
	}
	data[get_global_id(0)] = y;
}
