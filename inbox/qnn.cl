#ifndef N
#define N 64
#endif

// qnn унарные операции
__kernel tanh_f32(__global float *r, __global float * x){
	uint  gid = get_global_id(0);
	r+=N*gid; x+=N*gid;
	for(int i=0;i<N; i++)
		r[i] = tanh(x[i]);
}
__kernel relu_f32(__global float *r, __global float * x){
	uint  gid = get_global_id(0);
	r+=gid; x+=gid;
	for(int i=0;i<N; i++)
		r[i] = max(x[i], 0f);
}
__kernel sqrt_f32(__global float *r, __global float * x){
	uint  gid = get_global_id(0);
	r+=gid; x+=gid;
	for(int i=0;i<N; i++)
		r[i] = sqrt(x[i]);
}
__kernel sigmoid_f32(__global float *r, __global float * x){
	uint  gid = get_global_id(0);
	r+=gid; x+=gid;
	for(int i=0;i<N; i++)
		r[i] = native_recip(1f + native_exp(x[i]));
}
__kernel gelu_quick_f32(__global float *r, __global float * x){
	uint  gid = get_global_id(0);
	r+=gid; x+=gid;
	for(int i=0;i<N; i++)
		r[i] = x[i]*native_recip(1f + native_exp(-1.702f*x[i]));
}
__kernel silu_f32(__global float *r, __global float * x){
	uint  gid = get_global_id(0);
	r+=gid; x+=gid;
	for(int i=0;i<N; i++) 
		r[i] = x[i]*native_recip(1f + native_exp(-x[i]));
}
__kernel gelu_f32(__global float *r, __global float * x){
	uint  gid = get_global_id(0);
	r+=gid; x+=gid;
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
	for(int i=0;i<N; i++) {
		float xi = x[i];
		r[i] = 0.5f*xi*(1f + tanh(SQRT_2_OVER_PI*xi*(1f+GELU_COEF_A*xi*xi)));
	}
}
__kernel void exp_f32(__global float *r, __global float * x){
	r[i] = exp(x[i]);
}
__kernel void step_f32(__global float *r, __global float * x){
	r[i] = (x[i]>0f)? 1f: 0f;
}
__kernel void hardsigmoid_f32(__global float *r, __global float * x){
	r[i] = min(1.0f, max(0.0f, (x[i] + 3.0f) / 6.0f));
}
__kernel void hardswish_f32(__global float *r, __global float * x){
	r[i] = x[i]*min(1.0f, max(0.0f, (x[i] + 3.0f) / 6.0f));
}
// групповые операции
__kernel mean_f32(__global float* r, __global float *x){

}
__kernel void rms_norm_f32(__global float* r, __global float *x){
	uint group_size = get_local_size(0);
	float sum=0;
	for(int i=0;i<N; i++) 
		sum += x[i]*x[i];
	float mean = work_group_reduce_add(sum)/ group_size;
	float scale= rsqrt(mean);
	for(int i=0;i<N; i++) 
		r[i] = x[i]*scale;
}
__kernel void norm_f32(__global float* r, __global float *x){
	uint group_size = get_local_size(0);
	float sum=0;
	for(uint i=0;i<N; i+=group_size) 
		sum += x[i];
	float mean = work_group_reduce_add(sum)/(N*group_size);
	sum = 0;
	for(uint i=0;i<N; i+=group_size) 
		sum += (x[i]-mean)*(x[i]-mean);
	float dev   = work_group_reduce_add(sum)/(N*group_size);
	float scale = rsqrt(dev+eps);
	for(uint i=0;i<N; i+=group_size) 
		r[i] = (x[i]-mean)*scale;
}
// методы L1, минимальная кратность N*64
__kernel void sdot (global float *x, global float * y, global float* r, 
	ulong x_offset, ulong y_offset, ulong r_offset){
	x += x_offset;
	y += x_offset;
	uint id = get_local_id(0);
	uint sz = get_local_size(0);
	
	float sum=0;
	for(uint i=id; i<N; i+=sz) 
		sum += x[i]*y[i];
	sum = work_group_reduce_add(sum);
	if (id == 0)
		r[id] = sum;
}
// matrix-vector mul
__kernel void sgemv(float *a, float *v, float* b){
	uint x  = get_local_id(0);
	uint y  = get_local_id(1);
	uint dx = get_local_size(0);
	uint dy = get_local_size(1);
	a += (y*dx + x)*N; v += x;

	float sum=0.0f;
	for(int i=0;i<N; i++) 
		sum  = fma(a[i], v[i], sum);
	sum = work_group_reduce_add(sum);
	result[y] = sum + b[y];
}
// произведение матрицы на матрицу A B
__kernel void sgemm(__global float* c, __constant float* A, __constant float* B) {
	uint x = get_global_id(0);
	uint y = get_global_id(1);
	uint size = get_global_size(0);

	float sum = 0.0f;
	for (uint k = 0; k < N; k++) {
		sum = fma(A[y * size + k], B[k * size + x], sum);
	}
	c[y * size + x] = sum;
}
// произведение адамара матрицы на матрицу A B^T
__kernel void sgemm_t(__global float* c, __constant float* A, __constant float* B) {
	uint x = get_global_id(0);
	uint y = get_global_id(1);
	uint sz = get_global_size(0);

	float sum = 0.0f;
	for (uint k = 0; k < sz; k++)
		sum = fma(A[y * sz + k], B[y * sz + k], sum);
	c[y * sz + x] = sum;
}

// деквантизация
#define QK4_0 32
#define QK8_0 32
#define QK8_1 32
#define QK_K 256
typedef struct {
    half 	d;       // delta
    char2  qs[QK8_0/2]; //[QK8_0] -- quants
} block_q8_0;
typedef struct {
    half d;
    uchar qs[QK4_0/2];
} block_q4_0;
typedef struct {
    half d; // delta
	half s; // d * sum(qs[i])
    char4 qs[QK8_1/4]; // quants
} block_q8_1;
typedef struct {
    uchar ql[QK_K/2];      // quants, lower 4 bits
    uchar qh[QK_K/4];      // quants, upper 2 bits
    char  scales[QK_K/16]; // scales, quantized with 8 bits
    half d;                // super-block scale
} block_q6_K;

__kernel  void dequantize_bf16 (__global float2 *r, __global ushort *x){
	uint  gid = get_global_id(0);
	r+=gid; x+=gid;
	for(int i=0;i<32; i++)
		r[i] = as_float( (uint)x[i]<<16);
}
__kernel  void dequantize_f16 (__global float *r, __global half *x){
	uint  gid = get_global_id(0);
	r+=gid; x+=gid;
	for(int i=0;i<32; i++)
		r[i] = x[i];
}
__kernel  void dequantize_q8_0(__global float2 *r, __global block_q8_0 *x){
	uint  gid = get_global_id(0);
	r+=gid; x+=gid;
	float  d = x->d;
	for(int i=0;i<QK8_0/2; i++){
		r[i] = x->qs[i] * d;
	}
}
__kernel  void dequantize_q4_0(__global float2 *r, __global block_q8_0 *x){
	uint  gid = get_global_id(0);
	r+=gid; x+=gid;
	float  d = x->d;
	for(uint i=0;i<QK4_0/2; i++){
		r[i].lo = ((x->qs[i]&0xF)-8) * d;
		r[i].hi = (((x->qs[i]>>4)&0xF)-8) * d;
	}
	const float dm = -8*d;
	for (int l = 0; l < 4; ++l) {
        r[l+ 0] = d * (q[l] & 0xF) + dm;
        r[l+16] = d * (q[l] >>  4) + dm;
    }
}
__kernel  void dequantize_q4_1(__global float *y, __global block_q4_1 *x){
    const uint8_t * q = x->qs + 4*il;
    for (int l = 0; l < QK4_1/2; ++l) {
        y[l+ 0] = d.x * (q[l] & 0xF) + d.y;
        y[l+16] = d.x * (q[l] >>  4) + d.y;
    }
}
__kernel  void dequantize_q6_K(__global float *y, __global block_q6_K *x){

	global uint8_t * q1 = x->qs;
	block_idx = get_sub_group_id();
	x += block_idx;// номер блока
	ql 
	= get_sub_group_local_id();// номер блока
	= get_max_sub_group_size();
	qh = ;
	for (uint i=0; i<Nb; i++){// по числу блоков
		y[ 0] = d * sc[0] * ((char)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
		y[32] = d * sc[2] * ((char)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
		y[64] = d * sc[4] * ((char)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
		y[96] = d * sc[6] * ((char)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
	}
}

void quantize_q8_0(__global float * x, __global block_q8_0 * y) {
//    const int nb = k / QK8_0;

	float amax = 0.0f; // absolute max
	for (int i = 0; i < QK8_0; i++) 
		amax = max(amax, abs(x[i]));

	const float d = amax / 127;
	const float id = d ? 1.0f/d : 0.0f;

	y[i].d = d;

	for (int j = 0; j < QK8_0; ++j) {
		y->qs[j] = as_int_roundf(x[i]*id);
	}
}
float vec_dot_q6_K_q8_1_impl_mmvq(
     int vl, int vh, const int *  u, const int8_t *  scales,
     float & d, const float * d8
	) {

    float sumf = 0.0f;

    for (int i = 0; i < QR6_K; ++i) {
        const int sc  = scales[4*i];
        const int vil =  (vl >> (4*i)) & 0x0F0F0F0F;
        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;
		// VNNI
        const int vi  = __vsubss4((vil | vih), 0x20202020); // vi = (vil | vih) - 32

        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d*sumf;
}