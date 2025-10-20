#include "kernel.hpp" // note: unbalanced round brackets () are not allowed and string literals can't be arbitrarily long, so periodically interrupt with )+R(
string opencl_c_container() { return R( // ########################## begin of OpenCL C code ####################################################################



int dp4a(const char4 a, const char4 b, const int c) { // 4-wide byte dot product and accumulate
)+"#if cl_nv_compute_capability>=61"+R( // use hardware-supported dp4a on Nvidia Pascal or newer GPUs with inline PTX assembly
	int d;)+"asm(\"dp4a.s32.s32\t%0,%1,%2,%3;\":\"=r\"(d):\"r\"(as_int(a)),\"r\"(as_int(b)),\"r\"(c));"+R(return d;
)+"#elif defined(__opencl_c_integer_dot_product_input_4x8bit)"+R( // use hardware-supported dp4a on some Intel GPUs
	return  c+dot(a,b); // c+dot(a, b);   dot_acc_sat(a, b, c); -- is slow
)+"#elif __has_builtin(__builtin_amdgcn_sdot4)"+R( // use hardware-supported dp4a on some AMD GPUs
    return __builtin_amdgcn_sdot4(as_int(a), as_int(b), c, false);
)+"#elif defined(cl_arm_integer_dot_product_accumulate_int8)"+R( // use hardware-supported dp4a on some ARM GPUs
    return arm_dot_acc(a, b, c);
)+"#else"+R( // fallback emulation (compilers will turn this into hardware-supported dp4a instruction if available)
	return c + a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;
)+"#endif"+R(
}



)+"#ifdef cl_khr_fp64"+R( // OpenCL C defines don't work in R() stringification macro
kernel void kernel_double(global float* data) {
	double x = (double)get_global_id(0);
	double y = (double)get_local_id(0);
	for(uint i=0u; i<128u; i++) {
#if defined(FP_FAST_FMA)
		x = fma(y, x, y); // 2 operations
		y = fma(x, y, x); // 2 operations
#else
		x = y*x+y; // 2 operations
		y = x*y+x; // 2 operations
#endif
	}
	data[get_global_id(0)] = (float)y;
}
)+"#endif"+R( // cl_khr_fp64

kernel void kernel_float(global float* data) {
	float x = (float)get_global_id(0);
	float y = (float)get_local_id(0);
	for(uint i=0u; i<512u; i++) {
#if defined(FP_FAST_FMAF)
		x = fma(y, x, y); // 2 operations
		y = fma(x, y, x); // 2 operations
#else
		x = y*x+y; // 2 operations
		y = x*y+x; // 2 operations
#endif
	}
	data[get_global_id(0)] = y;
}

)+"#ifdef cl_khr_fp16"+R( // OpenCL C defines don't work in R() stringification macro
kernel void kernel_half(global float* data) {
	half2 x = (half2)((float)get_global_id(0), (float)get_local_id(0));
	half2 y = (half2)((float)get_local_id(0), (float)get_global_id(0));
	float z = 0;//
	for(uint i=0u; i<512u; i++) {
#if defined(FP_FAST_FMA_HALF)
		x = fma(y,x, y);
		y = fma(x,y, x);
#else
		x = y*x+y; // 4 operations
		y = x*y+x; // 4 operations
#endif
	}
	data[get_global_id(0)] = (float)y.x+(float)y.y;
}
)+"#endif"+R( // cl_khr_fp16

kernel void kernel_long(global float* data) {
	long x = (long)get_global_id(0);
	long y = (long)get_local_id(0);
	for(uint i=0u; i<8u; i++) {
		x = y*x+y; // 2 operations
		y = x*y+x; // 2 operations
	}
	data[get_global_id(0)] = as_float((int)y);
}

kernel void kernel_int(global float* data) {
	int x = get_global_id(0);
	int y = get_local_id(0);
	for(uint i=0u; i<512u; i++) {
		x = y*x+y; // 2 operations
		y = x*y+x; // 2 operations
	}
	data[get_global_id(0)] = as_float(y);
}
kernel void kernel_mad(global float* data) {
	uint x = ~get_global_id(0);
	uint y = ~get_local_id(0);
	for(uint i=0u; i<512u; i++) {
		x = mad_hi(y,x,y); // 2 operations
		y = mad_hi(x,y,x); // 2 operations
	}
	data[get_global_id(0)] = as_float(y);
}


kernel void kernel_mod_uint(global float* data) {
	int ix = get_sub_group_local_id();
	int x = as_int(data[ix+get_sub_group_id()*get_max_sub_group_size()+128]);
	const ushort A = 0xFFA0;
	const int P = (A<<16)+1;
	for(uint i=0u; i<256u; i++) {
		short2 d = as_short2(data[ix]);
		x+= d.s0;
		x = as_short2(x).s1 - as_ushort2(x).s0*A; // 2 operations
		x+= d.s1;
		x = as_short2(x).s1 - as_ushort2(x).s0*A; // 2 operations
//		x = as_ushort2(x).s0*A + as_ushort2(x).s1; // 2 operations
//		x = as_ushort2(x).s0*A + as_ushort2(x).s1; // 2 operations
//		x = min(as_uint(x-P), x);
	}
	int s1 = sub_group_reduce_add(x>>16);
	uint s0 = sub_group_reduce_add(x & 0xFFFFu);
//	x = min(as_uint(x-P), x);
	if (ix==0) {
		s1 = s1+(s0>>16) - (s0&0xFFFFu)*A;
		data[get_sub_group_id()] = as_float(s1);
	}
}
kernel void kernel_shoup_uint(global float* data) {
	int ix = get_sub_group_local_id();
	uint x = as_uint(data[ix+get_sub_group_id()*get_max_sub_group_size()+128]);
	const ushort A = 0xFFA0;
	const uint P = (A<<16)+1;
	ulong r = 0;
	uint2 d = (uint)(data[ix]);
	for(uint i=0u; i<256u; i++) {
		uint q = mul_hi(x, d.s1);
		uint r0= x*d.s0;
		uint r1= q*P;
		r0 -=r1;
		r+= min(r0, r0-P);
	}
	uint s1 = sub_group_reduce_add(x>>16);
	uint s0 = sub_group_reduce_add(x & 0xFFFFu);
	if (ix==0) {
		s1 = s1+(s0>>16) - (s0&0xFFFFu)*A;
		data[get_sub_group_id()] = as_float(s1);
	}
}
kernel void kernel_tnn_uint(global float* data) {
	int ix = get_sub_group_local_id();
	ushort2 x = as_ushort2(data[ix+get_sub_group_id()*get_max_sub_group_size()]);
	ushort2 d = 0;
	float v = data[ix];
	for(uint i=0u; i<1024u; i++) {
		ushort2 y = as_ushort2(v);
		ushort2 r0 = x&y;
		ushort2 r1 = x&y.s10;
		x = r1;
		d += popcount(r0);
		d -= popcount(r1);
	}
//	data[get_global_id(0)] = d.x+d.y;
	uint s = sub_group_reduce_add(d.x+d.y);
	if (ix==0) 
		data[get_sub_group_id()] = as_float(s);
}
kernel void kernel_tnn_uint2(global uint2* data) {
	int ix = get_sub_group_local_id();
	uint2 x = as_uint2(data[ix+get_sub_group_id()*get_max_sub_group_size()]);
	uint2 d = 0;
	uint2 v = data[ix];
	for(uint i=0u; i<512u; i++) {
		uint2 y = as_uint2(v);
		uint2 r0 = x&y;
		uint2 r1 = x&y.s10;
		x = r1;
		d += popcount(r0);
		d -= popcount(r1);
	}
	data[get_global_id(0)] = d.x+d.y;
//	uint s = sub_group_reduce_add(d.x+d.y);
//	if (ix==0) 
//		data[get_sub_group_id()] = as_float(s);
}
kernel void kernel_short(global float* data) {
	short2 x = as_short2((uint)get_global_id(0));
	short2 y = as_short2((uint)get_local_id(0));
	for(uint i=0u; i<512u; i++) {
		x = y*x+y; // 4 operations
		y = x*y+x; // 4 operations
	}
	data[get_global_id(0)] = as_float(y);
}
kernel void kernel_mod_short(global float* data) {
	short2 x = as_short2((uint)get_global_id(0));
	short2 y = as_short2((uint)get_local_id(0));
	for(uint i=0u; i<128u; i++) {
		x = (y%x)+y; // 4 operations
		y = (x%y)+x; // 4 operations
	}
	data[get_global_id(0)] = as_float(y);
}

kernel void kernel_char(global float* data) {
	char4 x = as_char4((uint)get_global_id(0));
	char4 y = as_char4((uint)get_local_id(0));
	for(uint i=0u; i<512u; i++) {
		x = as_char4(dp4a(y, x, as_int(y))); // 8 operations
		y = as_char4(dp4a(x, y, as_int(x))); // 8 operations
	}
	data[get_global_id(0)] = as_float(y);
}

//cl_khr_subgroup_clustered_reduce
//	int a = dot((char4){ 0, 2,0, 0}, d);
//	float c = sub_group_clustered_reduce_add(a*u,4)*0.5f;
// Кубическая интерполяция по четырем точкам
static inline float cubic_u8_interp(uchar4 d, float dx){
	float4 u = {1.f,dx, dx*dx, dx*dx*dx};
	int4 a; 
	a[0] = dot((char4){ 0, 2,0, 0},d);
	a[1] = dot((char4){-1, 0,1, 0},d);
	a[2] = dot((char4){ 2,-5,4,-1},d);
	a[3] = dot((char4){-1, 3,-3,1},d);
	return (a[0]*u[0] + a[1]*u[1] + a[2]*u[2] + a[3]*u[3])*0.5f;
}
static inline uint Sum0(uint x) {
	return rotate(x,32u-2) ^ rotate(x,32u-13) ^ rotate(x,32u-22);
}
static inline uint Sum1(uint x) {
	return rotate(x,32u-6) ^ rotate(x,32u-11) ^ rotate(x,32u-25);
}
static inline uint Sbox(uint x) {
	return (x ^ (rotate(~x,1u) & rotate(x,2u)));
}
kernel void kernel_sigma(global float* data) {
	uint x = get_global_id(0);
	uint y = get_local_id(0);
	for(uint i=0u; i<512u; i++) {
#if 1
		x = Sum0(x)+y;
		y = Sum1(y)+x;
#else
		x = Sbox(x)+y;
		y = Sbox(y)+x;
#endif
	}
	data[get_global_id(0)] = as_float(y);
}
static inline float _matrix_mad_tf32_tf32_k16( int v, int8 b, float acc )
{
static inline float _matrix_mad_f16_f16_k32( int v, int8 b, float acc )
{
)+"#if defined(cl_intel_subgroup_matrix_multiply_accumulate)"+R(
#if 0//
	acc = intel_sub_group_f16_f16_matrix_mad_k16(as_short2(v).x,b,acc);
	acc = intel_sub_group_f16_f16_matrix_mad_k16(as_short2(v).y,b,acc);
	return acc;
#elif 0
	acc = intel_sub_group_bf16_bf16_matrix_mad_k16(as_short2(v).x,b,acc);
	acc = intel_sub_group_bf16_bf16_matrix_mad_k16(as_short2(v).y,b,acc);
	return acc;
#elif 0
//	acc = as_int(intel_sub_group_u2_u4_matrix_mad_k64(as_uchar4(v).x,as_uint8(b),as_int(acc)));
//	acc = as_int(intel_sub_group_u2_i8_matrix_mad_k32(as_uchar4(v).x,as_int8(b),as_int(acc)));
	acc = as_int(intel_sub_group_u8_u8_matrix_mad_k32(as_ushort2(v).x,as_uint8(b),as_int(acc)));
	acc = as_int(intel_sub_group_u8_u8_matrix_mad_k32(as_ushort2(v).y,as_uint8(b),as_int(acc)));
	return acc;
#elif 1
	return intel_sub_group_tf32_tf32_matrix_mad_k8(as_float(v),as_float8(b),acc);
#elif 1
	acc = as_int(intel_sub_group_u8_u8_matrix_mad_k32(as_ushort2(v).x,as_uint8(b),as_int(acc)));
	acc = as_int(intel_sub_group_u8_u8_matrix_mad_k32(as_ushort2(v).y,as_uint8(b),as_int(acc)));
	return acc;
#endif
)+"#else"+R(
    float result = acc;
    for (int k=0;k<8; ++k)
        result += dot(as_half2(sub_group_broadcast( v, k )), as_half2( b[k] ));
    return result;
)+"#endif"+R(
}
static inline float _matrix_mad_bf16_bf16_k16( int v, int8 b, float acc )
{
}
__attribute__((intel_reqd_sub_group_size(32)))
kernel void kernel_mma(global float8* data) {
	int8 x = get_global_id(0);
	int8 y = get_local_id(0);
	for(uint i=0u; i<8u; i++) { y[i]+=i; x[i] *=i; }
	for(uint r=0u; r<64u; r++)
	for(uint i=0u; i<8u; i++) {// 512 rounds x 2*k16
		x[i] = as_int(_matrix_mad_f16_f16_k32(x[i],y,as_float(x[i])));
		y[i] = as_int(_matrix_mad_f16_f16_k32(y[i],x,as_float(y[i])));
	}
	data[get_global_id(0)] = as_float(y[0])+as_float(y[1])+as_float(y[2])+as_float(y[3])+as_float(y[4])+as_float(y[5])+as_float(y[6])+as_float(y[7]);
}
inline float _sub_group_block_read(global float* yy){
    float y;
    int ix = get_sub_group_local_id();
    return yy[ix];
}
inline float _sub_group_block_write(global float* yy, float val){
    int ix = get_sub_group_local_id();
    return yy[ix] = val;
}
kernel void kernel_block_read(global float* data) {
	const uint n = sub_group_broadcast(get_global_id(0),0);
	const uint sz = get_max_sub_group_size();
	float x = 0.0f;
	for(uint i=0u; i<def_M; i++) x += _sub_group_block_read(data+i*sz+n);
	x = sub_group_reduce_add(x);
	if (get_sub_group_local_id()==0) 
		data[n] = x;
}
kernel void kernel_block_write(global float* data) {
	const uint n = sub_group_broadcast(get_global_id(0),0);
	const uint sz = get_max_sub_group_size();
	int ix = get_sub_group_local_id();
	for(uint i=0u; i<def_M; i++) _sub_group_block_write(data+i*sz+n, as_float(ix)); // block write
}

kernel void kernel_coalesced_write(global float* data) {
	const uint n = get_global_id(0);
	for(uint i=0u; i<def_M; i++) data[i*def_N+n] = as_float(n); // coalesced write
}
kernel void kernel_coalesced_read(global float* data) {
	const uint n = get_global_id(0);
	float x = 0.0f;
	for(uint i=0u; i<def_M; i++) x += data[i*def_N+n]; // coalesced read
	data[n] = x;
}
kernel void kernel_misaligned_write(global float* data) {
	const uint n = get_global_id(0);
	for(uint i=0u; i<def_M; i++) data[n*def_M+i] = as_float(n); // misaligned write
}
kernel void kernel_misaligned_read(global float* data) {
	const uint n = get_global_id(0);
	float x = 0.0f;
	for(uint i=0u; i<def_M; i++) x += data[n*def_M+i]; // misaligned read
	data[n] = x;
}



);} // ############################################################### end of OpenCL C code #####################################################################