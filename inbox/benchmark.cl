/* тесты заимствованы из 
	https://github.com/ProjectPhysX/OpenCL-Benchmark/blob/master/src/kernel.cpp

Запуск тестов выполняется утилитой ./ocl
Методика тестирования - запустить много раз, выбрать минимальные по времени исполнения результаты. 
Пока GPU включается, поднимает напряжение, 
Метод тестирования запустить множество операций типа mac - multiply accumulate с зацеплением,
чтобы компилятор не оптимизировал эту операцию на этапе компиляции. Кернел должен выполнять не менее 64 циклов
без обращения к общей памяти. В слуае векторных операций число циклов снижается пропорционально числу компонент в векторе.
Расчет дается в TFLOPs (операциях с плавающей точкой, включая операции умножения  и сложения, 
операция mac считается за две операции) или в TIOPs - операциях целочисленных. 
Хост запускает кернел с размерностью 3 и размером: 1024*1024*1, 521*512*4, 256*256*16 или 128*128*64. 
Размер выбирается исходя из возможностей GPU. 
Во всех случаях запускается 1М копий и 1024 операций в цикле кернела. Всего для простоты расчета получается, 
что каждый тест выполняет 1G операций. 



раскручивает вентиляторы охлаждения, поднимает частоту ядра, может пройти некоторое время.
Если на GPU помимо теста обрабатывается графика, результаты будут занижены. 

Пример запуска тестов 
$ ./ocl -i benchmark.cl -k kernel_float4 -d 1 -v

 -ffp-contract=off

#pragma OPENCL FP_CONTRACT OFF // prevents implicit FMA optimizations
#define fma(a, b, c) ((a) * (b) + (c))  // shadows OpenCL explicit function fma()
*/
#if 1
kernel void kernel_float_(global float* data) {
	float x = (float)get_global_id(0);
	float y = (float)get_local_id(0);
	for(uint i=0u; i<512u; i++) {
		x = fma(y, x, y);
		y = fma(x, y, x);
	}
	data[get_global_id(0)] = y;
}
#if 0
kernel void kernel_half_(global float* data) {
	half2 x = (half2)((float)get_global_id(0), (float)get_local_id(0));
	half2 y = (half2)((float)get_local_id(0), (float)get_global_id(0));
	for(uint i=0u; i<512u; i++) {
		x = fma(y, x, y);
		y = fma(x, y, x);
	}
	data[get_global_id(0)] = as_float(y);
}
#endif
kernel void kernel_long(global float* data) {
	long x = (long)get_global_id(0);
	long y = (long)get_local_id(0);
	for(uint i=0u; i<8u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[get_global_id(0)] = as_float((int)y);
}

kernel void kernel_int_(global float* data) {
	int x = get_global_id(0);
	int y = get_local_id(0);
	for(uint i=0u; i<512u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[get_global_id(0)] = as_float(y);
}

kernel void kernel_short_(global float* data) {
	short2 x = as_short2((int)get_global_id(0));
	short2 y = as_short2((int)get_local_id(0));
	for(uint i=0u; i<128u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[get_global_id(0)] = as_float(y);
}

kernel void kernel_char_(global float* data) {
	char4 x = as_char4((int)get_global_id(0));
	char4 y = as_char4((int)get_local_id(0));
	for(uint i=0u; i<64u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[get_global_id(0)] = as_float(y);
}
#endif
kernel void kernel_float4(global float4* data, global float4 *X, global float4 *Y) {
	uint gid = get_global_id(0);
	float4 x = X[gid];
	float4 y = Y[gid];
	for(uint i=0u; i<128u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[gid] = y;
}
kernel void kernel_float3(global float3* data, global float3 *X, global float3 *Y) {
	uint gid = get_global_id(0);
	float3 x = X[gid];
	float3 y = Y[gid];
	for(uint i=0u; i<512u/3; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[gid] = y;
}
kernel void kernel_float2(global float2* data, global float2 *X, global float2 *Y) {
	uint gid = get_global_id(0);
	float2 x = X[gid];
	float2 y = Y[gid];
	for(uint i=0u; i<256u; i++) {
		x = fma(y,x,y);
		y = fma(x,y,x);
	}
	data[gid] = y;
}
kernel void kernel_float(global float* data, global float *X, global float *Y) {
	uint gid = get_global_id(0);
	float x = X[gid];
	float y = Y[gid];
	for(uint i=0u; i<512u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[gid] = y;
}
// для выполнения этих операций должны поддерживаться операции FP16 
#if 1//defined(cl_khr_fp16)
kernel void kernel_half(global half* data, global half *X, global half *Y) {
	uint gid = get_global_id(0);
	half x = X[gid];
	half y = Y[gid];
	for(uint i=0u; i<512u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[gid] = y;
}
kernel void kernel_half2(global half2* data, global half2 *X, global half2 *Y) {
	uint gid = get_global_id(0);
	half2 x = X[gid];
	half2 y = Y[gid];
	for(uint i=0u; i<256u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[gid] = y;
}
#if 0
kernel void kernel_mad_half4(global half4* data, global half4 *X, global half4 *Y) {
	uint gid = get_global_id(0);
	half4 x = X[gid];
	half4 y = Y[gid];
	for(uint i=0u; i<128u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[gid] = y;
}

kernel void kernel_mad_half8(global half8* data, global half8 *X, global half8 *Y) {
	uint gid = get_global_id(0);
	half8 x = X[gid];
	half8 y = Y[gid];
	for(uint i=0u; i<64u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[gid] = y;
}
#endif
kernel void kernel_half4(global half4* data, global half4 *X, global half4 *Y) {
	uint gid = get_global_id(0);
	half4 x = X[gid];
	half4 y = Y[gid];
	for(uint i=0u; i<128u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[gid] = y;
}
kernel void kernel_half8(global half8* data, global half8 *X, global half8 *Y) {
	uint gid = get_global_id(0);
	half8 x = X[gid];
	half8 y = Y[gid];
	for(uint i=0u; i<64u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[gid] = y;
}
#endif
kernel void kernel_short(global short* data, global short *X, global short *Y) {
	uint gid = get_global_id(0);
	short x = X[gid];
	short y = Y[gid];
	for(uint i=0u; i<512u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[gid] = y;
}
kernel void kernel_short2(global short2* data, global short2 *X, global short2 *Y) {
	uint gid = get_global_id(0);
	short2 x = as_short2((uint)get_global_id(0));//X[gid];
	short2 y = as_short2((uint)get_local_id(0));//Y[gid];
	for(uint i=0u; i<256u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[gid] = y;
}
kernel void kernel_short4(global short4* data, global short4 *X, global short4 *Y) {
	uint gid = get_global_id(0);
	short4 x = get_global_id(0);//X[gid];
	short4 y = get_local_id(0);//Y[gid];
	for(uint i=0u; i<128u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[gid] = y;
}
kernel void kernel_short8(global short8* data, global short8 *X, global short8 *Y) {
	uint gid = get_global_id(0);
	short8 x = (short)get_global_id(0);//X[gid];
	short8 y = (short)get_global_id(1);//Y[gid];
	for(uint i=0u; i<64u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[gid] = y;
}
kernel void kernel_uint(global uint* data, global uint *X, global uint *Y) {
	uint gid = get_global_id(0);
	uint x = X[gid];
	uint y = Y[gid];
	for(uint i=0u; i<512u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[gid] = y;
}
kernel void kernel_uint2(global uint2* data, global uint2 *X, global uint2 *Y) {
	uint gid = get_global_id(0);
	uint2 x = X[gid];
	uint2 y = Y[gid];
	for(uint i=0u; i<256u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[gid] = y;
}
kernel void kernel_uint4(global uint4* data, global uint4 *X, global uint4 *Y) {
	uint gid = get_global_id(0);
	uint4 x =get_global_id(0);// X[gid];
	uint4 y =get_global_id(1);// Y[gid];
	for(uint i=0u; i<128u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[gid] = y;
}

kernel void kernel_int(global int* data, global int *X, global int *Y) {
	uint gid = get_global_id(0);
	int x = X[gid];
	int y = Y[gid];
	for(uint i=0u; i<512u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[gid] = y;
}
kernel void kernel_int2(global int2* data, global int2 *X, global int2 *Y) {
	uint gid = get_global_id(0);
	int2 x = X[gid];
	int2 y = Y[gid];
	for(uint i=0u; i<256u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[gid] = y;
}

kernel void kernel_int4(global int4* data, global int4 *X, global int4 *Y) {
	uint gid = get_global_id(0);
	int4 x = X[gid];
	int4 y = Y[gid];
	for(uint i=0u; i<128u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[gid] = y;
}
kernel void kernel_char(global char* data, __constant char *X, __constant char *Y) {
	uint gid = get_global_id(0);
	char x = X[gid];
	char y = Y[gid];
	for(uint i=0u; i<1024u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[gid] = y;
}
kernel void kernel_char2(global char2* data, global char2 *X, global char2 *Y) {
	uint gid = get_global_id(0);
	char2 x = X[gid];
	char2 y = Y[gid];
	for(uint i=0u; i<512u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[gid] = y;
}
kernel void kernel_char4(global char4* data, global char4 *X, global char4 *Y) {
	uint gid = get_global_id(0);
	char4 x = X[gid];
	char4 y = Y[gid];
	for(uint i=0u; i<256u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[gid] = y;
}
kernel void kernel_char8(global char8* data, global char8 *X, global char8 *Y) {
	uint gid = get_global_id(0);
	char8 x = X[gid];
	char8 y = Y[gid];
	for(uint i=0u; i<128u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[gid] = y;
}
kernel void kernel_char16(global char16* data, global char16 *X, global char16 *Y) {
	uint gid = get_global_id(0);
	char16 x = X[gid];
	char16 y = Y[gid];
	for(uint i=0u; i<64u; i++) {
		x = (y*x)+y;
		y = (x*y)+x;
	}
	data[gid] = y;
}
static inline int _dp4a(char4 a, char4 b, int acc){

#if __has_builtin(__builtin_amdgcn_sdot4)
	return __builtin_amdgcn_sdot4(as_int(a), as_int(b), acc, false);
#elif defined(cl_arm_integer_dot_product_accumulate_int8)
	return arm_dot_acc(a, b, acc);
#elif defined(__opencl_c_integer_dot_product_input_4x8bit)
	return  acc+dot(a,b);
//	return dot_acc_sat(a,b,acc);
#else 
	return acc+a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;
#endif
}
kernel void kernel_dp4a(global int* data, global int *X, global int *Y) {
	uint gid = get_global_id(0);
	int x = X[gid];
	int y = Y[gid];
	for(uint i=0u; i<512u; i++) {
		x = _dp4a(as_char4(y),as_char4(x),y);
		y = _dp4a(as_char4(x),as_char4(y),x);
	}
	data[gid] = y;
}
kernel void kernel_mod_uint(global float* data) {
	int ix = get_sub_group_local_id();
	int x = as_int(data[ix+get_sub_group_id()*get_max_sub_group_size()+128]);
	const ushort A = 0xFFA0;
	const int P = ((int)A<<16)+1;
	for(uint i=0u; i<512u; i++) {
		short2 d = as_short2(data[ix]);
		x+= d.s0;
		x = as_short2(x).s1 - as_ushort2(x).s0*A; // 2 operations
		x+= d.s1;
		x = as_short2(x).s1 - as_ushort2(x).s0*A; // 2 operations
	}
	int s1 = sub_group_reduce_add(x>>16);
	uint s0 = sub_group_reduce_add(x & 0xFFFFu);
//	x = min(as_uint(x-P), x);
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
//	data[get_global_id(0)] = d.x+d.y;
	uint s = sub_group_reduce_add(d.x+d.y);
	if (ix==0) 
		data[get_sub_group_id()] = as_float(s);
}

static half _mma_f16_f16_k16_sg16( half v, half16 b, float acc ){
    float result = acc;
    for (int i= 0; i<16; i++) 
		result += sub_group_broadcast(v,i) * b[i];
	return result;
}
__attribute__((intel_reqd_sub_group_size(16)))
kernel void kernel_mma_f16(global float* data, global float *X, global half16 *Y) {
	uint gid = get_global_id(0);
	half   x = X[gid];
	half16 y = Y[gid];
	float acc =data[gid];
	for(uint i=0u; i<1024u; i++) {
		acc = intel_sub_group_f16_f16_matrix_mad_k16(as_short(x), as_int8(y), acc);
		//acc = _mma_f16_f16_k16_sg16(x, y, acc);
	}
	data[gid] = acc;
}
kernel void kernel_mma_bf16(global float* data, global short *X, global int8 *Y) {
	uint gid = get_global_id(0);
	short x = X[gid];
	int8  y = Y[gid];
	float acc =data[gid];
	for(uint i=0u; i<1024u; i++)
		acc = intel_sub_group_bf16_bf16_matrix_mad_k16(as_short(x), as_int8(y), acc);
	data[gid] = acc;
}
kernel void kernel_mma4_bf16(global float4* data, global short4 *X, global int8 *Y) {
	uint gid = get_global_id(0);
	short4 x = X[gid];
	int8  y = Y[gid];
	float4 acc =data[gid];
	for(uint i=0u; i<1024u; i++)
		acc = intel_sub_group_bf16_bf16_matrix_mad_k16(as_short4(x), as_int8(y), acc);
	data[gid] = acc;
}
#if defined(cl_intel_subgroup_matrix_multiply_accumulate_tf32)
kernel void kernel_mma_tf32(global float* data, global float *X, global float8 *Y) {
	uint gid = get_global_id(0);
	float  x = X[gid];
	float8 y = Y[gid];
	float acc =data[gid];
	for(uint i=0u; i<1024u; i++)
		acc = intel_sub_group_tf32_tf32_matrix_mad_k8(as_float(x), as_float8(y), acc);
	data[gid] = acc;
}
kernel void kernel_mma4_tf32(global float4* data, global float2 *X, global float8 *Y) {
	uint gid = get_global_id(0);
	float2 x = X[gid];
	float8 y = Y[gid];
	float4 acc =data[gid];
	for(uint i=0u; i<1024u; i++)
		acc = intel_sub_group_tf32_tf32_matrix_mad_k8(as_float2(x), as_float8(y), acc);
	data[gid] = acc;
}
#endif
kernel void kernel_mma_i8(global int* data, global short *X, global int8 *Y) {
	uint gid = get_global_id(0);
	short x = X[gid];
	int8  y = Y[gid];
	int acc =data[gid];
	for(uint i=0u; i<1024u; i++)
		acc = intel_sub_group_i8_i8_matrix_mad_k32(as_short(x), as_int8(y), acc);
	data[gid] = acc;
}
kernel void kernel_mma_i8_sg8(global int* data, global int *X, global int8 *Y) {
	uint gid = get_global_id(0);
	int x = X[gid];
	int8  y = Y[gid];
	int acc =data[gid];
	for(uint i=0u; i<1024u; i++)
		acc = intel_sub_group_i8_i8_matrix_mad_k32(as_int(x), as_int8(y), acc);
	data[gid] = acc;
}
kernel void kernel_mma4_i8_sg8(global int4* data, global int4 *X, global int8 *Y) {
	uint gid = get_global_id(0);
	int4 x = X[gid];
	int8  y = Y[gid];
	int4 acc =data[gid];
	for(uint i=0u; i<1024u; i++)
		acc = intel_sub_group_i8_i8_matrix_mad_k32(as_int4(x), as_int8(y), acc);
	data[gid] = acc;
}
kernel void kernel_mma8_i8_sg8(global int8* data, global int8 *X, global int8 *Y) {
	uint gid = get_global_id(0);
	int8 x = X[gid];
	int8 y = Y[gid];
	int8 acc =data[gid];
	for(uint i=0u; i<1024u; i++)
		acc = intel_sub_group_i8_i8_matrix_mad_k32(as_int8(x), as_int8(y), acc);
	data[gid] = acc;
}
kernel void kernel_mma_i4(global int* data, global short *X, global int8 *Y) {
	uint gid = get_global_id(0);
	short x = X[gid];
	int8  y = Y[gid];
	int acc =data[gid];
	for(uint i=0u; i<1024u; i++)
		acc = intel_sub_group_i4_i4_matrix_mad_k64(as_short(x), as_int8(y), acc);
	data[gid] = acc;
}
kernel void kernel_mma_i2(global int* data, global char *X, global int4 *Y) {
	uint gid = get_global_id(0);
	char  x = X[gid];
	int4  y = Y[gid];
	int acc =data[gid];
	for(uint i=0u; i<1024u; i++)
		acc = intel_sub_group_i2_i2_matrix_mad_k64(as_char(x), as_int4(y), acc);
	data[gid] = acc;
}
kernel void kernel_mma8_i2(global int8* data, global char8 *X, global int4 *Y) {
	uint gid = get_global_id(0);
	char8  x = X[gid];
	int4  y = Y[gid];
	int8 acc =data[gid];
	for(uint i=0u; i<1024u; i++)
		acc = intel_sub_group_i2_i2_matrix_mad_k64(as_char8(x), as_int4(y), acc);
	data[gid] = acc;
}
kernel void kernel_mma4_i2(global int4* data, global char4 *X, global int4 *Y) {
	uint gid = get_global_id(0);
	char4  x = X[gid];
	int4  y = Y[gid];
	int4 acc =data[gid];
	for(uint i=0u; i<1024u; i++)
		acc = intel_sub_group_i2_i2_matrix_mad_k64(as_char4(x), as_int4(y), acc);
	data[gid] = acc;
}
kernel void kernel_mma4_i4(global int4* data, global short4 *X, global int8 *Y) {
	uint gid = get_global_id(0);
	short4 x = X[gid];
	int8  y = Y[gid];
	int4 acc =data[gid];
	for(uint i=0u; i<1024u; i++)
		acc = intel_sub_group_i4_i4_matrix_mad_k64(as_short4(x), as_int8(y), acc);
	data[gid] = acc;
}

__attribute__((intel_reqd_sub_group_size(16)))
kernel void kernel_mma2(global float2* data, global half2 *X, global half16 *Y) {
	uint gid = get_global_id(0);
	half2  x = X[gid];
	half16 y = Y[gid];
	float2 acc =data[gid];
	for(uint i=0u; i<1024u; i++) {
		acc = intel_sub_group_bf16_bf16_matrix_mad_k16(as_short2(x), as_int8(y), acc);
		//acc = intel_sub_group_f16_f16_matrix_mad_k16(as_short2(x), as_int8(y), acc);
		//acc = _mma_f16_f16_k16_sg16(x, y, acc);
	}
	data[gid] = acc;
}
kernel void kernel_mma4(global short4* data, global short4 *X, global half16 *Y) {
	uint gid = get_global_id(0);
	short4 x = X[gid];
	half16 y = Y[gid];
	short4 acc =data[gid];
	for(uint i=0u; i<1024u; i++)
		acc = intel_sub_group_bf16_bf16_matrix_mad_k16(as_short4(x), as_int8(y), acc);
	data[gid] = acc;
}

kernel void kernel_mma8(global float8* data, global short8 *X, global half16 *Y) {
	uint gid = get_global_id(0);
	short8  x = X[gid];
	half16 y = Y[gid];
	float8 acc =data[gid];
	for(uint i=0u; i<1024u; i++)
		acc = intel_sub_group_bf16_bf16_matrix_mad_k16(as_short8(x), as_int8(y), acc);
	data[gid] = acc;
}
kernel void kernel_dot(global float* data, global float *X, global float *Y) {
	uint gid = get_global_id(0);
	float x = X[gid];
	float y = Y[gid];
	for(uint i=0u; i<512u; i++) {
		x = dot(as_half2(y),as_half2(x))+y;
		y = dot(as_half2(x),as_half2(y))+x;
	}
	data[gid] = y;
}

kernel void kernel_dot_half4(global half4* data, global half4 *X, global half4 *Y) {
	uint gid = get_global_id(0);
	half4 x = X[gid];
	half4 y = Y[gid];
	for(uint i=0u; i<256u; i++) {
		x = dot(y,x)+y;
		y = dot(x,y)+x;
	}
	data[gid] = y;
}

kernel void kernel_mu(global float4* data, global float4 *X, global float4 *Mu) {
	uint gid = get_global_id(0);
	float4 mu = Mu[gid];
	float4 E = (1.f);
	float4 x = X[gid];
	float4 y = (0.5f);// Y[gid];
	for(uint i=0u; i<256u; i++) {
		x = dot((E-mu),x)+dot(mu,y);
		y = dot((E-mu),y)+dot(mu,x);
	}
	data[gid] = y;
}
kernel void kernel_lerp(global float4* data, global float4 *X, global float4 *Mu) {
	uint gid = get_global_id(0);
	float4 mu = Mu[gid];
	float4 x = X[gid];
	float4 y = (0.5f);// Y[gid];
	for(uint i=0u; i<256u; i++) {
		x = mix(x,y,mu);
		y = mix(y,x,mu);
	}
	data[gid] = y;
}
kernel void kernel_lerp_half(global half4* data, global half4 *X, global half4 *Mu) {
	uint gid = get_global_id(0);
	half4 mu = Mu[gid];
	half4 x = X[gid];
	half4 y = (0.5f);// Y[gid];
	for(uint i=0u; i<256u; i++) {
		x = mix(x,y,mu);
		y = mix(y,x,mu);
	}
	data[gid] = y;
}


kernel void kernel_mu_half(global half4* data, global half4 *X, global half4 *Mu) {
	uint gid = get_global_id(0);
	half4 mu = Mu[gid];
	half4 E = (1.f);
	half4 x = X[gid];
	half4 y = (0.5f);// Y[gid];
	for(uint i=0u; i<256u; i++) {
		x = (E-mu)*x+mu*y;
		y = (E-mu)*y+mu*x;
	}
	data[gid] = y;
}


#define def_M 1024
#define def_N (def_M*def_M)

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
	for(uint i=0u; i<def_M; i++) 
		x += _sub_group_block_read(data+i*sz+n);
	x = sub_group_reduce_add(x);
	if (get_sub_group_local_id()==0) 
		data[n] = x;
}

kernel void kernel_coalesced_write(global float* data) {
	const uint n = get_global_id(0);
	for(uint i=0u; i<def_M; i++) data[i*def_N+n] = 0.0f; // coalesced write
}
kernel void kernel_coalesced_read(global float* data) {
	const uint n = get_global_id(0);
	float x = 0.0f;
	for(uint i=0u; i<def_M; i++) x += data[i*def_N+n]; // coalesced read
	data[n] = x;
}
kernel void kernel_misaligned_write(global float* data) {
	const uint n = get_global_id(0);
	for(uint i=0u; i<def_M; i++) data[n*def_M+i] = 0.0f; // misaligned write
}
kernel void kernel_misaligned_read(global float* data) {
	const uint n = get_global_id(0);
	float x = 0.0f;
	for(uint i=0u; i<def_M; i++) x += data[n*def_M+i]; // misaligned read
	data[n] = x;
}



#define rotr(v,i) (((v)<<(32-i)) | ((v)>>(i)))
#define Sum0(x) (rotate(x,32u-2) ^ rotate(x,32u-13) ^ rotate(x,32u-22))
#define Sum1(x) (rotate(x,32u-6) ^ rotate(x,32u-11) ^ rotate(x,32u-25))

kernel void kernel_sigma(global uint* data, constant uint* X, constant uint* Y) {
	uint gid = get_global_id(0);//+get_global_id(1)*get_global_size(0);
	uint x =get_global_id(0);// X[gid];
	uint y =get_global_id(1);// Y[gid];
	for(uint i=0u; i<512u; i++) {
		x = Sum0(x)+y;
		y = Sum1(y)+x;
	}
	data[gid] = y+x;
}
kernel void kernel_sigma4(global uint4* data, constant uint4* X, constant uint4* Y) {
	uint gid = get_global_id(0);//+get_global_id(1)*get_global_size(0);
	uint4 x = X[gid];
	uint4 y = Y[gid];
	for(uint i=0u; i<128u; i++) {
		x = Sum0(x)+y;
		y = Sum1(y)+x;
	}
	data[gid] = y+x;
}
kernel void kernel_sigma8(global uint8* data, constant uint8* X, constant uint8* Y) {
	uint gid = get_global_id(0);//+get_global_id(1)*get_global_size(0);
	uint8 x = X[gid];
	uint8 y = Y[gid];
	for(uint i=0u; i<64u; i++) {
		x = Sum0(x)+y;
		y = Sum1(y)+x;
	}
	data[gid] = y+x;
}
kernel void kernel_sigma16(global uint16* data, constant uint16* X, constant uint16* Y) {
	uint gid = get_global_id(0);//+get_global_id(1)*get_global_size(0);
	uint16 x = X[gid];
	uint16 y = Y[gid];
	
//	__attribute__((opencl_unroll_hint(32)))
	for(uint i=0u; i<32u; i++) {
		x = Sum0(x)+y;
		y = Sum1(y)+x;
	}
	data[gid] = y+x;
}

// Bit Field Extract
#define BEXTR(x, n, len) (((x) >> (n)) & ((1 << (len))-1))

kernel void kernel_magma(global uint* data)
{
	const uchar16 sbox[8] = {
/* π0' = */ {12, 4, 6, 2, 10, 5, 11, 9, 14, 8, 13, 7, 0, 3, 15, 1},
/* π1' = */ {6, 8, 2, 3, 9, 10, 5, 12, 1, 14, 4, 7, 11, 13, 0, 15},
/* π2' = */ {11, 3, 5, 8, 2, 15, 10, 13, 14, 1, 7, 4, 12, 9, 6, 0},
/* π3' = */ {12, 8, 2, 1, 13, 4, 15, 6, 7, 0, 10, 5, 3, 14, 9, 11},
/* π4' = */ {7, 15, 5, 10, 8, 1, 6, 13, 0, 9, 3, 14, 11, 4, 2, 12},
/* π5' = */ {5, 13, 15, 6, 9, 2, 12, 10, 11, 7, 8, 1, 4, 3, 14, 0},
/* π6' = */ {8, 14, 2, 5, 6, 9, 1, 12, 15, 4, 11, 0, 13, 10, 3, 7},
/* π7' = */ {1, 7, 14, 13, 0, 5, 8, 3, 4, 15, 10, 6, 9, 12, 11, 2}
};
	uint gid = get_global_id(0);//+get_global_id(1)*get_global_size(0);
	uint a = data[gid];
	uint r = 0;
	__attribute__((opencl_unroll_hint(8)))
	for (int i=0; i<8; i++){
		uint s = sbox[i][BEXTR(a,(i*4),4)];
		r |= (s)<<(i*4);
		//r = bitselect(r, s<<(i*4), (uint)0xF<<(i*4));
	}
	data[gid] = r;
}
