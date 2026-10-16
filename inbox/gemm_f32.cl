#define SIZE (256)
// произведение матрицы на транспонированную матрицу B
__kernel void GEMM(__global float* c, __constant float* A, __constant float* B) {
	uint x = get_global_id(0);
	uint y = get_global_id(1);
	uint size = get_global_size(0);

	float sum = 0.0f;
	//#pragma unroll 
	for (uint k = 0; k < 1024; k++) {
		sum += A[y * SIZE + k]* B[k * SIZE + x];
	}
	c[y * SIZE + x] = sum;
}
__kernel void GEMM4(__global float* c, __constant float4* A, __constant float4* B) {
	uint x = get_global_id(0);
	uint y = get_global_id(1);
	uint size = get_global_size(0);

	float4 sum = 0.0f;
	//#pragma unroll 
	for (uint k = 0; k < 1024/4; k++) {
		sum += A[y * SIZE + k]* B[x * SIZE + k];
	}
	c[y * SIZE + x] = sum.x+sum.y+sum.z+sum.w;
}
__kernel void GEMM8(__global float* c, __constant float8* A, __constant float8* B) {
	uint x = get_global_id(0);
	uint y = get_global_id(1);
	uint size = get_global_size(0);

	float8 sum = 0.0f;
	//#pragma unroll 
	for (uint k = 0; k < 1024/8; k++) {
		sum += A[y * SIZE + k]* B[x * SIZE + k];
	}
	float4 s4 = sum.lo+sum.hi;
	float2 s2 = s4.lo+s4.hi;
	c[y * SIZE + x] = s2.lo+s2.hi;
}