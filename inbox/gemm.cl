#define SIZE (1024/4)
// произведение матрицы на транспонированную матрицу B
__kernel void GEMM(__global float4* A, __global float4* B, __global float* c) {
	uint x = get_global_id(0);
	uint y = get_global_id(1);
	uint size = get_global_size(0);

	float4 sum = 0.0f;
	//#pragma unrol 4
	for (uint k = 0; k < SIZE; k++) {
		sum += A[y * SIZE + k] * B[x * SIZE + k];
	}
	c[y * size + x] = sum.x+sum.y+sum.z+sum.w;
}