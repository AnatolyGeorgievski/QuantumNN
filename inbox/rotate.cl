/*! Поворот матриц
LU- разложение матриц и решение систем уравнений с использованием LU разложения
	Copyright (c) 2024 Anatoly Georgievskii <anatoly.georgievski@gmail.com>
 */
#define SHUFFLE2(v, w, a,b,c,d) shuffle2((v),(w), (uint4)(a,b,c,d))
#define N 8
__kernel void lu_decomp (__global float *LL, __global float *UU, __constant float *A)
{
	uint gid = get_global_id(0)+get_global_id(1)*get_global_size(0);
	LL+= gid*N*N;
	UU+= gid*N*N;
	A += gid*N*N;

	float L[N][N] = {0};
	float U[N][N] = {0};
	for (int i=0; i<N; i++)	L[i][i] = 1;
	
	for (int i=0; i<N; i++)
	for (int j=0; j<N; j++){
		float s=0;
		if (i<=j) {
			for(int k=0; k<i;k++)
				s+= L[i][k]*U[k][j];
			U[i][j] =  A[i*N+j] - s;
		} else {
			for(int k=0; k<j;k++)
				s+= L[i][k]*U[k][j];
			L[i][j] = (A[i*N+j] - s)/U[j][j];
		}
	}
	for (int i=0; i<N; i++) 
	for (int j=0; j<N; j++)
		LL[i*N+j] = L[i][j];
	for (int i=0; i<N; i++) 
	for (int j=0; j<N; j++)
		UU[i*N+j] = U[i][j];
}
__kernel void lu_decomp2 (__global float *lu, __constant float *A)
{
	uint gid = get_global_id(0)+get_global_id(1)*get_global_size(0);
	lu+= gid*N*N;
	A += gid*N*N;
	float L[N*N];
	float U[N*N];
	for (int i=0; i<N; i++)
	for (int j=0; j<N; j++){
		float s=0;
		if (i<=j) {
			for(int k=0; k<i;k++)
				s+= L[i*N+k]*U[k*N+j];
			U[i*N+j] =  A[i*N+j] - s;
		} else {
			for(int k=0; k<j;k++)
				s+= L[i*N+k]*U[k*N+j];
			L[i*N+j] = (A[i*N+j] - s)/U[j*N+j];
		}
	}
	for (int i=0; i<N; i++)
	for (int j=0; j<N; j++)
		lu[i*N+j] = (j<i)?L[i*N+j]:U[i*N+j];
}
__kernel void lu_decomp1 (__global float *lu, __constant float *A)
{
	uint gid = get_global_id(0)+get_global_id(1)*get_global_size(0);
	lu+= gid*N*N;
	A += gid*N*N;

	float M[N*N];
	for(int i=0; i<N*N; i++)
		M[i] = A[i];
	for(int i=0; i<N; i++) { // decompose M in M=L*U
		for(int j=i+1; j<N; j++) {
			M[N*j+i] /= M[N*i+i];
			for(int k=i+1; k<N; k++) 
				M[N*j+k] -= M[N*j+i]*M[N*i+k];
		}
	}
	for(int i=0; i<N*N; i++)
		lu[i] = M[i];
}
#undef N
/* solves system of N linear equations M*x=b within dimensionality Nsol<=N 
Применяется 5x5 и 2x2
 */
#define N 5
#define Nsol 5
void lu_solve(float* M, float* x, float* b/*, const int N, const int Nsol*/) 
{
	for(int i=0; i<Nsol; i++) { // decompose M in M=L*U
		for(int j=i+1; j<Nsol; j++) {
			M[N*j+i] /= M[N*i+i];
			for(int k=i+1; k<Nsol; k++) M[N*j+k] -= M[N*j+i]*M[N*i+k];
		}
	}
	for(int i=0; i<Nsol; i++) { // find solution of L*y=b
		x[i] = b[i];
		for(int k=0; k<i; k++) x[i] -= M[N*i+k]*x[k];
	}
	for(int i=Nsol-1; i>=0; i--) { // find solution of U*x=y
		for(int k=i+1; k<Nsol; k++) x[i] -= M[N*i+k]*x[k];
		x[i] /= M[N*i+i];
	}
}
__kernel void transpose_4x4f (__global float4 *r, __constant float4 *s)
{
	float4 x[4];
    x[0] = (float4)(s[0].lo, s[1].lo);//SHUFFLE2 (s[0], s[1], 0, 1, 4, 5);
    x[1] = (float4)(s[0].hi, s[1].hi);//SHUFFLE2 (s[0], s[1], 2, 3, 6, 7);
    x[2] = (float4)(s[2].lo, s[3].lo);//SHUFFLE2 (s[2], s[3], 0, 1, 4, 5);
    x[3] = (float4)(s[2].hi, s[3].hi);//SHUFFLE2 (s[2], s[3], 2, 3, 6, 7);
    r[0] = (float4)(x[0].even, x[2].even);//SHUFFLE2 (x0, x2, 0, 2, 4, 6);
    r[1] = (float4)(x[0].odd,  x[2].odd );//SHUFFLE2 (x0, x2, 1, 3, 5, 7);
    r[2] = (float4)(x[1].even, x[3].even);//SHUFFLE2 (x1, x3, 0, 2, 4, 6);
    r[3] = (float4)(x[1].odd,  x[3].odd );//SHUFFLE2 (x1, x3, 1, 3, 5, 7);
}

__kernel void transpose_4x4 (__global uint4 *r, __constant uint4 *s)
{
	uint4 x[4];
    x[0] = (uint4)(s[0].lo, s[1].lo);//SHUFFLE2 (s[0], s[1], 0, 1, 4, 5);
    x[1] = (uint4)(s[0].hi, s[1].hi);//SHUFFLE2 (s[0], s[1], 2, 3, 6, 7);
    x[2] = (uint4)(s[2].lo, s[3].lo);//SHUFFLE2 (s[2], s[3], 0, 1, 4, 5);
    x[3] = (uint4)(s[2].hi, s[3].hi);//SHUFFLE2 (s[2], s[3], 2, 3, 6, 7);
    r[0] = (uint4)(x[0].even, x[2].even);//SHUFFLE2 (x0, x2, 0, 2, 4, 6);
    r[1] = (uint4)(x[0].odd,  x[2].odd );//SHUFFLE2 (x0, x2, 1, 3, 5, 7);
    r[2] = (uint4)(x[1].even, x[3].even);//SHUFFLE2 (x1, x3, 0, 2, 4, 6);
    r[3] = (uint4)(x[1].odd,  x[3].odd );//SHUFFLE2 (x1, x3, 1, 3, 5, 7);
}
__kernel void transpose_4x4s (__global uint4 *r, __global uint4 *s)
{
    uint4 x0 = SHUFFLE2 (s[0], s[1], 0, 1, 4, 5); // 0 1 4 5
    uint4 x1 = SHUFFLE2 (s[0], s[1], 2, 3, 6, 7); // 2 3 6 7
    uint4 x2 = SHUFFLE2 (s[2], s[3], 0, 1, 4, 5); // 8 9 12 13
    uint4 x3 = SHUFFLE2 (s[2], s[3], 2, 3, 6, 7); // 10 11 14 15
    r[0] = SHUFFLE2 (x0, x2, 0, 2, 4, 6);// 0 4  8 12
    r[1] = SHUFFLE2 (x0, x2, 1, 3, 5, 7);// 1 5  9 13
    r[2] = SHUFFLE2 (x1, x3, 0, 2, 4, 6);// 2 6 10 14
    r[3] = SHUFFLE2 (x1, x3, 1, 3, 5, 7);// 3 7 11 15
}

__kernel void rotate_cw_4x4(__global uint4 *r, __global uint4 *s)
{
    uint4 s0 = SHUFFLE2 (s[0], s[1], 0, 1, 5, 6); // 0 1 4 5
    uint4 s1 = SHUFFLE2 (s[0], s[1], 2, 3, 7, 4); // 2 3 6 7
    uint4 s2 = SHUFFLE2 (s[2], s[3], 2, 3, 7, 4); // 8 9 12 13
    uint4 s3 = SHUFFLE2 (s[2], s[3], 0, 1, 5, 6); // 10 11 14 15
    r[0] = SHUFFLE2 (s0, s2, 0, 2, 4, 6);// 0 4  8 12
    r[1] = SHUFFLE2 (s0, s2, 1, 3, 5, 7);// 1 5  9 13
    r[2] = SHUFFLE2 (s1, s3, 0, 2, 4, 6);// 2 6 10 14
    r[3] = SHUFFLE2 (s1, s3, 1, 3, 5, 7);// 3 7 11 15
}

__kernel void rotate_ccw_4x4(__global uint4 *r, __global uint4 *s)
{
	uint4 x0 = SHUFFLE2 (s[0], s[1], 0, 1, 4, 5); // 0 1 4 5
    uint4 x1 = SHUFFLE2 (s[0], s[1], 2, 3, 6, 7); // 2 3 6 7
    uint4 x2 = SHUFFLE2 (s[2], s[3], 0, 1, 4, 5); // 8 9 12 13
    uint4 x3 = SHUFFLE2 (s[2], s[3], 2, 3, 6, 7); // 10 11 14 15
    r[0] = SHUFFLE2 (x0, x2, 0, 2, 4, 6);// 0 4  8 12
    r[1] = SHUFFLE2 (x0, x2, 7, 1, 3, 5);// 1 5  9 13
    r[2] = SHUFFLE2 (x1, x3, 4, 6, 0, 2);// 2 6 10 14
    r[3] = SHUFFLE2 (x1, x3, 3, 5, 7, 1);// 3 7 11 15
}
