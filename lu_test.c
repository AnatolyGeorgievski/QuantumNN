/*! Много численных методов разложения матриц и решения СЛАУ 
	см LINALG.md
	
	§²×°‖≡≠≈∂∆ᵀᴴ ÷×°±²∞∏∑‾≤≥ λμ αβγδε ζηθξ
 */

/*! LU разложение матриц и решение систем СЛАУ методом LU
Методы вычисления LU-разложения:
• метод Гаусса;
(низкая точность для разреженных матриц, большой объем вычислений)
• алгоритм Дулиттла;
(L - унитреугольная, U - верхняя треугольная)
• алгоритм Кроута.
(L - нижняя треугольная, U - унитреугольная)

LUP-разложение (LUP-декомпозиция) — представление данной матрицы 
𝐴 в виде произведения 𝑃𝐴=𝐿𝑈 где матрица 
𝐿 является нижнетреугольной с единицами на главной диагонали, 
𝑈 — верхнетреугольная общего вида, а 𝑃 — т. н. матрица перестановок, 
получаемая из единичной матрицы путём перестановки строк или столбцов. 

LUP-разложение это LU-разложения матрицы A с частичным выбором ведущего
элемента (перестановка по строкам), которое записывается следующим образом:
P A = L U ,
а, в случае перестановок по столбцам LUP-разложение записывается в виде:
A Q = L U ,
а, в наиболее общем виде:
P A Q = L U ,
где:
P - матрица перестановок по строкам;
Q - матрица перестановок по столбцам;
L - нижняя унитреугольная матрица;
U - верхняя треугольная матрица.

LDU-разложение матрицы A - это ее представление в виде произведения трех
матриц:
A = L D U ,
где:
D - диагональная матрица;
L - нижняя унитреугольная матрица;
U - верхняя унитреугольная матрица.

LL-разложение (Разложение Холецкого)
LL-разложение положительно определенной матрицы A - это ее представление в
виде произведений двух матриц:
A = L L⃰
A = U⃰U
где:
L - нижняя треугольная матрица; U - верхняя треугольная матрица.
Для вещественных матриц: L⃰ = Lᵀ и U⃰ = Uᵀ.

## QR - разложение матрицы

QR - разложение матрицы — представление матрицы в виде произведения унитарной 
(или ортогональной матрицы) и верхнетреугольной матрицы. QR-разложение является 
основой одного из методов поиска собственных векторов и чисел матрицы — QR-алгоритма

Полный набор ортогональных векторов

	Процесс Грама-Шмидта
Процесс Грама-Шмидта — это метод вычисления ортогональной матрицы Q, которая состоит из ортогональных или независимых единичных векторов и занимает такое же пространство, что и матрица X.

1. Выбрать вектор столбец u_1 = x_1, x1!=0
2. Рассчитать для всех столбцов u_k = x_k - sum_{j=1}^{k-1} proj(u_j, x_k)
3. Для всех векторов Q_k = u_k/ norm(u_k) -- нормализация

proj - считает ортогональный вектор u_k к другим методом найти такую константу чтобы u_2 = x_2 - c u_2

QR = X
R = Q^-1 X = Q^T X

https://www-users.cse.umn.edu/~olver/aims_/qr.pdf


Процесс Грама ― Шмидта может быть истолкован как разложение невырожденной квадратной матрицы в произведение ортогональной (или унитарной в случае эрмитова пространства) и верхнетреугольной матрицы с положительными диагональными элементами ― QR-разложение

### Спектральное разложение матрицы

Спектральное разложение матрицы A - это ее представление в виде произведения трех матриц:
A = V Λ V^{−1},
где:
V - матрица с собственными векторами матрицы A;
Λ - диагональная матрица с собственными значениями 𝜆(A).
Только матрицы, имеющие полный набор собственных значений, могут быть представлены в виде спектрального разложения.

###Каноническая форма Жордана

Каноническая форма Жордана - это представление квадратной матрицы A в виде:
A = C J C^{−1},
где:
C - матрица перехода к новому базису;
J - матрица Жордана.
По сути, это разложение - обобщение спектрального разложения на случай кратных собственных значений.
Если кратность всех собственных значений матрицы A равна единице, то матрица
J - диагональная. В противном случае матрица J является блок-диагональной и
состоит из Жордановых блоков.

### Разложение Шура

Разложение Шура матрицы A - это ее представление в виде произведения трех матриц:
A = U T U*,
где:
U - унитарная матрица;
U*- эрмитово-сопряженная матрица;
T - верхняя треугольная матрица с Tii = 𝜆i(A).
Для случая вещественных чисел разложение имеет вид:
A = V T Vᵀ ,
где:
V - ортогональная матрица.

### Унитарная матрица

Унитарная матрица - это квадратная матрица, в общем случае состоящая из
комплексных чисел, произведение которой на эрмитово-сопряженную матрицу
равно единичной матрице, т.е.:
U⃰U = U U⃰ = I

### Ортогональная матрица

Ортогональная матрица - это унитарная матрица, но только с вещественными
числами, для которой справедливо:
Aᵀ A = A Aᵀ = I ,
и как следствие:
Aᵀ = A^{−1}

### Эрмитова матрица
Эрмитова матрица - это квадратная матрица, в общем случае состоящая из 
комплексных чисел, которая равна своей эрмитово-сопряженной (сопряженно-
транспонированной матрице) и равна транспонированной комплексно-сопряженной
матрице, т.е.:
A = (A¯)ᵀ = A⃰,
A⃰ -- это и транспонирование и поэлементное комплексное сопряжение

	det(Aᵀ)= det(A) 
	det(A)=det(LU)=det(L)det(U)

Определитель, вычисленный det мера масштабного коэффициента линейного преобразования, 
описанного матрицей. Когда определитель ниже нуля, матрица сингулярна, и никакая инверсия не существует.
Определитель треугольной матрицы - произведение диагональных элементов. 


Некоторые матрицы почти сингулярны, и несмотря на то, что обратная матрица существует, 
вычисление восприимчиво к числовым ошибкам. cond функция вычисляет число обусловленности 
для инверсии, которая дает индикацию относительно точности результатов матричной инверсии. 
Число обусловленности лежит в диапазоне от 1 для численно устойчивой матрицы к Inf для 
сингулярной матрицы.

c = cond(A)

### Дополнительный минор

Дополнительный минор M¯_ij квадратной матрицы A - это определитель, 
составленный из элементов матрицы A путем удаления строк и столбцов c номерами ij


### Собственные вектора и собственные значения матрицы

Собственным вектором x и соответствующим ему собственном значением 𝜆 
матрицы A называются величины, для которых следующее равенство имеет 
ненулевое решение:
A v = 𝜆 v

В общем случае определение собственных значений и собственных векторов 
требует решения матричного уравнения:
(A − 𝜆 I) v = 0 ,

Матрица комплексная может быть разложена на произведение 
A = U R U⃰
R - Треугольной матрицы, содержащей собственные значения на главной диагонали
U - унитарная матрица U* = U^-1
U*- транспонированная комплексно-сопряженная матрица 
Если A - нормальная то AA* = A*A, то матрица R- диагональная содержит собственные значения
Если A нормальная эрмитова матрица то собственные значения - вещественные числа. 


### След матрицы

След матрицы tr(A) - это скалярное значение, которое
равно сумме всех диагональных элементов этой матрицы
След матрицы равен сумме ее собственных значений
 */

#include <stdio.h>
#include <math.h>
#define N 3 // параметризация
// операции с матрицами
void  print_m (const float *A);
void  print_mn(const float* A, unsigned m, unsigned n);
void  mov_m	 (const float* a, float *r);
void  mov_mn (const float* a, float *r, unsigned m, unsigned n);
void  mul_m  (const float* A, const float* B, float* R);
void  mul_mn (const float* A, const float* B, float* R, unsigned m, unsigned n);
void  gemv   (const float* A, const float* x, const float* b,  float* r);
void  axpy	 (const float* A, const float* x, float* y);
int   cmp_m  (const float* a, const float* b);
int   cmp_eps_m (const float* a, const float* b, float eps);
int cmp_eps_mn(const float* a, const float* b, float eps, unsigned m, unsigned n);
float det_m  (const float *lu);
float trace_m(const float *a);
void balance  (float* A, float *D, unsigned m, unsigned n);
void unbalance(float* A, float *D, unsigned m, unsigned n);
// восстановить A=LU, A=LDU, A=LDL^T
void lu_comp	(const float *lu, float* r);
void ldu_comp	(const float *lu, float* r);
void ldl_comp	(const float *lu, float* r);
// выполнить разложение A=LU
void lu_decomp  (float *L, float *U, const float *A);
void lu_decomp1 (float *M);
void lu_decomp2 (float *LU, const float *A);
void lu_decomp3 (float *LU, const float *A);
void ldu_decomp (float *LDU,const float *A);
// решение систем уравнений Ax=C
void lu_solve   (const float *LU, float *x, const float *c);
void lup_solve	(const float *lu, const int *p, float *z, const float *c);
void ldu_solve  (const float *LDU,float *x, const float *c);
void lu_inv 	(const float *LU, float *A_inv);
void lup_inv 	(const float *LU, const int *P, float *A_inv);
void ldu_inv 	(const float *LDU,float *A_inv);
void lu_inv_l 	(const float *LU, float *L_inv);
void lu_inv_u 	(const float *LU, float *U_inv);
float lu_det 	(const float *LU);
// разложение Холецкого для симметричных и эрмитовых матриц
void cholesky_decomp(float * L, const float *A);
void cholesky_ldl_decomp(float * L, const float *A);
void cholesky_ldl_solve(const float * L, float *z, const float* c);
void cholesky_ldl_inv  (const float * L, float *z);
float cholesky_ldl_det (const float * L);
// разложение Холецкого для анти-симметричных матриц
// разложение QR
int  qr_cgs_decomp(float* a, float* r, unsigned m, unsigned n);
int  qr_decomp	  (float* a, float* r, unsigned m, unsigned n);
void qr_block_decomp(float* a, float* r, unsigned m, unsigned n);
int  qr_givens  (float* a, float* r, unsigned m, unsigned n);
int  qr_house   (float* a, float* r, unsigned m, unsigned n);
int  qr_house2  (float* a, float* tau, unsigned m, unsigned n);
int  qr_house_unpack(float * a, float * tau,  float * q, float * r, unsigned m,  unsigned n);
void qr_house_bidi(float * a, float * tau, float * tav, unsigned m,  unsigned n);
void qr_house_bidi_unpack(float * a, float * tau, float * tav, float * U, float * V, unsigned m,  unsigned n);
float qr_det(float* r, unsigned n);
// разложение Шура
// сингулярное разложение SVD
// определение матриц
int  is_hermitian  (const float* A, unsigned n);
int  is_symmetric  (const float* A, unsigned n);
int  is_skew_symmetric(const float* A, unsigned n);
int  is_symplectic (const float* A, unsigned n);// блочная матрица 2n 2n
int  is_hamiltonian(const float* A, unsigned n);
int  is_orthogonal (const float* A, unsigned n);
int  is_orthonormal(const float* A, unsigned m, unsigned n);
int  is_normal     (const float* A, unsigned n);// нормальная матрица AᵀA == AAᵀ
int  is_hessenberg (const float* A, unsigned n);// определить матрицу Хессенберга, все нули под-диагональю
int  is_tridiagonal(const float* A, unsigned n);
int  is_singular   (const float* A, unsigned n);// может матрица сингулярная? Строки или столбцы матрицы линейно зависимы.

static void _set_zero(float* a, unsigned m, unsigned n, unsigned lda);
static void _set_identity(float* a, unsigned m, unsigned n);
/* A is Hermitian  \iff A={\overline{A}^{\mathsf{T}}}.

$${\displaystyle A{\text{ is Hermitian}}\quad \iff \quad A={\overline {A^{\mathsf{T}}}}.}$$
 */

int main()
{
	float eps = N*__FLT_EPSILON__;
	float A [3*3]= {
	25, 5, 4,
	10, 8,16,
	8 ,12,22,
	};
	float B [3*3]= {
	25, 5, 4,
	 0, 8,16,
	 0,12,22,
	};
	float A1[3*3]= {
	10,-7, 0,
	-3, 6, 2,
	 5,-1, 5,
	};
	float LU1[3*3]= {
	10,-7, 0,
	-0.3, -0.1, 6,
	 0.5, -25, 155,
	};

	float C[3*3]= {
	4,12, -16,
	12,37, -43,
	-16,-43,98,
	};
	float LC[3*3]= {
	2,0,0,
	6,1,0,
	-8,5,3,
	};
	float D[3*3]= {
	1.4, 1, 1,
	1, 0.9, 1,
	1, 1, 1.4,
	};
	float LD[3*3]= {
	1.4, 0, 0,
	0.7143, 0.1857, 0,
	0.7143, 1.5385, 0.2462,
	};
	float L1[3*3]= {
	2, 0, 0,
	8,-7, 0,
	4, 9,-27,
	};
	float E1[3*3]= {
	1, 0, 0,
	0, 1, 0,
	0, 0, 1,
	};
		
	
	float L [3*3];
	float U [3*3];
	float M [3*3];
	float M1[3*3];
	lu_decomp(L, U, A);
	print_m (L);
	print_m (U);
	
	mul_m(L,U, M);
	if(cmp_m(M,A)) printf("..ok\n");
	else printf("..fail\n");
	
//	printf("LU decomp1\n");
	lu_decomp1(M);
	print_m (M);
//	if(cmp_m(M,M1)) printf("..ok\n");
//	else printf("..fail\n");

	printf("LU decomp2\n");
	lu_decomp2(M1,A);
	print_m (M1);
	if(cmp_eps_m(M,M1, 2*eps)) printf("..ok\n");
	else printf("..fail\n");

	printf("LU compose\n");
	lu_comp(M1,B);
	print_m (B);
	if(cmp_eps_m(B,A, 2*eps)) printf("..ok\n");
	else printf("..fail\n");

	printf("LU decomp3\n");
	lu_decomp3(M1,A);
	print_m (M1);
	if(cmp_eps_m(M,M1,2*eps)) printf("..ok\n");
	else printf("..fail\n");

printf("\n");
	lu_decomp(L, U, B);
	print_m (L);
	print_m (U);
	mul_m(L,U, M);
	if(cmp_m(M,B)) printf("..ok\n");
	else printf("..fail\n");

	printf("LU decomp1\n");
	mov_m(B,M);
	lu_decomp1(M);
	print_m (M);
	if(cmp_eps_m(M,M1, 2*eps)) printf("..ok\n");
	else printf("..fail\n");

	lu_decomp2(M,B);
	print_m (M);
	printf("inversion\n");
	float B_inv[3*3];
	float E[3*3];
	lu_inv(M,B_inv);
	mul_m(B,B_inv, E);
	print_m (E);
	if(cmp_eps_m(E,E1, N*eps)) printf("..ok\n");
	else printf("..fail\n");

	printf("inversion L matrix\n");
	lu_decomp(L, U, B_inv);
	print_m (L);
	printf("inversion L_inv\n");
	lu_decomp2(M,B);
	lu_inv_l(M, B_inv);
	print_m (B_inv);
	lu_decomp(L, U, B);
	mul_m(L,B_inv, E);
	//print_m (E);
	if(cmp_eps_m(E,E1, eps)) printf("..ok\n");
	else printf("..fail\n");

	printf("inversion U matrix\n");
	lu_inv_u(M, B_inv);
	mul_m(U,B_inv, E);// если есть деление, теряется точность
	if(cmp_eps_m(E,E1, eps)) printf("..ok\n");
	else printf("..fail\n");

// \todo аналогичную серию ldu_inv
	printf("LDU inversion\n");
	ldu_decomp(M,B);
	ldu_inv(M,B_inv);
	mul_m(B,B_inv, E);
	if(cmp_eps_m(E,E1, N*eps)) printf("..ok\n");
	else printf("..fail\n");

	printf("Cholesky LL decomp\n");
	cholesky_decomp(L, C);
	print_m (L);
	if(cmp_m(L,LC)) printf("..ok\n");
	else printf("..fail\n");

	printf("Cholesky LDL decomp\n");
	cholesky_ldl_decomp(L, D);
	print_m (L);
	if(cmp_eps_m(L,LD, N*eps)) printf("..ok\n");
	else printf("..fail\n");

	printf("Cholesky LDL solve\n");
	float Ev[3] = {1,1,1};
	float  v[3];
	cholesky_ldl_solve(L, v, Ev);
	printf("v = %1.8f %1.8f %1.8f\n",v[0],v[1],v[2]);

	printf("LU solve\n");
	lu_decomp2(L, D);
	lu_solve(L, v, Ev);
	printf("v = %1.8f %1.8f %1.8f\n",v[0],v[1],v[2]);
	printf("LDU solve\n");
	ldu_decomp(L, D);
	ldu_solve(L, v, Ev);
	printf("v = %1.8f %1.8f %1.8f\n",v[0],v[1],v[2]);

	printf("Test decomp-mul\n");

	lu_decomp(L,U,A1);
	print_m (L);
	print_m (U);
	mul_m(L,U, M);
	if(cmp_m(M,A1)) printf("..ok\n");

	float Q[3*3] = {
		1, 1, 2,
		1, 0,-2,
	   -1, 2, 3
	};
	float R[3*3]={0};// пусто
	printf("QR decomp (QR Modified Gram-Schmidt)\n");
	mov_m(Q,A1);
	qr_decomp(Q, R, N, N);
	print_m (Q);
	print_m (R);
	mul_m(Q,R,M);
	print_m (M);
	if(cmp_eps_m(M,A1, eps)) printf("..ok\n");
	if (is_orthogonal(Q, N)) printf("..is_orthogonal\n");
	if (is_orthonormal(Q, N, N)) printf("..is_orthonormal\n");
	if (is_normal(Q, N)) printf("..is_normal\n");
	
	printf("QR decomp (QR Classical Gram-Schmidt)\n");
	float Q1[3*3] = {
		1, 1, 2,
		1, 0,-2,
	   -1, 2, 3
	};
	qr_cgs_decomp(Q1, R, N, N);
	print_m (Q1);
	print_m (R);
	if (is_orthogonal(Q1, N)) printf("..is_orthogonal\n");
	if (is_orthonormal(Q1, N, N)) printf("..is_orthonormal\n");
	if (is_normal(Q1, N)) printf("..is_normal\n");
	
	float G[3*3] = {
		1, 1, 2,
		1, 0,-2,
	   -1, 2, 3
	};
	printf("QR decomp (QR Givens)\n");
	qr_givens(G,R, N, N);

	float H[3*3] = {
		1, 1, 2,
		1, 0,-2,
	   -1, 2, 3
	};
	printf("QR decomp (QR Householder)\n");
	float tau[N];
	mov_m(H,A1);
	qr_house (H,tau, N, N);

	print_m (H);
	qr_house_unpack(H, tau, Q, R, N, N);
	print_m (Q);
	print_m (R);
	mul_m(Q,R,M);
	print_m (M);
	if(cmp_eps_m(M,A1, eps)) printf("..ok\n");
	if (is_orthogonal(Q, N)) printf("..is_orthogonal\n");
	if (is_orthonormal(Q, N, N)) printf("..is_orthonormal\n");
	if (is_normal(Q, N)) printf("..is_normal\n");

	float H1[3*3] = {
		1, 1, 2,
		1, 0,-2,
	   -1, 2, 3
	};

	printf("QR decomp (QR Householder)\n");

	qr_house (H1,tau, N, N);
	print_m (H1);
	qr_house_unpack(H1, tau, Q, R, N, N);
	mul_m(Q,R,M);
	if(cmp_eps_m(M,A1, eps)) printf("..ok\n");
	if (is_orthogonal(Q, N)) printf("..is_orthogonal\n");
	if (is_orthonormal(Q, N, N)) printf("..is_orthonormal\n");
	if (is_normal(Q, N)) printf("..is_normal\n");

	float A4[] = { 
/*		2,1,0,0,
		1,2,1,0,
		0,1,2,1,
		0,0,1,2}; */
		0.18, 0.60, 0.57, 0.96,
		0.41, 0.24, 0.99, 0.58,
		0.14, 0.30, 0.97, 0.66,
		0.51, 0.13, 0.19, 0.85 }; 
	float tau4[4];
	float tav4[4];
	float B4[4*4];
	float Q4[4*4];
	float R4[4*4];
	float M4[4*4];
	mov_mn(A4, B4, 4, 4);
	mov_mn(A4, Q4, 4, 4);
	qr_decomp(Q4,R4, 4, 4);
//	qr_cgs_decomp(Q4,R4, 4, 4);
	mul_mn(Q4,R4,M4,4,4);
	print_mn(M4, 4, 4);
	if(cmp_eps_mn(M4,B4, eps, 4,4)) printf("..ok\n");
	qr_house(A4,tau4, 4, 4);
	qr_house_unpack(A4, tau4, Q4, R4, 4, 4);
	print_mn(Q4, 4, 4);
	print_mn(R4, 4, 4);
	mul_mn(Q4,R4,M4,4,4);
	if(cmp_eps_mn(M4,B4, eps, 4,4)) printf("..ok\n");
	if (is_orthogonal(Q4, 4)) printf("..is_orthogonal\n");

	printf("QR block decomp (QR block)\n");
	mov_mn(B4, Q4, 4, 4);
	qr_block_decomp(Q4,R4, 4, 4);
	print_mn(R4, 4, 4);
	mul_mn(Q4,R4,M4,4,4);
	if(cmp_eps_mn(M4,B4, eps, 4,4)) printf("..ok\n");
/*!
octave> A = [ 0.18, 0.60, 0.57, 0.96;
              0.41, 0.24, 0.99, 0.58;
              0.14, 0.30, 0.97, 0.66;
              0.51, 0.13, 0.19, 0.85 ];

octave> x = [ -4.05205; -12.6056; 1.66091; 8.69377];

octave> A * x
ans =
  1.0000
  2.0000
  3.0000
  4.0000 */
  	float T4[] = {
		1, 1, 0, 0,
		0, 2, 1, 0, 
		0, 0, 3, 1,
		0.001, 0, 0, 4
	};
	float H4[] = {
		1,    1./2, 1./3, 1./4,
        1./2, 1./5, 1./4, 1.f/5,
        1./3, 1./4, 1./6, 1.f/7,
        1./4, 1./5, 1./7, 1.f/9 };
	float D4[4];
	mov_mn(T4, A4, 4, 4);
	balance(A4, D4, 4,4);
	unbalance(A4, D4, 4,4);

	printf("Balance\n");
	for(int i=0; i<4; i++) printf("\t%f", D4[i]);
	printf("\nA=\n");
	print_mn(A4, 4, 4);

	printf("QR Householder Bidiagonalization\n");

	float A45[] = {
		4, 3, 0, 2, 5,
		2, 1, 2, 1, 6,
		4, 4, 0, 3, 0,
		5, 6, 1, 3, 7};
//	mov_mn(B4, Q4, 4, 4);
	qr_house_bidi(A45, tau4, tav4, 4,5);
//	print_mn(Q4, 4, 4);
	float A10[] = {// 10 x 5
	0.8594598509, 0.8886035203, 0.8149294811, 0.7431045200, 0.8032585254,
	0.0587533356, 0.7245921139, 0.5380305406, 0.7342256338, 0.6982547215,
	0.7176400044, 0.0539911194, 0.3670289037, 0.9701228316, 0.8404100032,
	0.4112932913, 0.3075223914, 0.5798244230, 0.0015286701, 0.7890766996,
	0.9781337455, 0.2921431712, 0.0432923459, 0.9428416709, 0.9646959945,
	0.0354323143, 0.4898468039, 0.4513681016, 0.2107982126, 0.4445287671,
	0.8115565467, 0.7058405790, 0.5527189195, 0.5410537042, 0.9117912347,
	0.1149175267, 0.8406228190, 0.6040554044, 0.4260203703, 0.2376075180,
	0.2164094832, 0.1800869710, 0.7479251262, 0.0009715103, 0.8810979640,
	0.8647838791, 0.5856765260, 0.0127644690, 0.5744975219, 0.1985024847};
	printf("QR Bidi A10\n");
	float B10[10*5];
	float Q10[10*5];
	float R10[5*5];
	float M10[10*5];
	float tau10[10];
	float tav5[5];
	mov_mn(A10, B10, 10, 5);
#if 0
	qr_decomp(A10, R10, 10, 5);
	print_mn(A10, 6, 5);
	print_mn(R10, 5, 5);
	mul_mn(A10,R10, M10, 10, 5);
	print_mn(M10, 6, 5);
	if(cmp_eps_mn(M10,B10, eps, 10,5)) printf("..ok\n");
#elif 1
	qr_house(A10, tau10, 10, 5);
	qr_house_unpack(A10, tau10, Q10, R10, 10, 5);
	mul_mn(Q10,R10, M10, 10, 5);
	print_mn(M10, 6, 5);
	print_mn(R10, 5, 5);
	if(cmp_eps_mn(M10,B10, 2*eps, 10, 5)) printf("..ok\n");
#else
	qr_house_bidi(A10, tau10, tav5, 10,5);
	print_mn(A10, 6, 5);
#endif

/* Результат QR разложения должен быть:
2.288      1.517      1.607      1.892      1.183      
0          1.105      0.7235     0.07972    0.07877    
0          0          0.6674     0.299      -0.4158    
0          0          0          0.4826     0.6031     
0          0          0          0          0.9661 
*/
	return 0;
}
/*! \brief вывод на экран печать матриц NxN
 */
void  print_m(const float *A)
{
	for(int i=0; i<N; i++){
		for(int j=0; j<N; j++){
			printf("\t%1.3f,", A[i*N + j]);
		}
		printf("\n");
	}
	printf("\n");
}
/*! \brief печатает матрицу m x n
	\param A - матрица
	\param m - число строк 
	\param n - число столбцов
 */
void print_mn(const float* A, unsigned m, unsigned n){
	for(int i=0; i<m; i++){
		for(int j=0; j<n; j++){
			printf("\t%f,", A[i*n + j]);
		}
		printf("\n");
	}
	printf("\n");	
}
/* копировать матрицу */
void  mov_m	 (const float* a, float *r)
{
	for(int i=0;i<N*N; i++){
		r[i] = a[i];
	}
}
void  mov_mn (const float* a, float *r, unsigned m, unsigned n)
{
	unsigned i, j;
	for(i=0;i<m; i++)
	for(j=0;j<n; j++){
		r[i*n+j] = a[i*n+j];
	}
}
/*! \brief умножение матриц  
	\param r результат R = AB
 */
void mul_m(const float* a, const float* b, float* r)
{
	for (int i=0; i<N; i++)
	for (int j=0; j<N; j++){
		float s = 0;
		for (int k=0; k<N; k++)
			s += a[i*N+k]*b[k*N+j];
		r[i*N+j] = s; 
	}	
}
void mul_mn(const float* a, const float* b, float* r, unsigned m, unsigned n)
{
	unsigned i,j,k;
	for (i=0; i<m; i++)
	for (j=0; j<n; j++){
		float s = 0;
		for ( k=0; k<n; k++)// тут другая размерность
			s += a[i*n+k]*b[k*n+j];
		r[i*n+j] = s; 
	}	
}
/*! \brief умножение матрицы на вектор 
	\param r - результат r = Ax+b
 */
void gemv(const float* a, const float* x, const float* b,  float* r)
{
	for (int i=0; i<N; i++) {
		float s = 0;
		for (int k=0; k<N; k++)
			s += a[i*N+k]*x[k];
		r[i] = s + b[i]; 
	}	
}
/*! \brief умножение матрицы на вектор 
	\param y - результат y = Ax+y
 */
void axpy(const float* a, const float* x, float* y)
{
	for (int i=0; i<N; i++) {
		float s = y[i];
		for (int k=0; k<N; k++)
			s += a[i*N+k]*x[k];
		y[i] = s; 
	}	
}
/*! \brief вычисление детерминанта - определитель матрицы 
	Для расчета используются диагональные элементы LU разложения. 
	
	det(A)=det(LU)=det(L)det(U)
	Определитель унитреугольной матрицы (на диагонали единицы) равен единице
	Одна из матриц LU - унитреугольная
*/
float det_m(const float *lu){
	float d = 1.f;
	for (int i=0; i<N; i++){
		d *= lu[i*N+i];
	}
	return d;
}
/*! \brief След матрицы (trace) - cумма диагональных элементов */
float trace_m(const float *a){
	float d = 0.0f;
	for (int i=0; i<N; i++){
		d += a[i*N+i];
	}
	return d;
}
/*! \brief сравнение двух матриц */
int cmp_m(const float* a, const float* b)
{
	for (int i=0; i<N; i++)
	for (int j=0; j<N; j++){
		if (a[i*N+j]!=b[i*N+j]) return 0;
	}
	return 1;
}
int cmp_eps_m(const float* a, const float* b, float eps)
{
	for (int i=0; i<N; i++)
	for (int j=0; j<N; j++){
		if (fabs(a[i*N+j]-b[i*N+j]) >= eps) return 0;
	}
	return 1;
}
int cmp_eps_mn(const float* a, const float* b, float eps, unsigned m, unsigned n)
{
	unsigned i,j;
	for (i=0; i<m; i++)
	for (j=0; j<n; j++){
		if (fabs(a[i*n+j]-b[i*n+j]) >= eps) return 0;
	}
	return 1;
}

/*! \brief умножение матриц R = LU получаем исходную матрицу
	\param lu разложение матрицы L нижняя унитреугольная (единицы на диагонали), U - верхняя треугольная
	\param r - результат умножения LU, матрица NxN
 */
void lu_comp(const float *lu, float* r)
{
	for (int i=0; i<N; i++)
	for (int j=0; j<N; j++){
		int m = i<=j? i:j;
		float s = 0;
		for (int k=0; k<m; k++) 	//if (k<i && k<j)
			s = fmaf(lu[i*N+k],lu[k*N+j],s);
		if (j< i) s += (lu[i*N+j]*lu[j*N+j]);// строка меньше
		else s += lu[i*N+j]; 	// добавить диагональный элемент, умножается на единицу
		r[i*N+j] = s; 
	}	
}
/*! \brief Решение системы системы линейных уравнений СЛАУ через LU разложение 
  \param lu - две матрицы треугольные матрицы - результат разложения в одном, диагональные элементы относятся к верхней треугольной матрице U, для выделения L - нижней унитреугольной матрицы следует использовать единичные элменты на диагонали.

  Решение A x = b
  LU x = b
  1.  L y = b -- находим вектор y
  2.  U x = y -- находим вектор x

 */
void lu_solve (const float *lu, float *z, const float *c){
	z[0] = c[0];
	for (int i=1; i<N; i++){//  L y = c
		float s = 0;
		for (int k=0; k<i; k++)
			s = fmaf(lu[i*N+k],z[k],s);
		z[i] = c[i]-s;
	}
	z[N-1] /= lu[N*N-1];
	for (int i=N-2; i>=0; i--){// U z = y
		double s = 0;
		for (int k=i+1; k<N; k++)
			s = fmaf(lu[i*N+k],z[k],s);
		z[i] = (z[i] - s)/lu[i*N+i];
	}
}
/*! 
\see https://en.wikipedia.org/wiki/LU_decomposition
 */
void lup_solve (const float *lu, const int *p, float *z, const float *c){
	z[0] = c[p[0]];
	for (int i=1; i<N; i++){//  L y = c
		z[i] = c[p[i]];
		float s = 0;
		for (int k=0; k<i; k++)
			s += lu[i*N+k]*z[k];
		z[i] = z[i]-s;
	}
	z[N-1] /= lu[N*N-1];
	for (int i=N-2; i>=0; i--){// U z = y
		float s = 0;
		for (int k=i+1; k<N; k++)
			s += lu[i*N+k]*z[k];
		z[i] = (z[i]-s)/lu[i*N+i];
	}
}
/*! \brief Решение системы системы линейных уравнений СЛАУ через LDU разложение 
	\param ldu - две матрицы треугольные матрицы, 
  L - нижняя унитреугольная матрица, 
  D - диагональная матрица
  U - верхняя унитреугольная матрица, 
  
  Решение A x = b
  LDU x = b
  1.  L y = b -- находим вектор y
  2.  D z = y -- находим вектор z
  2.  U x = z -- находим вектор x

 */
void ldu_solve (const float *ldu, float *z, const float *c){
	z[0] = c[0];
	for (int i=1; i<N; i++){//  L y = c
		z[i] = c[i];
		float s = 0;
		for (int k=0; k<i; k++)
			s = fmaf(ldu[i*N+k],z[k],s);
		z[i] = z[i]-s;
	}
	for (int i=0; i<N; i++)		// D z = y
		z[i] /= ldu[N*i+i];
	for (int i=N-2; i>=0; i--){	// U x = z
		float s = 0;
		for (int k=i+1; k<N; k++)
			s = fmaf(ldu[i*N+k],z[k],s);
		z[i] = z[i] - s;
	}
}
/*! \brief Метод инверсии матрицы с использованием LDU разложения
	\note столбцы матрицы можно вычислять параллельно
 */
void ldu_inv (const float *ldu, float *z){
	for (int i=0; i<N; i++){
		for (int j=0; j<N; j++)	z[i*N+j] = 0;
		z[i*N+i] = 1;
	}
	for (int j=0; j<N; j++) {// столбец
		for (int i=1; i<N; i++){//  L y = c
			float s = 0;
			for (int k=0; k<i; k++)
				s += ldu[i*N+k]*z[k*N+j];
			z[i*N+j] = z[i*N+j]-s;
		}
		for (int i=0; i<N; i++)		// D z = y
			z[i*N+j] /= ldu[N*i+i];
		for (int i=N-2; i>=0; i--){	// U x = z
			float s = 0;
			for (int k=i+1; k<N; k++)
				s += ldu[i*N+k]*z[k*N+j];
			z[i*N+j] = z[i*N+j] - s;
		}
	}
}
/*! \brief нахождение обратной матрицы 
	тем же методом, что и решение системы для диагональной матрицы E: AX = E
	
	Обратная матрица существует если детерминант det(A) !=0 
	Иначе будет деление на ноль!
	
  \param [in] lu - две треугольные матрицы - результат LU разложения в одном, диагональные элементы относятся к верхней треугольной матрице U, для выделения L - нижней унитреугольной матрицы следует использовать единичные элменты на диагонали.
  \param [out] z - обратная матрица. 
  
 */
void lu_inv (const float *lu, float *z){
	for (int i=0; i<N; i++){
		for (int j=0; j<N; j++)	z[i*N+j] = 0;
		z[i*N+i] = 1;
	}
	for (int j=0; j<N; j++) {// столбец
		for (int i=1; i<N; i++){//  L y = c
			float s = 0;
			for (int k=0; k<i; k++)
				s += lu[i*N+k]*z[k*N+j];
			z[i*N+j] = z[i*N+j]-s;
		}
	}
	for (int j=0; j<N; j++) {// столбец
		z[(N-1)*N+j] /= lu[N*N-1];
		for (int i=N-2; i>=0; i--){// U z = y
			float s = 0;
			for (int k=i+1; k<N; k++)
				s += lu[i*N+k]*z[k*N+j];
			z[i*N+j] = (z[i*N+j]-s)/lu[i*N+i];
		}
	}
}
/*! \brief нахождение обратной матрицы с использованием LU разложения
	\param p перестановка строк
	\param z обратная матрица
 */
void lup_inv   (const float *lu, const int *p, float *z){
	for (int i=0; i<N; i++){
		for (int j=0; j<N; j++)	
			z[i*N+j] = p[j]==i?1.0:0.0;
	}
	for (int j=0; j<N; j++) {// столбец
		for (int i=1; i<N; i++){//  L y = c
			float s = 0;
			for (int k=0; k<i; k++)
				s += lu[i*N+k]*z[k*N+j];
			z[i*N+j] = z[i*N+j]-s;
		}
	}
	for (int j=0; j<N; j++) {// столбец
		z[(N-1)*N+j] /= lu[N*N-1];
		for (int i=N-2; i>=0; i--){// U z = y
			float s = 0;
			for (int k=i+1; k<N; k++)
				s += lu[i*N+k]*z[k*N+j];
			z[i*N+j] = (z[i*N+j]-s)/lu[i*N+i];
		}
	}
}
float lu_det (const float *lu){
	float d = lu[0];
	for (int i=1;i<N;i++){
		d*=lu[i*N+i];
	}
	return d;
}
/*! 
\todo реализовать метод поиска двух обратных матриц U^{-1}L^{-1} для  LU разложения 
обратная матрица для L - нижняя унитреугольная. Для U - верхняя треугольная.
\todo надо тестировать, не отлажено
 */
void lu_inv_l (const float *lu, float *z){

	for (int i=0; i<N; i++){
		for (int j=0; j<N; j++)	z[i*N+j] = 0;
		z[i*N+i] = 1;
	}
	for (int j=0; j<N; j++) {// столбец
		for (int i=1; i<N; i++){//  L y = c
			float s = 0;
			for (int k=0; k<i; k++)
				s += lu[i*N+k]*z[k*N+j];
			z[i*N+j] = z[i*N+j]-s;
		}
	}
}

void lu_inv_u (const float *lu, float *z){

	for (int i=0; i<N; i++){
		for (int j=0; j<N; j++)	z[i*N+j] = 0;
		z[i*N+i] = 1;
	}
	for (int j=0; j<N; j++) {// столбец
		z[(N-1)*N+j] /= lu[N*N-1];
		for (int i=N-2; i>=0; i--){// U z = y
			float s = 0;
			for (int k=i+1; k<N; k++)
				s = fmaf(lu[i*N+k],z[k*N+j],s);
			z[i*N+j] = (z[i*N+j]-s)/lu[i*N+i];
		}
	}
}


/*! \brief Метод LU разложения без копирования
 */
void lu_decomp1 (float *a)
{
	for(int i=0; i<N; i++) { // decompose M in M=L*U
		for(int j=i+1; j<N; j++) {
			float s = a[N*j+i] / a[N*i+i];
			a[N*j+i] = s;
			for(int k=i+1; k<N; k++) 
				a[N*j+k] -= s*a[N*i+k];
		}
	}
}
/*! \brief Метод LU разложения матрицы A = LU (Алгоритм Дулиттла)
	\param LU - матрица составлена из двух треугольных матриц
	L - нижняя унитреугольная матрица, 
	U - верхняя треугольная матрица, диагональные элементы LU относятся к U
	\param A -- исходная матрица
	
	Алгоритм Кроута - симметрично считает LU по столбцам
	U - унитренугольной, L - треугольной. 
	Для получения алгоритма надо поменять i j в расчете и считать в транспонированном виде.
 */
void lu_decomp2 (float *LU, const float *A)
{
	for (int i=0; i<N; i++)
	for (int j=0; j<N; j++){
		float s=0;
		if (i<=j) {
			for(int k=0; k<i;k++)
				s += (LU[i*N+k]*LU[k*N+j]);
			LU[i*N+j] =  A[i*N+j] - s;
		} else {
			for(int k=0; k<j;k++)
				s += (LU[i*N+k]*LU[k*N+j]);
			LU[i*N+j] = (A[i*N+j] - s)/LU[j*N+j];
		}
	}
}
// lu = L+U-I
void lu_decomp3(float *L, const float *A)
{
	for (int i=0; i<N; i++) {// L -- нижняя часть
		for (int j=0; j<i; j++){
			float s=0;
			for(int k=0; k<j;k++)
				s = fmaf(L[i*N+k],L[k*N+j],s);
			L[i*N+j] = (A[i*N+j] - s)/L[j*N+j];
		}
		for (int j=i; j<N; j++){// U1 -- верхняя часть матрицы
			float s=0;
			for(int k=0; k<i;k++)
				s+= L[i*N+k]*L[k*N+j];
			L[i*N+j] = (A[i*N+j] - s);
		}
	}
}
/*! \brief Метод LDU разложения матрицы A = LDU 
	\param LU - матрица составлена из двух треугольных матриц и диагональной
	L - нижняя унитреугольная матрица, 
	U - верхняя унитреугольная матрица, 
	D - диагональные элементы LU

 */
void ldu_decomp (float *L, const float *A)
{
	for (int i=0; i<N; i++) {// L -- нижняя часть
		for (int j=0; j<i; j++){
			float s=0;
			for(int k=0; k<j;k++)
				s= fmaf(L[i*N+k],L[k*N+j],s);
			L[i*N+j] = (A[i*N+j] - s)/L[j*N+j];
		}
		for (int j=i; j<N; j++){// U1 -- верхняя часть матрицы
			float s=0;
			for(int k=0; k<i;k++)
				s= fmaf(L[i*N+k],L[k*N+j],s);
			L[i*N+j] = (A[i*N+j] - s);
		}
/*		
			if (i<=j) {	// U1 -- верхняя часть матрицы
				for(int k=0; k<i;k++)
					s+= L[i*N+k]*L[k*N+j];
				L[i*N+j] = (A[i*N+j] - s);
			} else {	// L -- нижняя часть
				for(int k=0; k<j;k++)
					s+= L[i*N+k]*L[k*N+j];
				L[i*N+j] = (A[i*N+j] - s)/L[j*N+j];
			}
		}*/
	}
	for (int i=0; i<N-1; i++){// Выделить U1 = DU, где U - унитреугольная верхняя
		float d = 1/ L[i*N+i];
		// vector_t v = subrow(L, i, i+1, N-(i+1));
		// BLAS(scal)(v, d);
		for (int j=i+1; j<N; j++)
			L[i*N+j] *= d;
	}
}
/*! \brief Метод LU разложение матрицы A = LU 
 */
void lu_decomp (float *LL, float *UU, const float *A)
{
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
void lu_decomp_ (float *LU, const float *A)
{
	float L[N][N];
	float U[N][N];
	
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
		LU[i*N+j] = j<i?L[i][j]:U[i][j];
}
/*! \brief Разложение Холецкого A = LL^T, последующее разложение LDL^T

### LL-разложение (Разложение Холецкого)
LL-разложение положительно определенной матрицы A - это ее 
представление в виде произведений двух матриц:
A = L L*
A = U* U
где:
L - нижняя треугольная матрица; U - верхняя треугольная матрица.
Для вещественных матриц: L* = L^⊤ и U* = U^⊤.

Для несимметричных матрицы разложение Холецкого принципиально можно получить, 
но оно не будет соответствовать исходной матрице A

Существует несколько алгоритмов получения разложения Холецкого:
∙ классический алгоритм Холецкого;
(в его основе метод исключения Гаусса при приведении исходной матрицы к нижнетреугольному виду)
∙ алгоритм Холецкого-Банахевича;
(алгоритм вычисления матрицы L начиная с первой строки и далее итерационно по каждой строке)
∙ алгоритм Холецкого-Кроута;
(алгоритм вычисления матрицы U начиная с первого столбца и далее итерационно по каждому столбцу)
Алгоритм Кроута в обобщенном виде является алгоритмом вычисления LU-разложения,
где L - нижняя треугольная, а U - унитарная верхняя треугольная.

 */
void cholesky_decomp(float * L, const float *A)
{
    //__builtin_bzero(L, N*N*sizeof(float));
    for (int i = 0; i < N; i++)
	for (int j = 0; j <= i; j++) {
		float s = 0;
		for (int k = 0; k < j; k++)
			s += L[i*N+k] * /*conj*/(L[j*N+k]);// комплексно-сопряженное число
		if (j == i) {
			L[i*N+j] = sqrt(A[i*N+j] - s);
		} else {
			L[i*N+j] = (A[i*N+j] - s)/L[j*N+j];
			L[j*N+i] = 0;// /*conj*/(L[i*N+j]);// можно не заполнять вовсе
		}
	}
}
/*! \brief LDL-разложение A = LDL^T
	\param LD - на диагонали распоолжены элементы матрицы D
LDL-разложение (модификация LL-разложения) положительно определенной матрицы A - это ее представление в виде произведения трех матриц:
A = L D L*
где:
D - диагональная матрица;
L - нижняя унитреугольная матрица.

При вычислении LDL-разложения, в отличии от LL-разложения, не требуется вычисление квадратных корней, а условием существования разложения является
отличие от нуля всех угловых миноров исходной матрицы A.

$LDL = L D_1 D_1^T L^T$

Theorem #1(Eigendecomposition). A square matrix A ∈ R^{n×n} can be factored into
A = P D P^{−1}, 
where P ∈ R^{n×n} and D is a diagonal matrix whose diagonal entries are
the eigenvalues of A, if and only if the eigenvectors of A form a basis of R^n.

Theorem #2. A symmetric matrix S ∈ R^{n×n} can always be diagonalized

Theorem #3. Given a matrix A ∈ R^{m×n}, we can always obtain a symmetric, positive semidefinite matrix S ∈ R^{n×n} by defining S := A^⊤ A .
*Remark*. If rk(A) = n, then S := A^⊤ A is symmetric, positive definite

Theorem #4(Spectral Theorem). If A ∈ R^{n×n} is symmetric, there exists an orthonormal basis of the corresponding vector space V consisting of
eigenvectors of A, and each eigenvalue is real.

 */
void cholesky_ldl_decomp(float * L, const float *A)
{
    //__builtin_bzero(L, N*N*sizeof(float));
    for (int i = 0; i < N; i++)
	for (int j = 0; j <= i; j++) {
		float s = 0;
		for (int k = 0; k < j; k++)
			s += L[k*N+k] * L[i*N+k]* /*conj*/(L[j*N+k]);// комплексно-сопряженное число
		if (j == i) {// диагональный элемент
			L[i*N+j] = (A[i*N+j] - s);
		} else {
			L[i*N+j] = (A[i*N+j] - s)/L[j*N+j];
			L[j*N+i] = 0;// /*conj*/(L[i*N+j]);// можно не заполнять вовсе
		}
	}
}

/*!	\brief Метод разложения Холецкого

Метод разложения Холецкого - это метод решения СЛАУ, в котором сначала определяется разложение Холецкого матрицы A, т.е. матрица L, а затем вычисляется
решение x последовательным решением двух уравнений:
L y = b
D z = y
L^T x = z

Условие применимости:
∙ A - положительно-определенная матрицы.

Несимметричность матрицы A не накладывает ограничений на применимость метода, но полученное решение будет не соответствовать истинному.

 */
void cholesky_ldl_solve(const float * L, float *z, const float* c)
{
	z[0] = c[0];
	for (int i=1; i<N; i++){//  L y = c
		float s = 0;
		for (int k=0; k<i; k++)
			s = fmaf(L[i*N+k],z[k],s);
		z[i] = c[i] - s;
	}
	for (int i=0; i<N; i++)		// D z = y
		z[i] /= L[N*i+i];
	for (int i=N-2; i>=0; i--){	// U x = z
		float s = 0;
		for (int k=i+1; k<N; k++)// использует только нижнюю треугольную матрицу
			s = fmaf(/*conj*/(L[k*N+i]),z[k],s);
		z[i] = z[i] - s;
	}
}
/*! \brief вычисление детерминта из разложения Холецкого

В случае A=LL^T разложения получается две треугольные матрицы
 det(A) = det(L)*det(L^T)
		= \prod_{k=1}^{N} l_{kk}^2
В случае A=LDL^T разложения
 det(A) = \prod_{k=1}^{N} d_{kk}
 */
float cholesky_ldl_det(const float * L){
	float d = L[0];
	for (int i=1; i<N; i++)
		d *= L[i*N+i];
	return d;
}

/*! Раздел полиномы, хотим реализовать шаблон класса полиномы с учетом замены операции 
- на множестве рациональных чисел (представленных дробью из целых и натуральных чисел).
- на множестве чисел в конечном поле с арифметикой Галуа 
- на множестве чисел по модулю большого числа
- на множестве вещественных чисел
- на множестве комплексных чисел
- на вероятностном множестве, когда сумма коэффициентов разложения дает единицу. 
- рациональные числа, корни полинома представлены в виде дроби P/Q так, что P/M+Q/M =1. 
 */
/*! \brief вычисление значения полинома s = r(x)
	\param a - вектор коэффициентов при степенях, a[0]x^n + a[1]x^n-1 + ... + a[n]
 */
static inline float v_poly_eval(const float* a, size_t len_a, float x)
{
	float s = 0;
	for(int i=0; i< len_a; i++)
		s = s*x + a[i];
	return s;
}
/*! \brief обнуление коэффициентов полинома
 */
static inline void v_poly_zero(float* r, size_t offs, size_t len){
	for (int i=offs; i< len; i++) r[i]=0;
}
/*! \brief сложение  полиномов r = r(x) + a(x)
 */
static inline void v_poly_add(float* r, const float* a, size_t len_a){
	for (int i=0; i< len_a; i++) r[i]+=a[i];
}
/*! \brief вычитание полиномов r = r(x) - a(x)
 */
static inline void v_poly_sub(float* r, const float* a, size_t len_a){
	for (int i=0; i< len_a; i++) r[i]-=a[i];
}
/*! \brief сложение  полиномов r = r(x) + c a(x)
	\param c - скаляр
 */
static inline void v_poly_mac1(float* r, const float* a, size_t len_a, float c){
	for (int i=0; i< len_a; i++) r[i]+=a[i]*c;
}
/*! \brief вычитание полиномов r = r(x) - c a(x)
	\param c - скаляр
 */
static inline void v_poly_mus1(float* r, const float* a, size_t len_a, float c){
	for (int i=0; i< len_a; i++) r[i]-=a[i]*c;
}
/*! \brief произведение на скаляр r(x) = c a(x)
	\param a - вектор коэффициентов при степенях, a[0]x^n + a[1]x^n-1 + ... + a[n]
	\param c - скаляр
 */
static inline void v_poly_mul1(float* r, const float* a, size_t len_a, float c){
	for (int i=0; i< len_a; i++) r[i] =a[i]*c;
}
/*! \brief произведение полиномов 
	\param a  - вектор коэффициентов при степенях, a[0]x^n + a[1]x^n-1 + ... + a[n]
	Степень полинома на выходе len_a + len_b -1
 */
void v_poly_mul(float* r, const float* a, size_t len_a, const float* b, size_t len_b)
{
	v_poly_mul1(r, a, len_a, b[0]);
	v_poly_zero(r, len_a, len_a + len_b - 1);
	for (int i=1; i<len_b; i++)
		v_poly_mac1(r+i, a, len_a, b[i]);
}
/*! \brief деление полиномов, понижение степени полинома 
	На выходе r содержит q - результат деления в первой части (len_r - (len_a -1))
	и остаток от деления в последних (len_a-1) числах
	\param r - вектор коэффициентов при степенях
	\param len_r - степень полинома r+1, длина вектора
	\param a - вектор коэффициентов при степенях, a[0]x^n + a[1]x^n-1 + ... + a[n], где a[0]=1
	\param len_a - степень полинома a+1, длина вектора 
 */
void v_poly_div(float* r, size_t len_r, const float* a, size_t len_a)
{
	for (int i=0; i<=(len_r - len_a); i++){
		float q = r[i];
		if (q!=0) 
			v_poly_mus1(r+i, a, len_a, q);
		r[i] = q;
	}
}
/*! \brief нахождение собственных векторов и собственных значений матрицы

	Решение существует если P(r) = det(A - 𝜆I)=0 
	det(A − 𝜆I) = det(U - 𝜆L^{-1})
	можно составить полином и найти корни полинома - собственные числа 𝜆_n.

	Нахождение корней полинома...
Основная теорема алгебры утверждает, что каждый многочлен над полем комплексных чисел 
представим в виде произведения линейных многочленов, причём единственным образом 
с точностью до постоянного множителя и порядка следования сомножителей.
 */
#undef N

/*! Вычисление нормы суммы квадратов от колонки прямоугольной матрицы
	\param n - размер вектора
	\param m - размер по ширине, для векторов ширина принимается равной 1.
	\param off - номер столбца матрицы
 */
static float norm_col(const float *a, int m, int n, int j)
{
	float s=0;
	for(int i=0; i<m; i++) s = fmaf(a[i*n + j], a[i*n + j], s);
	return sqrtf(s);
}
/*! Вычисление скалярного произведения от двух колонок матрицы
	\param m - число строк
	\param n - число столбцов, m x n размер матрицы
	\param j - номер колонки
	\param k - номер колонки
 */
static float dot_col(const float *a, int m, int n, int j, int k)
{
	float s=0;
	for(int i=0; i<m; i++) s = fmaf(a[i*n + j], a[i*n + k], s);
	return s;
}
static float dot_row(const float *A, unsigned m, unsigned n, unsigned j, unsigned k)
{
	const float *a = A + n*j;
	const float *b = A + n*k;
	float s=0;
	for(unsigned i=0; i<m; i++) s = fmaf(a[i], b[i], s);
	return s;
}
static
void scal_col(float d, float* a, unsigned M, unsigned N, int k){
	for (int j=0; j<M; ++j)
		a[j*N+k] *= d;
}
static
void scal_row(float d, float* a, unsigned M, unsigned N, int k){
	for (int j=0; j<N; ++j)
		a[k*N+j] *= d;
}
static
void pow_col(float p, float* a, unsigned M, unsigned N, int k){
	for (unsigned j=0; j<M; ++j)
		a[j*N+k] = ldexpf(a[j*N+k], p);
}
static
void pow_row(float p, float* a, unsigned M, unsigned N, int k){
	for (unsigned j=0; j<N; ++j)
		a[k*N+j] = ldexpf(a[k*N+j], p);
}
/*! \brief определяет симметричную матрицу A = Aᵀ */
int  is_symmetric(const float* A, unsigned N){
	unsigned i,j;
	for (i=0; i<N-1; i++)
		for (j=i+1; j<N; j++){// по строке
			if (A[i*N+j]!=A[j*N+i]) return 0;
		}
	return 1;
}
/*! \brief определяет анти-симметричную матрицу A =-Aᵀ */
int  is_skew_symmetric(const float* A, unsigned N){
	unsigned i,j;
	for (i=0; i<N; i++)
		for (j=i; j<N; j++){// по строке
			if (A[i*N+j]!=-A[j*N+i]) return 0;
		}
	return 1;
}
/*! \brief определяет эрмитову матрицу - комплексно сопряженная транспонированная 
	доделать
 */
int  is_hermitian(const float* A, unsigned N){
	unsigned i,j, lda = N+N;
	for (i=0; i<N; i+=1)
		for (j=i; j<i; j+=1){// по строке
			if (/*conj*/A[i*lda+j]!=A[j*lda+i] || A[i*lda+j+1]!=-A[j*lda+i+1]) return 0;
		}
	return 1;
}
int  is_unitary(const float* A, unsigned N){
	return 1;
}
/*! \brief определяет ортогональную матрицу 

квадратная матрица Q называется ортогональной, если QᵀQ = I. 
квадратная матрица U называется унитарной, если UᴴU=I
 */
int is_orthogonal(const float* A, unsigned N){
	const float eps = N*__FLT_EPSILON__;
	const unsigned int M = N;
	unsigned i,j;
	for (j=0; j<N; j++)
	for (i=0; i<N; i++){
		float s = dot_col(A, M, N, i,j);// dot product column vectors
//		float r = dot_row(A, M, N, i,j);// dot product row vectors
//		if(fabsf(r-s)>=eps) return 0;
		if(i!=j && fabsf(s-0.0f)>=eps) {
			//printf("i,j=%d,%d\n", i,j);
			return 0;
		}
		if(i==j && fabsf(s-1.0f)>=eps) {// нормированные столбцы
			//printf("i,j=%d,%d %1.8f\n", i,j, s);
			return 0;
		}
	}
	return 1;
}
/*! \brief определяет ортогональную симплектическую структуру матрицы

Матрица A размером 2N×2N состоит из блоков симметричных и антисимметричных матриц N×N
[V U; -U V]

 */
int is_symplectic (const float* A, unsigned N){
 	unsigned lda = N+N;
	unsigned i, j;
	const float *a,*b;
	a = A+N; b = A+N*lda;// анти-диагональные блоки
	for (i=0; i<N; ++i)
	for (j=0; j<N; ++j)
		if(a[i*lda+j] != -b[i*lda+j]) return 0;
	a = A; b = A+N*lda+N;// диагональные блоки
	for (i=0; i<N; ++i)
	for (j=0; j<N; ++j)
		if(a[i*lda+j] != b[i*lda+j]) return 0;
	return 1;
}
/*! \brief определяет гамильтонову структуру блочной матрицы H = [A B; C D]
	1. анти-диагональные блоки симметричны C=Cᵀ, B=Bᵀ
	2. диагональные блоки A + Dᵀ = 0
 */
int is_hamiltonian (const float* H, unsigned N){
	unsigned lda = N+N;
	unsigned i, j;
	const float *a, *b;
	a = H+N; b = H+N*lda;// анти-диагональные блоки
	for (i=0; i<N-1; ++i)
	for (j=i+1; j<N; ++j) {// симметричны A=Aᵀ, B=Bᵀ
		if(a[j*lda+i] != a[i*lda+j]) return 0;
		if(b[j*lda+i] != b[i*lda+j]) return 0;
	}
	a = H; b = H+N*lda+N;// диагональные блоки A + Bᵀ = 0
	for (i=0; i<N; ++i)
	for (j=0; j<N; ++j)
		if(a[j*lda+i] + b[i*lda+j] != 0) return 0;
	return 1;
}
int is_equal(const float* A, const float* B, unsigned M, unsigned N, unsigned lda){
	unsigned i, j;
	for (i=0; i<M; ++i)
	for (j=0; j<N; ++j){
		if(A[i*lda+j] != B[i*lda+j]) return 0;
	}
	return 1;
}
/*! 
	если матрица состоит из нормированных векторов-столбцов ‖A(:,j)‖ = 1
 */
int is_orthonormal(const float* A, unsigned M, unsigned N){
	const float eps = N*__FLT_EPSILON__;
	for (int i=0; i<N; i++){
		float s = norm_col(A, M, N, i);
		if(fabsf(s-1.0f)>=eps) return 0;
	}
	return 1;
}
/*! 
	матрица нормальная, если выполнено условие Aᵀ·A = A·Aᵀ. 
	Для комплексных матриц Aᴴ·A = A·Aᴴ.
 */
int is_normal(const float* A, unsigned N){
	const float eps = N*__FLT_EPSILON__;
	const unsigned M = N;
	unsigned i,j,k;
	for (i=0; i<N; i++){
		for (j=0; j<N; j++){
			float s = dot_col(A, M, N, i,j);// dot product column vectors
			float r = dot_row(A, M, N, i,j);
			if(fabsf(r-s)>=eps) return 0;
		}
	}
	return 1;
}
/*! QR- разложение прямоугольной вещественной матрицы 
	Q - ортогональная, R- верхняя треугольная



Свойство ортогональной матрицы $Q^T Q = I$
Ортогональная матрица сохраняет норму |Qx| = |x|

 Any nonsingular matrix A can be factored, A = QR, into the product
of an orthogonal matrix Q and an upper triangular matrix R. The factorization is unique
if all the diagonal entries of R are assumed to be positive.

<https://www-users.cse.umn.edu/~olver/aims_/qr.pdf>

Theorem 5.1. Any nonsingular matrix A can be factored, A = QR, into the product of 
an orthogonal matrix Q and an upper triangular matrix R. 
The factorization is unique if all the diagonal entries of R are assumed to be positive.

Algorithm 5.2.6 (Modified Gram-Schmidt) This algorithm requires 2mn2 flops. 
A \in R^{m \times n} 
R \in R^{n \times n} upper triangular
*/

int qr_decomp(float* a, float* r, unsigned M, unsigned N)
{
	float d;
	unsigned i,j,k;
	for(j=0; j<N;j++){// по колонкам
		for (k=0; k<j; k++) r[j*N+k] = 0;
		r[j*N+j] = d = norm_col(a, M, N, j);
		//if (r[j*N+j]==0) return j;// линейно зависимая колонка
		if (d!=0) scal_col(1/d, a, M ,N, j);
		for (k=j+1; k<N; k++) {
			r[j*N+k] = d = dot_col(a, M, N, j, k);
			for (i=0; i<M; i++) // column(k)-= u(j)*(a_j^T a_k)// y = y -d x
				a[i*N+k] -= a[i*N+j]*d;
		}
	}
	return -1;
}
/*! \brief QR- разложение прямоугольной вещественной матрицы (классический алгоритм Грама-Шмидта)

	Q - ортогональная, R- верхняя треугольная

Algorithm 5.2.7 (Classical Gram-Schmidt)  [Golub & van Loan]
*/
int qr_cgs_decomp(float* a, float* r, unsigned M, unsigned N)
{
	float d;
	r[0*N+0] = d = norm_col(a, M, N, 0);
	if (d!=0.0f) scal_col(1/d, a, M ,N, 0);
	for (int k=1; k<N; ++k){
		for (int i=0; i<k; ++i)  // R(1:k−1, k) = Q(1:m, 1:k−1)^T A(1:m, k)
			r[i*N+k] = dot_col(a, M, N, i, k);
		for (int j=0; j<M; ++j){
			float s = 0;
			for (int i=0; i<k; ++i)
				s = fmaf(a[j*N+i],r[i*N+k],s);
			a[j*N+k] -= s;
		}
		r[k*N+k] = d = norm_col(a, M, N, k);
		scal_col(1/d, a, M ,N, k);
	}
	return 0;
}
static inline
float givens(float a, float b, float*c, float*s){
	float r = sqrtf(a*a + b*b);
	float d = 1.0f/r;
	*c =  a*d; *s = b*d;
	return r;
}
/* \see Golub & van Loan. 5.1.9 Applying Givens Rotations

Я немного доработал алгоритм, поменял знак синуса и цикл стал не от нуля
*/
static 
void givens_left (float* a, unsigned  N, unsigned i, unsigned k, float c, float s, float v){
	a[i*N+i] = v;
	a[k*N+i] = 0;
	for (unsigned j=i+1; j < N; ++j){// цикл по строке
		float t1 = a[i*N+j];
		float t2 = a[k*N+j];
		a[i*N+j] = c*t1 + s*t2;
		a[k*N+j] = c*t2 - s*t1;
	}
}
static 
void givens_right(float* a, unsigned N, unsigned i, unsigned k, float c, float s, float r){
	a[i*N+i] = r;
	a[i*N+k] = 0;
	for (unsigned j=i+1; j < N; ++j){// цикл по столбцу матрицы
		float t1 = a[j*N+i];
		float t2 = a[j*N+k];
		a[j*N+i] = c*t1 + s*t2;
		a[j*N+k] = c*t2 - s*t1;
	}
}

/*! Получение QR разложения методом вращения Гивенса 
доделать Q = 
 */
int qr_givens(float* a, float* q, unsigned M, unsigned N)
{
	unsigned i,j;
	float c,s;
	if(q!=0){
		_set_identity(q, N,N);
	}
	for (i=1; i < M; ++i){// номер строки
		for (j=0; j < i; ++j){// номер столбца x
			//if (a[i*N+j]==0) continue;
			float v = givens(a[j*N+j], a[i*N+j], &c,&s);
			givens_left(a, N, j, i, c, s, v);
			if (q!=NULL) {
				givens_right(q, N, j, i, c, s, v);
			}
			printf("[%d;%d]\n", i,j);
			print_mn(a, M, N);
		}
	}
	return 0;
}

typedef float  Ftype;
typedef struct _vector vector_t;
typedef struct _matrix matrix_t;
struct _vector {
	unsigned N;		//!< число элементов
	unsigned stride;
	Ftype * data;
};
struct _matrix {
	unsigned M,N;	//!< размер матрицы M-строк, N - столбцов
	unsigned lda;	//!< размер строки матрицы с учетом выравнивания
	Ftype * data;	//!< данные
};
struct _tensor {
	unsigned typ;	//!< тип матрицы, HE, SY, TR, ...; формат упаковки Q8_0 Q8_1 F16, BF16, F32... квантизация
	unsigned M,N;	//!< размер матрицы M-строк, N - столбцов
	unsigned lda;	//!< размер строки матрицы с учетом выравнивания
	Ftype * data;	//!< данные
};
static inline
matrix_t _submatrix(Ftype* a, unsigned i, unsigned j, unsigned M, unsigned N, unsigned lda){
	return (matrix_t){.lda = lda, .M = M, .N = N, .data = a + i*lda + j};
}
/*! класс вектор - это то что выделено из матрицы (подстрока или колонка)*/
static inline
vector_t _subrow(Ftype* a, unsigned i, unsigned j, unsigned size, unsigned N){
	return (vector_t){.data = a+i*N+j, .N = size, .stride = 1};
}
static inline
vector_t _subcolumn(Ftype* a, unsigned i, unsigned j, unsigned size, unsigned N){
	vector_t v = {.data = a+i*N+j, .N = size, .stride = N};
	return v;
}
/*! выделить диагональный вектор из матрицы */
static inline
vector_t _diag(Ftype* a, unsigned i, unsigned j, unsigned size, unsigned N){
	vector_t v = {.data = a+i*N+j, .N = size, .stride = N+1};
	return v;
}

static inline
vector_t _subvector(Ftype* x, unsigned i, unsigned size, unsigned N){
	return (vector_t){.data = x+i*N+0, .N = size, .stride = N};
}
static inline
void  _vector_set(vector_t *v, unsigned i, Ftype value){
	v->data[i*v->stride] = value;
}
static inline
Ftype _vector_get(vector_t *v, unsigned i, Ftype value){
	return v->data[i*v->stride];
}
static inline
Ftype _vector_exchange(vector_t *v, unsigned i, Ftype value){
	Ftype *r = v->data+i*v->stride;
	Ftype vi = *r;
	*r = value;
	return vi;
}
static
void _set_zero(Ftype* a, unsigned M, unsigned N, unsigned lda){
	unsigned i, j;
	for (i=0; i<M; i++)
	for (j=0; j<N; j++)
		a[i*lda+j] = 0;
}
static
void _set_identity(Ftype* a, unsigned M, unsigned N){
	for (unsigned i=0; i<M; ++i)
	for (unsigned j=0; j<N; ++j)
		a[i*N+j] = 0;
	if (M>N) M = N;
	for (unsigned i=0; i<M; ++i)
		a[i*N+i] = 1;
}
// Методы BLAS по шаблону для класса Ftype
#define BLAS(n) cblas_##n
#define CblasNoTrans 0
#define CblasTrans 	 1
static 
Ftype  BLAS(nrm2)(const Ftype *v, unsigned N, unsigned stride){
	double sum = 0.0f;
	for (unsigned i=0; i<N; ++i)
		sum += v[i*stride]*v[i*stride];
	return sqrtf(sum);
}
static 
Ftype  BLAS(asum)(const Ftype *v, unsigned N, unsigned stride){
	double sum = 0.0f;
	for (unsigned i=0; i<N; ++i)
		sum += fabsf(v[i*stride]);
	return sum;
}
static 
void   BLAS(scal)(Ftype s,  Ftype *v, unsigned N, unsigned stride){
	for (unsigned i=0; i<N; ++i) v[i*stride] *= s;
}
static 
void   BLAS(axpy)(Ftype alpha, const Ftype *x, int dx, 
Ftype *y, int dy, unsigned N){
	for (unsigned i=0; i<N; ++i)
		y[i*dy] += alpha*x[i*dx];
}
static 
void   BLAS(lerp)(const Ftype *mu, int dm, const Ftype *x, int dx, 
Ftype *y, int dy, unsigned N){
	for (unsigned i=0; i<N; ++i)
		y[i*dy] += mu[i*dm] * (x[i*dx] - y[i*dy]);
}
static 
void  BLAS(ger)(Ftype beta, const Ftype *X, int dx, const Ftype*Y, int dy, 
Ftype*A, int lda, unsigned M, unsigned N)
{
    for (unsigned j = 0; j < M; j++) {
      const Ftype tmp = beta * Y[j*dy];
      for (unsigned i = 0; i < N; i++)
        A[i + lda * j] += X[i*dx] * tmp;
    }
}
static
void BLAS(gemv)(Ftype beta,  Ftype* Y, int dy,
Ftype alpha, Ftype* X, int dx,
Ftype*A, int lda, unsigned M, unsigned N, int Trans)
{
	unsigned lenX, lenY;
  if (Trans == CblasNoTrans) {
    lenX = N;
    lenY = M;
  } else {
    lenX = M;
    lenY = N;
  }
	if (beta==0.0f){
		for (unsigned i=0; i< lenY; i++) Y[i*dy] = 0;
	} else if (beta!=1.0f){
		for (unsigned i=0; i< lenY; i++) Y[i*dy]*= beta;
	}
	if (alpha==0.0) return;
  if (Trans == CblasTrans){ /* form  y := alpha*Aᵀ*x + y */
    for (unsigned j = 0; j < lenX; j++) {
      const Ftype temp = alpha * X[j*dx];
      if (temp != 0.0) {
        for (unsigned i = 0; i < lenY; i++)
          Y[i*dy] = fmaf(temp , A[lda * j + i], Y[i*dy]);
      }
    }		
  } else {/* form  y := alpha*A*x + y */
    for (unsigned i = 0; i < lenY; i++) {
      Ftype temp = 0.0;
      for (unsigned j = 0; j < lenX; j++)
        temp = fmaf(X[j*dx], A[lda * i + j], temp);
      Y[i*dy] += alpha * temp;
    }
  }
}

#define GSL_SIGN(x)    ((x) >= 0.0 ? 1 : -1)
static 
Ftype house (Ftype *v, int N, unsigned stride){
	if (N==1) return 0.0f;
	Ftype vdot=0;
	for(unsigned i=1; i< N;++i) vdot = fmaf(v[i*stride], v[i*stride],vdot);// nrm2
//	if (norm==0.0f) return 0.0f;
	Ftype alpha = v[0];
	Ftype beta = GSL_SIGN(alpha)*sqrtf(alpha*alpha+vdot);//sqrtf(alpha*alpha + norm*norm);
	Ftype s  =(alpha+beta);
	v[0] = -beta;
	if (s!=0.0f) for(unsigned i=1; i< N;++i) v[i*stride] /= s;// scal
	return (alpha+beta)/beta;
}
static
void house_mh  (Ftype beta, vector_t* v, matrix_t* A){
	if (beta==0.0f) return;
	unsigned i, j, k, M = A->M, N = A->N, lda = A->lda;
	for (i = 0; i < M; i++){
		Ftype w = A->data[i*lda+0];
		for (j=1; j<N; j++) // sdot
			w = fmaf(A->data[i*lda+j],v->data[v->stride*j], w);
		w *= -beta;
		A->data[i*lda+0] += w;
		for (j=1; j<N; j++) /* A = A - tau w v' */
			A->data[i*lda+j] += w*v->data[v->stride*j];
	}
}
static
void house_hm  (Ftype beta, vector_t* v, matrix_t* A){
	if (beta==0.0f) return;
	unsigned i, j, k, M = A->M, N = A->N, lda = A->lda;
	for (j = 0; j < N; j++){
		Ftype w = A->data[0*lda+j];
		for (k=1; k<M; k++) // sdot
			w = fmaf(A->data[k*lda+j],v->data[v->stride*k],w);
		w *= -beta;
		A->data[0*lda+j] += w;
		for (i=1; i<M; i++) /* Aij = Aij - tau vi wj */
			A->data[i*lda+j] = fmaf(w,v->data[v->stride*i], A->data[i*lda+j]);
	}
}
static
void house_hm1 (Ftype beta, vector_t* v, matrix_t* A){
	unsigned i, j, M = A->M, N = A->N, lda = A->lda;
	if (beta==0.0f) {
		A->data[0] = 1;
		for (j=1; j<N; ++j) A->data[0*lda+j] = 0;
		for (i=1; i<M; ++i) A->data[i*lda+0] = 0;
		return;
	}
	for (j = 0; j < N; j++){
		Ftype w = 0;//A->data[0*lda+j];
		for (i=1; i<M; i++) // sdot
			w = fmaf(A->data[i*lda+j],v->data[v->stride*i],w);
		w *= -beta;
		A->data[0*lda+j] = w;
		for (i=1; i<M; i++) /* Aij = Aij - tau vi wj */
			A->data[i*lda+j] = fmaf(w,v->data[v->stride*i], A->data[i*lda+j]);
	}
	A->data[0*lda+0] = 1-beta;
	for (i=1; i<M; i++)
		A->data[i*lda+0] *= -beta;
}
static
void house_left(Ftype beta, vector_t* v, matrix_t* A,  vector_t* w){

	if (beta==0.0f) return;
	Ftype v0 = v->data[0]; v->data[0] = 1.0f;
#if 1
	BLAS(gemv)(0.0f, w->data, w->stride, 1.0f, v->data, v->stride, A->data, A->lda, A->M, A->N, CblasTrans);// w = A^T v
	BLAS(ger) (-beta, w->data, w->stride, v->data, v->stride, A->data, A->lda, A->M, A->N); //A = A - beta v (v^T A) = A - beta v (A^T v)^T
#else // без использования BLAS
	const unsigned M = A->M;
	const unsigned N = A->N;
	const unsigned lda = A->lda;
	for (int k=0;k<N; k++){
		Ftype s = 0;
		for (int i=0;i<M; ++i){
			s += A->data[i*lda+k]*v->data[i*v->stride];
		}
		w->data[k*w->stride] = s;
	}
	for (int i=0;i<M; ++i){
		Ftype d = -beta*v->data[i*v->stride];
		for (int k=0;k<N; k++)
			A->data[i*lda+k] += d*w->data[k*w->stride];
	}
#endif
	v->data[0] = v0;
}
static
void house_right(Ftype beta, vector_t* v, matrix_t* A,  vector_t* w){
	if (beta==0.0f) return;
	Ftype v0 = v->data[0]; v->data[0] = 1.0f;
	BLAS(gemv)(0.0f, w->data, w->stride, 1.0f, v->data, v->stride, 
	A->data, A->lda, A->M, A->N, CblasNoTrans);// w = A^T v
	BLAS(ger) (-beta, v->data, v->stride, w->data, w->stride, A->data, A->lda, A->M, A->N); 
	//A = A - beta v (v^T A) = A - beta v (A^T v)^T
	v->data[0] = v0;
}
/*! \brief QR разложение методом отражения Хаусхолдера */
int qr_house(Ftype * a, Ftype * tau, unsigned M,  unsigned N)
{
	const unsigned lda = N;
	for (int j=0; j< N; ++j){
		vector_t v = _subcolumn(a, j, j, M-j, lda);
		Ftype tau_j = house(a+lda*j+j, M-j, lda);
		tau[j] = tau_j;
		if (j+1<M){
			vector_t w = _subvector(tau,  j+1, N-(j+1), 1);
			matrix_t m = _submatrix(a, j, j+1, M-j, N-(j+1), lda);
			house_left(tau_j, &v, &m, &w);
		}
	}
	return 0;
}
static 
void BLAS(gemm)(int flags,  Ftype alpha, matrix_t* A, matrix_t* B, Ftype beta, matrix_t* Y){
	unsigned i,j,k;
	const unsigned M = Y->M;
	const unsigned N = Y->N;
	const unsigned L = B->M;
	if (flags == CblasTrans){
		for (i=0; i<M; i++)
		for (j=0; j<N; j++){
			float s=0;
			for (k=0; k<L; k++)
				s += A->data[k*A->lda+i]*B->data[k*B->lda+j];
			Y->data[i*Y->lda+j] = beta*Y->data[i*Y->lda+j] + alpha*s;
		}
	} else {
		for (i=0; i<M; i++)
		for (j=0; j<N; j++){
			float s=0;
			for (k=0; k<L; k++)
				s += A->data[i*A->lda+k]*B->data[k*B->lda+j];
			Y->data[i*Y->lda+j] = beta*Y->data[i*Y->lda+j] + alpha*s;
		}
	}
}
#if 1
/*! \brief QR-decomposition (Modified Gram-Schmidt) повторно
 */
static
int qr_mgs_decomp(float* a, float* r, unsigned M, unsigned N, unsigned lda)
{
	float d;
	unsigned i,j,k;
	for(j=0; j<N;j++){// по колонкам
		for (k=0; k<j; k++) r[j*lda+k] = 0;
		r[j*lda+j] = d = norm_col(a, M, lda, j);
		//if (r[j*N+j]==0) return j;// линейно зависимая колонка
		if (d!=0) scal_col(1/d, a, M, lda, j);
		for (k=j+1; k<N; k++) {
			r[j*lda+k] = d = dot_col(a, M, lda, j, k);
			for (i=0; i<M; i++) // column(k)-= u(j)*(a_j^T a_k)// y = y -d x
				a[i*lda+k] -= a[i*lda+j]*d;
		}
	}
	return 0;
}

/*! \brief блочный рекурсивный алгоритм QR
	\param a матрица MxN
	\param r матрица NxN
	\return 0 - если процесс завершен
	\see Golub & van Loan. Algorithm 5.2.3 (Recursive Block QR)
 */
int qr_block(matrix_t* a, matrix_t* r)
{
	const unsigned Nb= 2; 
	const unsigned M = a->M;
	const unsigned N = a->N;
	
	const unsigned N1 = N/2;// SPLIT(N)
	if (N1<Nb) // порог перехода к линейному алгоритму
		return qr_mgs_decomp(a->data, r->data, M, N, a->lda);
	matrix_t A1 = _submatrix(a->data, 0,  0,   M,   N1, a->lda);
	matrix_t A2 = _submatrix(a->data, 0, N1,   M, N-N1, a->lda);
	matrix_t R11= _submatrix(r->data, 0,  0,   N1,  N1, r->lda);
	matrix_t R12= _submatrix(r->data, 0, N1,   N1,N-N1, r->lda);
	matrix_t R22= _submatrix(r->data, N1,N1, N-N1,N-N1, r->lda);
	_set_zero(r->data+r->lda*N1, N-N1, N1, r->lda);// R21
	qr_block(&A1, &R11);
	//print_mn(A1.data, N1, N);
	BLAS(gemm)(CblasTrans,    1, &A1, &A2, 0, &R12); // R_{12} = Q_1^T A_2
	BLAS(gemm)(CblasNoTrans, -1, &A1, &R12,1, &A2); //  A_2  = A2 - Q_1 R_{12}
	qr_block(&A2, &R22);
	return 0;
}
void qr_block_decomp(float* a, float* r, unsigned M, unsigned N){
	const unsigned Nb= 4; 
	if (N<4){ // порог перехода к линейному алгоритму
		qr_mgs_decomp(a, r, M, N, N);
		return;
	}
	matrix_t A = _submatrix(a, 0, 0, M, N, N);
	matrix_t R = _submatrix(r, 0, 0, N, N, N);
	qr_block(&A, &R);
}
#endif
/*! \brief Восстановить матрицу Q и R из компактной записи QR разложения, 
полученной методом вращений Хаусхолдера 
	\param r - матрица R может быть NULL
*/
int qr_house_unpack(Ftype * a, Ftype * tau,  Ftype * q, Ftype * r, unsigned M,  unsigned N){
	unsigned i,j,k;
	unsigned lda = N;
	_set_identity(q, M, N);
	for (j=N; j-- >0;){
		vector_t v = _subcolumn(a, j, j, M-j, lda);
		matrix_t m = _submatrix(q, j, j, M-j, N-j, lda);
		house_hm(tau[j], &v, &m);
	}
	if (r!=NULL){
		for (i=0; i<N; ++i){
			for (j = 0; j < i && j < N; j++)
				r[i*N+j] = 0;
			for (j = i; j < N; j++)
				r[i*N+j] = a[i*lda+j];
		}
	}
}

/*! \brief Би-диагонализация
	\param tau - вектор размер N
	\param tav - вектор размер M
	\param M - число строк матрицы a
	\param N - число столбцов матрицы a
 */
void qr_house_bidi(Ftype * a, Ftype * tau, Ftype * tav, unsigned M,  unsigned N)
{
	const unsigned lda = N;
	for (unsigned j=0; j< N; ++j){
		vector_t v = _subcolumn(a, j, j, M-j, lda);
		Ftype tau_j = house(a+lda*j+j, M-j, lda);
		tau[j] = tau_j;
		if (j+1<N){		
			vector_t w = _subvector(tau,  j, N-(j+1), 1);
			matrix_t m = _submatrix(a, j, j+1, M-j, N-(j+1), lda);
			//float v0 = _vector_exchange(&v, 0, 1.0f);
			house_left(tau_j, &v, &m, &w);
			//_vector_set(&v, 0, v0);
		}

		if (j+1<N){
			vector_t v = _subrow(a, j,  j+1, N-j-1, lda);
			Ftype tau_j = house(a+lda*j+j+1, N-j-1, 1);
			if (j+1<M) {
				vector_t w = _subvector(tav,  j+1, M-(j+1), 1);
				matrix_t m = _submatrix(a, j+1, j+1, M-(j+1), N-(j+1), lda);
				house_right(tau_j, &v, &m, &w);
			}
			tav[j] = tau_j;
		}
		print_mn(a, M, N);
	}
}
/*! \brief Би-диагонализация
	\param a - на входе результат разложения UBV^T. На выходе матрица U (MxN)
	\param tau - вектор размер N, туда помещается диагональ B (N)
	\param tav - вектор размер N, туда копируются элемента над диагональю B (N-1)
	\param V - матрица NxN 
	\param M - число строк матрицы a
	\param N - число столбцов матрицы a
 */
void qr_house_bidi_unpack(Ftype * a, Ftype * tau, Ftype * tav, Ftype * U, Ftype * V, unsigned M,  unsigned N)
{
	unsigned i,j,k;
	unsigned lda = N;
	if (V!=NULL) {
		_set_identity(V, N, N);
		for (j=N-1; j-- >0;){
			vector_t v = _subrow   (a, j,   j+1, N-(j+1), lda);
			matrix_t m = _submatrix(V, j+1, j+1, N-(j+1), N-(j+1), lda);
			house_hm(tav[j], &v, &m);
		}
	}
	/* Копировать наддиагональные элементы в tav */
	for (j = 0; j < N - 1; j++)
		tav[j] = a[j*lda+j+1];
	// if U != A
	if (U==a || U==NULL){
		for (j=N; j-- >0;){
			vector_t v = _subcolumn(a, j, j, M-j, lda);
			matrix_t m = _submatrix(a, j, j, M-j, N-j, lda);
			Ftype beta = tau[j];
			tau[j] = a[j*lda+j];// копировать диагональные элементы
			house_hm1(beta, &v, &m);
		}
	} else {
		_set_identity(U, M, N);
		for (j=N; j-- >0;){
			vector_t v = _subcolumn(a, j, j, M-j, lda);
			matrix_t m = _submatrix(U, j, j, M-j, N-j, lda);
			Ftype beta = tau[j];
			tau[j] = a[j*lda+j];// копировать диагональные элементы
			house_hm(beta, &v, &m);
		}
	}
}
/*! \brief вычисление детерминанта после QR разложения. 

det(A)=det(Q)det(R) = prod R_{ii}, det(Q)=1 
 */
float qr_det(float* r, unsigned N){
	float d = 1;
	for (unsigned i=0; i<N; i++, r+=N+1)
		d *= *r;
	return d;
}
/*! \brief решение системы уравнений Ax=b методом QR 
	\param a матрица M x N заменяется на R, верхнюю треугольную
	\param x вектор размером M должен содержать на входе `b`, на выходе `x`
 */
int qr_house_svx(Ftype * a, Ftype * x, unsigned M,  unsigned N){
	for (unsigned i=0; i<N; i++){
		Ftype aii = a[i*N+i];
		Ftype r=0;
		for (unsigned k=i+1; k<M; k++){// норма (dot) вектора A(i+1:m, i)
			Ftype aki = a[k*N+i];
			r += aki*aki;
		}
		if (r==0.0f) return i;
		r += aii*aii;// v^T v
		Ftype alpha = copysignf(sqrtf(r),aii);//*GSL_SIGN(aii); 
		Ftype ak = 1.0f/(r+alpha*aii);
		
		a[i*N+i] = aii+alpha;
		for (unsigned k = i + 1; k < N; k++){
			Ftype f =0;
			for (unsigned j = i; j < M; j++)// скалярное произведение dot_col(a, k, i)
				f += a[j*N+k]*a[j*N+i];
			f *= ak;
			for (unsigned j = i; j < M; j++)
				a[j*N+k] -= f*a[j*N+i];
		}
		/* Perform update of right-hand side `b` */
		Ftype f = 0;
        for (unsigned j = i; j < M; j++)
        	f += x[j] * a[j*N+i];
		f *= ak;
		for (unsigned j = i; j < M; j++)
			x[j] -= f*a[j*N+i];

		a[i*N+i] =-alpha;
	}
    /* Perform back-substitution. */

    for (unsigned i = N; i-- > 0;)
    {
        Ftype sum = 0.0;
        for (unsigned k = i + 1; k < N; k++)
            sum += a[i*N+k] * x[k];
        x[i] = (x[i] - sum) / a[i*N+i]; // d[i] = -alpha_i
    }
}
/*! \brief QR разложение методом отражения Хаусхолдера (2)
	\param a - матрица размером M x N заменяется на упакованный результат QR -разложения,
	где R - Верхняя треугольная матрица, а нижняя часть содержит вектора Хаусхолдера.
	\param tau - вектор размером N для хранения промежуточных результатов
	\param M - число строк
	\param M - число столбцов матрицы
 */
int qr_house2(Ftype * a, Ftype* tau, unsigned M,  unsigned N)
{
	unsigned i,j,k;
	for (i=0; i<N; i++){
		Ftype aii = a[i*N+i];
		Ftype r=0;
		for (k=i+1; k<M; k++){// норма (dot) вектора A(i+1:m, i)
			Ftype aki = a[k*N+i];
			r += aki*aki;
		}
		if (r==0.0f) return i;
		r += aii*aii;// v^T v
		Ftype alpha = copysignf(sqrtf(r),aii);//*GSL_SIGN(aii); 
		Ftype ak = 1.0f/(r+alpha*aii);
		a[i*N+i] = aii+alpha;
		tau[i]   = (aii+alpha)/alpha;
		for (k = i + 1; k < N; k++){
			Ftype f =0;
			for (j = i; j < M; j++)// скалярное произведение dot_col(a, k, i)
				f += a[j*N+k]*a[j*N+i];
			f *= ak;
			for (j = i; j < M; j++)
				a[j*N+k] -= f*a[j*N+i];
		}
		Ftype s  = a[i*N+i];
		a[i*N+i] =-alpha; // диагональный элемент верхней треугольной матрицы
		/* сохранить вектор Хаусхолдера (не обязательно). 
		   Нижняя унитреугольная матрица содержит вектора {1.0, v[j+1:m]} */
		if (s!=0.0f) for (k = i + 1; k < N; k++) 
			a[k*N+i] /=s;
	}
	return 0;
}
/*! Algorithm 5.3.2 (Householder LS Solution)

Use Algorithm 5.2.1 to overwrite A with its QR factorization.
 */
void ls_house(Ftype * a, Ftype* tau, Ftype* b, unsigned M,  unsigned N)
{
	for (int j=0; j<N;++j){
		vector_t v = _subcolumn(a, j, j, M-j, N);
		Ftype v0 = _vector_exchange(&v, 0, 1.0f);
		Ftype beta = tau[j];
		_vector_set(&v, 0, v0);
	}
}
#if 0
/*! Решение СЛАУ Ax=b через QR 
	Q^T QRx = Q^T b => Rx = Q^T b
 */
/*! Сингулярное разложение матриц
	
 */
/*! \brief Бидиагональное разложение 


	Число операций 4mn^2 − 4n^3/3 flops
	\see Golub & Van Loan, Matrix Computations (3rd ed.).
 */
int qr_bidiag_decomp (matrix_t * A, vector_t * tau_U, vector_t * tau_V)
{
	const unsigned M = A->M;
	const unsigned N = A->N;
	const unsigned lda = A->lda;
	vector_t * tmp = _vector_alloc(M);

	for (size_t j = 0 ; j < N; j++)
	{	/* apply Householder transformation to current column */
		vector_t v = _subcolumn(A, j, j, M - j, lda);
		float tau_j = house(&v);

		if (j + 1 < N) {/* apply the transformation to the remaining columns */
			matrix_t m = _submatrix (A, j, j + 1, M - j, N - j - 1);
			vector_t w = _subvector(tau_U, j, N - j - 1);
			house_left (tau_j, &v, &m, &w);
		}
		_vector_set (tau_U, j, tau_j);            
		if (j + 1 < N) {/* apply Householder transformation to current row */
			v = _matrix_subrow (A, j, j + 1, N - j - 1);
			tau_j = house (&v);
			if (j + 1 < M) {/* apply the transformation to the remaining rows */
                matrix_t m = _submatrix(A, j + 1, j + 1, M - j - 1, N - j - 1);
                vector_t w = _subvector(tmp, 0, M - j - 1);
				house_right(tau_j, &v, &m, &w);
			}
			_vector_set (tau_V, j, tau_j);
		}
	}
    _vector_free(tmp);
	return 0;
}
#endif
/*! \brief Балансировка матрицы D^{-1}AD
	\param A матрица MxN
	\param D диагональные элементы матрицы D 
	\param M число строк, 
	\param N число столбцов
 */
void balance(Ftype* A, Ftype *D, unsigned M, unsigned N){
	Ftype c,r,s,f;
	int p;
	for (unsigned i=0; i<N; i++) D[i]=1;
	int converged=0;
	while( converged==0 ){
		converged = 1;
		for (unsigned i=0; i<N; i++){
			r = c = - fabsf(A[i*N+i]); 
			for (unsigned j=0; j<N; j++){// 1-norm
				//if (i!=j)
				{
					c += fabsf(A[j*N+i]);
					r += fabsf(A[i*N+j]);
				}
			}
			if (c==0 || r==0) continue;
			s = c+r;
			p = 0;
			if (c<r)
				while (ldexpf(c,p+1) < ldexpf(r,-p)) p++;
			else
				while (ldexpf(c,p-1) > ldexpf(r,-p)) p--;

			if ((ldexpf(c,p) + ldexpf(r,-p))<0.95*s){
				converged = 0;
				if (p!=0){
					f = ldexpf(1.0f, p);
					D[i] *= f;// можно сохранить только степень `p`
					// printf("f=%f\n", f);
					scal_row(1/f, A, M,N, i);
					scal_col(  f, A, M,N, i);
				}
			}
		}
	}
}
void unbalance(Ftype* A, Ftype *D, unsigned M, unsigned N){
	for (unsigned i=0; i<N; i++)
	{
		scal_col(1/D[i], A, M,N, i);
		scal_row(  D[i], A, M,N, i);
	}

}
/*! \brief Algorithm 2: Balancing (Parlett and Reinsch) с использованием 2-нормы

Note that $D^{−1}AD$ can be calculated without roundoff. 
When A is balanced, the computed eigenvalues are usually more accurate
although there are exceptions. See Parlett and Reinsch (1969) and Watkins(2006)
*/
void balance2(Ftype* A, Ftype *D, unsigned M, unsigned N){
	const Ftype b = 2;// where β is the floating point base. 
	const Ftype b2 = b*b;
	Ftype c,r,s;
	int p;
	for (unsigned i=0; i<N; i++) D[i]=1;
	int converged=0;
	while( converged==0 ){
		converged = 1;
		for (unsigned i=0; i<N; i++){
			r = c = - A[i*N+i]*A[i*N+i];// без диагонального элемента
			c += dot_col(A, M, N, i, i);
			r += dot_row(A, M, N, i, i);
			s = c+r;
			p = 0;
			while (c*b2 < r){
				c *= b2; r /= b2; p++;
			}
			while (c >= r*b2){
				c /= b2; r *= b2; p--;
			}
			if ((c+r)<0.95*s){
				converged = 0;
				D[i] = ldexpf(D[i],p);
				Ftype f = ldexpf(1.0f,p);
				//printf("f=%f\n", f);
				pow_row(-p, A, M,N, i);
				pow_col( p, A, M,N, i);
			}
		}
	} 
}
/*! \brief Algorithm 5.2.5 (Hessenberg QR) 
	\param H - матрица Хессенберга, с ненулевыми элементами под диагональю
 */
void Hessenberg_qr(Ftype* H, unsigned N){
	unsigned k;
	float c[N-1],s[N-1],v[N-1];
	for(k=0; k< N-1;++k){
		v[k] = givens(H[k*N+k], H[(k+1)*N+k], &c[k], &s[k]);
		givens_left(H, N, k, k+1, c[k],s[k],v[k]);
	}
}
/*! \brief Algorithm 7.4.1 (Hessenberg QR Step)

If H is an n-by-n upper Hessenberg matrix, then this algorithm
overwrites H with H+ = RQ where H = QR is the QR factorization of H
*/
void Hessenberg_qr_step(Ftype* H, unsigned N){
	unsigned k;
	float c[N-1],s[N-1],v[N-1];
	for(k=0; k< N-1;++k){//Algorithm 5.2.5 (Hessenberg QR) 
		v[k] = givens(H[k*N+k], H[(k+1)*N+k], &c[k], &s[k]);
		givens_left(H, N, k, k+1, c[k],s[k],v[k]);
	}
	for(k=0; k< N-1;++k){
		givens_right(H, N, k, k+1, c[k],s[k],v[k]);
	}
}
/*! \brief Algorithm 7.4.2 (Householder Reduction to Hessenberg Form)
 */
void Hessenberg_reduction(Ftype* H, unsigned N){
	unsigned k, lda = N;
	for (k=0; k<N-2;++k){
		vector_t v = _subcolumn(H, k+1, k, N-(k+1), lda);
		Ftype b = house(H+(k+1)*lda+k, N-(k+1), lda);
		matrix_t m = _submatrix(H, k+1, k, N-(k+1), N-k, lda);
		house_hm(b, &v, &m);
		matrix_t h = _submatrix(H, 0, k+1, N, N-(k+1), lda);
		house_mh(b, &v, &h);
	}
}