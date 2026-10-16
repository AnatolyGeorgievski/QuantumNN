/* потенциал Леннарда-Джонса */
#if !defined(Sigma) // значения должны быть заданы в шапке
#define Sigma2 1.0f // сигма в квадрате - 
#define Dt2_m  1.0f // dt^2 *(4*eps/m)
#define D2_MAX 1.0f // макс дистанция в квадрате
// Размер хеш таблицы по осям XYZ
#define DIM_X 128
#define DIM_Y 128
#define DIM_Z 128

#define MIN_Y 0.0f // минимальное и максимальное положения по Y
#define MAX_Y 1.0f

// Геометрический размер ячейки хеш таблицы, nm
#define Scale_X 1.0f
#define Scale_Y 1.0f
#define Scale_Z 1.0f
// параметры распределения по скоростям
#define Temp	1.0f // Температура среды, К
#define Ma  	1.0f // Масса атома среды, К
#define Sigma   1.0f // Из температуры и массы атома высчитываются параметры распределения
#define Mu	    0.1f // Смещения центральной точки распределения

#endif

/*! Декларация функций распределения случайной величины */
float2 gaussian(ulong *state);
float  uniform (ulong *state);
/*! Задание начальных условий - скорости  определяется температурой и дополнительным требованием 

лавные граничные условия:
1. Для атомов вдоль верхней и нижней границы - силы приравнять нулю
2. Для группы мобильных атомов (середина модели) - принять рандомные скорости, приняв температуру за 0.1К. 
Остальным атомам не добавлять
3. Для верхней группы атомов принять скорость равную величине Vmax, для нижней границы - ноль. Для атомов между ними (мобильная группа) - добавить к существующим значениям от нуля до Vmax.
*/
__kernel void v_init(__global float3 *p, __constant float3 *p0, __global float3 *a, __constant ulong* states)
{
	uint gid = get_global_id(0);
	ulong state = states[gid];
	float2 v1 = Sigma*gaussian(&state) + mix(0,Mu, (p0[gid].y-MIN_Y)/(MAX_Y-MIN_Y));// граничные условия на скорость, тянем
	float2 v2 = Sigma*gaussian(&state) ;
	p[gid] = (float3)(v1,v2.x);
}
/*  Lennard-Jones potential 4*eps*((e/r)^12 - (e/r)^6) */
float3 v_force(float3 p1, float3 p2)
{
	float3 p = p1-p2; 
	float r = dot(p,p);// сумма квадратов
	//if (r>D2_MAX) return (float3)(0);// быстрее без проверки
	p = p*rsqrt(r); // единичный вектор
	r = Sigma2/r;
	r = (r*r*r);
	return ((r*r) - r)*p;
}

#define IDX(x,y,z) (x + y*(DIM_X) + z*(DIM_X*DIM_Y))
const float3 Scale = {Scale_X,Scale_Y,Scale_Z};// заполнить константы снаружи
/* Хеш-функция */
static uint hash(float3 p){
	uint3 v = convert_uint3_rtz(p*Scale);// округлить к меньшему
	return IDX(v.x,v.y,v.z);
}

#define FIN UINT_MAX
#define Nb 26 // число ячеек в кубике 3x3x3 исключая центральный элемент

__kernel void update(__global float3 *pp, __constant float3 *p0, __constant float3 *p1, __global float3 *a, 
	__constant uint* cells, __constant uint* next)
{
	const int Index[Nb] = {// элементы по ребрам вдоль X по 3 шт
		IDX(-1,-1,-1), IDX(1,-1,-1), IDX(0,-1,-1), 
		IDX(-1, 1,-1), IDX(1, 1,-1), IDX(0, 1,-1), 
		IDX(-1,-1, 1), IDX(1,-1, 1), IDX(0,-1, 1), 
		IDX(-1, 1, 1), IDX(1, 1, 1), IDX(0, 1, 1), 
		IDX(-1, 0,-1), IDX(1, 0,-1), IDX(0, 0,-1),  
		IDX(-1, 0, 1), IDX(1, 0, 1), IDX(0, 0, 1),  
		IDX(-1,-1, 0), IDX(1,-1, 0), IDX(0,-1, 0),
		IDX(-1, 1, 0), IDX(1, 1, 0), IDX(0, 1, 0),
		IDX(-1, 0, 0), IDX(1, 0, 0)//IDX(0, 0, 0)
	};
	const short3 Bound[Nb] = {// элементы по ребрам вдоль X по 3 шт
		{-1,-1,-1}, {1,-1,-1}, {0,-1,-1}, 
		{-1, 1,-1}, {1, 1,-1}, {0, 1,-1}, 
		{-1,-1, 1}, {1,-1, 1}, {0,-1, 1}, 
		{-1, 1, 1}, {1, 1, 1}, {0, 1, 1}, 
		{-1, 0,-1}, {1, 0,-1}, {0, 0,-1},  
		{-1, 0, 1}, {1, 0, 1}, {0, 0, 1},  
		{-1,-1, 0}, {1,-1, 0}, {0,-1, 0},
		{-1, 1, 0}, {1, 1, 0}, {0, 1, 0},
		{-1, 0, 0}, {1, 0, 0}//{0, 0, 0}
	};

	size_t gid = get_global_id(0);
	float3 p = pp[gid];
	uint   h = hash(p);
	short3  v = convert_short3_rtz(p*Scale);
	float3 fs = 0;// сила действует на атом с координатами p
	
	for (int k=0; k<Nb;k++){// по всем направлениям
		if (all(convert_ushort3(v + Bound[k])<(ushort3){DIM_X,DIM_Y,DIM_Z})){
			uint idx = h + Index[k];// вычисляем индекс в хеш таблице соседей
			uint i = cells[idx];// хеш таблица
			while (i != FIN){// расчет потенциала (сил) для всех точек которые попадают в ячейку
				fs += v_force(pp[i], p);
				i = next[i];
			}
		}
	}
	// отдельно считаем для центральной ячейки
	uint i = cells[h];// хеш таблица
	while (i != FIN){
		fs += (i!=gid)? v_force(pp[i], p) : 0;
		i = next[i];
	}

/* a- ускорение v0 = p0-p1  -- вектор скорости */
	a[gid] = fs*(Dt2_m);
	p = p0[gid];
	pp[gid] = p + (p - p1[gid]) + a[gid];

}

/* раcсчитать предсказанные положения атомов (pp) и хеш таблицу */
__kernel void predict(__global float3 *pp, __constant float3 *p0, __constant float3 *p1, __constant float3 *a, 
	__global uint* cells, __global uint* next)
{
	uint gid = get_global_id(0);
	float3 p = p0[gid];
	p += (p - p1[gid]) + a[gid];// схема интеграла Верле
	pp[gid] = p;
	// составить хеш таблицу, атомарные операции
	uint idx = hash(p); // преобразовать в hash
	next[gid] = atomic_xchg(&cells[idx], gid);// список
}

/* генератор псевдо случайных чисел, используется для начальных значений скорости. 
обратное число = 2,3283064370807973754314699618685e-10
enum{ A = 4294883355UL };    // A "special" constant 2^32 -1 = 4294967295
ulong x;                     // Holds current RNG State
x=(x&0xFFFFFFFF)*A+(x>>32);  // Advance RNG to the next state

 */
 
enum{ MWC64X_A = 4294883355U };
enum{ MWC64X_M = 18446383549859758079UL };


uint mwc64x(uint2 *state)
{
    uint x=(*state).x, c=(*state).y;  // Unpack the state
    uint res=x^c;                     // Calculate the result
    x=x*MWC64X_A+c;
    c=mad_hi(x,MWC64X_A, (uint)(x<c));   	  // Step the RNG
    *state=(uint2)(x,c);              // Pack the state back up
    return res;                       // Return the next result
}
/*! Генератор последовательности псевдо-случайных чисел */
#define MWC_A1 0xFFEBB71D94FCDAF9UL
ulong mwc128_next(ulong* state) {
	ulong r = state[0];
	ulong t = (MWC_A1* r + state[1]);
	ulong c = mad_hi(MWC_A1, r, state[1]);
	state[0] = t;
	state[1] = c;
	return r;
}
__kernel void k_mwc128(__global ulong* state, __global uint* data) {
	uint gid = get_local_id(0);
	ulong x = state[2*gid+0];
	ulong c = state[2*gid+1];
	for (int i=0; i<256; i++){
		ulong t;
		t = (MWC_A1* x + c);
		c = mad_hi(MWC_A1, x, c);
		x = t;
	}
	data[2*gid+0] = x;
	data[2*gid+1] = c;
}

__kernel void k_mwc64x(__global ulong * state, __global float* data)
//uint mwc64x(ulong *state, uint* data)
{
	uint gid = get_local_id(0);
	uint2 x = state[gid];
	data+= 2*gid;
	for (int i=0; i<2; i++){
		data[i] = ((x.x ^ x.y)>>8)* 0x1.0p-24f;
		ulong t =  4294883355UL*x.x + x.y;
		x = as_uint2(t);
	}
	//state[gid] = as_ulong(x);
	
    // return ((x>>32) ^ (x&0xFFFFFFFFU));
}
/*
float2 gaussian()
{
	float fac;
	do{
		v1 = 2.0*uniform()-1.0;
		v2 = 2.0*uniform()-1.0;
		rsq = v1*v1 + v2*v2;
    } while ((rsq >= 1.0) || (rsq == 0.0));
    fac = sqrt(-2.0*log(rsq)/rsq);
	return fac*(v1,v2);
}
*/
// однородное распрелеление [0,1)
float uniform (ulong *state) {
	uint x = mwc64x(state);
	return x * 0x1.0p-24f;
}
/* Box-Muller transform 
чтобы получилось распределение 
*/
float2 gaussian(ulong *state)
{
	float u1 = 1.0f-uniform(state);	// распределение (0,1]
	float u2 = uniform(state);
    return sqrt(-2*log(u1))*(float2)(cospi(u2), sinpi(u2));
}
// инициализация скоростей - происходит в хосте

// инициализация переменных, в начальный момент времени a=0
__kernel void acc_init(__global float3 *a)
{
	uint gid = get_global_id(0);
	a[gid] = (float3)(0);
}
// инициализация хеш таблицы, для размерности DIM_X*DIM_Y*DIM_Z
__kernel void cells_init(__global uint *cells)
{
	uint gid = get_global_id(0);
	cells[gid] = FIN;
}

/* Генератор случайных чисел см 
	https://github.com/clMathLibraries/clRNG/blob/master/src/include/clRNG/private/mrg31k3p.c.h
 */
#define mrg31k3p_M1 2147483647             /* 2^31 - 1 */
#define mrg31k3p_M2 2147462579             /* 2^31 - 21069 */

#define mrg31k3p_MASK12 511                /* 2^9 - 1 */
#define mrg31k3p_MASK13 16777215           /* 2^24 - 1 */
#define mrg31k3p_MASK2 65535               /* 2^16 - 1 */
#define mrg31k3p_MULT2 21069

#define mrg31k3p_NORM_cl_double 4.656612873077392578125e-10  /* 1/2^31 */
#define mrg31k3p_NORM_cl_float  4.6566126e-10

__kernel void clrngMrg31k3pNextState(__global uint3 * state)
{
	uint gid = get_global_id(0);
	uint3 g1 = state[gid*2+0];
	uint3 g2 = state[gid*2+1];
	
	uint y1, y2;

	// first component
	y1 = ((g1.y & mrg31k3p_MASK12) << 22) + (g1.y >>  9)
	   + ((g1.z & mrg31k3p_MASK13) <<  7) + (g1.z >> 24);

	if (y1 >= mrg31k3p_M1) y1 -= mrg31k3p_M1;

	y1 += g1.z;
	if (y1 >= mrg31k3p_M1) y1 -= mrg31k3p_M1;
	
	g1 = (uint3)(y1,g1.xy);

	// second component
	y1 = ((g2.x & mrg31k3p_MASK2) << 15) + (mrg31k3p_MULT2 * (g2.x >> 16));
	if (y1 >= mrg31k3p_M2)	y1 -= mrg31k3p_M2;
	y2 = ((g2.z & mrg31k3p_MASK2) << 15) + (mrg31k3p_MULT2 * (g2.z >> 16));
	if (y2 >= mrg31k3p_M2)	y2 -= mrg31k3p_M2;
	y2 += g2.z;
	if (y2 >= mrg31k3p_M2)	y2 -= mrg31k3p_M2;
	y2 += y1;
	if (y2 >= mrg31k3p_M2)	y2 -= mrg31k3p_M2;
	g2 = (uint3)(y2,g2.xy);

	state[gid*2+0] = g1;
	state[gid*2+1] = g2;

/*
	if (g1[0] <= g2[0])
		return (g1[0] - g2[0] + mrg31k3p_M1);
	else
		return (g1[0] - g2[0]); */
}
