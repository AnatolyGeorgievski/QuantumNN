/*! \brief ML-KEM 

    (с)2025 Георгиевский А.M.

    * [FIPS.203](https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.203.pdf)
    * (https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-227.ipd.pdf)
параметры схемы:
    du --
    dv -- 
    𝜂1 -- specify the distribution for generating the vectors 𝐬 and e
    𝜂2 -- specify the distribution for generating the vectors e1 and e2
    k  -- determines the dimensions of the matrix A
    q  -- модуль

|           | n | q  | k | 𝜂1 | 𝜂2 | du | dv | strength (bits) |
|-----------|---|----|---|----|----|----|----|----------|
|ML-KEM-512  256 3329  2   3    2    10   4    128
|ML-KEM-768  256 3329  3   2    2    10   4    192
|ML-KEM-1024 256 3329  4   2    2    11   5    256

В реализации используется упаковка SIMD uint16x2_t, поскольку вычисления ведутся в сопряженных парах.

uint16x8_t, uint16x16_t, uint16x32_t для ускорения вычислений на современных процессорах.
Длина вектора определяется при компиляции и задается параметром VL=32,16,8.

Алгоритмы используют криптографический хэш и XOF генерацию, см. shake256.c:
XOF = SHAKE128, 
PRF = SHAKE256, 
H   = SHA-3-256, 
J   = SHAKE256,
G   = SHA-3-512

Предполагается возможность замены криптографических хэшей на другие алгоритмы, 
позволяющие реализовать XOF и PRF. В частности, Poseidon2

API:
* K_PKE_KeyGen()
* K_PKE_Encrypt()
* K_PKE_Decrypt()

-- функции шифрования не предполагается использовать вне процедуры KEM. 



Отдельно могут использоваться с фиксированным модулем и N:
* NTT()
* iNTT()
* MultiplyNTTs()
* poly_mul() -- умножение полиномов в кольце Z_q[x]/(x^n+1)
* poly_add() -- сложение  полиномов (векторов)
* poly_sub() -- вычитание полиномов (векторов)

тестирование:
$ gcc -DTEST_NTT -march=native -O3 -o test ml_kem.c shake256.c
$ echo "" | gcc -dM -E - -march=native 

Ряд простых чисел $q=a 2^s +1$, где a-простое число, для которых работает:
Shoup q=  257 (0x0101) a=  1 s= 8 w=(b*0xff00ff01uLL)>>24 zeta=3
Shoup q=  769 (0x0301) a=  3 s= 8 w=(b*0xaa71da0duLL)>>25 zeta=7
Shoup q= 3329 (0x0d01) a= 13 s= 8 w=(b*0x9d7dbb41uLL)>>27 zeta=17
Shoup q= 7937 (0x1f01) a= 31 s= 8 w=(b*0x841cc544uLL)>>28 zeta=71
Shoup q= 9473 (0x2501) a= 37 s= 8 w=(b*0xdd61cceduLL)>>29 zeta=187
Shoup q=11777 (0x2e01) a= 23 s= 9 w=(b*0xb21263c7uLL)>>29 zeta=347
Shoup q=12289 (0x3001) a=  3 s=12 w=(b*0xaaa71c85uLL)>>29 zeta=145
Shoup q=13313 (0x3401) a= 13 s=10 w=(b*0x9d86d11auLL)>>29 zeta=225
Shoup q=19457 (0x4c01) a= 19 s=10 w=(b*0xd7915fc6uLL)>>30 zeta=89
-- для этх чисел метод модульного умножения Шоупа работает со всеми входными значениями 
до 0xFFFF. Константы для замены деления выбраны таким образом чтобы выходные значения 
метода модульного умножения Шоупа работали для всех входных значений b < q.

Тестирование Алгоритмов схемы ML-KEM:
1. Обратимость операций Compress/Decompress
2. Обратимость ByteDecode/ByteEncode для 1,2,3,12
3. Обратимость NTT/iNTT
4. Обратимость операции шифрования и расшифрования PKE

A[i,j] = SampleNTT(rho, i,j) 
s^T \circ A^T \circ A \circ s;
*/
#include <stdint.h>
#include <stdio.h>
// \see (shake256.c)
typedef struct _XOF_ctx XOF_ctx_t;
extern void shake128(uint8_t *data, size_t len, uint8_t *tag, int d);
extern void shake256(uint8_t *data, size_t len, uint8_t *tag, int d);
extern void sha3_256(uint8_t *data, size_t len, uint8_t *tag);
extern void sha3_512(uint8_t *data, size_t len, uint8_t *tag);

struct _XOF_ctx {
    uint64_t S[256/8];
    unsigned int len; // длина сообщения в буфере
    unsigned int tlen;
};
extern XOF_ctx_t* XOF_init(XOF_ctx_t* ctx);
extern void XOF_absorb (XOF_ctx_t* ctx, uint8_t* data, size_t len);
extern void XOF_squeeze(XOF_ctx_t* ctx, uint8_t* data, size_t len);


#define MLKEM_512  1
#define MLKEM_768  2
#define MLKEM_1024 3

#define K 2
#define ETA1 3
#define ETA2 2

// Q_PRIME 12289 и N = 512 - другой вариант см. BLISS
// Q_PRIME  3329 и N = 256 - ML-KEM
#define N       256
#define Q_PRIME 3329// (13<<8) +1

#if (Q_PRIME == 3329)
#define ZETA    17  // корень из единицы N-й степени
#define N_INV   3303// ≡ 128^{−1} mod Q
#define Q_MONT  3327// -q^{-1} mod 2^{16} = 3327
#define U_BARRETT 40317 // 2^{27}/Q_PRIME
#elif (Q_PRIME == 12289)
#define ZETA    145  // корень из единицы N-й степени
#define Q_MONT  53249
#endif

typedef uint16_t uint16x2_t __attribute__((vector_size(4)));
typedef  int16_t  int16x2_t __attribute__((vector_size(4)));

typedef uint16_t uint16x8_t  __attribute__((vector_size(16)));
typedef uint16_t uint16x16_t __attribute__((vector_size(32)));
typedef uint16_t uint16x32_t __attribute__((vector_size(64)));

/*! Алгоритм заменяет целочисленное деление на модуль на умножение (mul_hi) и сдвиг. 

    Есть сомнение, как правильно высчитывать константу, можно округлять RNE или RTP 
(в большую сторону), чтобы не проверять отрицательные значения. 
    В данном случае используется округление к большему значению. 
    Если используется signed_shoup, то округление должно быть RNE.
 */
uint32_t shoup_div(uint32_t b){
#if Q_PRIME == 3329
//    return (((b*0xBB41uL)>>16) + b*0x9D7DuL)>>(43-32);
    return (b*0x9D7DBB41uLL)>>27;
//    b<<=4;
//    return ((b*0x3AFB7681uLL)>>32) + b;
#elif Q_PRIME == 12289
    return (b*0xAAA71C85uLL)>>(45-16);
#else
    return ((uint32_t)b<<16)/Q_PRIME;
#endif
}
uint16_t mod1(uint32_t x, uint16_t p){
    uint32_t d = (x*0x9D7DBB41uLL)>>43;
    return x - d*Q_PRIME;
}

static inline uint16x2_t SUBM(uint16x2_t a, uint16x2_t b){
    uint16x2_t r = (a-b);
    return r + ((r >= Q_PRIME)&Q_PRIME);
}
static inline uint16x2_t ADDM(uint16x2_t a, uint16x2_t b){
    uint16x2_t r = (a+b);
    return r - ((r >= Q_PRIME)&Q_PRIME);// _mm512_min_epu16 (r, r-Q_PRIME)
}
static inline uint16x2_t MULM(uint16x2_t a, uint16x2_t b){
    uint16x2_t r;
    r[0] = (a[0]*(uint32_t)b[0])%Q_PRIME;
    r[1] = (a[1]*(uint32_t)b[1])%Q_PRIME;
    return r;
}
#include <x86intrin.h>
#if defined(__AVX512F__)
#define VL 32 // vector length, epi16 elements per vector register
static inline uint16x32_t VMOD1(uint16x32_t r, uint16x32_t p){
    return (uint16x32_t)_mm512_min_epu16((__m512i)r, _mm512_sub_epi16((__m512i)r, (__m512i)p));
}
static inline uint16x32_t VADDM(uint16x32_t a, uint16x32_t b, uint16x32_t p){
    __m512i r = _mm512_add_epi16((__m512i)a, (__m512i)b);
    return (uint16x32_t)_mm512_min_epu16(r, _mm512_sub_epi16(r, (__m512i)p));
}
static inline uint16x32_t VSUBM(uint16x32_t a, uint16x32_t b, uint16x32_t p){
    __m512i r = _mm512_sub_epi16((__m512i)a, (__m512i)b);
    return (uint16x32_t)_mm512_min_epu16(r, _mm512_add_epi16(r, (__m512i)p));
}
// Умножение по методу Шаупа (Shoup) с предварительным вычислением w = b*2^{16}/Q
static inline uint16x32_t VMULM(uint16x32_t a, uint16x32_t b, uint16x32_t w, uint16x32_t p){
    __m512i q = _mm512_mulhi_epu16((__m512i)a, (__m512i)w);
    __m512i r0= _mm512_mullo_epi16((__m512i)a, (__m512i)b);
    __m512i r1= _mm512_mullo_epi16(q, (__m512i)p);
    __m512i r = _mm512_sub_epi16(r0, r1);
// эта операция не требуется? 
    r = _mm512_min_epu16(r, _mm512_sub_epi16(r, (__m512i)p));
    return (uint16x32_t)r;
}
static inline uint16x32_t VMULM_barrett(uint16x32_t a, uint16x32_t b, uint16x32_t u, uint16x32_t q){
    __m512i z1 = _mm512_mulhi_epu16((__m512i)a, (__m512i)b);
    __m512i z0 = _mm512_mullo_epi16((__m512i)a, (__m512i)b);
    __m512i Ur = _mm512_set1_epi16(U_BARRETT);
#if defined(__AVX512_VBMI2__)
    __m512i c1 = _mm512_shrdi_epi16(z0,z1, 11);
#else
    __m512i c1 = _mm512_or_si512(_mm512_srli_epi16(z0, 11), _mm512_slli_epi16(z1, 16-11));
#endif
    __m512i c2 = _mm512_mulhi_epu16(Ur, c1);
    __m512i c4 = _mm512_mullo_epi16(c2,  (__m512i)q);
    __m512i r  = _mm512_sub_epi16(z0, c4);
    return (uint16x32_t)_mm512_min_epu16(r, _mm512_sub_epi16(r, (__m512i)q));
}
#elif defined(__AVX2__)
#define VL 16
static inline uint16x16_t VMOD1(uint16x16_t r, uint16x16_t p){
    return (uint16x16_t)_mm256_min_epu16((__m256i)r, _mm256_sub_epi16((__m256i)r, (__m256i)p));
}
static inline uint16x16_t VADDM(uint16x16_t a, uint16x16_t b, uint16x16_t p){
    __m256i r = _mm256_add_epi16((__m256i)a, (__m256i)b);
    return (uint16x16_t)_mm256_min_epu16(r, _mm256_sub_epi16(r, (__m256i)p));
}
static inline uint16x16_t VSUBM(uint16x16_t a, uint16x16_t b, uint16x16_t p){
    __m256i r = _mm256_sub_epi16((__m256i)a, (__m256i)b);
    return (uint16x16_t)_mm256_min_epu16(r, _mm256_add_epi16(r, (__m256i)p));
}
/*! Shoup's multiplication with precomputed w = (b<<16)/Q_PRIME for faster computation */
static inline uint16x16_t VMULM(uint16x16_t a, uint16x16_t b, uint16x16_t w, uint16x16_t p){
    __m256i q = _mm256_mulhi_epu16((__m256i)a, (__m256i)w);
    __m256i r0= _mm256_mullo_epi16((__m256i)a, (__m256i)b);
    __m256i r1= _mm256_mullo_epi16(q, (__m256i)p);
    __m256i r = _mm256_sub_epi16(r0, r1);
// эта операция не требуется? 
    r = _mm256_min_epu16(r, _mm256_sub_epi16(r, (__m256i)p));
    return (uint16x16_t)r;
}
/*! \brief Barrett reduction
    \param a multiplicand
    \param b multiplier
    \param u precomputed value for Barrett reduction Ur = 2^{L} / q (L=Q+16, Q=11, L=27, Ur=40317)
    \param q modulus
    \return a*b mod q */

static inline uint16x16_t VMULM_barrett(uint16x16_t a, uint16x16_t b, uint16x16_t u, uint16x16_t q){
    __m256i z1 = _mm256_mulhi_epu16((__m256i)a, (__m256i)b);
    __m256i z0 = _mm256_mullo_epi16((__m256i)a, (__m256i)b);
//    __m256i Ur = _mm256_set1_epi16(U_BARRETT);
#if defined(__AVX512_VBMI2__) && defined(__AVX512VL__)// AVX10.1
    __m256i c1 = _mm256_shrdi_epi16(z0,z1, 11);
#else 
    __m256i c1 = _mm256_or_si256(_mm256_srli_epi16(z0, 11), _mm256_slli_epi16(z1, 16-11));
#endif
    __m256i c2 = _mm256_mulhi_epu16((__m256i)u, c1);
    __m256i c4 = _mm256_mullo_epi16(c2,  (__m256i)q);
    __m256i r  = _mm256_sub_epi16(z0, c4);
    return (uint16x16_t)_mm256_min_epu16(r, _mm256_sub_epi16(r, (__m256i)q));
}
#elif defined(__AVX__) && defined(__SSE4_1__) && defined(__SSE2__)
#define VL 8
static inline uint16x8_t VMOD1(uint16x8_t r, uint16x8_t p){
    return (uint16x8_t)_mm_min_epu16((__m128i)r, _mm_sub_epi16((__m128i)r, (__m128i)p));
}
static inline uint16x8_t VADDM(uint16x8_t a, uint16x8_t b, uint16x8_t p){
    __m128i r = _mm_add_epi16((__m128i)a, (__m128i)b);
    return (uint16x8_t)_mm_min_epu16(r, _mm_sub_epi16(r, (__m128i)p));
}
static inline uint16x8_t VSUBM(uint16x8_t a, uint16x8_t b, uint16x8_t p){
    __m128i r = _mm_sub_epi16((__m128i)a, (__m128i)b);
    return (uint16x8_t)_mm_min_epu16(r, _mm_add_epi16(r, (__m128i)p));
}
/*! Shoup's multiplication with precomputed w = (b<<16)/Q_PRIME for faster computation */
static inline uint16x8_t VMULM(uint16x8_t a, uint16x8_t b, uint16x8_t w, uint16x8_t p){
    __m128i q = _mm_mulhi_epu16((__m128i)a, (__m128i)w);
    __m128i r0= _mm_mullo_epi16((__m128i)a, (__m128i)b);
    __m128i r1= _mm_mullo_epi16(q, (__m128i)p);
    __m128i r = _mm_sub_epi16(r0, r1);
    return (uint16x8_t)_mm_min_epu16(r, _mm_sub_epi16(r, (__m128i)p));
}
static inline uint16x8_t VMULM_barrett(uint16x8_t a, uint16x8_t b, uint16x8_t u, uint16x8_t q){
    __m128i z1 = _mm_mulhi_epu16((__m128i)a, (__m128i)b);
    __m128i z0 = _mm_mullo_epi16((__m128i)a, (__m128i)b);
    __m128i c1 = _mm_or_si128(_mm_srli_epi16(z0, 11), _mm_slli_epi16(z1, 16-11));
    __m128i c2 = _mm_mulhi_epu16((__m128i)u, c1);
    __m128i c4 = _mm_mullo_epi16(c2,  (__m128i)q);
    __m128i r  = _mm_sub_epi16(z0, c4);
    return (uint16x8_t)_mm_min_epu16(r, _mm_sub_epi16(r, (__m128i)q));
}
#else // Векторизация не используется
#define VL 2
static inline uint16_t shoup_MULM(uint16_t a, uint16_t b, uint16_t w, uint16_t p);
static uint16x2_t VMULM(uint16x2_t a, uint16x2_t b, uint16x2_t w, uint16x2_t q){
    uint16x2_t r;
    r[0] = shoup_MULM(a[1], b[0], w[0], Q_PRIME);
    r[1] = shoup_MULM(a[1], b[1], w[1], Q_PRIME);
    return r;
}
#endif
#if VL==32
#define uint16xVL_t uint16x32_t
#define VSET1(x) {x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x}
static inline uint16x32_t VROTL(uint16x32_t a, uint16x32_t b){
    return __builtin_shufflevector(a,b, 63, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
                                        15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30);
}
static const uint16x32_t xzv[] = {
{    1, 2642, 1062, 1919,  296, 2447, 1339, 1476,  289,  331, 3253, 1756, 1197, 2304, 2277, 2055,   17, 2761,  583, 2649, 1637,  723, 2288, 1100, 1409, 2662, 3281,  233,  756, 2156, 3015, 3050,},
{ 1729,  630,  193,  797, 3046,   56, 2240, 1333,  650, 1977, 2513,  632, 2865,   33, 1320, 1915, 1703, 1651, 2789, 1789, 1847,  952, 1461, 2687,  939, 2308, 2437, 2388,  733, 2337,  268,  641,},
{ 2580, 1897, 2786, 3260, 1426, 2094,  535, 2882, 2319, 1435,  807,  452, 1438, 2868, 1534, 2402, 1584, 2298, 2037, 3220,  375, 2549, 2090, 1645, 1063,  319, 2773,  757, 2099,  561, 2466, 2594,},
{ 3289,  848,  569, 1746, 2393, 2879, 1974,  821, 2647, 2617, 1481,  648, 2474, 3110, 1227,  910, 2804, 1092,  403, 1026, 1143, 2150, 2775,  886, 1722, 1212, 1874, 1029, 2110, 2935,  885, 2154,},
};
static const uint16x32_t wzv[] = {
{   19,52011,20906,37778, 5827,48172,26360,29057, 5689, 6516,64039,34569,23564,45357,44825,40455,  334,54354,11477,52149,32226,14233,45042,21655,27738,52405,64591, 4586,14882,42443,59354,60043,},
{34037,12402, 3799,15690,59964, 1102,44097,26241,12796,38919,49471,12441,56401,  649,25986,37699,33525,32502,54905,35218,36360,18741,28761,52897,18485,45436,47975,47011,14430,46007, 5275,12618,},
{50790,37345,54846,64177,28072,41223,10532,56736,45652,28249,15886, 8898,28309,56460,30198,47286,31183,45239,40101,63390, 7382,50180,41144,32384,20926, 6279,54590,14902,41321,11044,48546,51066,},
{64748,16694,11201,34372,47109,56677,38860,16162,52109,51519,29155,12756,48704,61224,24155,17914,55200,21497, 7933,20198,22501,42325,54629,17442,33899,23859,36892,20257,41538,57779,17422,42404,},
};
static inline uint16x2_t* V(uint16x2_t *a){
    uint16x32_t *v = (uint16x32_t*)a;
    uint16x32_t f = __builtin_shufflevector(v[0], v[1], 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,
                        32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62);
    uint16x32_t g = __builtin_shufflevector(v[0], v[1], 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,
                        33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63);
    v[0] = f, v[1] = g;
    return a;
}
void NTT_CT_butterfly_2xVL(uint16x32_t *f_, const uint16x32_t z, const uint16x32_t w, uint16x32_t p){
    uint16x32_t a,b,f,g,zv,wv;
    f = f_[0], g = f_[1]; 
    zv = __builtin_shuffle(z, (uint16x32_t){1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1});
    wv = __builtin_shuffle(w, (uint16x32_t){1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1});
    g = VMULM(g, zv,wv, p);
        a = VADDM(f, g, p);
        b = VSUBM(f, g, p);
    f = __builtin_shuffle(a, b, (uint16x32_t){ 0, 1, 2, 3,  4, 5, 6, 7,  8, 9,10,11, 12,13,14,15, 32,33,34,35, 36,37,38,39, 40,41,42,43, 44,45,46,47});
    g = __builtin_shuffle(a, b, (uint16x32_t){16,17,18,19, 20,21,22,23, 24,25,26,27, 28,29,30,31, 48,49,50,51, 52,53,54,55, 56,57,58,59, 60,61,62,63});
    zv = __builtin_shuffle(z, (uint16x32_t){2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3});
    wv = __builtin_shuffle(w, (uint16x32_t){2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3});
    g = VMULM(g, zv,wv, p);
        a = VADDM(f, g, p);
        b = VSUBM(f, g, p);
    f = __builtin_shuffle(a, b, (uint16x32_t){0,1, 2, 3,  4, 5, 6, 7, 32,33,34,35, 36,37,38,39, 16,17,18,19, 20,21,22,23, 48,49,50,51, 52,53,54,55});
    g = __builtin_shuffle(a, b, (uint16x32_t){8,9,10,11, 12,13,14,15, 40,41,42,43, 44,45,46,47, 24,25,26,27, 28,29,30,31, 56,57,58,59, 60,61,62,63});
    zv = __builtin_shuffle(z, (uint16x32_t){4,4,4,4,4,4,4,4, 5,5,5,5,5,5,5,5, 6,6,6,6,6,6,6,6, 7,7,7,7,7,7,7,7});
    wv = __builtin_shuffle(w, (uint16x32_t){4,4,4,4,4,4,4,4, 5,5,5,5,5,5,5,5, 6,6,6,6,6,6,6,6, 7,7,7,7,7,7,7,7});
    g = VMULM(g, zv,wv, p);
        a = VADDM(f, g, p);
        b = VSUBM(f, g, p);
    f = __builtin_shuffle(a, b, (uint16x32_t){ 0, 1, 2, 3, 32,33,34,35,  8, 9,10,11, 40,41,42,43, 16,17,18,19, 48,49,50,51, 24,25,26,27, 56,57,58,59});
    g = __builtin_shuffle(a, b, (uint16x32_t){ 4, 5, 6, 7, 36,37,38,39, 12,13,14,15, 44,45,46,47, 20,21,22,23, 52,53,54,55, 28,29,30,31, 60,61,62,63});
    zv = __builtin_shuffle(z, (uint16x32_t){8,8,8,8, 9,9,9,9, 10,10,10,10, 11,11,11,11, 12,12,12,12, 13,13,13,13, 14,14,14,14, 15,15,15,15});
    wv = __builtin_shuffle(w, (uint16x32_t){8,8,8,8, 9,9,9,9, 10,10,10,10, 11,11,11,11, 12,12,12,12, 13,13,13,13, 14,14,14,14, 15,15,15,15});
    g = VMULM(g, zv,wv, p);
        a = VADDM(f, g, p);
        b = VSUBM(f, g, p);
    f = __builtin_shuffle(a, b, (uint16x32_t){ 0, 1, 32,33, 4, 5, 36,37,  8, 9, 40,41, 12,13, 44,45, 16,17, 48,49, 20,21, 52,53, 24,25, 56,57, 28,29, 60,61});
    g = __builtin_shuffle(a, b, (uint16x32_t){ 2, 3, 34,35, 6, 7, 38,39, 10,11, 42,43, 14,15, 46,47, 18,19, 50,51, 22,23, 54,55, 26,27, 58,59, 30,31, 62,63});
    zv = __builtin_shuffle(z, (uint16x32_t){16,16, 17,17, 18,18, 19,19, 20,20, 21,21, 22,22, 23,23, 24,24, 25,25, 26,26, 27,27, 28,28, 29,29, 30,30, 31,31});
    wv = __builtin_shuffle(w, (uint16x32_t){16,16, 17,17, 18,18, 19,19, 20,20, 21,21, 22,22, 23,23, 24,24, 25,25, 26,26, 27,27, 28,28, 29,29, 30,30, 31,31});
    g = VMULM(g, zv,wv, p);
        a = VADDM(f, g, p);
        b = VSUBM(f, g, p);
    f = __builtin_shuffle(a, b, (uint16x32_t){0,32, 2,34, 4,36, 6,38, 8,40,10,42,12,44,14,46,16,48,18,50,20,52,22,54,24,56,26,58,28,60,30,62});
    g = __builtin_shuffle(a, b, (uint16x32_t){1,33, 3,35, 5,37, 7,39, 9,41,11,43,13,45,15,47,17,49,19,51,21,53,23,55,25,57,27,59,29,61,31,63});
    f_[0] = f, f_[1] = g;
}
void NTT_GS_butterfly_2xVL(uint16x32_t *f_, const uint16x32_t z, const uint16x32_t w, uint16x32_t p){
    uint16x32_t a,b,f,g,zv,wv;
    a = f_[0], b = f_[1]; 
    f = __builtin_shuffle(a, b, (uint16x32_t){0,32, 2,34, 4,36, 6,38, 8,40,10,42,12,44,14,46,16,48,18,50,20,52,22,54,24,56,26,58,28,60,30,62});
    g = __builtin_shuffle(a, b, (uint16x32_t){1,33, 3,35, 5,37, 7,39, 9,41,11,43,13,45,15,47,17,49,19,51,21,53,23,55,25,57,27,59,29,61,31,63});
    a = VADDM(g, f, p);
    b = VSUBM(g, f, p);
    zv = __builtin_shuffle(z, (uint16x32_t){31,31, 30,30, 29,29, 28,28, 27,27, 26,26, 25,25, 24,24, 23,23, 22,22, 21,21, 20,20, 19,19, 18,18, 17,17, 16,16});
    wv = __builtin_shuffle(w, (uint16x32_t){31,31, 30,30, 29,29, 28,28, 27,27, 26,26, 25,25, 24,24, 23,23, 22,22, 21,21, 20,20, 19,19, 18,18, 17,17, 16,16});
    b = VMULM(b, zv, wv, p);
    f = __builtin_shuffle(a, b, (uint16x32_t){ 0, 1, 32,33, 4, 5, 36,37,  8, 9, 40,41, 12,13, 44,45, 16,17, 48,49, 20,21, 52,53, 24,25, 56,57, 28,29, 60,61});
    g = __builtin_shuffle(a, b, (uint16x32_t){ 2, 3, 34,35, 6, 7, 38,39, 10,11, 42,43, 14,15, 46,47, 18,19, 50,51, 22,23, 54,55, 26,27, 58,59, 30,31, 62,63});
    a = VADDM(g, f, p);
    b = VSUBM(g, f, p);
    zv = __builtin_shuffle(z, (uint16x32_t){15,15,15,15,14,14,14,14, 13,13,13,13,12,12,12,12, 11,11,11,11,10,10,10,10, 9,9,9,9,8,8,8,8});
    wv = __builtin_shuffle(w, (uint16x32_t){15,15,15,15,14,14,14,14, 13,13,13,13,12,12,12,12, 11,11,11,11,10,10,10,10, 9,9,9,9,8,8,8,8});
    b = VMULM(b, zv, wv, p);
    f = __builtin_shuffle(a, b, (uint16x32_t){ 0, 1, 2, 3, 32,33,34,35,  8, 9,10,11, 40,41,42,43, 16,17,18,19, 48,49,50,51, 24,25,26,27, 56,57,58,59});
    g = __builtin_shuffle(a, b, (uint16x32_t){ 4, 5, 6, 7, 36,37,38,39, 12,13,14,15, 44,45,46,47, 20,21,22,23, 52,53,54,55, 28,29,30,31, 60,61,62,63});
    a = VADDM(g, f, p);
    b = VSUBM(g, f, p);
    zv = __builtin_shuffle(z, (uint16x32_t){7,7,7,7,7,7,7,7, 6,6,6,6,6,6,6,6, 5,5,5,5,5,5,5,5, 4,4,4,4,4,4,4,4});
    wv = __builtin_shuffle(w, (uint16x32_t){7,7,7,7,7,7,7,7, 6,6,6,6,6,6,6,6, 5,5,5,5,5,5,5,5, 4,4,4,4,4,4,4,4});
    b = VMULM(b, zv, wv, p);
    f = __builtin_shuffle(a, b, (uint16x32_t){0,1, 2, 3,  4, 5, 6, 7, 32,33,34,35, 36,37,38,39, 16,17,18,19, 20,21,22,23, 48,49,50,51, 52,53,54,55});
    g = __builtin_shuffle(a, b, (uint16x32_t){8,9,10,11, 12,13,14,15, 40,41,42,43, 44,45,46,47, 24,25,26,27, 28,29,30,31, 56,57,58,59, 60,61,62,63});
    a = VADDM(g, f, p);
    b = VSUBM(g, f, p);
    zv = __builtin_shuffle(z, (uint16x32_t){3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3, 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});
    wv = __builtin_shuffle(w, (uint16x32_t){3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3, 2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2});
    b = VMULM(b, zv, wv, p);
    f = __builtin_shuffle(a, b, (uint16x32_t){ 0, 1, 2, 3,  4, 5, 6, 7,  8, 9,10,11, 12,13,14,15, 32,33,34,35, 36,37,38,39, 40,41,42,43, 44,45,46,47});
    g = __builtin_shuffle(a, b, (uint16x32_t){16,17,18,19, 20,21,22,23, 24,25,26,27, 28,29,30,31, 48,49,50,51, 52,53,54,55, 56,57,58,59, 60,61,62,63});
    a = VADDM(g, f, p);
    b = VSUBM(g, f, p);
    zv = __builtin_shuffle(z, (uint16x32_t){1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1});
    wv = __builtin_shuffle(w, (uint16x32_t){1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1});
    b = VMULM(b, zv, wv, p);
    f_[0] = a, f_[1] = b;
}
#elif VL==16
typedef uint16x16_t uint16xVL_t;
#define VSET1(x) {x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x}
static inline uint16x16_t VROTL(uint16x16_t a, uint16x16_t b){
    return __builtin_shufflevector(a,b, 31, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14);
}
static const uint16x16_t xzv[] = {
{    1, 1062,  296, 2447,  289,  331, 3253, 1756,   17, 2761,  583, 2649, 1637,  723, 2288, 1100,},
{ 1729, 1919, 1339, 1476, 1197, 2304, 2277, 2055, 1409, 2662, 3281,  233,  756, 2156, 3015, 3050,},
{ 2580,  193, 3046,   56,  650, 1977, 2513,  632, 1703, 1651, 2789, 1789, 1847,  952, 1461, 2687,},
{ 3289,  797, 2240, 1333, 2865,   33, 1320, 1915,  939, 2308, 2437, 2388,  733, 2337,  268,  641,},
{ 2642, 2786, 1426, 2094, 2319, 1435,  807,  452, 1584, 2298, 2037, 3220,  375, 2549, 2090, 1645,},
{  630, 3260,  535, 2882, 1438, 2868, 1534, 2402, 1063,  319, 2773,  757, 2099,  561, 2466, 2594,},
{ 1897,  569, 2393, 2879, 2647, 2617, 1481,  648, 2804, 1092,  403, 1026, 1143, 2150, 2775,  886,},
{  848, 1746, 1974,  821, 2474, 3110, 1227,  910, 1722, 1212, 1874, 1029, 2110, 2935,  885, 2154,},
};
static const uint16x16_t wzv[] = {
{   19,20906, 5827,48172, 5689, 6516,64039,34569,  334,54354,11477,52149,32226,14233,45042,21655,},
{34037,37778,26360,29057,23564,45357,44825,40455,27738,52405,64591, 4586,14882,42443,59354,60043,},
{50790, 3799,59964, 1102,12796,38919,49471,12441,33525,32502,54905,35218,36360,18741,28761,52897,},
{64748,15690,44097,26241,56401,  649,25986,37699,18485,45436,47975,47011,14430,46007, 5275,12618,},
{52011,54846,28072,41223,45652,28249,15886, 8898,31183,45239,40101,63390, 7382,50180,41144,32384,},
{12402,64177,10532,56736,28309,56460,30198,47286,20926, 6279,54590,14902,41321,11044,48546,51066,},
{37345,11201,47109,56677,52109,51519,29155,12756,55200,21497, 7933,20198,22501,42325,54629,17442,},
{16694,34372,38860,16162,48704,61224,24155,17914,33899,23859,36892,20257,41538,57779,17422,42404,},
};
/*! \brief Переводит внешнее представление вектора во внутреннее представление 
    внешнее - сопряженные пары значений, внутреннее - сопряженные вектора
 */
static inline uint16x2_t* V(uint16x2_t *a){
    uint16x16_t *v = (uint16x16_t*)a;
    uint16x16_t f = __builtin_shufflevector(v[0], v[1], 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30);
    uint16x16_t g = __builtin_shufflevector(v[0], v[1], 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31);
    v[0] = f, v[1] = g;
    return a;
}
/*! \brief Оптимизированная версия NTT_CT_butterfly_2xVL для использования с векторами x16. 

    Идея данного алгоритма в том, чтобы использовать перестановки так чтобы на каждом шаге размещение элементов вектора f и g 
    соответствовало входному порядку элементов вектора. Перестановки из операции BaseCaseMultiply_2xVL исключаются за счет выходных перестановок. 
    
 */
void NTT_CT_butterfly_2xVL(uint16x16_t *f_, const uint16x16_t z, const uint16x16_t w, uint16x16_t p){
    uint16x16_t a,b,f,g,zv,wv;
    f = f_[0], g = f_[1]; 
    zv = __builtin_shuffle(z, (uint16x16_t){1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1});
    wv = __builtin_shuffle(w, (uint16x16_t){1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1});
    g = VMULM(g, zv,wv, p);
        a = VADDM(f, g, p);
        b = VSUBM(f, g, p);
    f = __builtin_shuffle(a, b, (uint16x16_t){0,1, 2, 3,  4, 5, 6, 7, 16,17,18,19, 20,21,22,23});
    g = __builtin_shuffle(a, b, (uint16x16_t){8,9,10,11, 12,13,14,15, 24,25,26,27, 28,29,30,31});
    zv = __builtin_shuffle(z, (uint16x16_t){2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3});
    wv = __builtin_shuffle(w, (uint16x16_t){2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3});
    g = VMULM(g, zv,wv, p);
        a = VADDM(f, g, p);
        b = VSUBM(f, g, p);
    f = __builtin_shuffle(a, b, (uint16x16_t){ 0, 1, 2, 3,16,17,18,19, 8, 9,10,11,24,25,26,27});
    g = __builtin_shuffle(a, b, (uint16x16_t){ 4, 5, 6, 7,20,21,22,23,12,13,14,15,28,29,30,31});
    zv = __builtin_shuffle(z, (uint16x16_t){4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7});
    wv = __builtin_shuffle(w, (uint16x16_t){4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7});
    g = VMULM(g, zv,wv, p);
        a = VADDM(f, g, p);
        b = VSUBM(f, g, p);
    f = __builtin_shuffle(a, b, (uint16x16_t){ 0, 1,16,17, 4, 5,20,21, 8, 9,24,25,12,13,28,29});
    g = __builtin_shuffle(a, b, (uint16x16_t){ 2, 3,18,19, 6, 7,22,23,10,11,26,27,14,15,30,31});
    zv = __builtin_shuffle(z, (uint16x16_t){8,8,9,9, 10,10,11,11, 12,12,13,13, 14,14,15,15});
    wv = __builtin_shuffle(w, (uint16x16_t){8,8,9,9, 10,10,11,11, 12,12,13,13, 14,14,15,15});
    g = VMULM(g, zv,wv, p);
        a = VADDM(f, g, p);
        b = VSUBM(f, g, p);
    f = __builtin_shuffle(a, b, (uint16x16_t){0,16, 2,18, 4,20, 6,22, 8,24,10,26,12,28,14,30});
    g = __builtin_shuffle(a, b, (uint16x16_t){1,17, 3,19, 5,21, 7,23, 9,25,11,27,13,29,15,31});
    f_[0] = f, f_[1] = g;
}
/*! \brief Векторная бабочка обратного преобразования NTT для векторов длины 2xVL 
    \param f_ вектора для преобразования
    \param z вектор степеней корня из 1, используются значения того же вектора, 
    что и в прямом преобразовании, но в обратном порядке

    zr[i] = _shuffle(z[7-i], {0,1,3,2,7,6,5,4,15,14,13,12,11,10,9,8});
    zr*z = -1 (mod q)

    \param w константы для Shoup multiplication

    \note модульную операцию сложения и вычитания можно заменить на обычное сложение, 
    в тех случаях когда поле операции идет умножение и редуцирование по модулю (VMULM). 
 */
void NTT_GS_butterfly_2xVL(uint16x16_t *f_, const uint16x16_t z, const uint16x16_t w, uint16x16_t p){
    uint16x16_t a,b,f,g,zv,wv;
    a = f_[0], b = f_[1]; 
    f = __builtin_shuffle(a, b, (uint16x16_t){0,16, 2,18, 4,20, 6,22, 8,24,10,26,12,28,14,30});
    g = __builtin_shuffle(a, b, (uint16x16_t){1,17, 3,19, 5,21, 7,23, 9,25,11,27,13,29,15,31});
    a = VADDM(g, f, p);
    b = VSUBM(g, f, p);
    zv = __builtin_shuffle(z, (uint16x16_t){15,15,14,14, 13,13,12,12, 11,11,10,10, 9,9,8,8});
    wv = __builtin_shuffle(w, (uint16x16_t){15,15,14,14, 13,13,12,12, 11,11,10,10, 9,9,8,8});
    b = VMULM(b, zv, wv, p);
    f = __builtin_shuffle(a, b, (uint16x16_t){ 0, 1,16,17, 4, 5,20,21, 8, 9,24,25,12,13,28,29});
    g = __builtin_shuffle(a, b, (uint16x16_t){ 2, 3,18,19, 6, 7,22,23,10,11,26,27,14,15,30,31});
    a = VADDM(g, f, p);
    b = VSUBM(g, f, p);
    zv = __builtin_shuffle(z, (uint16x16_t){7,7,7,7, 6,6,6,6, 5,5,5,5, 4,4,4,4});
    wv = __builtin_shuffle(w, (uint16x16_t){7,7,7,7, 6,6,6,6, 5,5,5,5, 4,4,4,4});
    b = VMULM(b, zv, wv, p);
    f = __builtin_shuffle(a, b, (uint16x16_t){ 0, 1, 2, 3,16,17,18,19, 8, 9,10,11,24,25,26,27});
    g = __builtin_shuffle(a, b, (uint16x16_t){ 4, 5, 6, 7,20,21,22,23,12,13,14,15,28,29,30,31});
    a = VADDM(g, f, p);
    b = VSUBM(g, f, p);
    zv = __builtin_shuffle(z, (uint16x16_t){3,3,3,3,3,3,3,3, 2,2,2,2,2,2,2,2});
    wv = __builtin_shuffle(w, (uint16x16_t){3,3,3,3,3,3,3,3, 2,2,2,2,2,2,2,2});
    b = VMULM(b, zv, wv, p);
    f = __builtin_shuffle(a, b, (uint16x16_t){0,1, 2, 3,  4, 5, 6, 7, 16,17,18,19, 20,21,22,23});
    g = __builtin_shuffle(a, b, (uint16x16_t){8,9,10,11, 12,13,14,15, 24,25,26,27, 28,29,30,31});
    a = VADDM(g, f, p);
    b = VSUBM(g, f, p);
    zv = __builtin_shuffle(z, (uint16x16_t){1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1});
    wv = __builtin_shuffle(w, (uint16x16_t){1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1});
    b = VMULM(b, zv, wv, p);
    f_[0] = a, f_[1] = b;
}
#elif VL==8
#define uint16xVL_t uint16x8_t
#define VSET1(x) {x,x,x,x, x,x,x,x}
static inline uint16x8_t VROTL(uint16x8_t a, uint16x8_t b){
    return __builtin_shufflevector(a,b, 15,0,1,2, 3,4,5,6);
}
static inline uint16x2_t* V(uint16x2_t *a){
    uint16x8_t *v = (uint16x8_t*)a;
    uint16x8_t f = __builtin_shufflevector(v[0], v[1], 0,2,4,6,8,10,12,14);
    uint16x8_t g = __builtin_shufflevector(v[0], v[1], 1,3,5,7,9,11,13,15);
    v[0] = f, v[1] = g;
    return a;
}
static const uint16x8_t xzv[] = {
{    1,  296,  289,  331,   17, 2761,  583, 2649,},
{ 1729, 2447, 3253, 1756, 1637,  723, 2288, 1100,},
{ 2580, 1339, 1197, 2304, 1409, 2662, 3281,  233,},
{ 3289, 1476, 2277, 2055,  756, 2156, 3015, 3050,},
{ 2642, 3046,  650, 1977, 1703, 1651, 2789, 1789,},
{  630,   56, 2513,  632, 1847,  952, 1461, 2687,},
{ 1897, 2240, 2865,   33,  939, 2308, 2437, 2388,},
{  848, 1333, 1320, 1915,  733, 2337,  268,  641,},
{ 1062, 1426, 2319, 1435, 1584, 2298, 2037, 3220,},
{ 1919, 2094,  807,  452,  375, 2549, 2090, 1645,},
{  193,  535, 1438, 2868, 1063,  319, 2773,  757,},
{  797, 2882, 1534, 2402, 2099,  561, 2466, 2594,},
{ 2786, 2393, 2647, 2617, 2804, 1092,  403, 1026,},
{ 3260, 2879, 1481,  648, 1143, 2150, 2775,  886,},
{  569, 1974, 2474, 3110, 1722, 1212, 1874, 1029,},
{ 1746,  821, 1227,  910, 2110, 2935,  885, 2154,},
};
static const uint16x8_t wzv[] = {
{   19, 5827, 5689, 6516,  334,54354,11477,52149,},
{34037,48172,64039,34569,32226,14233,45042,21655,},
{50790,26360,23564,45357,27738,52405,64591, 4586,},
{64748,29057,44825,40455,14882,42443,59354,60043,},
{52011,59964,12796,38919,33525,32502,54905,35218,},
{12402, 1102,49471,12441,36360,18741,28761,52897,},
{37345,44097,56401,  649,18485,45436,47975,47011,},
{16694,26241,25986,37699,14430,46007, 5275,12618,},
{20906,28072,45652,28249,31183,45239,40101,63390,},
{37778,41223,15886, 8898, 7382,50180,41144,32384,},
{ 3799,10532,28309,56460,20926, 6279,54590,14902,},
{15690,56736,30198,47286,41321,11044,48546,51066,},
{54846,47109,52109,51519,55200,21497, 7933,20198,},
{64177,56677,29155,12756,22501,42325,54629,17442,},
{11201,38860,48704,61224,33899,23859,36892,20257,},
{34372,16162,24155,17914,41538,57779,17422,42404,},
};
void NTT_CT_butterfly_2xVL(uint16x8_t *f_, const uint16x8_t z, const uint16x8_t w, uint16x8_t p){
    uint16x8_t a,b,f,g,zv,wv;
    f = f_[0], g = f_[1]; 
    zv = __builtin_shuffle(z, (uint16x8_t){1,1,1,1,1,1,1,1});
    wv = __builtin_shuffle(w, (uint16x8_t){1,1,1,1,1,1,1,1});
    g = VMULM(g, zv,wv, p);
        a = VADDM(f, g, p);
        b = VSUBM(f, g, p);
    f = __builtin_shuffle(a, b, (uint16x8_t){0,1, 2, 3,  8, 9,10,11});
    g = __builtin_shuffle(a, b, (uint16x8_t){4,5, 6, 7, 12,13,14,15});
    zv = __builtin_shuffle(z, (uint16x8_t){2,2,2,2, 3,3,3,3});
    wv = __builtin_shuffle(w, (uint16x8_t){2,2,2,2, 3,3,3,3});
    g = VMULM(g, zv,wv, p);
        a = VADDM(f, g, p);
        b = VSUBM(f, g, p);
    f = __builtin_shuffle(a, b, (uint16x8_t){ 0, 1,  8, 9, 4, 5, 12,13});
    g = __builtin_shuffle(a, b, (uint16x8_t){ 2, 3, 10,11, 6, 7, 14,15});
    zv = __builtin_shuffle(z, (uint16x8_t){4,4, 5,5, 6,6, 7,7});
    wv = __builtin_shuffle(w, (uint16x8_t){4,4, 5,5, 6,6, 7,7});
    g = VMULM(g, zv,wv, p);
        a = VADDM(f, g, p);
        b = VSUBM(f, g, p);
    f = __builtin_shuffle(a, b, (uint16x8_t){0,8, 2,10, 4,12, 6,14});
    g = __builtin_shuffle(a, b, (uint16x8_t){1,9, 3,11, 5,13, 7,15});
    f_[0] = f, f_[1] = g;
}
void NTT_GS_butterfly_2xVL(uint16x8_t *f_, const uint16x8_t z, const uint16x8_t w, uint16x8_t p){
    uint16x8_t a,b,f,g,zv,wv;
    a = f_[0], b = f_[1]; 
    f = __builtin_shuffle(a, b, (uint16x8_t){0,8, 2,10, 4,12, 6,14});
    g = __builtin_shuffle(a, b, (uint16x8_t){1,9, 3,11, 5,13, 7,15});
    a = VADDM(g, f, p);
    b = VSUBM(g, f, p);
    zv = __builtin_shuffle(z, (uint16x8_t){7,7, 6,6, 5,5, 4,4});
    wv = __builtin_shuffle(w, (uint16x8_t){7,7, 6,6, 5,5, 4,4});
    b = VMULM(b, zv, wv, p);
    f = __builtin_shuffle(a, b, (uint16x8_t){ 0, 1,  8, 9, 4, 5, 12,13});
    g = __builtin_shuffle(a, b, (uint16x8_t){ 2, 3, 10,11, 6, 7, 14,15});
    a = VADDM(g, f, p);
    b = VSUBM(g, f, p);
    zv = __builtin_shuffle(z, (uint16x8_t){3,3,3,3, 2,2,2,2});
    wv = __builtin_shuffle(w, (uint16x8_t){3,3,3,3, 2,2,2,2});
    b = VMULM(b, zv, wv, p);
    f = __builtin_shuffle(a, b, (uint16x8_t){0, 1, 2, 3,  8, 9,10,11});
    g = __builtin_shuffle(a, b, (uint16x8_t){4, 5, 6, 7, 12,13,14,15});
    a = VADDM(g, f, p);
    b = VSUBM(g, f, p);
    zv = __builtin_shuffle(z, (uint16x8_t){1,1,1,1,1,1,1,1});
    wv = __builtin_shuffle(w, (uint16x8_t){1,1,1,1,1,1,1,1});
    b = VMULM(b, zv, wv, p);
    f_[0] = a, f_[1] = b;
}
#elif VL==4
typedef uint16x4_t uint16xVL_t;
#define VSET1(x) {x,x,x,x}
static inline uint16x2_t VROTL(uint16x2_t a, uint16x2_t b){
    return __builtin_shufflevector(a,b, 7, 0,1,2,3);
}
static inline uint16x2_t* V(uint16x2_t *a){
    uint16x4_t *v = (uint16x2_t*)a;
    uint16x4_t f = __builtin_shufflevector(v[0], v[1], 0,2,4,6);
    uint16x4_t g = __builtin_shufflevector(v[0], v[1], 1,3,5,7);
    v[0] = f, v[1] = g;
    return a;
}
#else // используется для тестирования
typedef uint16x2_t uint16xVL_t; 
#define VSET1(x) {x,x}
static inline uint16x2_t VROTL(uint16x2_t a, uint16x2_t b){
    return __builtin_shufflevector(a,b, 3, 0);
}
static inline uint16x2_t* V(uint16x2_t *a){
    uint16x2_t *v = (uint16x2_t*)a;
    uint16x2_t f = __builtin_shufflevector(v[0], v[1], 0,2);
    uint16x2_t g = __builtin_shufflevector(v[0], v[1], 1,3);
    v[0] = f, v[1] = g;
    return a;
}

#endif


static inline uint32_t MOD1(uint32_t r, uint32_t p) {
    return ((uint32_t)(r-p)<r)? r-p:r;
}

/*! Редуцирование по модулю простого числа, с использованием Барретта 
Метод проверяем на всех числах 0<2^15
Константа Q выбирается как число бит простого числа минус 1. 
 */
static uint32_t MODB(uint32_t a, uint32_t q, uint16_t U) {
const int Q = 11;//31-__builtin_clz(q);//11;
const int L = Q+16;
    uint16_t Ur = (1uL<<L)/q;
    uint32_t c2 = Ur*(a>>Q);// mul hi
    int16_t c4 = a - (int16_t)(q*(c2>>(L-Q)));
    if (c4>=q) c4 -= q;
    return c4;//((uint16_t)(c4-q)<(uint16_t)c4)? c4-q: c4;
}
/*! signed Montogomery multiplication with precomputed qm = -q^{-1} mod 2^{16} */
static inline int16_t  mont_MULM(int16_t a, int16_t b){

    int32_t z = (a*(int32_t)b);// signed high 16 bit
    int16_t z0=  a*(int32_t)b; // signed low 16 bit
    int32_t q = Q_PRIME;
    int16_t p = Q_MONT; // mod_inverse(Q_PRIME); // 1/q mod 2^{32}
    int16_t m = z0 * p; // low product z_0 (1/q)
    z = (z + m*q)>>16; // high product
    if (z<0) z+=q;
    
    return  (int16_t)z;
}
static inline uint16_t shoup_MULM(uint16_t a, uint16_t b, uint16_t w, uint16_t p){
    uint16_t q = (a*(uint32_t)w)>>16;
    uint16_t r0= (a*b);
    uint16_t r1=  q*p;
    uint16_t r = r0 - r1;
    return ((uint16_t)(r-p)<r)? r-p:r;
}
static inline void vec_mulm_u(uint16x2_t* r_, const uint16x2_t* a_, uint16_t b, unsigned int len){
    uint16xVL_t* r = (uint16xVL_t*)r_;
    uint16xVL_t* a = (uint16xVL_t*)a_;
    const uint16_t w = shoup_div(b);
    uint16xVL_t wv = VSET1(w);
    uint16xVL_t bv = VSET1(b);
    uint16xVL_t prime = VSET1(Q_PRIME);
    for(int i=0; i<len/VL; i++){
        r[i] = VMOD1(VMULM(a[i], bv, wv, prime),prime);
    }
}
static inline 
void vec_xtime_madd(uint16xVL_t* r, uint16xVL_t* a, uint16_t b, unsigned int len){
    uint16_t w  = shoup_div(b);
    uint16xVL_t bv = VSET1(b);
    uint16xVL_t wv = VSET1(w);
    uint16xVL_t p  = VSET1(Q_PRIME);
    uint16xVL_t c = p - r[len/VL-1];
    for (int i=0; i<len/VL; i++){
        uint16xVL_t v = VROTL(r[i], c);
        c = r[i];
        r[i] = VADDM(v, VMOD1(VMULM(a[i],bv,wv, p),p), p);
    }
}
// Сложение полиномов по модулю
static void poly_add(uint16x2_t *r_, const uint16x2_t *a_, const uint16x2_t *b_){
    const uint16xVL_t* a = (const uint16xVL_t*)a_;
    const uint16xVL_t* b = (const uint16xVL_t*)b_;
    uint16xVL_t* r = (uint16xVL_t*)r_;
    const uint16xVL_t p  = VSET1(Q_PRIME);
    for (int i = 0; i<N/VL; i++){
        r[i] = VADDM(a[i], b[i], p);
    }
}
// Вычитание полиномов по модулю
static void poly_sub(uint16x2_t *r_, const uint16x2_t *a_, const uint16x2_t *b_){
    const uint16xVL_t* a = (const uint16xVL_t*)a_;
    const uint16xVL_t* b = (const uint16xVL_t*)b_;
    uint16xVL_t* r = (uint16xVL_t*)r_;
    const uint16xVL_t p  = VSET1(Q_PRIME);
    for (int i = 0; i<N/VL; i++){
        r[i] = VSUBM(a[i], b[i], p);
    }
}
static void poly_mul(uint16x2_t *r, const uint16x2_t *a, const uint16_t *b)
{
    vec_mulm_u(r, a, b[N-1], N);
    for (int i = N-2; i>=0; i--){
        vec_xtime_madd((uint16xVL_t*)r, (uint16xVL_t*)a, b[i], N);
    }
}
/*
> The Compress and Decompress algorithms satisfy two important properties. First, decompression
followed by compression preserves the input. That is, Compress_𝑑(Decompress_𝑑(𝑦)) = 𝑦 for
all 𝑦 ∈ ℤ_{2𝑑} and all 𝑑 < 12.
> Division and rounding in the computation of these functions are performed in the set of rational numbers.
Floating-point computations shall not be used.
*/
static inline uint32_t compress(uint32_t x, unsigned int d){
    return (uint32_t)((((1u<<(d+1))*x + Q_PRIME)/Q_PRIME)>>1) & ((1u<<d)-1);
}
static inline uint32_t decompress(uint32_t y, unsigned int d){
    return (Q_PRIME*y + (1u<<(d-1)))/(1u<<d);
}
static uint16_t* Decompress(uint16_t* a,  unsigned int d){
    for(int i=0; i<N; i++){
        a[i] = decompress(a[i], d);
    }
    return a;
}
static uint16_t* Compress(uint16_t* a,  unsigned int d){
    for(int i=0; i<N; i++){
        a[i] = compress(a[i], d);
    }
    return a;
}
/*! \brief Algorithm 6 ByteDecode_𝑑(B) 

    Decodes a byte array into an array of 𝑑-bit integers for 1 ≤ 𝑑 ≤ 12.
 */
uint16_t* ByteDecode(uint16_t* dst, const uint8_t *s, unsigned int d){
    uint32_t mask = (1u<<d)-1;
    int bits = 0;
    uint32_t v = 0;
    for(int i=0; i<N; i++){
        while (bits<d) {
            v |= (*s++)<<bits;
            bits+=8;
        }
        dst[i] = (uint16_t) v & mask;
        v>>=d;
        bits-=d;
    }
    return dst;
}
/*! \brief Algorithm 5 ByteEncode_𝑑(F) 

    Encodes an array of 𝑑-bit integers into a byte array for 1 ≤ 𝑑 ≤ 12.
    \param s - выходной буфер, байты
    \param src - входной вектор, целые числа
    \param d - количество бит
 */
uint8_t* ByteEncode(uint8_t * s, const uint16_t* src, unsigned int d){
    uint16_t mask = (1u<<d)-1;
    int bits = 0;
    uint32_t v = 0;
    for(int i=0; i<N; i++){
        v |= (src[i] & mask)<<bits;
        bits+=d;
        while (bits>=8) {
            *s++ = (uint8_t) v;
            v>>=8;
            bits-=8;
        }
    }
    return s;
}
#if 1
/*! \brief Algorithm 7 SampleNTT(𝐵)
    Takes a 32-byte seed and two indices as input and outputs a pseudorandom element of 𝑇𝑞.
*/
void SampleNTT(uint16_t *a, uint8_t *b, int i, int j){
    const int n = 12; // 2^n > Q_PRIME
    const int mask = (1<<n)-1;
    XOF_ctx_t ctx;
    XOF_init(&ctx);
    b[32] = i;
    b[33] = j; 
    XOF_absorb(&ctx, b, 34);
    j =0;
    uint32_t c=0; 
    uint16_t d0, d1;
    while (j<N){
        XOF_squeeze(&ctx, (uint8_t*)&c, 3);
        d0 = (c   ) & mask;
        d1 = (c>>n) & mask;
        if (d0<Q_PRIME) {
            a[j++] = d0;
        } else
        if (d1<Q_PRIME && j<N) {
            a[j++] = d1;
        }
    }
}
#endif
static uint32_t sum_bits(uint8_t *b, int offs, int eta) {
    uint32_t bit_sum = 0;
    for (uint32_t i=offs; i<offs+eta; i++) {
        bit_sum += b[i>>3]>>(i&7) & 1u;
    }
    return bit_sum;
}
static inline uint16_t _subm(uint16_t a, uint16_t b, uint32_t q){
    return (a-b)%q;
}
/*! \brief Algorithm 8 SamplePolyCBD_𝜂 (𝐵) -- Sampling from the centered binomial distribution.
Takes a seed as input and outputs a pseudorandom sample from the distribution D𝜂(𝑅𝑞). */
void SamplePolyCBD(uint16_t *f,  uint8_t *b, int eta, uint32_t q)
{
    uint16_t x, y;
    for (int i=0; i<N; i++){
        x = sum_bits(b, 2*eta*i, eta);
        y = sum_bits(b, 2*eta*i+eta, eta);
        f[i] = _subm(x, y, q);
    }
}

// Gentleman-Sande (GS) butterfly
static void NTT_GS_butterfly(uint16x2_t *f_, unsigned int len, uint16_t zeta){
//if (len>=VL)
{
    uint16xVL_t *f = (uint16xVL_t*)(f_);
    uint16xVL_t *g = (uint16xVL_t*)(f_+len/2);
    uint16xVL_t t;
    uint16xVL_t z = VSET1(zeta);
    uint16xVL_t p = VSET1(Q_PRIME);
    uint32_t w = shoup_div(zeta);
    uint16xVL_t wv = VSET1(w);
    for (int i=0; i<len/VL; i++){
        t = f[i];
        f[i] = VADDM(g[i],t, p);
        g[i] = VMULM(g[i]+p-t, z, wv, p); 
    }
}
#if 0
else {
    uint16x2_t *f = f_;
    uint16x2_t *g = f_+len/2;
    uint16x2_t z = {zeta, zeta};
    uint16x2_t t;
    for (int i=0; i<len/2; i++){
        t = f[i];
        f[i] = ADDM(g[i],t);
        g[i] = MULM(SUBM(g[i],t), z);
    }
}
#endif
}

/*! Cooley-Tukey (CT) bufferfly */
static void NTT_CT_butterfly(uint16x2_t *f_, unsigned int len, uint16_t zeta){
//if (len>=VL)
{
    uint16xVL_t *f = (uint16xVL_t*)(f_);
    uint16xVL_t *g = (uint16xVL_t*)(f_+len/2);
    uint16xVL_t t;
    uint16xVL_t z = VSET1(zeta);
    uint16xVL_t p = VSET1(Q_PRIME);
    uint32_t w = shoup_div(zeta);
    uint16xVL_t wv = VSET1(w);
    for (int i=0; i<len/VL; i++){
        t = VMULM(g[i], z, wv, p);
        g[i] = VSUBM(f[i], t, p);
        f[i] = VADDM(f[i], t, p); 
    }
}
#if 0
else {
    printf("l=%d,%d \n", len, zeta);
    uint16x2_t *f = f_;
    uint16x2_t *g = f_+len/2;
    uint16x2_t z = {zeta, zeta};
    uint16x2_t t;
    for (int i=0; i<len/2; i++){
        t = MULM(g[i], z);
        g[i] = SUBM(f[i], t);
        f[i] = ADDM(f[i], t);
    }
}
#endif
}


static uint16_t zeta2[] = {// степени zeta^{2 BitRev(i)+1} mod q
  17, 3312,2761,  568, 583, 2746,2649,  680,
1637, 1692, 723, 2606,2288, 1041,1100, 2229,
1409, 1920,2662,  667,3281,   48, 233, 3096,
 756, 2573,2156, 1173,3015,  314,3050,  279,
1703, 1626,1651, 1678,2789,  540,1789, 1540,
1847, 1482, 952, 2377,1461, 1868,2687,  642,
 939, 2390,2308, 1021,2437,  892,2388,  941,
 733, 2596,2337,  992, 268, 3061, 641, 2688,
1584, 1745,2298, 1031,2037, 1292,3220,  109,
 375, 2954,2549,  780,2090, 1239,1645, 1684,
1063, 2266, 319, 3010,2773,  556, 757, 2572,
2099, 1230, 561, 2768,2466,  863,2594,  735,
2804,  525,1092, 2237, 403, 2926,1026, 2303,
1143, 2186,2150, 1179,2775,  554, 886, 2443,
1722, 1607,1212, 2117,1874, 1455,1029, 2300,
2110, 1219,2935,  394, 885, 2444,2154, 1175,
};
static uint16_t wzeta2[]= {// w = (z<<16)/q - коэффициенты для Shoup's multiplication
  334,65201,54354,11181,11477,54058,52149,13386,
32226,33309,14233,51302,45042,20493,21655,43880,
27738,37797,52405,13130,64591,  944, 4586,60949,
14882,50653,42443,23092,59354, 6181,60043, 5492,
33525,32010,32502,33033,54905,10630,35218,30317,
36360,29175,18741,46794,28761,36774,52897,12638,
18485,47050,45436,20099,47975,17560,47011,18524,
14430,51105,46007,19528, 5275,60260,12618,52917,
31183,34352,45239,20296,40101,25434,63390, 2145,
 7382,58153,50180,15355,41144,24391,32384,33151,
20926,44609, 6279,59256,54590,10945,14902,50633,
41321,24214,11044,54491,48546,16989,51066,14469,
55200,10335,21497,44038, 7933,57602,20198,45337,
22501,43034,42325,23210,54629,10906,17442,48093,
33899,31636,23859,41676,36892,28643,20257,45278,
41538,23997,57779, 7756,17422,48113,42404,23131,
};
/*! \brief Algorithm 9 NTT(𝑓) forward number theoretic transform 
Computes NTT representation 𝑓 of the given polynomial 𝑓 ∈ 𝑅_𝑞.

Алгоритм можно разделить на две части:
1. Прямое преобразование (CT) на регистрах с длиной 16 и более, когда используется одно значение zeta на вектор. 
2. Преобразование CT NTT_CT_butterfly_2xVL на длины VL и менее. 
Для этого используются перестановки значений zeta внутри вектора. 

*/
uint16x2_t* NTT(uint16x2_t *f){
    int i=1, len;
    for (len=N/2; len>VL; len>>=1)// 128, 64, 32
    for (int off=0; off<N; off+=2*len){
        uint16_t z = xzv[i][0];//g^{BitRev7(i)} 
        i++;
        NTT_CT_butterfly(f+off/2, len, z);
    }
    uint16xVL_t q = VSET1(Q_PRIME);
    for (int off=0, j=0; off<N; off+=2*VL, j++)
        NTT_CT_butterfly_2xVL((uint16xVL_t*)(f+off/2), xzv[j], wzv[j], q);
    return f;
}
/*! \brief Algorithm 10 iNTT(𝑓) Computes inverse NTT of the polynomial representation 𝑓 ∈ 𝑅_𝑞 that corresponds to the given NTT representation 𝑓 ∈ 𝑇_𝑞. */
uint16x2_t* iNTT(uint16x2_t *f){
    uint16xVL_t q = VSET1(Q_PRIME);
    for (int off=0, j=N/2/VL-1; off<N; off+=2*VL, j--)
        NTT_GS_butterfly_2xVL((uint16xVL_t*)(f+off/2), xzv[j], wzv[j], q);
#if 0
    int i=N/2-1;
    int len;
    for (len=2; len<=VL; len<<=1)
    for (int off=0; off<N; off+=2*len){
        uint16_t z = zeta[i];//g[BitRev7(i)]; 
        i--;
        NTT_GS_butterfly(f+off/2, len, z);
    }
#endif
    int i=N/2/VL -1;
    for (int len =2*VL; len<=N/2; len<<=1)
    for (int off=0; off<N; off+=2*len){
        uint16_t z = xzv[i][0];//zeta[i];//g[BitRev7(i)]; 
        i--;
        NTT_GS_butterfly(f+off/2, len, z);
    }
    vec_mulm_u(f, f, N_INV, N);
    return f;
}
/*! \brief Algorithm 12 BaseCaseMultiply(𝑎0, 𝑎1, 𝑏0, 𝑏1, 𝛾) 

 - работает как умножение комплексных чисел, где 𝛾 - мнимая единица. */
static uint16x2_t BaseCaseMultiply(uint16x2_t a, uint16x2_t b, uint16_t g){
    uint16x2_t c;
    c[0] = (a[0]*(uint32_t)b[0] + a[1]*(((uint32_t)b[1]*g)%Q_PRIME)) % Q_PRIME;
    c[1] = (a[1]*(uint32_t)b[0] + a[0]*(uint32_t)b[1]  ) % Q_PRIME;
    return c;
}
/*! \brief Algorithm 12 BaseCaseMultiply(𝑎0, 𝑎1, 𝑏0, 𝑏1, 𝛾) -- векторная версия 
    - работает как умножение комплексных чисел, где 𝛾 - мнимая единица.
    \param z - коэффициенты для умножения по модулю Q. 
    \param w - коэффициенты для Shoup's multiplication. 
    \param p - модуль Q. 
    \param u - константа Барретта для умножения по модулю Q. 

    \todo исключить перестановки вектора внутри функции, согласовать с выходом CT_Butterfly_2xVL. 
 Можно сделать операцию (a0*b0 + a1*b1) mod q с одним редуцированием.
 */
static void BaseCaseMultiply_2xVL(uint16xVL_t *a, uint16xVL_t *b, uint16xVL_t z, uint16xVL_t w, uint16xVL_t p, uint16xVL_t u){
/*  uint16x16_t a0 = __builtin_shufflevector(a[0], a[1], 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30);
    uint16x16_t a1 = __builtin_shufflevector(a[0], a[1], 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31);
    uint16x16_t b0 = __builtin_shufflevector(b[0], b[1], 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30);
    uint16x16_t b1 = __builtin_shufflevector(b[0], b[1], 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31);
*/
    uint16xVL_t r00 = VMULM_barrett(a[0], b[0], u, p);
    uint16xVL_t r11 = VMULM_barrett(a[1], b[1], u, p);
    uint16xVL_t r01 = VMULM_barrett(a[0], b[1], u, p);
    uint16xVL_t r10 = VMULM_barrett(a[1], b[0], u, p);
                r11 = VMULM(r11, z, w, p);
                r11 = VMOD1(r11, p);
    uint16xVL_t c0 = VADDM(r00, r11, p);
    uint16xVL_t c1 = VADDM(r10, r01, p);
    a[0] = c0, a[1] = c1;
//    a[0] = __builtin_shufflevector(c0, c1, 0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23);
//    a[1] = __builtin_shufflevector(c0, c1, 8,24,9,25,10,26,11,27,12,28,13,29,14,30,15,31);
}
/*! Algorithm 11 MultiplyNTTs(𝑓,𝑔)
    Computes the product (in the ring 𝑇_𝑞) of two NTT representations. 
    \param lda линейное смещение при умножении
 */
static void MultiplyNTTs(uint16x2_t *h, uint16x2_t *a, uint16x2_t *b, unsigned lda){
    if (lda!=1){
        for (int i=0; i<N/2; i++)
            h[i] = BaseCaseMultiply(a[i*lda], b[i],  zeta2[i]);
    } else {
        uint16xVL_t p = VSET1(Q_PRIME);
        uint16xVL_t u = VSET1(U_BARRETT);
        for (int i=0; i<N/2/VL; i++){
            uint16xVL_t* av = (uint16xVL_t*)(a+VL*i);
            uint16xVL_t* bv = (uint16xVL_t*)(b+VL*i);
            uint16xVL_t z = *(uint16xVL_t*)( zeta2+VL*i);
            uint16xVL_t w = *(uint16xVL_t*)(wzeta2+VL*i);
            BaseCaseMultiply_2xVL(av, bv, z, w, p, u);
        }
    }
}
/*! \brief Функция генерации псевдослучайного вектора
    \param s - случайный вектор 256 бит (32 байта) 
    \param eta - параметр распределения {2,3}
    \return вектор 256 бит (32 байта)
    */
static uint8_t* PRF(uint8_t*tag, uint8_t* s, uint8_t i, int eta){
    s[32] = i;
    shake256(s, 33, tag, 64*eta);
    return tag;
}
static uint8_t* H(uint8_t*tag, uint8_t* s, size_t len){
    sha3_256(s, len, tag);
    return tag;
}
static uint8_t* G(uint8_t*tag, uint8_t* s, size_t len){
    sha3_512(s, len, tag);
    return tag;
}
static uint8_t* J(uint8_t*tag, uint8_t* s, uint8_t len){
    shake256(s, len, tag, 32);
    return tag;
}
#if 0
/*! \brief 
    \param ek ключ шифрования, 384*k+32 байта $A\circ s + e$
    \param dk ключ расшифрования, 384*k+32 байта $s$
    \param d случайность 32 байта (+1 байт резервируется)

    \note транспонированная матрица A^T = A^{-1} в NTT представлении
    */
uint8_t* K_PKE_KeyGen(uint8_t* ek_PKE, uint8_t* dk_PKE, uint8_t* d, 
        int k, int du, int dv, int eta1, int eta2){
    uint8_t r[64];
    uint8_t tag[64*eta1];
    uint8_t* rho = r;
    uint8_t* sigma = r+32;
    d[32] = k;
    G(r, d, 33);
    __builtin_memcpy(ek_PKE+(N*12/8)*k, r, 32);
    uint16x2_t a[N/2], s[k*N/2], e[N/2];
    for (int i=0; i<k; i++){
        SamplePolyCBD(s+i*N/2, PRF(tag, sigma,  i  , eta1), eta1, Q_PRIME);
        NTT(s+i*N/2);
        ByteEncode(dk_PKE+(N*12/8)*i, s, 12);
    }
    for (int i=0; i<k; i++){
        SamplePolyCBD(e, PRF(tag, sigma,  i+k, eta1), eta1, Q_PRIME);
        NTT(e);
        poly_zero(t);
        for (int j=0; j<k; j++) {
            SampleNTT(a, rho, j, i);// генерация обратной матрицы A^{-1} в NTT представлении
            MultiplyNTTs(a, a, s+j*N/2, 1);
            poly_add(e, e, a);
        }
        ByteEncode(ek_PKE+(N*12/8)*i, e, 12);
    }
    return ek_PKE;
}
#endif
/*! \brief 
    \param dk_PKE ключ шифрования, k*384 байта= N*12бит
    \param ct зашифрованный текст 32*(k*du+dv) байта
    \param m расшифрованный текст 256 бит (32 байта)

    параметры схемы:
    k  -- размер матрицы A
    du -- размер вектора u
    dv -- размер вектора v
*/
uint8_t* K_PKE_Decrypt(uint8_t* m, uint8_t* ct, uint8_t* dk_PKE, int k, int du, int dv){
    uint16x2_t u[N/2], s[N/2], v[N/2], w[N/2] = {0};
    for (int i=0; i<k; i++){
        Decompress(ByteDecode((uint16_t*)u,ct+(N*du/8)*i,du), du);
        NTT(u);
        ByteDecode((uint16_t*)s, dk_PKE+(N*12/8)*i, 12);
        MultiplyNTTs(u, u, V(s), 1);
        poly_add(w, w, u);// операция dot product
    }
    Decompress(ByteDecode((uint16_t*)v,ct+(N*du/8)*k, dv), dv);
    poly_sub(w, v, iNTT(w));
    ByteEncode(m, Compress((uint16_t*)w,1),1);
    return m;
}
#if 0
/*! \brief Алгоритм 14 Шифрование сообщения

    Uses the encryption key to encrypt a plaintext message using the randomness 𝑟.
    \param m не шифрованный текст 256 бит (32 байта)
    \param ct зашифрованный текст 32*(k*du+dv) байта
    \param dk_PKE ключ шифрования, k*384 байта= N*12бит
    \param r рандомность 256 бит (32 байта)
 */
uint8_t* K_PKE_Encrypt(uint8_t* m, uint8_t* ct, uint8_t* dk_PKE, uint8_t* r,
        int k, int du, int dv, int eta1, int eta2){
    uint16x2_t u[N/2], s[N/2], v[N/2] = {0};
    uint8_t tag[64*MAX(eta1,eta2)];
    uint8_t b[32];
    uint8_t* rho = dk_PKE + 384*k;
    for (int i=0; i<k; i++){
        SamplePolyCBD(u,  PRF(tag, r,  i, eta1), eta1, Q_PRIME);
         NTT(u);
        uint16x2_t* a=s;
        SampleNTT(a, rho, i, j);// столбец матрицы A|j|i
        MultiplyNTTs(u,u,V(a),1);
        iNTT(u);
        uint16x2_t *e1 = s;
        SamplePolyCBD(e1, PRF(tag, r,k+i, eta2), eta2, Q_PRIME);
        poly_add(s, u, e1);
        ByteEncode(ct+(N*du/8)*i, Compress(s,du),du);
        ByteDecode(s, dk_PKE+(N*12/8)*i, 12);
        MultiplyNTTs(u, u, V(s), 1);
        poly_add(v, v, u);// операция dot product
    }
    iNTT(v);
    uint16x2_t *e2 = s;
    SamplePolyCBD(e2, PRF(tag, r,k+k, eta2), eta2, Q_PRIME);
    poly_add(v, v, e2);
    uint16x2_t *mu = s;
    mu = Decompress(ByteDecode(mu, m, 1), 1);
    poly_add(v, v, mu);
    ByteEncode(ct+(N*du/8)*k, Compress(v,dv),dv);
    return ct;
}
#endif
#if 0
/*! \brief Алгоритм 16 Обмен ключами ML-KEM Internal Key Generation

    Uses randomness to generate an encapsulation key and a corresponding decapsulation key.
    \param ek encapsulation key 384*k+32 байта
    \param dk decapsulation key 384*k+32 байта
    \param d ramndomness 32 байт
    \param z ramndomness 32 байт, rejection value
 */
uint8_t* ML_KEM_KeyGen_internal(uint8_t* ek, uint8_t* dk, uint8_t* d, uint8_t* z, 
        int k, int du, int dv, int eta1, int eta2){
    uint8_t* dk_PKE = dk;
    uint8_t* ek_PKE = dk+32;
    K_PKE_KeyGen(ek_PKE, dk_PKE, d);
    if (ek!=NULL && ek != ek_PKE) {
        __builtin_memcpy(ek, dk_PKE, 32); // copy dk_PKE to ek 
    }
    uint8_t *h = dk+768*k+32;
    H(h, ek, 384*k+32);
    if (z != dk+768*k+64) 
        __builtin_memcpy(dk+768*k+64, z, 32); // (H(ek) || z) 
    return ek;
}
/*! \brief Алгоритм 17 Обмен ключами ML-KEM Internal Encapsulation

    Uses the encapsulation key and randomness to generate a key and an associated ciphertext

    \param ss shared secret 32 байта
    \param ct ciphertext 32*(k*du+dv) байта
    \param ek encapsulation key 384*k байта
    \param m message 32 байта
    \param k, du, dv, eta1, eta2 параметры схемы
    \return ciphertext 32*(k*du+dv) байта
 */
uint8_t* ML_KEN_Encaps_internal(uint8_t* ss, uint8_t* ct, uint8_t* ek, uint8_t *m, 
        int k, int du, int dv, int eta1, int eta2){
    uint8_t *r = m+32;// randomness 256 бит (32 байта)
    H(r, ek, 32);
    G(ss, m, 64);
    K_PKE_Encrypt(m, ct, ek, r, k, du, dv, eta1, eta2);
    return ct;
}
/*! \brief Алгоритм 17 Обмен ключами ML-KEM Internal Decapsulation

    Uses the decapsulation key to produce a shared secret key from a ciphertext.
    \param dk decapsulation key 768*k+96 байта
    \param ct ciphertext 32*(k*du+dv) байта
    \param m message 32 байта
    \return shared secret Key 32 байта

 */
uint8_t* ML_KEN_Decaps_internal(uint8_t *ss, uint8_t* ct, uint8_t* dk,  uint8_t *m, 
        int k, int du, int dv, int eta1, int eta2){
    uint8_t *dk_PKE = dk;
    uint8_t *ek_PKE = dk + 384*k;
    uint8_t *h = dk+768*k+32;
    uint8_t *z = dk+768*k+64; // implicit rejection value
    K_PKE_Decrypt(m, ct, dk_PKE, k, du, dv);
    G(ss, m, 64);
    uint8_t *r = ss+32;
    K_PKE_Encrypt(m, ct1, ek_PKE, r, k, du, dv, eta1, eta2);
    if (__builtin_memcmp(ct,ct1)==0)
            J(ss, z, ct);
    return ss;
}
#endif

#include <stdio.h>
static uint8_t BitRev7(uint8_t x);
static void v_print(uint16x16_t f, uint16x16_t g){
    for (size_t i = 0; i < 16; i++){
        printf("%2d,", f[i]);
    }
    printf(" ");
    for (size_t i = 0; i < 16; i++){
        printf("%2d,", g[i]);
    }
    printf("\n");
    
}
#if 0
// этот тестовый пример для отладки перестановок векторов в NTT_CT_butterfly_2xVL
static void NTT_CT_butterfly_2xVL_test(){
    uint16x16_t a,b;
    uint16x16_t p = VSET1(Q_PRIME);
    uint16x16_t z = VSET1(17);
    uint16x16_t w = VSET1(334);
    uint16x16_t f = { 0, 1, 2, 3,  4, 5, 6, 7,  8, 9,10,11, 12,13,14,15};
    uint16x16_t g = {16,17,18,19, 20,21,22,23, 24,25,26,27, 28,29,30,31};
    v_print(f,g);
        z = __builtin_shuffle(z, (uint16x16_t){1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1});
        w = __builtin_shuffle(w, (uint16x16_t){1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1});
    a = f, b = g;
    f = __builtin_shuffle(a, b, (uint16x16_t){0, 1, 2, 3,  4, 5, 6, 7, 16,17,18,19, 20,21,22,23});
    g = __builtin_shuffle(a, b, (uint16x16_t){8, 9,10,11, 12,13,14,15, 24,25,26,27, 28,29,30,31});
    v_print(f,g);
        z = __builtin_shuffle(z, (uint16x16_t){2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3});
        w = __builtin_shuffle(w, (uint16x16_t){2,2,2,2,2,2,2,2, 3,3,3,3,3,3,3,3});
    a = f, b = g;
    f = __builtin_shuffle(a, b, (uint16x16_t){0, 1, 2, 3,  8, 9,10,11, 16,17,18,19, 24,25,26,27});
    g = __builtin_shuffle(a, b, (uint16x16_t){4, 5, 6, 7, 12,13,14,15, 20,21,22,23, 28,29,30,31});
    v_print(f,g);
        z = __builtin_shuffle(z, (uint16x16_t){4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7});
        w = __builtin_shuffle(w, (uint16x16_t){4,4,4,4, 5,5,5,5, 6,6,6,6, 7,7,7,7});
    a = f, b = g;
    f = __builtin_shuffle(a, b, (uint16x16_t){0, 1, 4, 5,  8, 9,12,13, 16,17,20,21, 24,25,28,29});
    g = __builtin_shuffle(a, b, (uint16x16_t){2, 3, 6, 7, 10,11,14,15, 18,19,22,23, 26,27,30,31});
    v_print(f,g);
        z = __builtin_shuffle(z, (uint16x16_t){8,8,9,9, 10,10,11,11, 12,12,13,13, 14,14,15,15});
        w = __builtin_shuffle(w, (uint16x16_t){8,8,9,9, 10,10,11,11, 12,12,13,13, 14,14,15,15});
    a = f, b = g;
    f = __builtin_shuffle(a, b, (uint16x16_t){0, 2, 4, 6,  8,10,12,14, 16,18,20,22, 24,26,28,30});
    g = __builtin_shuffle(a, b, (uint16x16_t){1, 3, 5, 7,  9,11,13,15, 17,19,21,23, 25,27,29,31});
    v_print(f,g);// обратный проход

    int res =1;// проверка правильности перестановки
    for (int i=0;i<16;i++) res = res && (BitRev7(i)>>2)==f[i];
    for (int i=0;i<16;i++) res = res && (BitRev7(i+16)>>2)==g[i];
    if (res) printf(".. ok\n");

    a = f, b = g;
    f = __builtin_shuffle(a, b, (uint16x16_t){0,16, 1,17,  2,18, 3,19,  4,20, 5,21,  6,22, 7,23});
    g = __builtin_shuffle(a, b, (uint16x16_t){8,24, 9,25, 10,26,11,27, 12,28,13,29, 14,30,15,31});

    v_print(f,g);
    a = f, b = g;
    f = __builtin_shuffle(a, b, (uint16x16_t){0, 1,16,17,  2, 3,18,19,  4, 5,20,21,  6, 7,22,23});
    g = __builtin_shuffle(a, b, (uint16x16_t){8, 9,24,25, 10,11,26,27, 12,13,28,29, 14,15,30,31});
    v_print(f,g);
    a = f, b = g;
    f = __builtin_shuffle(a, b, (uint16x16_t){0, 1, 2, 3, 16,17,18,19,  4, 5, 6, 7, 20,21,22,23});
    g = __builtin_shuffle(a, b, (uint16x16_t){8, 9,10,11, 24,25,26,27, 12,13,14,15, 28,29,30,31});
    v_print(f,g);
    a = f, b = g;
    f = __builtin_shuffle(a, b, (uint16x16_t){0, 1, 2, 3,  4, 5, 6, 7, 16,17,18,19, 20,21,22,23});
    g = __builtin_shuffle(a, b, (uint16x16_t){8, 9,10,11, 12,13,14,15, 24,25,26,27, 28,29,30,31});
    v_print(f,g);
// эти вращения применяются в NTT_CT_butterfly_2xVL и NTT_GS_butterfly_2xVL -- они обратимы
    a = f, b = g;
    f = __builtin_shuffle(a, b, (uint16x16_t){0,1, 2, 3,  4, 5, 6, 7, 16,17,18,19, 20,21,22,23});
    g = __builtin_shuffle(a, b, (uint16x16_t){8,9,10,11, 12,13,14,15, 24,25,26,27, 28,29,30,31});
    a = f, b = g;
    f = __builtin_shuffle(a, b, (uint16x16_t){ 0, 1, 2, 3,16,17,18,19, 8, 9,10,11,24,25,26,27});
    g = __builtin_shuffle(a, b, (uint16x16_t){ 4, 5, 6, 7,20,21,22,23,12,13,14,15,28,29,30,31});
    a = f, b = g;
    f = __builtin_shuffle(a, b, (uint16x16_t){ 0, 1,16,17, 4, 5,20,21, 8, 9,24,25,12,13,28,29});
    g = __builtin_shuffle(a, b, (uint16x16_t){ 2, 3,18,19, 6, 7,22,23,10,11,26,27,14,15,30,31});
    a = f, b = g;
    f = __builtin_shuffle(a, b, (uint16x16_t){0,16, 2,18, 4,20, 6,22, 8,24,10,26,12,28,14,30});
    g = __builtin_shuffle(a, b, (uint16x16_t){1,17, 3,19, 5,21, 7,23, 9,25,11,27,13,29,15,31});
    a = f, b = g;
    f = __builtin_shuffle(a, b, (uint16x16_t){0,16, 2,18, 4,20, 6,22, 8,24,10,26,12,28,14,30});
    g = __builtin_shuffle(a, b, (uint16x16_t){1,17, 3,19, 5,21, 7,23, 9,25,11,27,13,29,15,31});
    a = f, b = g;
    f = __builtin_shuffle(a, b, (uint16x16_t){ 0, 1,16,17, 4, 5,20,21, 8, 9,24,25,12,13,28,29});
    g = __builtin_shuffle(a, b, (uint16x16_t){ 2, 3,18,19, 6, 7,22,23,10,11,26,27,14,15,30,31});
    a = f, b = g;
    f = __builtin_shuffle(a, b, (uint16x16_t){ 0, 1, 2, 3,16,17,18,19, 8, 9,10,11,24,25,26,27});
    g = __builtin_shuffle(a, b, (uint16x16_t){ 4, 5, 6, 7,20,21,22,23,12,13,14,15,28,29,30,31});
    a = f, b = g;
    f = __builtin_shuffle(a, b, (uint16x16_t){0,1, 2, 3,  4, 5, 6, 7, 16,17,18,19, 20,21,22,23});
    g = __builtin_shuffle(a, b, (uint16x16_t){8,9,10,11, 12,13,14,15, 24,25,26,27, 28,29,30,31});
    printf("sh 1: ");v_print(f,g);
    a = f, b = g;
    f = __builtin_shufflevector(a, b, 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30);
    g = __builtin_shufflevector(a, b, 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31);
    a = f, b = g;
    f = __builtin_shufflevector(a, b, 0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23);
    g = __builtin_shufflevector(a, b, 8,24,9,25,10,26,11,27,12,28,13,29,14,30,15,31);
    printf("sh 2: ");v_print(f,g);
    a = f, b = g;
    f = __builtin_shuffle(a, b, (uint16x16_t){0, 1,16,17,  2, 3,18,19,  4, 5,20,21,  6, 7,22,23});
    g = __builtin_shuffle(a, b, (uint16x16_t){8, 9,24,25, 10,11,26,27, 12,13,28,29, 14,15,30,31});
    a = f, b = g;
    f = __builtin_shuffle(a, b, (uint16x16_t){0, 1,4,5,  8, 9,12,13, 16,17,20,21, 24,25,28,29});
    g = __builtin_shuffle(a, b, (uint16x16_t){2, 3,6,7, 10,11,14,15, 18,19,22,23, 26,27,30,31});
    printf("sh 3: ");v_print(f,g);
    a = f, b = g;
    f = __builtin_shuffle(a, b, (uint16x16_t){ 0, 1,16,17, 4, 5,20,21, 8, 9,24,25,12,13,28,29});
    g = __builtin_shuffle(a, b, (uint16x16_t){ 2, 3,18,19, 6, 7,22,23,10,11,26,27,14,15,30,31});
    a = f, b = g;
    f = __builtin_shuffle(a, b, (uint16x16_t){0,16, 2,18, 4,20, 6,22, 8,24,10,26,12,28,14,30});
    g = __builtin_shuffle(a, b, (uint16x16_t){1,17, 3,19, 5,21, 7,23, 9,25,11,27,13,29,15,31});
    a = f, b = g;
    f = __builtin_shuffle(a, b, (uint16x16_t){0,16, 2,18, 4,20, 6,22, 8,24,10,26,12,28,14,30});
    g = __builtin_shuffle(a, b, (uint16x16_t){1,17, 3,19, 5,21, 7,23, 9,25,11,27,13,29,15,31});
    a = f, b = g;
    f = __builtin_shuffle(a, b, (uint16x16_t){ 0, 1,16,17, 4, 5,20,21, 8, 9,24,25,12,13,28,29});
    g = __builtin_shuffle(a, b, (uint16x16_t){ 2, 3,18,19, 6, 7,22,23,10,11,26,27,14,15,30,31});
    printf("sh 4: ");v_print(f,g);


#if 0 
// две операции заменяются на одну...
    a = f, b = g;
    f = __builtin_shuffle(a, b, (uint16x16_t){0, 1,16,17,  2, 3,18,19,  4, 5,20,21,  6, 7,22,23});
    g = __builtin_shuffle(a, b, (uint16x16_t){8, 9,24,25, 10,11,26,27, 12,13,28,29, 14,15,30,31});
    a = f, b = g;
    f = __builtin_shufflevector(a, b, 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30);
    g = __builtin_shufflevector(a, b, 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31);
    printf("sh 1: ");v_print(f,g);
#endif

}
#endif

static uint32_t POWM(const uint32_t b, uint32_t a, const uint32_t q)
{
	uint32_t r = b;
	uint32_t s = 1;
    while (a!=0) {
		if (a&1) 
			s = ((uint64_t)s*r)%q;
		r = ((uint64_t)r*r)%q;
		a>>=1;
	}
	return s;
}
static inline uint32_t INVM(uint32_t a, const uint32_t q){
    return POWM(a, q-2, q);
}
/*! \brief Функция для вычисления q^{-1} mod 2^{32} */
uint32_t mod_inverse(uint32_t q) {
    uint32_t q_inv = 1;
    for (int i = 1; i < 16; i++) {
        if (((q * q_inv) & ((~0uL)>>(31-i))) != 1) {
            q_inv += (1u<<i);
        }
    }
    return q_inv;
}
uint32_t div_c(uint32_t b, int *nd) 
{
	uint32_t C;
	int k = __builtin_ctz(b);// количество ноликов в младщей части числа
	b>>=k;
	if (b==1) {
		*nd = 31;
		C = 0x80000000UL;
	} else 
	{
		*nd = 63 - __builtin_clz(b);// количество ноликов в старшей части числа
		C = ((1ULL<<(*nd))/b)+1;
	}
	*nd+=k;
	return C;
}
static uint32_t div_c0(uint32_t b, int *nd) 
{
	uint32_t C;
	int k = __builtin_ctz(b);// количество ноликов в младщей части числа
	b>>=k;
	if (b==1) {
		*nd = 32;
		C = 0;//0x1ULL<<32;
	} else
	{
		*nd = 64 - __builtin_clz(b);// количество ноликов в старшей части числа
		C = (uint64_t)(((1ULL<<32)-b)<<(*nd-32))/b + (1ULL<<(*nd-32)) + 1;
	}
	*nd+=k;
	return C & 0xFFFFFFFFUL;
}
/*! \brief Обратный битовый порядок для 8 битных чисел */
static uint8_t BitRev7(uint8_t x){
    x = ((x & 0x55) << 1) | ((x & 0xAA) >> 1);
    x = ((x & 0x33) << 2) | ((x & 0xCC) >> 2);
    x = ((x & 0x0F) << 4) | ((x & 0xF0) >> 4);
    return x>>1;
}
static uint16_t zeta[] = {// степени zeta^{k} mod q
   1,1729,2580,3289,2642, 630,1897, 848,
1062,1919, 193, 797,2786,3260, 569,1746,
 296,2447,1339,1476,3046,  56,2240,1333,
1426,2094, 535,2882,2393,2879,1974, 821,
 289, 331,3253,1756,1197,2304,2277,2055,
 650,1977,2513, 632,2865,  33,1320,1915,
2319,1435, 807, 452,1438,2868,1534,2402,
2647,2617,1481, 648,2474,3110,1227, 910,
  17,2761, 583,2649,1637, 723,2288,1100,
1409,2662,3281, 233, 756,2156,3015,3050,
1703,1651,2789,1789,1847, 952,1461,2687,
 939,2308,2437,2388, 733,2337, 268, 641,
1584,2298,2037,3220, 375,2549,2090,1645,
1063, 319,2773, 757,2099, 561,2466,2594,
2804,1092, 403,1026,1143,2150,2775, 886,
1722,1212,1874,1029,2110,2935, 885,2154,
};
#define BIT(x,n) (((x)>>(n))&1)
#define SWAP(x,y) do {    \
   __typeof__(x) _x = x;  \
   __typeof__(y) _y = y;  \
   x = _y;                \
   y = _x;                \
 } while(0)
static int jacobi(uint32_t a, uint32_t m) 
{
	a = a%m;
	int t = 1;
	unsigned m1= BIT(m,1);
	while (a!=0){
		int z = __builtin_ctz(a);
		a = a>>z;
		unsigned a1= BIT(a,1);
		if((BIT(z,0)&(m1^BIT(m,2))) ^ (a1&m1)) t = -t;
		SWAP(a,m);
		m1= a1;
		a = a%m;
	}
	if (m!=1) return 0;
	return t;
}
/*! \brief Поиск корня степени 2N= 2^s по модулю простого числа q
    \param N степень полинома (x^N + 1)
    \param q простое число, нечетное $q = 1 mod 2N$
    \return (r^2)^N = 1 mod q
 */
uint32_t ntt_root(uint32_t n, uint32_t q){
    uint32_t gen =  3;// выбор генератора - квадратичный не-вычет

    while (jacobi(gen,q)!=-1) gen+=2;
    uint32_t r = gen;
#if 0 // этот путь самый быстрый
    r = POWM(r, (q-1)/n, q);
    // if (POWM(r, N/2, q)==q-1) printf("..ok\n");
    return r;
#else 
    while (POWM(r, N/2, q)!=q-1) r+=2;
    return r;
#endif
}
int main(int argc, char** argv)
{
    uint32_t qm = mod_inverse(Q_PRIME);
    printf ("Q_PRIME = %d, Q_MONT =%d %d Ur=%d\n", Q_PRIME, qm, (Q_PRIME*qm)&0xFFFF, U_BARRETT);
    int nd, nd0;
    uint32_t Q_DIV =div_c(Q_PRIME, &nd);
    uint32_t Q_DIV_C0 =div_c0(Q_PRIME, &nd0);
    int s = 8;
    //Q_DIV = (Q_DIV +(0u<<(s-1)))>>s; nd-=s;
    printf ("Q_DIV = x%04X>>%d, C0=x%04X>>%d\n", Q_DIV, nd, Q_DIV_C0, nd0);
    for (uint32_t a=0; a<=0xFFFF; a++){
        uint32_t r = ((uint64_t)Q_DIV*(a<<16))>>nd;
        if (r!=(a<<16)/Q_PRIME) {
            printf("fail div %d %d\n", r, (a<<16)/Q_PRIME);
            break;
        }
    }
    #define Q_PRIME2 12289

    qm = mod_inverse(Q_PRIME2);
    printf ("Q_PRIME = %d, Q_MONT =%d %d Ur=%d\n", Q_PRIME2, qm, (Q_PRIME2*qm)&0xFFFF, U_BARRETT);

    uint32_t Q_DIV2 =div_c(Q_PRIME2, &nd);
    printf ("Q_DIV = x%04X>>%d\n", Q_DIV2, nd);
    for (uint32_t a=0; a<=0xFFFF; a++){
        uint32_t r = ((uint64_t)Q_DIV2*(a<<16))>>nd;
        if (r!=(a<<16)/Q_PRIME2) {
            printf("fail div %d %d\n", r, (a<<16)/Q_PRIME2);
            break;
        }
    }

    uint32_t g = ZETA, w;
    zeta[BitRev7(0)] = 1;
    zeta[BitRev7(1)] = g;
    for (int i=2; i<N/2; i++){
        g = (g*ZETA)%Q_PRIME;
        zeta[BitRev7(i)] = g;
    }
    printf("wzeta =\n");
    for (int i=0; i<N/2; i++){
        g = zeta[i];
        // w = ((((uint32_t)g<<17)+Q_PRIME)/Q_PRIME)>>1;
        w = ((uint32_t)g<<16)/Q_PRIME;
        printf("%5d,", w);
        if (i%8 ==7) printf("\n");
    }

    uint32_t z_ = INVM(ZETA, Q_PRIME);
#if 0
    //r_zeta[BitRev7(0)] = g = 1;
    for (int i=127; i>0; i--){
        r_zeta[i] = POWM(ZETA, Q_PRIME-BitRev7(i)-1, Q_PRIME);
    }

    printf("r_zeta =\n");
    for (int i=0; i<N/2; i++){
        //printf("%4d,", r_zeta[i]);
        printf("%4d(%d),", r_zeta[i], zeta[i]*(uint32_t)r_zeta[i]%Q_PRIME);
        if (i%8 ==7) printf("\n");
        
    }
#endif
    printf("zeta2 =\n");
    //uint32_t z2 = (ZETA*ZETA)%Q_PRIME;
    for (int i=0; i<N/2; i++){
        uint16_t g,w;
        zeta2[i] = g = POWM(ZETA, 2*BitRev7(i)+1, Q_PRIME);
        w = ((uint32_t)g<<16)/Q_PRIME;
        wzeta2[i] = w;
    }
    for (int i=0; i<N/2; i+=2){
        printf("%4d,%5d,", zeta2[i], zeta2[i+1]);
        if (i%8 ==6) printf("\n");
    }
    printf("wzeta2=\n");
    for (int i=0; i<N/2; i+=2){
        printf("%5d,%5d,", wzeta2[i], wzeta2[i+1]);
        if (i%8 ==6) printf("\n");
    }
    printf("r_zeta2 =%d\n", INVM(ZETA, Q_PRIME));
    uint16x2_t a[N/2];
    for (int i=0; i<N/2; i++)
        a[i] = (uint16x2_t){i,i+1};
    
    uint32_t ct,pt;
    for (int d=1; d<12; d++)
    for (uint32_t i = 0; i < 1u<<d; i++) {
        pt = decompress(i, d);
        ct = compress(pt, d);
        if (ct!=i) printf("fail compression %d != %d\n", i, ct);
    }
    for (int c=0; c<Q_PRIME; c++){
        uint16x2_t b = {c,c};
        for (int d=0; d<Q_PRIME; d++){
            uint16x2_t a = {d,d};
            uint16x2_t r = ADDM(a,b);
            if (r[0]!= (d + c)%Q_PRIME)printf("fail add %d + %d = %d != %d\n", d, c, r[0], (d+c)%Q_PRIME);
            if (r[1]!= (d + c)%Q_PRIME)printf("fail add %d + %d = %d != %d\n", d, c, r[1], (d+c)%Q_PRIME);
            r = SUBM(a,b);
            if (r[0]!= (d +Q_PRIME- c)%Q_PRIME)printf("fail sub %d - %d = %d != %d\n", d, c, r[0], (d-c)%Q_PRIME);
            if (r[1]!= (d +Q_PRIME- c)%Q_PRIME)printf("fail sub %d - %d = %d != %d\n", d, c, r[1], (d-c)%Q_PRIME);
        }
    }
if (1) {// проверка умножения полиномов
    uint16x2_t a[N/2] = {0};
    uint16x2_t b[N/2] = {0}; 
    uint16x2_t e[N/2] = {0}; 
    uint16x2_t r[N/2] = {0};
    for (int i=0; i<N/2; i++){
        a[i] = (uint16x2_t){i+5, i+2586};
        b[i] = (uint16x2_t){i+1, i+582};
        e[i] = (uint16x2_t){1, 1};
    }
    b[0]=(uint16x2_t){0,1};
    poly_mul(r, a, (uint16_t*)b);
    for (int i=0; i<N/2; i++){
        printf("%4d,%4d, ", r[i][0], r[i][1]);
        if (i%8 ==7) printf("\n");
    }
    printf("NTT =\n");
    for (int i=0; i<N/2; i++){
        printf("%4d,%4d, ", a[i][0], a[i][1]);
        if (i%8 ==7) printf("\n");
    }
    printf("inv NTT =\n");
    NTT(a);
    NTT(b);
    MultiplyNTTs(a, a, b, 1);
    iNTT(a);
    int res = 1;
    for (int i=0; i<N/2; i++){
        printf("%4d,%4d, ", a[i][0], a[i][1]);
        if (i%8 ==7) printf("\n");
        res = res && (a[i][0] == r[i][0] && a[i][1] == r[i][1]);
    }
    if (res) printf("NTT mul OK\n");
}
// Проверка операций
    if (1) {// проверка редукции Барретта
        uint32_t p = (13<<8)+1;
        uint64_t Ur = ((uint64_t)((1uLL<<16)-p)<<16)/p;
        printf("Barrett q=%d, Ur=%d %x\n", p, Ur, p*Ur);
        
        uint16_t r;
        for (uint16_t a=0; a<p; a++){
            for (uint16_t b=0; b<p; b++){
                r = MODB(a*(uint32_t)b, p, Ur);
                if (r != (a*(uint32_t)b)%p) {
                    printf("fail mont mul %d * %d = %d != %d\n", a, b, r, (a*(uint32_t)b)%p);
                    //break;
                    return 0;
                }
            }
        }
        printf("Barrett OK\n");
//        return 0;
    }
    if (1) {// проверка редукции Барретта
        uint32_t p = Q_PRIME;
        uint16xVL_t P = VSET1(Q_PRIME);
        uint16xVL_t Ur = VSET1(U_BARRETT);
        uint16_t r;
        for (uint16_t a=0; a<p*2; a++){
            for (uint16_t b=0; b<p; b++){
                uint16xVL_t av = VSET1(a);
                uint16xVL_t bv = VSET1(b);
                uint16xVL_t r = VMULM_barrett(av,bv, Ur, P);
                if (r[0] != (a*(uint32_t)b)%p) {
                    printf("fail mont mul %d * %d = %d != %d\n", a, b, r, (a*(uint32_t)b)%p);
                    //break;
                    return 0;
                }
            }
        }
        printf("Barrett OK\n");
    }    
    if (1) {// расчет коэффициентов Shoup для операции NTT_CT_butterfly_2xVL
        uint16_t g,w,z,p;
        printf("xzetav[]=\n");
        z = ZETA;
        p = Q_PRIME;
        int vl=8;
        int L=N/2/vl;
        for (int k=0; k<L; k++){
            int i = L+k;
            g = POWM(z, BitRev7(k), p);
            w = ((uint32_t)(g) <<16)/p;
            printf("{%2d,", k);
//            printf("{%5d,", g);
            for (int len=vl, l=0; len>=2; len>>=1, l++){// стадия разложения длина
                for (int offs=0,j=0; offs<vl; offs+=len, j++){// смещение число шагов N/2len
                    g = POWM(z, BitRev7(i+j), p);
                    w  = ((uint32_t)g <<16)/p;
//                    printf("%5d,", g);
                    printf("%2d ", i+j);
                }
                i += (L+k)<<l;
            }
            printf("},\n");
        }
        printf("wzetav[]=\n");
        for (int k=0; k<L; k++){
            int i = L+k;
            g = POWM(z, BitRev7(k), p);
            w = ((uint32_t)(g) <<16)/p;
            printf("{%5d,", w);
            for (int len=vl, l=0; len>=2; len>>=1, l++){// стадия разложения длина
                for (int offs=0,j=0; offs<vl; offs+=len, j++){// смещение число шагов N/2len
                    g = POWM(z, BitRev7(i+j), p);
                    w  = ((uint32_t)g <<16)/p;
                    printf("%5d,", w);
                }
                i += (L+k)<<l;
            }
            printf("},\n");
        }
    }
    if (0) {
        // NTT_CT_butterfly_2xVL_test();
        uint16xVL_t p = VSET1(Q_PRIME);
        uint16xVL_t u = VSET1(U_BARRETT);
        uint16xVL_t v, b;
        for (int i=0; i<8; i++){
        }
        printf("\n");
        return 0;
    }
#if 0
    if (0) {// проверка редукции Монтгомери
        uint32_t p = Q_PRIME;
        printf("Montgomery q=%d, qm=%d\n", p, Q_MONT);
        uint16x16_t P = VSET1(Q_PRIME);
        uint16x16_t qm = VSET1(Q_MONT);
        uint16_t r;
        for (uint16_t a=0; a<p; a++){
            for (uint16_t b=0; b<p; b++){
                uint16x16_t av = VSET1(a);
                uint16x16_t bv = VSET1(b);
                uint16x16_t r = VMULM_mont(av,bv, qm, P);
                if (((int32_t)r[0]<<16)%Q_PRIME != (a*(uint32_t)b)%p) {
                    printf("fail vmont mul %d * %d = %d != %d\n", a, b, ((int32_t)r[0]<<16)%Q_PRIME, (a*(uint32_t)b)%p);
                    //break;
                    //return 0;
                }
            }
        }
        printf("Montgomery OK\n");
    }
#endif
    if (1) {// проверка умножения методом Монтгомери
        uint16_t qm = -mod_inverse(Q_PRIME);
        printf("Mont q=%d qm=%d\n", Q_PRIME, qm);
        int16_t r;
        for (int16_t a=0; a<0x7FFF; a++){
            for (int32_t b=0; b<=0x7FFF; b++){
                r = mont_MULM(a, b);
                if (((int32_t)r<<16)%Q_PRIME != (a*(int32_t)b)%Q_PRIME) 
                    printf("fail mont mul %d * %d = %d != %d\n", a, b, r, (a*b)%Q_PRIME);
            }
        }
    }
    if (1) {// проверка умножения методом Shoup
        printf("Shoup q=%d\n", Q_PRIME);
        for (uint32_t a=0; a<=0xFFFF; a++){
            for (uint16_t b=0; b<Q_PRIME; b++){
                uint16_t w = shoup_div(b);//((uint32_t)b<<16)/Q_PRIME;
                uint16_t r = shoup_MULM(a, b, w, Q_PRIME);
                //uint16_t r = mod1(a*b,Q_PRIME);
                if (r != (a*(uint32_t)b)%Q_PRIME) 
                    printf("fail shoup mul %d * %d = %d != %d\n", a, b, r, (a*b)%Q_PRIME);
            }
        }
    }
    if (1) {// проверка умножения методом Shoup
        uint16_t primes[] = {
            0x101, 0x301, 0xd01, 0x1f01, 0x2501, 0x2e01, 0x3001, 0x3401, 0x4c01, 25601,
//            257, 769, 3329, 7681, 7937, 9473, 10753, 11777, 12289, 13313, 14081, 14593, 
//            15361, 17921, 18433, 19457, 22273, 23041, 23297, 25601, 26113, 26881, 30977, 31489, 32257,
            // 36353, 37889, 40193, 40961, 41729, 45569, 46337, 49409, 51713, 57089, 59393,
        };
        for (int k=0; k<sizeof(primes)/sizeof(primes[0]); k++){
            uint16_t p = primes[k];//Q_PRIME2;
            if (p%N!=1) continue;
            uint32_t z = ntt_root(N, p);
            int nd;
            uint32_t C0 = div_c0(p, &nd);
            uint32_t C = div_c(p, &nd);
            int n = __builtin_clz(p)-16;
            int s = __builtin_ctz(p-1);
            printf("Shoup q=%5d (0x%04x) a=%3d s=%2d w=(b*%08xuLL)>>%d zeta=%d\n", p, p, p>>s, s, C, nd-16, z);
            for (uint32_t a=0; a<=0xFFFF; a++){
                for (uint16_t b=0; b<p; b++){
//                    uint16_t w = ((uint32_t)b<<16)/p;
                    uint16_t w = (b*(uint64_t)C)>>(nd-16);

//                    uint16_t w = ((b*(uint64_t)C0)>>(32-n)) + (b<<n);
                    uint16_t r = shoup_MULM(a, b, w, p);
                    uint16_t r2 = (a*(uint32_t)b)%p;
                    if (r != r2) {
                        printf("fail shoup mul %d * %d = %d != %d\n", a, b, r, r2);
                        break;
                    }
                }
            }            
        }
    }


    return 0;
}