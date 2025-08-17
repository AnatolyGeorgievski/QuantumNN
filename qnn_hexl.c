/*!
    Implementation of modular arithmetic and polynomial operations for Ring-LWE based cryptographic schemes.
    Optimized for x86-64 with AVX512 instructions, focusing on the prime modulus q = 2^23 - 2^13 + 1 = 8380417.
    The code supports operations in the polynomial ring R_q = Z_q[x]/(x^256 + 1).

    Оптимизировано для модулей $q = 2^{32} - A⋅2^{16} - 1$, модули одновременно используются в алгоритмах MWC32 (Multiply-with-carry)

    Сборка
$ gcc -march=native -O3 -o test qnn_hexl.c

Все алгоритмы используют векторные инструкции 64бит. Это связано с тем что при редуцировании чисел 32 бит используется 64 битная арифметика. 

Оптимизация
На x86 есть единственная операция умножения векторов 32х32=64 бита. Нет операции mullo и mulhi для epu32, чтобы оставаться в 32 битах.
Есть операция для работы с 52 битными целыми числами, но операнды должны быть с выравниванием на 64 бит.

При работе с векторами используются операции:
1. Сложение модульное
2. Вычитание модульное
3. Умножение на константу 32 бита.
4. Поэлементное умножение векторов по модулю. 

__m512i _mm512_cvtepu32_epi64 (__m256i a) -- преобразование вектора 32х8 в 64х8

References:
* [[2306.01989](https://arxiv.org/pdf/2306.01989)] Optimized Vectorization Implementation of CRYSTALS-Dilithium
* [[2103.16400](https://arxiv.org/pdf/2103.16400)] Intel HEXL: Accelerating Homomorphic Encryption with Intel AVX512-IFMA52
* [[2018/039](https://eprint.iacr.org/2018/039.pdf)] Faster AVX2 optimized NTT multiplication for Ring-LWE lattice cryptography

* [[NIST:fips.203](https://nvlpubs.nist.gov/nistpubs/fips/nist.fips.203.pdf)] Module-Lattice-Based Key-Encapsulation Mechanism Standard. Tech. rep. National Institute of Standards and Technologies, 2024.\
(http://dx.doi.org/10.6028/NIST.FIPS.203)
* [[NIST:fips.204](https://nvlpubs.nist.gov/nistpubs/fips/nist.fips.204.pdf)] Module-Lattice-Based Digital Signature Standard. Tech. rep. National Institute of Standards and Technologies, 2024.\
(http://dx.doi.org/10.6028/NIST.FIPS.204)

Редуцирование по модулю:
y = x - ⌊x/q⌋*q - остаток от деления x на q
⌊x/q⌋ = заменяется на умножение на обратное число, Ur = ⌊2^L/q⌋
⌊x/q⌋ = ((x>>32)*Ur)>>(32+n) - умножение на обратное число по модулю q
-- это не полное редуцирование, требуется ещё одна проверка: 
min(y-q, y) = (y-q)<y? y-q: y;

Чтобы уложиться в 64 бита при вычислении Ur для 32 битных prime, 
константа рассчитывается, как ⌊(2^{64}-q)/q⌋.

Стандарты NIST серии PQC применяют Ring-LWE для построения схемы цифровой подписи и 
схемы выработки ключей для симметричной криптографии. 
Используют параметры:
* The prime number $𝑞 = 2^{23} − 2^{13} + 1 = 8380417$
* кольцо полиномов $\mathcal{R}_q = \mathbb{Z}_q[x]/(x^{256} + 1)$

*/

#include <x86intrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>


// Prime modulus q = 2^23 - 2^13 + 1
// #define Q_PRIME 8380417
// Prime modulus q = A0*2^16 - 1
#define Q_PRIME (((0xFFA0u)<<16)-1)
// Barrett constant u = ⌊(2^64 - q)/q⌋
#define U_BARRETT ((uint32_t)(((uint64_t)(-Q_PRIME) << 32) / Q_PRIME))

#ifdef __AVX512F__
typedef uint32_t uint32x16_t __attribute__((__vector_size__(64)));
// модуль q
static inline __m512i _mod1(__m512i x, __m512i q){
    return _mm512_min_epu32(x, _mm512_sub_epi32(x, q));
}

static inline __m512i _addm(__m512i a, __m512i b, __m512i q){
    __m512i d = _mm512_add_epi32(a, b);
    __m512i t = _mm512_sub_epi32(d, q);
    return _mm512_min_epu32(d, t);
//    return _mod1(_mm512_add_epi32(a, b), q);
}
static inline __m512i _subm(__m512i a, __m512i b, __m512i q){
    __m512i d = _mm512_sub_epi32(a, b);
    __m512i t = _mm512_add_epi32(d, q);
    return _mm512_min_epu32(d, t);
}
static inline __m512i _mulhi(__m512i a, __m512i b, __m512i q){
    __m512i d0 = _mm512_mul_epu32(a, b);
    __m512i d1 = _mm512_mul_epu32(_mm512_srli_epi64 (a, 32), _mm512_srli_epi64 (b, 32));
    return _mm512_mask_mov_epi32(_mm512_srli_epi64(d0, 32), (__mmask16)0xAAAA, d1);
}
/*! \brief Barrett reduction
    \param d - vector 64 bits x8
    \param q - modulus $2^{31} < q < 2^{32}$
    \param u - barrett constant \lfloor (2^{64}-q)/q \rfloor 
    \return $d \mod q$

В нашей версии алгоритма применяются иные сдвиги выровненные на 32 бита, иначе считается константа обратной величины (к), что позволяет использовать операцию mulhi.
$u = \lfloor (2^{64}-q)/q \rfloor$ подобрали выражение в таком виде. 

1. function Partial_Reduction(𝑑, 𝑞, u, 𝑄=32, 𝐿=64)
2.   𝑐_1 ← 𝑑 ≫ Q
3.   𝑐_2 ← d + 𝑘⋅𝑐_1
4.   𝑐_3 ← 𝑐_2 ≫ (L-Q)
5.   𝑐_4 ← 𝑑 − 𝑞⋅𝑐_3
6.   if 𝑐_4 ≥ q then
7.      𝑐_4 ← 𝑐_4 − 𝑞
8.   end if
9.   return $𝑐_4$
10.end function

 */
static inline __m512i _barret(__m512i d, __m512i q, __m512i u){
    __m512i c1, c2, c3, c4;
    c1 = _mm512_srli_epi64(d, 32);
    c2 = _mm512_add_epi64 (d, _mm512_mul_epu32(c1, u));
    c3 = _mm512_srli_epi64(c2, 32);
    c4 = _mm512_sub_epi64 (d, _mm512_mul_epu32(c3, q));
    return _mm512_min_epu64(c4, _mm512_sub_epi64(c4, q));
}
#define Q 32

/*! \brief Multiplication modulo q 

Вектор uint32x16_t
    \param a - first operand
    \param b - second operand
    \param q - modulus $2^{31} < q < 2^{32}$
    \param u - barrett constant \lfloor (2^{64}-q)/q \rfloor
    \return $a \cdot b \mod q$
 */
static inline __m512i _mulm(__m512i a, __m512i b, __m512i q, __m512i u){
    __m512i d0 = _mm512_mul_epu32(a, b);
    __m512i d1 = _mm512_mul_epu32(_mm512_srli_epi64(a,32), _mm512_srli_epi64(b,32));
    d0 =  _barret(d0, q, u);
    d1 =  _barret(d1, q, u);
    return _mm512_unpacklo_epi32(d0, d1);
}
static inline __m512i _macm(__m512i a, __m512i b, __m512i c, __m512i q, __m512i u){
    //a = _mm512_madd52lo_epu64(a, b, c);// a+ b*c
    a = _mm512_add_epi64(a, _mm512_mul_epu32(b,c));
    return _barret(a, q, u);
}
/*! \brief Square modulo q 
    \param a - first operand
    \param q - modulus $2^{31} < q < 2^{32}$
    \param u - barrett constant \lfloor (2^{64}-q)/q \rfloor 
    \return $a^2 \mod q$
 */
static inline __m512i _sqrm(__m512i a, __m512i q, __m512i u){
    __mm512i b = _mm512_srli_epi64(a, 32);
    a = _mm512_mul_epu32(a, a);
    b = _mm512_mul_epu32(b, b);
    a = _barret(a, q, u);
    b = _barret(b, q, u);
    return _mm512_unpacklo_epi32(a, b);
}

/*! \brief Element-wise polynomial multiplication modulo q.
    \param a First polynomial (array of 256 uint32_t coefficients).
    \param b Second polynomial (array of 256 uint32_t coefficients).
    \param result Output polynomial (array of 256 uint32_t coefficients).

    \note Processes 16 coefficients at a time using AVX512, for polynomials in R_q = Z_q[x]/(x^{256} + 1).
 */
void poly_mulm(const uint32_t *a, const uint32_t *b, uint32_t *result) {
    __m512i q = _mm512_set1_epi32(Q_PRIME);
    __m512i u = _mm512_set1_epi64(U_BARRETT);
    for (int i = 0; i < 256; i += 16) {
        __m512i va = _mm512_loadu_epi32((const void *)(a + i));
        __m512i vb = _mm512_loadu_epi32((const void *)(b + i));
        __m512i vr = _mulm(va, vb, q, u);// алгоритм умножения по модулю работает с числами 64 бита.
        _mm512_storeu_epi32(result + i, vr);
    }
}
static inline void poly_mulm_u(const uint32_t *a, const uint32_t b, uint32_t *result) {
    __m512i q = _mm512_set1_epi32(Q_PRIME);
    __m512i u = _mm512_set1_epi64(U_BARRETT);
    __m512i vb = _mm512_set1_epi32(b);
    for (int i = 0; i < 256; i += 16) {
        __m512i va = _mm512_loadu_epi32((const void *)(a + i));
        __m512i vr = _mulm(va, vb, q, u);// алгоритм умножения по модулю работает с числами 64 бита.
        _mm512_storeu_epi32(result + i, vr);
    }
}
/*! \brief Element-wise polynomial addition modulo q.
    \param a First polynomial (array of 256 uint32_t coefficients).
    \param b Second polynomial (array of 256 uint32_t coefficients).
    \param result Output polynomial (array of 256 uint32_t coefficients).
    
    \note Processes 16 coefficients at a time using AVX512, for polynomials in R_q = Z_q[x]/(x^{256} + 1).
 */
static inline void poly_addm(const uint32_t *a, const uint32_t *b, uint32_t *result) {
    __m512i q = _mm512_set1_epi32(Q_PRIME);
    for (int i = 0; i < 256; i += 16) {
        __m512i va = _mm512_loadu_epi32((const void *)(a + i));
        __m512i vb = _mm512_loadu_epi32((const void *)(b + i));
        __m512i vr = _addm(va, vb, q);
        _mm512_storeu_epi32(result + i, vr);
    }
}
/*! \brief сдвиг и редуцирование полинома (b) по модулю (x^N + 1) и сложение с полином (a)
    \param a First polynomial (array of 256 uint32_t coefficients).
    \param b Second polynomial (array of 256 uint32_t coefficients).
    \param r Output polynomial (array of 256 uint32_t coefficients).
 */
static inline void poly_xtime_addm(const uint32_t *a, const uint32_t *b, uint32_t *r) {
    __m512i q = _mm512_set1_epi32(Q_PRIME);
    __m512i c = _mm512_loadu_epi32((const void *)(b + 256-16)); // перенос
    c = _mm512_sub_epi32(q, c);         // c = q - c
    for (int i = 0; i < 256; i += 16) {
        __m512i va = _mm512_loadu_epi32((const void *)(a + i));
        __m512i vb = _mm512_loadu_epi32((const void *)(b + i));
        c = _mm512_alignr_epi32 (vb,c, 15);
        va = _addm(va, c, q);
        _mm512_storeu_epi32(r + i, va);
        c = vb;
    }
}
/*! \brief Операция умножения полинома на скаляр и сложение с результатом r = r*x + a*\beta
    \param a First polynomial (array of 256 uint32_t coefficients).
    \param b scalar (uint32_t).
    \param r Output polynomial (array of 256 uint32_t coefficients).
 */
static inline void poly_xtime_madd(const uint32_t *a, const uint32_t b, uint32_t *r) {
    __m512i q = _mm512_set1_epi32(Q_PRIME);
    __m512i u = _mm512_set1_epi64(U_BARRETT);
    __m512i vb= _mm512_set1_epi32(b);
    __m512i c = _mm512_loadu_epi32((const void *)(r + 256-16)); // перенос
    c = _mm512_sub_epi32(q, c);
    for (int i = 0; i < 256; i += 16) {
        __m512i va = _mm512_loadu_epi32((const void *)(a + i)); 
        __m512i vr = _mm512_loadu_epi32((const void *)(r + i));
        va = _mulm(va, vb, q, u);
        c  = _mm512_alignr_epi32 (vr,c, 15);
        va = _addm(va, c, q);
        _mm512_storeu_epi32(r + i, va);
        c = vr;
    }
}
#elif defined(__AVX2__)
typedef uint32_t uint32x8_t __attribute__((__vector_size__(32)));
static inline __m256i _mod1_avx2(__m256i x, __m256i q){
    return _mm256_min_epu32(x, _mm256_sub_epi32(x, q));
}
static inline __m256i _barret_avx2(__m256i d, __m256i q, __m256i u) {
    __m256i c1 = _mm256_srli_epi64(d, 32);
    __m256i c2 = _mm256_add_epi64(d, _mm256_mul_epu32(c1, u));
    __m256i c3 = _mm256_srli_epi64(c2, 32);
    __m256i c4 = _mm256_sub_epi64(d, _mm256_mul_epu32(c3, q));
    __m256i c4_minus_q = _mm256_sub_epi64(c4, q);       // Compare c4 < c4_minus_q (unsigned comparison)
    __m256i cmp = _mm256_cmpgt_epi64(c4_minus_q, c4);   // c4_minus_q > c4
    return _mm256_blendv_epi8(c4_minus_q, c4, cmp);     // Select: if c4 < c4_minus_q, take c4; else take c4_minus_q
}
static inline __m256i _addm_avx2(__m256i a, __m256i b, __m256i q) {
    __m256i d = _mm256_add_epi32(a, b);
    __m256i t = _mm256_sub_epi32(d, q);
    return _mm256_min_epu32(d, t);
}
static inline __m256i _subm_avx2(__m256i a, __m256i b, __m256i q) {
    __m256i d = _mm256_sub_epi32(a, b);
    __m256i t = _mm256_add_epi32(d, q);
    return _mm256_min_epu32(d, t);
}
static inline __m256i _mulm_avx2(__m256i a, __m256i b, __m256i q, __m256i u){
    __m256i d0 = _mm256_mul_epu32(a, b);
    __m256i d1 = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), _mm256_srli_epi64(b,32));
    d0 = _barret_avx2(d0, q, u);
    d1 = _barret_avx2(d1, q, u);
    return _mm256_unpacklo_epi32(d0, d1);
}
/*! \brief Element-wise polynomial multiplication modulo q.
    \param a First polynomial (array of 256 uint32_t coefficients).
    \param b Second polynomial (array of 256 uint32_t coefficients).    
    \param result Output polynomial (array of 256 uint32_t coefficients).
    
    \note Processes 8 coefficients at a time using AVX2, for polynomials in R_q = Z_q[x]/(x^{256} + 1).
 */
static inline void poly_mulm(const uint32_t *a, const uint32_t *b, uint32_t *result) {
    __m256i q = _mm256_set1_epi32(Q_PRIME);
    __m256i u = _mm256_set1_epi64x(U_BARRETT);
    for (int i = 0; i < 256; i += 8) {
        __m256i va = _mm256_loadu_si256((__m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((__m256i *)(b + i));
        __m256i vr = _mulm_avx2(va, vb, q, u);
        _mm256_storeu_si256((__m256i *)(result + i), vr);
    }
}
/*! редуцирование (b) и сложение полиномов x^N + 1
N = 256 
 */
static inline void poly_xtime_addm(const uint32_t *a, const uint32_t *b, uint32_t *r, unsigned int N) {
    __m256i q = _mm256_set1_epi32(Q_PRIME);
    __m256i c = _mm256_loadu_si256((__m256i *)(b + 256-8)); // перенос
    c = _mm256_sub_epi32(q, c);         // c = q - c
    for (int i = 0; i < 256; i += 8) {
        __m256i va = _mm256_loadu_si256((__m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((__m256i *)(b + i));
        c = _mm256_alignr_epi32 (vb,c, 7);
        va = _addm_avx2(va, c, q);
        _mm256_storeu_si256((__m256i *)(r + i), va);
        c = vb;
    }
}

/*! \brief Операция умножения полинома на скаляр и сложение с результатом r = r*x + a*\beta
    \param a First polynomial (array of 256 uint32_t coefficients).
    \param b scalar (uint32_t).
    \param r Output polynomial (array of 256 uint32_t coefficients).
    \param N степень полинома
 */
static inline 
void poly_xtime_madd(const uint32_t *a, const uint32_t b, uint32_t *r, unsigned int N) {
    __m256i q = _mm256_set1_epi32 (Q_PRIME);
    __m256i u = _mm256_set1_epi64x(U_BARRETT);
    __m256i vb= _mm256_set1_epi32 (b);
    __m256i c = _mm256_loadu_si256((__m256i *)(r + 256-8)); // перенос
    for (int i = 0; i < N; i += 8) {
        __m256i va = _mm256_loadu_si256((__m256i *)(a + i));
        __m256i vr = _mm256_loadu_si256((__m256i *)(r + i));
        va = _mulm_avx2(va, vb, q, u);
        c  = _mm256_alignr_epi32 (vr,c, 7);// уточнить
        va = _addm_avx2(va, c, q);
        _mm256_storeu_si256((__m256i *)(r + i), va);
        c = vr;
    }
}
/*! \brief Element-wise polynomial addition modulo q.
    \param a First polynomial (array of 256 uint32_t coefficients).
    \param b Second polynomial (array of 256 uint32_t coefficients).    
    \param result Output polynomial (array of 256 uint32_t coefficients).
    \param N степень полинома (x^N + 1)
    
    \note Processes 8 coefficients at a time using AVX2, for polynomials in R_q = Z_q[x]/(x^{N} + 1).
 */
static inline 
void poly_addm(const uint32_t *a, const uint32_t *b, uint32_t *result, unsigned int N) {
    __m256i q = _mm256_set1_epi32(Q_PRIME);
    for (int i = 0; i < N; i += 8) {
        __m256i va = _mm256_loadu_si256((__m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((__m256i *)(b + i));
        __m256i vr = _addm_avx2(va, vb, q);
        _mm256_storeu_si256((__m256i *)(result + i), vr);
    }
}
/*! \brief Element-wise polynomial substraction modulo q.
    \param a First polynomial (array of 256 uint32_t coefficients).
    \param b Second polynomial (array of 256 uint32_t coefficients).    
    \param result Output polynomial (array of 256 uint32_t coefficients).
    \param N степень полинома (x^N + 1)
 */
static inline void poly_subm(const uint32_t *a, const uint32_t *b, uint32_t *result, unsigned int N) {
    __m256i q = _mm256_set1_epi32(Q_PRIME);
    for (int i = 0; i < N; i += 8) {
        __m256i va = _mm256_loadu_si256((__m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((__m256i *)(b + i));
        __m256i vr = _subm_avx2(va, vb, q);
        _mm256_storeu_si256((__m256i *)(result + i), vr);
    }
}
#endif
/*! \brief Специальный вид инверсии для алгоритма редуцирования $\lfloor (2^{64}-q)/q \rfloor$ 
    Ur = 2^{32}+INVL(q);
 */
static inline uint32_t INVL(uint32_t v) {
    return ((unsigned __int64)(-v)<<32)/v;
}
static inline uint32_t MULM(uint32_t a, uint32_t b, uint32_t q) {
    return ((unsigned __int64)a*b)%q;
}
static inline uint32_t SQRM(uint32_t a, uint32_t q) {
    return ((unsigned __int64)a*a)%q;
}
static inline uint32_t MOD(uint64_t a, uint32_t q) {
    return a%q;
}
static inline uint32_t SLM(uint32_t a, uint32_t q) {
    return ((unsigned __int64)a<<1)%q;
}
static inline uint32_t SRM(uint32_t a, uint32_t q) {
    return (a&1)? (a+(unsigned __int64)q)>>1: (a>>1);
}
/*! Редуцирование по модулю простого числа, с использованием Барретта */
static inline uint32_t MODB(uint64_t a, uint32_t q, uint32_t U) {
    uint64_t c2 = a + U*(a >>32);
    uint64_t c4 = a - q*(c2>>32);
    return  (c4>= q)? c4 - q: c4;
}
/*! \brief Хеширование данных 
    \param h - хеш-код 0 <= h < q
    \param d - данные 
    \param q - модуль простого числа
    \param U - специальная константа Барретта U = ⌊(2^{64} −q)/q⌋ 
    \param K - сдвиговая константа, K = 2^N mod q
    \return хеш-код 32 бита
 */
static inline uint32_t FOLD(uint32_t h, uint32_t d, uint32_t q, uint32_t U, uint32_t K) {
    uint64_t c1 = d  + ((uint64_t)K*h); 
#if 0// Ur = ⌊2^{64}/q⌋ 
    uint64_t c2 = U*(c1>>31) + (c1>>31<<32);
    uint64_t c4 = c1 - q*(c2>>33);// (a<<16)+(c2>>33)
#else
    uint64_t c2 = c1 + U*(c1>>32);// {c1.l, 0} + {U, 1} * {c1.h, 0}-> {mad(), c1.h + mad_hi()}
    uint64_t c4 = c1 - q*(c2>>32);// {c1.l, c1.h} - {q, 0} * {c2.h, 0}-> {mad(), c1.h + mad_hi()}
#endif
    return  (c4-q< c4)? c4 - q: c4;
//    return  (c4>= q)? c4 - q: c4;
}
static inline uint32_t FOLD_(uint32_t h, uint32_t d, uint32_t q, uint32_t U, uint32_t K) {
    uint64_t c1 = d  + ((uint64_t)h<<32); 
    return  c1%q;
}
/*! Возведение в степень по модулю $m\cdot 2^a (\mod q)$
 */
static uint32_t ldexpm(const uint32_t m, uint32_t a, uint32_t q)
{
	uint32_t r = 2;
	uint32_t s = m;
    while (a!=0) {
		if (a&1) 
			s = ((uint64_t)s*r)%q;
		r = ((uint64_t)r*r)%q;// по таблице 
		a>>=1;
	}
	return s;
}
#include <math.h>
float convert_lwe_to_f32(uint32_t m, int ex, const uint32_t *Kr, uint32_t q){
    m = (m*(uint64_t)Kr[ex+148])%q; // умножение на сдвиговую константу по модулю `q`
    int i = (m>>31)?(int)m-q:m; // восстановить знак
    return ldexpf(i, ex-24);
}
uint32_t convert_f32_to_lwe(float f, int *ex, const uint32_t *K, uint32_t q){
    f = frexpf(f,ex);     // загрузка экспоненты, результат в интервале [0.5,1)
    int32_t i = (f*(1u<<24));// округление до ближайшего целого RNE
    //if (i<0) i = q+i;
    uint32_t m = (i<0)?q+i:i;
    m = (m*(uint64_t)K[*ex+148])%q;
    return m;
}
uint32_t convert_f32_to_lwe_(float f, const uint32_t *K, const uint32_t *Kr, uint32_t q){
    const uint32_t M = (1u<<24);
    union {
        float f;
        uint32_t u;
    } v = {.f = f};
    
    int ex;
    f = frexpf(f,&ex);     // загрузка экспоненты, результат в интервале [0.5,1)
    int32_t i = (f*M);// округление до ближайшего целого RNE
    if (i<0) i = q+i;
    uint32_t m = i;
    m = (m*(uint64_t)K[ex+148])%q;
    //return m;
    //printf ("f=0x%08X m=%1.2f exp=%d: %08X\n", v.u, f, ex, m);
    if (v.f != convert_lwe_to_f32(m, ex, Kr, q))
        printf("fail %f != %f\n", f, convert_lwe_to_f32(m, ex, Kr, q));
/*    m = (m*(uint64_t)Kr[ex])%q;
    if (m>>31) i=i-q;// восстановить знак
    if (v.f != ldexpf(i, ex-24)) printf("fail %f != %f\n", f, ldexpf(i, ex));
    */
    return m;    // умножение на сдвиговую константу по модулю `q`
}
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
/*! \brief Специальный вид инверсии для алгоритма редуцирования $\lfloor (2^{64}-q)/q \rfloor$ */
static inline uint64_t INVL128(uint64_t v) {
    return ((unsigned __int128)(-v)<<64)/v;
}
/*! Возводит в степень по модулю b^a mod P 
	очень быстрый тест!
 */
typedef unsigned int __attribute__((mode(TI)))   uint128_t;
static uint64_t POWM128(const uint64_t b, uint64_t a, const uint64_t P)
{
	uint64_t r;
	r = b;
	uint64_t s = 1;
	int i;
    while (a!=0) {
		if (a&1) 
			s = ((uint128_t)s*r)%P;
		r = ((uint128_t)r*r)%P;
		a>>=1;
		//if (r==b) return 0;
	}
	return s;
}
/// Reference implementation
#define A0  0xFEA0u// 0xFF80u
#define Q0  ((A0<<16)-1)
/*! функция хеширования эквивалентна FOLD(K=A), q=(A<<16)-1
 */
uint32_t mwc32_hash(uint32_t h, uint16_t d, uint32_t q, uint32_t a){
    h = (0xFFFF&h)*a + (h>>16) + d;// rotl(h, 16) - (0x10000 - a)*(uint16_t)h
    if (h>= q) h-=q;
    return h;
}

uint32_t mwc32_hash_16(uint32_t h, uint16_t d, uint32_t q, uint32_t a){
    h += d;
    h = (0xFFFF&h)*a + (h>>16);
//    if (h>= q) h-=q;
    return h;
}
/*! \brief Хэш MWC32 Сдвиг 2^{-16} - редуцирование
    \param h - хеш-код 32 бита
    \param d - вектор 16 с распаковкой на 32 бита x16
    \param q - модуль $2^{31} < q < 2^{32}$, 
    \param a - константа $a < 2^{16}$
    \return $h*a + d \mod q$
 */

#if 1

/*! \brief вычисляем константу для замены A/B == ((A*C0)>>32 + A)>>(n-32)
 */
static uint32_t div_c0(uint32_t b, int *nd) 
{
	uint32_t C;
	int k = __builtin_ctz(b);// количество ноликов в младшей части числа
		// count trailing zeros
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
}/*! \brief проверка по множеству целых чисел без знака 32 бита 
	для выражения uint32_t Q = (((A*(uint64_t)C0)>>32)+A)>>(n0-32);
 */
static int verify32(uint32_t B, uint32_t C0, int n0) {
	uint32_t A;
	for (A=(~0UL); A!=0; --A){// все числа 32 бит, кроме нуля
		uint32_t q = (((A*(uint64_t)C0)>>32)+A)>>(n0-32);
		if (q != (A/B)) {
			printf("fail A=0x%08X A/B=0x%08X Q=0x%08X \r\n", A, A/B, q);
			return 1;
		}
	}
	return 0;
}
// максимальный период
uint32_t mwc32_period(uint32_t a){
	return (a<<15)-2;
}
/*! Модульное уполовинивание */
static inline uint32_t hlvm(uint32_t v, uint32_t P){
	return (v&1)? ((uint64_t)v+P)>>1: (v>>1);
}
/*! Порядок мультипликативной группы по модулю простого числа q=A*2^{16} -1 
 */
uint32_t mwc32_length(uint32_t gen, uint32_t P){
    uint32_t a = gen;
    uint32_t i;
    for ( i=0; i<P; ++i) {// можно использовать модульное уполовинивание или удвоение
        a = hlvm(a, P);
        if (a == 1) break;
    }
    return i;
}
int main(int argc, char **argv){
    const uint32_t U0 = INVL(Q0);
#if __AVX512F__
    __m512i E = _mm512_set1_epi32(1);
    __m512i Z = _mm512_set1_epi32(0);
#endif
    uint64_t ur = (((uint128_t)1<<64))/Q0; // проверка (1<<32) + U0
    printf("Prime %08x, Ur %08x %08llx\n", Q0, U0, ur);
    if (1) {
        const uint32_t A1 = (0xFF80u);// период повтора 7fbffffe `-16`
        const uint32_t Q1 = (A1<<16)-1; 
        uint32_t K[256+24];
        uint32_t Kr[256+24];// обратные степени 2^{-n}
        uint32_t hlv = POWM(2, mwc32_period(A1), Q1);
        for (int i=0; i<256+24; i++){// заполнение таблицы констант
            K[i] = POWM(2, i, Q1);
            Kr[i]= POWM(hlv, i, Q1);
        }
        if (0) for (int i=0; i<=255; i++){
            float f = i*(-781.33f);
            uint32_t w = convert_f32_to_lwe_(f, K, Kr, Q1);
        }
        for (uint32_t i=0; i<~0u; i++){
            int ex;
            float f = *(float*)&i;
            if (isnan(f)) continue;
            //if (!isnormal(f)) continue;
            uint32_t w = convert_f32_to_lwe(f, &ex, K, Q1);
            float g = convert_lwe_to_f32(w,ex, Kr, Q1);
            if (f != g){
                printf("fail cvt %g != %g ^%d\n", f, g, ex);
            }
        }
    }
    if (1) {
        uint32_t q0 = 8380417;
        int len = mwc32_length(1, q0);
        printf("Prime %08x, order %08x\n", q0, len);
        
        uint64_t ur = (((uint128_t)1<<64)-0)/q0;
        //uint64_t ur = INVL(q0);
        int nd = 64 - __builtin_clz(q0);// количество ноликов в старшей части числа
        // uint64_t u0 = (uint64_t)(-q0)/q0;
        uint32_t u0 = div_c0(q0, &nd);
        int n0 = 32 - __builtin_clz(q0);
        
        printf("Prime %08x, Q = x%08x (%d bit) Ur=%08llx ", q0, u0, n0, ur);
        if (verify32(q0, u0, nd)) return 0;
        printf(" ..ok\n");
    }
    if (1) {
        uint32_t q0 = 8380417;
        int len = mwc32_length(1, q0);
        printf("Prime %08x, order %08x %s\n", q0, len, len==(q0/2-1)?"..ok":"");
        for (int i=0; i<=255; i++){
            uint32_t a = POWM(2, i, q0);
            // printf("%02x: %08x\n", i, a);
        }
    }
    if (1) {// найти сдвиговые константы
        uint32_t A1 = (0xFF80u);// период повтора 7fbffffe `-16`
        uint32_t Q1 = (A1<<16)-1; 

    }
    const int n = 31;
    uint32_t a1 = 1;
    uint32_t a2 = 1;
    uint32_t h1 = 1, h2 = 1, h3 = 1;
    for (uint64_t i=1; i<=(uint64_t)Q0<<(32-n); i++){// проверяем путем перебора всех значений
        if (0 && MODB(i<<n, Q0, U0) != MOD(i<<n, Q0)) {// проверка алгоритма MODB
            printf("%llx: %08x != %08x\n", i, MODB(i<<n, Q0, U0), MOD(i<<n, Q0));
            break;
        }
#if __AVX512F__
        if (1) {// проверка векторной реализации
            __m512i U = _mm512_set1_epi64(U0);   // barrett constant
            __m512i q = _mm512_set1_epi64(Q0);   // prime

            uint64_t a = (i<<n)-1;
            __m512i x = _mm512_set1_epi64(a);
            __m512i y = _barret(x, q, U);
            __m128i z = _mm512_extracti64x2_epi64(y, 0);
            uint64_t v = _mm_extract_epi64(z, 0);
            uint64_t w = MOD(a, Q0);
            if (v != w) {
                printf("%llx: %08llx %08llx\n", i, v, w);
                break;
            }
        }
#elif __AVX2__
        if (1) {// проверка векторной реализации
            __m256i U = _mm256_set1_epi64x(U0);   // barrett constant
            __m256i q = _mm256_set1_epi64x(Q0);   // prime

            uint64_t a = (i<<n)-1;
            __m256i x = _mm256_set1_epi64x(a);
            __m256i y = _barret_avx2(x, q, U);
            __m128i z = _mm256_extracti128_si256(y, 0);
            uint64_t v = _mm_extract_epi64(z, 0);
            uint64_t w = MOD(a, Q0);
            if (v != w) {
                printf("%llx: %08llx %08llx\n", i, v, w);
                break;
            }
        }
#endif
        if (1) {// проверка алгоритма FOLD
            uint64_t a = ~i;
            uint32_t f1 = FOLD(Q0-1, a, Q0, U0, -Q0);
            f1 = FOLD(f1, a, Q0, U0, -Q0);
            uint32_t f2 = FOLD_(Q0-1, a, Q0, U0, -Q0);
            f2 = FOLD_(f2, a, Q0, U0, -Q0);
            if (f1 != f2) {
                printf("fold %llx: %08x %08x\n", i, f1, f2);
                break;
            }
        }
        if (1) {
            uint32_t A1 = (0xFF80u);// период повтора 7fbffffe `-16`
            uint32_t Q1 = (A1<<16)-1; 
            a1 = SLM(a1, Q1);
            a2 = SRM(a2, Q1);
            if (a1 == 1) {
                printf("K_%llx: %08x\n", i, a1);
                //break;
            }
            if (a2 == A1) {
                printf("K_-%llx: %08x\n", i, a2);
                //break;
            }
        }
        h1 = mwc32_hash(h1, (uint16_t)i, Q0, A0);
        h3 = mwc32_hash_16(h3, (uint16_t)~i, Q0, A0);
        h2 = FOLD(h2, (uint16_t)i, Q0, U0, A0);
        if (h3 >= Q0) {
            printf("MWC_%llx: %08x >= q\n", i, h3);
            //break;
        }
        if (h1 != h2) {
            printf("MWC_%llx: %08x %08x\n", i, h1, h2);
            break;
        }
        
    }
    return 0;
}
#endif