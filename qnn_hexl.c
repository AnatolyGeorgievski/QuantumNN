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

Протестировать операции на простых числах
2^{31} -1 (Mersenne 31)
2^{31} -2^{27} +1 (Baby Bear)
2^{31} -2^{24} +1 (Koala Bear)
2^{31} -2^{30} +1 (Teddy Bear)
2^{64} -2^{32} +1 (Goldilocks)

https://eprint.iacr.org/2012/470.pdf

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
9.   return 𝑐_4
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
/*! \brief Signed Montgomery reduction 
    \see (https://arxiv.org/pdf/2306.01989)
*/
static inline __m512i _montgomery(__m512i d, __m512i q, __m512i qm) {
    __m512i m = _mm512_mul_epi32(d, qm);
    __m512i t = _mm512_mul_epi32(m, q);
    __m512i r = _mm512_sub_epi64(d, t);
    return r;// _mm256_srai_epi64(r, 32);
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
    __m512i b = _mm512_srli_epi64(a, 32);
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
/*! \brief ротация полинома (b) по модулю (x^N + 1) и сложение с полином (a)
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
/*! \brief ротация полинома (b) по модулю (x^N + 1) и вычитание из полинома (a)
    \param a First polynomial (array of 256 uint32_t coefficients).
    \param b Second polynomial (array of 256 uint32_t coefficients).
    \param r Output polynomial (array of 256 uint32_t coefficients).
 */
static inline void poly_xtime_subm(const uint32_t *a, const uint32_t *b, uint32_t *r) {
    __m512i q = _mm512_set1_epi32(Q_PRIME);
    __m512i c = _mm512_loadu_epi32((const void *)(b + 256-16)); // перенос    
    c = _mm512_sub_epi32(q, c);         // c = q - c
    for (int i = 0; i < 256; i += 16) {
        __m512i va = _mm512_loadu_epi32((const void *)(a + i));
        __m512i vb = _mm512_loadu_epi32((const void *)(b + i));
        c = _mm512_alignr_epi32 (vb,c, 15);
        va = _subm(va, c, q);
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
/*! \brief signed montgomery reduction
    \param d input value (uint64_t)
    \param q modulus (uint32_t)
    \param qm -q^{-1} mod 2^32 (uint32_t)

    \note используется в операциях умножения, объединяется методом _unpackhi_epi32
 */
static inline __m256i _montgomery_avx2(__m256i d, __m256i q, __m256i qm) {
    __m256i m = _mm256_mul_epi32(d, qm);
    __m256i t = _mm256_mul_epi32(m, q);
    __m256i r = _mm256_sub_epi64(d, t);
    return r;// _mm256_srai_epi64(r, 32);
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

/*! \defgroup _modular_math Группа методов для тестирования, не использует явную векторизацию.
 \{
 */
/*! \brief Специальный вид инверсии для алгоритма редуцирования $\lfloor (2^{64}-q)/q \rfloor$ 
    Ur = 2^{32}+INVL(q);
 */
static inline uint32_t INVL(uint32_t v) {
    return ((unsigned __int64)(-v)<<32)/v;
}
static inline uint32_t MULM(uint32_t a, uint32_t b, uint32_t q) {
    return ((unsigned __int64)a*b)%q;
}
static inline uint32_t ADDM(uint32_t a, uint32_t b, uint32_t q) {
    return ((unsigned __int64)a + b)%q;
}
static inline uint32_t SUBM(uint32_t a, uint32_t b, uint32_t q) {
    return ((unsigned __int64)a + q - b)%q;
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
/* Shoup modular multiplication. The most time-consuming primitive in NTT algorithms
is modular multiplication between the coefficients of a and the fixed (precomputed)
powers of ω. */
uint32_t soup_MULM(uint32_t a, uint64_t b, uint32_t p){
    uint64_t w = (double)(b<<32)/p;
    uint64_t q = (a*w)>>32;
    uint64_t r = a*b - q*p;
    return  (r-p< r)? r - p: r;// min(r, r-p)
}
uint32_t soup2_MULM(uint32_t a, uint64_t b, uint32_t p){
    uint64_t w = b;
    int l = __builtin_clzll(w);// количество ноликов в старшей части числа
    w = (double)(w<<(l))/p;// ⌊((w≪n)⋅β)/p⌉ округление к ближайшему целому

    if (w<(1uLL<<32)) {
        w<<=1; l++;
    }
    w -= (1uLL<<32);

    //printf ("%08llx %08llx %08llx", b, (b<<(l)), w);

    uint64_t q = (((a*w)>>32) + a)>>(l-32);
    uint64_t r = a*b - q*p;
    return  (r-p < r)? r - p: r;
}
/*! \brief Редукция Монтгомери 
    Работает на z<p, q<2^{31}

    \param q - prime
    \param p - 
    \return $zR \mod q$ , R = 2^{-32} 
 */
uint32_t  mont_modm(uint64_t z, uint32_t q, uint64_t p){
    uint32_t m = z * p; // low product z_0 (1/q)
    z = (z + m*(uint64_t)q)>>32; // high product
    if (z>=q) z-=q;
    return  (uint32_t)z;
}
// Функция для вычисления q^{-1} mod 2^32
uint32_t mod_inverse(uint32_t q) {
    uint32_t q_inv = 1;
    for (int i = 1; i < 32; i++) {
        if (((q * q_inv) & ((~0uL)>>(31-i))) != 1) {
            q_inv += (1u<<i);
        }
    }
    return q_inv;
}
/*! \brief Хеширование данных 
    \param h - хеш-код 0 <= h < q
    \param d - данные 
    \param q - модуль простого числа
    \param U - специальная константа Барретта U = ⌊(2^{64} -q)/q⌋ 
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
static inline uint32_t INVM(uint32_t a, const uint32_t q){
    return POWM(a, q-1, q);
}

/*! \brief Чисто-теоретическое преобразование на кольце $\mathbb{Z}_q/\langle x^N + 1\rangle$ 
    \param a First polynomial (array of 256 uint32_t coefficients). return NTT(a) in bit-reversed order.
    \param gamma store powers of gamma in bit-reverse ordering
    \param N power of 2
    \param q prime modulus $q \equiv 1 \mod 2N$


    * [2012.01968](https://arxiv.org/pdf/2012.01968)
    * [2103.16400](https://arxiv.org/pdf/2103.16400) 
*/
uint32_t* NTT(uint32_t *a, const uint32_t *gamma, unsigned int N, uint32_t q){
    unsigned int i, j, k, m;
    uint32_t t;
    t = N;
    for (m = 1; m < N; m = 2*m) {
        t = t/2;
        for (i=0; i<m; i++) {
            unsigned j_1 = 2*i*t;
            unsigned j_2 = j_1 + t;
            uint32_t w = gamma[m+i];
            for (j = j_1; j < j_2; j++) {
                uint32_t x_0 = a[j];
                uint32_t x_1 = a[j+t];
                uint32_t wx = MULM(x_1, w, q);
                a[j]   = ADDM(x_0, wx, q);
                a[j+t] = SUBM(x_0, wx, q);
            }
        }
    }
    return a;
}
/*! \brief Gentleman-Sande (GS) Radix-2 InvNTT 
    \param a First polynomial $a = (a_0, a_1, ..., a_{N-1})$ array of 256 uint32_t coefficients in bit-reverse ordering).
    \param gamma store powers of $\gamma^{-1}$ in bit-reverse ordering
    \param N power of 2
    \param N_inv inverse of N modulo $q$
    \param q prime modulus satisfying $q \equiv 1 \mod 2N$
 */
uint32_t* invNTT(uint32_t *a, const uint32_t *gamma, unsigned int N, uint32_t q){
    unsigned int i, j, k, m;
    unsigned int t, j_1, j_2;
    t = 1;
    for (m = N; m > 1; m = m/2) {
        j_1 = 0;
        unsigned int h = m/2;
        for (i=0; i<h; i++) {
            j_2 = j_1 + t;
            uint32_t w = gamma[h+i];
            for (j = j_1; j < j_2; j++) {
                uint32_t x_0 = a[j];
                uint32_t x_1 = a[j+t];
                a[j]   = ADDM(x_0, x_1, q);
                a[j+t] = MULM(SUBM(x_0, x_1, q), w, q);
            }
            j_1 = j_1 + 2*t;
        }
        t = t*2;
    }
    uint32_t N_inv = INVM(N, q); //8347681 = 256^{-1} mod q
    for (j = 0; j < N; j++) {// эту операцию совместить со следующей
        a[j] = MULM(a[j], N_inv, q);
    }
    return a;
}
uint8_t RevBits8(uint8_t x){
    x = ((x & 0x55) << 1) | ((x & 0xAA) >> 1);
    x = ((x & 0x33) << 2) | ((x & 0xCC) >> 2);
    x = ((x & 0x0F) << 4) | ((x & 0xF0) >> 4);
    return x;
}
uint32_t RevBits(uint32_t x){
    x = ((x & 0x55555555) << 1) | ((x & 0xAAAAAAAA) >> 1);
    x = ((x & 0x33333333) << 2) | ((x & 0xCCCCCCCC) >> 2);
    x = ((x & 0x0F0F0F0F) << 4) | ((x & 0xF0F0F0F0) >> 4);
    x = ((x & 0x00FF00FF) << 8) | ((x & 0xFF00FF00) >> 8);
    x = ((x & 0x0000FFFF) <<16) | ((x & 0xFFFF0000) >>16);
    return x;
}
/*! \brief Precompute powers of $\gamma$ and $\omega$ for NTT and InvNTT
 */
void nnt_precompute(uint32_t* gamma, uint32_t *omega, uint32_t* g_inv, uint32_t* o_inv,
    unsigned int N, uint32_t q) 
{
    uint32_t N_inv  = INVM(N, q);
    uint32_t gamma1 = gamma[1];
    uint32_t omega1 = omega[1];
    uint32_t o_inv1 = INVM(omega1, q);
    uint32_t g_inv1 = INVM(gamma1, q);
    gamma[0] = 1; g_inv[0] = 1; 
    omega[0] = 1; o_inv[0] = 1;
    uint32_t k = RevBits(1)>>8;
    gamma[k] = gamma1;
    omega[k] = omega1;
    g_inv[k] = g_inv1;
    o_inv[k] = o_inv1;

    uint32_t gm = gamma1;
    uint32_t om = omega1;
    uint32_t gi = g_inv1;
    uint32_t oi = o_inv1;
    for (uint32_t i=2; i<N; i++) {// обратный порядок бит.
        uint32_t k = RevBits(i)>>8;
        gamma[k] = gm = MULM(gm, gamma1, q);
        omega[k] = om = MULM(om, omega1, q);
        g_inv[k] = gi = MULM(gi, g_inv1, q);
        o_inv[k] = oi = MULM(oi, o_inv1, q);
    }
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
uint32_t mwc32_next(uint32_t h, const uint32_t A){
    h = (h&0xFFFFu)*A + (h>>16);
    return h;
}
int32_t mwc32s_next(int32_t h, const int16_t A){
    h = (h&0xFFFFu)*A - (h>>16);
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
    int n;
	int k = __builtin_ctz(b);// количество ноликов в младшей части числа
		// count trailing zeros
	b>>=k;
	if (b==1) {
		n = 0;
		C = 0;//0x1ULL<<32;
	} else
	{
		n = 32 - __builtin_clz(b);// количество ноликов в старшей части числа
		C = (uint64_t)(((1ULL<<32)-b)<<n)/b + (1ULL<<n) + 1;
	}
	*nd=k+n;
	return C & 0xFFFFFFFFUL;
}
static uint32_t div_c1(uint32_t b, int *nd) 
{
	uint32_t C;
    int n;
	int k = __builtin_ctz(b);
	b>>=k;
	if (b==1) {
		n = 0; C = 0;//0x1ULL<<32;
	} else
	{
		n = 32 - __builtin_clz(b);// количество ноликов в старшей части числа
		C = (uint64_t)(((1ULL<<32)-b)<<n)/b + (1ULL<<n);
	}
	*nd=k+n;
	return C & 0xFFFFFFFFUL;
}
/*! \brief проверка по множеству целых чисел без знака 32 бита 
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
uint32_t mwc32_root(uint32_t gen, uint32_t P){
    uint32_t a = gen;
    int i = 1;
    while (a!=1) {
        a = MULM(a, gen, P);
        i++;
    }
    return i;
}

/*! \brief наибольший общий делитель GCD, бинарный алгоритм */
uint32_t gcd(uint32_t u, uint32_t v) {
    if (u == 0) {
        return v;
    } else if (v == 0) {
        return u;
    }
    int i = __builtin_ctz(u);  u >>= i;
    int j = __builtin_ctz(v);  v >>= j;
    int k = (i<j)?i:j;

    while(u!=v) {// u,v нечетные
        if (u > v){// _swap(&u, &v);
			u -= v; // u теперь четное
			u >>= __builtin_ctz(u);
        } else {
			v -= u; // v теперь четное
			v >>= __builtin_ctz(v);
		}
    }
	return v << k;
}
/*! выбор генератора  - это утверждение эквивалентно символу якоби 
    
    jacobi(a, p) = -1, если a является квадратичным не-вычетом по модулю p.
    a^(p-1)/2 mod p -- эквивалентное утверждение. 
Legendre symbol:

$$\left({\frac {a}{p}}\right)\equiv a^{\tfrac{p-1}{2}}{\pmod {p}}.$$
    
 */
uint32_t mwc32_gen(uint32_t gen, uint32_t P){
    while (gcd(gen, P-1)!=1 || POWM(gen, (P-1)/2, P)==1)gen++;
    return gen;
}

#define BIT(x,n) (((x)>>(n))&1)
#define SWAP(x,y) do {    \
   typeof(x) _x = x;      \
   typeof(y) _y = y;      \
   x = _y;                \
   y = _x;                \
 } while(0)
/*! \brief Jacobi symbol 
    \param a - произвольное целое число
    \param m - положительное целое число, нечетное и не равное 1
    \return 1, если a является квадратичным вычетом по модулю m 
    (существует целое число x такое, что x^2 ≡ a (mod m)), 
    -1, если a является квадратичным невычетом по модулю m, 
    0, если a кратно m.

    Свойства
    jacobi(a,m) = jacobi(b,m) если a = b (mod m)
    jacobi(a,m) = 0, если a кратно m (gcd(a,m) \neq 1)
    jacobi(a,m) = jacobi(a, m/a) если a нечётно и a кратно m
    
    Си́мвол Яко́би — теоретико-числовая функция двух аргументов, введённая К. Якоби в 1837 году. 
    Является квадратичным характером в кольце вычетов.
    
    Символ Якоби обобщает символ Лежандра на все нечётные числа, большие единицы.
 */
int jacobi(uint64_t a, uint64_t m) 
{
	a = a%m;
	int t = 1;
	unsigned m1= BIT(m,1);
	while (a!=0){
		int z = __builtin_ctzll(a);
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
/*! выбор генератора - квадратичный не-вычет 

    \see (https://eprint.iacr.org/2012/470.pdf) 
    An algorithm for generating a quadratic non-residue modulo
    
 */
uint32_t generate_quadratic_non_residue(uint32_t gen, uint32_t P){
    if ((P&3)==3) return P-1;
    if ((P&7)==5) return 2;

    while (jacobi(gen,P)!=-1) gen++;
    return gen;
}
/*! \brief Поиск корня степени N= 2^s по модулю простого числа q
    \return (r^2)^N = 1 mod q
 */
uint32_t ntt_root(uint32_t N, uint32_t q){
    uint32_t gen =  3;// выбор генератора - квадратичный не-вычет
    while (jacobi(gen,q)!=-1) gen++;
    uint32_t r = gen;
    do {
        while ( POWM(r, N+N, q)!=1)
            r = MULM(r, gen, q);
    } while (POWM(r, N, q)==1);
    return r;
}
uint32_t primes[] = {
    (1u<<23) -(1u<<13)+1, // NIST
//            (1u<<31) -1, // Mersenne 31
    (1u<<31) -(1u<<27)+1,
    (1u<<31) -(1u<<25)+1,
            (1u<<31) -(1u<<24)+1,
    (1u<<31) -(1u<<19)+1,
    (1u<<31) -(1u<<17)+1,
    (1u<<31) -(1u<< 9)+1,

    (0x7efcu<<16) + 1,
    (0x7db2u<<16) + 1,
    (0x7bffu<<16) + 1,
    (0x7b27u<<16) + 1,
    (0x7a55u<<16) + 1,
    (0x7a46u<<16) + 1,
    (0x79efu<<16) + 1,
    (0x78c0u<<16) + 1,

    (0xff7bu<<16) + 1,
    (0xff03u<<16) + 1,
    (0xfe04u<<16) + 1,
    (0xfcf6u<<16) + 1,
    (0xfcd2u<<16) + 1,
    (0xfb13u<<16) + 1,
    (0xfa8fu<<16) + 1,
    (0xf9eau<<16) + 1,
    (0xf9d5u<<16) + 1,
    (0xf960u<<16) + 1,
    (0xf921u<<16) + 1,
    (0xf915u<<16) + 1,
    (0xf8d6u<<16) + 1,
    (0xf8c7u<<16) + 1,
    (0xf804u<<16) + 1,

    (0xFFF0u<<16)+1,// 2^32 - 2^20 +1
    (0xC000u<<16)+1,// 2^32 - 2^30 +1

    //0xff7b0001, 0xff030001, 0xfe040001, 0xfcf60001, 0xfcd20001,
    0x7f000001, 0x7e100001, 0x7e000001, 0x7d200001, 0x7ce00001, 0x7c800001, 0x7bd00001, 0x79500001, 0x78c00001, 0x78000001,
/*
prime=7f000001 gen=3 ord 007effff
prime=7e100001 gen=3 ord 03f07fff
prime=7e000001 gen=5 ord 00a7ffff
prime=7d200001 gen=3 ord 03e8ffff
prime=7ce00001 gen=7 ord 03e6ffff
prime=7c800001 gen=5 ord 014bffff
prime=7bd00001 gen=3 ord 008d7fff
prime=79500001 gen=5 ord 03ca7fff
prime=78c00001 gen=5 ord 01e2ffff
prime=78000001 gen=11 ord 03bfffff
*/
//    0xffdf0001, 0xffd50001, 0xffd30001, 0xff970001, 0xff930001, 0xff7b0001, 0xff6f0001, 0xff2b0001, 0xff0d0001, 0xff030001, 0xff010001,
    //0x7fe01001, 0x7fc5d001, 0x7fabf001, 0x7f555001, 0x7f3c9001, 0x7f0ff001, 

// не содержат корней от -1
#if 0
    (0xFFEAu<<16) - 1,
//    (0xFFD7u<<16) - 1,
//    (0xFFBDu<<16) - 1,
    (0xFFA8u<<16) - 1,
//    (0xFF9Bu<<16) - 1,
//    (0xFF81u<<16) - 1,
    (0xFF80u<<16) - 1,
//    (0xFF7Bu<<16) - 1,
//    (0xFF75u<<16) - 1,
    (0xFF48u<<16) - 1,
//    (0xFF3Fu<<16) - 1,
    (0xFF3Cu<<16) - 1,
    (0xFF2Cu<<16) - 1,
    (0xFF09u<<16) - 1,
    (0xFF03u<<16) - 1,
    (0xFF00u<<16) - 1,
//    (0xFEEBu<<16) - 1,
    (0xFEE4u<<16) - 1,
    (0xFEA8u<<16) - 1,
    (0xFEA5u<<16) - 1,
    (0xFEA0u<<16) - 1,
    (0xFE94u<<16) - 1,
    (0xFE8Bu<<16) - 1,
    (0xFE72u<<16) - 1,
    (0xFE4Eu<<16) - 1,
    (0xFE30u<<16) - 1,
    (0xFE22u<<16) - 1,
//    (0xFE15u<<16) - 1,
    (0xFE04u<<16) - 1,
    (0xFE00u<<16) - 1,
/*
prime=ffe9ffff A=ffea gen= 5 ord 7ff4fffe
prime=ffa7ffff A=ffa8 gen= 5 ord 7fd3fffe
prime=ff7fffff A=ff80 gen= 3 ord 7fbffffe
prime=ff47ffff A=ff48 gen=13 ord 7fa3fffe
prime=ff3bffff A=ff3c gen=13 ord 7f9dfffe
prime=ff2bffff A=ff2c gen= 3 ord 7f95fffe
prime=feffffff A=ff00 gen= 7 ord 7f7ffffe
prime=fee3ffff A=fee4 gen= 3 ord 7f71fffe
prime=fea7ffff A=fea8 gen= 3 ord 7f53fffe
prime=fe9fffff A=fea0 gen= 5 ord 7f4ffffe
prime=fe93ffff A=fe94 gen=11 ord 7f49fffe
prime=fe71ffff A=fe72 gen= 3 ord 7f38fffe
prime=fe4dffff A=fe4e gen= 3 ord 7f26fffe
prime=fe2fffff A=fe30 gen= 3 ord 7f17fffe
prime=fe21ffff A=fe22 gen= 5 ord 7f10fffe
prime=fe03ffff A=fe04 gen= 5 ord 7f01fffe
prime=fdffffff A=fe00 gen= 3 ord 7efffffe
*/
#endif
    };

int main(int argc, char **argv){
    const uint32_t U0 = INVL(Q0);
#if __AVX512F__
    __m512i E = _mm512_set1_epi32(1);
    __m512i Z = _mm512_set1_epi32(0);
#endif
    uint64_t ur = (((uint128_t)1<<64))/Q0; // проверка (1<<32) + U0
    printf("Prime %08x, Ur %08x %08llx\n", Q0, U0, ur);
    if (0) {
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

    if (1) {// mwc32s
        if (0) for (int k = 1; k<= 0x200; k++) {// MWC32
            uint32_t p = (1uLL<<32) - (k<<16) - 1;
            uint32_t h = 3;//p>>16;
            if ((p&3)==3) h = p-1;// выбор генератора по умолчанию
            else {
                h=2;
                while (jacobi(h, p)!=-1) h++;
            }
            uint32_t g = h;
            uint32_t a = (p>>16)+1;
            for(uint32_t i=0; i<(uint32_t)p+1; i++){
                h = mwc32_next(h, a);
                if (h==g) {
                    if ((i&0xFFFF)==0xFFFE && i==p/2-1)
                    printf("prime=%08x A=%04x gen=%2d ord %08x\n", p, a, g-p, i);
                    break;
                }
            }
        }
        if (1) for (int k = 1; k< 0x3FFF; k++) {// MWC32s
            int32_t p = (1uLL<<31) - (k<<16) + 1;
            int32_t h = 3;//p>>16;
            while (jacobi(h, p)!=-1) h++;
            int32_t g = h;
            for(uint32_t i=0; i<(uint32_t)p+1; i++){
                h = mwc32s_next(h, p>>16);
                if (h==g) {
                    if ((i&0x7FFE)==0x7FFE)
                    printf("prime=%08x gen=%2d ord %08x\n", p, g, i);
                    break;
                }
            }
        }
        for (int k = 0; k< sizeof(primes)/sizeof(primes[0]); k++) {
            const int32_t p = primes[k];
            int32_t h = 5;//p>>16;
            while (jacobi(h, p)!=-1) h++;
            // h = generate_quadratic_non_residue(3, (uint32_t)p);
            int32_t g = h;
            //printf("mwc32s prime=%08x gen=%d\n",(uint32_t)p, g);
            for(uint32_t i=0; i<(uint32_t)p+1; i++){
                h = mwc32s_next(h, p>>16);
/*
                if (p>0 && (h>=p || h<=-p)) {
                    printf("fail %08x: %08x\n", i, h);
                    break;
                }
                if (p<0 && (h>=-p || h<=p)) {
                    printf("fail %08x: %08x\n", i, h);
                    break;
                } */
                if (h==g) {
                    if ((i&0xFFFF)==0xFFFF)
                        printf("mwc32s prime=%08x gen=%2d ord %08x\n",p,g, i);
                    break;
                }
            }
        }
        return 0;
    }
    if (1) {// montgomery test
        uint32_t p = primes[0];
        uint32_t pi = -mod_inverse(p); 
        uint32_t qinv=4236238847;
        printf ("prime=%u pi = %u %x \n",p, pi, p*qinv);
        for (int k = 1; k< 15/* sizeof(primes)/sizeof(primes[0]) */; k++) {
            uint32_t p = primes[k];
            uint32_t pi = mod_inverse(p);
            printf ("p=%08x pi = %08x %08x \n",p, pi, p*(-pi));
            for (uint32_t a =p; a>0; a--)
            {
                uint32_t b  = ((uint64_t)a<<32)%p;
                uint64_t a2 = (uint64_t)a*b;

                uint32_t r  = mont_modm(a2, p, -pi);
                uint32_t r2 = ((uint64_t)a*a)%p;
                if (r2!=r) {
                    printf ("r = %08x %08x p=%08x \n", r,r2,a);
                     break;
                }
            }
            //break;
        }
        return 0;
    }
    if (1) {// soup test
        for (int k = 13; k< sizeof(primes)/sizeof(primes[0]); k++) {
            uint32_t p = primes[k];
            for (uint32_t a =p+0xFFFF; a>0; a--){
                uint32_t b = ((uint64_t)a+3)%p;
                //uint32_t c = soup2_MULM(a, b, p);
                uint32_t c = soup_MULM(a, b, p);
                uint32_t d = ((uint64_t)a*b)%p;
                if (c != d) {
                    printf("fail %08x * %08x = %08x != %08x\n", a, b, c, d);
                    //_Exit(1);
                }
            }
            printf("Prime %08x, soup test done\n", p);
        }

    }
    if (1) {
        uint32_t g=3;
        while (gcd(g, Q0 - 1) != 1) g++;
        int len = mwc32_period(A0);
        printf("Prime %08x, order %08x gen=%x\n", Q0, len, g);
        uint32_t l;
        g = mwc32_gen(2, Q0); // найти генератор группы (2, 3, 5, 11)
        l = 0;//mwc32_root(g, Q0);// найти корень k-й степени = q-1
        uint32_t r  = POWM(g, (Q0-1)/2, Q0);// найти корень квадратный
        // x ≡ ± a^{(p+1)/4} (mod p). 
        uint32_t r256 = 2;
        uint32_t k = 16;
        for (int i=0; i<sizeof(primes)/sizeof(primes[0]); i++) {
            uint32_t m31 = primes[i];
            //uint32_t gen =  mwc32_gen(2, m31);
            uint32_t gen =  generate_quadratic_non_residue(3, m31);
            // printf("poly %x quadratic residue %x\n", m31, gen2);
            if ((m31-1)%(2*k)!=0) {
                printf("poly %x has no k-th root\n", m31);
                continue;
            }
            if (m31%4==3) {
//                uint32_t sq = POWM(gen, (m31+1)/4, m31);
                uint32_t sq = POWM(gen, (m31-1)/2, m31);
                uint32_t s2 = MULM(sq,sq, m31);
                uint32_t s4 = MULM(s2,s2, m31);
                printf("poly %x has no imaginary roots 0x%08x %08x %08x \n", m31, sq, s2, s4); // p ≡ 3 (mod 4), then x² ≡ -1 (mod p) has no solutions.

                continue;
            }
            uint32_t g = ntt_root(k, m31);
            r256 = SQRM(g,m31);
/*            r256 = gen;
            while (POWM(r256, k, m31)!=1 || POWM(r256, k/2, m31)==1) //r256++;
                r256 = MULM(r256, gen, m31);
            //if (gcd(r256, m31-1)==1) continue;

*/

            if (1) for (int j = 0; j < k; j++) {
                uint32_t r = POWM(r256, j, m31);
                printf("%08x ", r); 
            }
            printf("\n"); 
//            uint32_t r2 = POWM(gen, MULM(Q0-1, INVM(k, m31), m31), m31);
            printf("prime %08x gen=%x root=%x r^%d=%x %x\n", m31, gen, r256, k, 
                POWM(r256, k, m31), POWM(MULM(r256,r256, m31), k/2, m31));
/*          uint32_t k_inv = POWM(k, m31-2, m31); -- так не работает
            uint32_t a = MULM(m31-1, k_inv, m31);
            if (POWM(gen, a, m31) == r256) 
                printf("root k-th %x\n", a); */
        }
        uint32_t r2 = POWM(g, (Q0+1)/4, Q0);// найти корень 4й r2^4 = g^2 g^(q+1)/2 ≡ g^2 = -5
        if (MULM(r, r, Q0) == 1) printf("root2 gen=%d r=%x r2=%x %x\n", g, r, r2, 
            POWM(r2,2,Q0));
        printf("root gen=%x l=%x\n", g, l);

        uint32_t k_inv = POWM(k, Q0-2, Q0);
        printf("inv 2^{-8} %x\n", k_inv);
        uint32_t a = MULM(Q0-1, k_inv, Q0);
        printf("l/k 2^{-8} %x\n", a);
        uint32_t omega = POWM(g, a, Q0);
        printf("omega %x %x\n", omega, POWM(omega, k, Q0));
        if (POWM(omega, k, Q0) == 1) 
            printf("k-th root %x\n", omega);
        return 0;
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
            printf("AVX2: Barret reduction\n");
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