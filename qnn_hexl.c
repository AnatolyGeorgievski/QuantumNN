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
/*! \brief Специальный вид инверсии для алгоритма редуцирования $\lfloor (2^{64}-q)/q \rfloor$ 
    Ur = 2^{32}+INVL(q);
 */
static inline uint64_t INVL(uint32_t v) {
    return ((unsigned __int64)(-v)<<32)/v;
}

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
    return _mm512_mask_blend_epi32((__mmask16)0xaaaau, d0, _mm512_slli_epi64(d1,32));
//    return _mm512_mask_shuffle_epi32(d0, (__mmask16)0xaaaau, d1, 0x80); 
//    return _mm512_unpacklo_epi32(d0, d1);
}
static inline __m512i _maddm(__m512i a, __m512i b, __m512i c, __m512i q, __m512i u){
    __m512i d0 = _mm512_mul_epu32(a, b);
    d0 = _mm512_add_epi64(d0, _mm512_maskz_mov_epi32(0x5555, c));
    __m512i d1 = _mm512_mul_epu32(_mm512_srli_epi64(a,32), _mm512_srli_epi64(b,32));
    d1 = _mm512_add_epi64(d1, _mm512_srli_epi64(c, 32));
    d0 =  _barret(d0, q, u);
    d1 =  _barret(d1, q, u);
    return _mm512_mask_blend_epi32((__mmask16)0xaaaau, d0, _mm512_slli_epi64(d1,32));
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
    return _mm512_mask_blend_epi32((__mmask16)0xaaaau, a, _mm512_slli_epi64(b,32));
//    return _mm512_mask_shuffle_epi32(a, (__mmask16)0xaaaau, b, 0x80); 
//    return _mm512_unpacklo_epi32(a, b);
}

/*! \brief Element-wise polynomial multiplication modulo q.
    \param a First polynomial (array of 256 uint32_t coefficients).
    \param b Second polynomial (array of 256 uint32_t coefficients).
    \param result Output polynomial (array of 256 uint32_t coefficients).

    \note Processes 16 coefficients at a time using AVX512, for polynomials in R_q = Z_q[x]/(x^{256} + 1).
 */
static inline 
void poly_mulm(uint32_t *result, const uint32_t *a, const uint32_t *b, unsigned int N, uint32_t p) {
    const uint32_t Ur = INVL(p);
    __m512i q = _mm512_set1_epi32(p);
    __m512i q_= _mm512_set1_epi64(p);
    __m512i u = _mm512_set1_epi64(Ur);
    for (int i = 0; i < N; i += 16) {
        __m512i va = _mm512_loadu_epi32((const void *)(a + i));
        __m512i vb = _mm512_loadu_epi32((const void *)(b + i));
        __m512i vr = _mulm(va, vb, q_, u);// алгоритм умножения по модулю работает с числами 64 бита.
        _mm512_storeu_epi32(result + i, vr);
    }
}
static inline 
void poly_mulm_u(uint32_t *r, const uint32_t *a, const uint32_t b, unsigned int N, const uint32_t p) {
    const uint32_t Ur = INVL(p);
    __m512i q  = _mm512_set1_epi32(p);
    __m512i q_ = _mm512_set1_epi64(p);
    __m512i u  = _mm512_set1_epi64(Ur);
    __m512i vb = _mm512_set1_epi32(b);
    for (int i = 0; i < N; i += 16) {
        __m512i va = _mm512_loadu_epi32((const void *)(a + i));
        __m512i vr = _mulm(va, vb, q_, u);// алгоритм умножения по модулю работает с числами 64 бита.
        _mm512_storeu_epi32(r + i, vr);
    }
}
/*! \brief Element-wise polynomial addition modulo q.
    \param r Output polynomial (array of N coefficients).
    \param a First  polynomial
    \param b Second polynomial
    \param N power of 2
    \param p prime
    
    \note Processes 16 coefficients at a time using AVX512, for polynomials in R_q = Z_q[x]/(x^{256} + 1).
 */
static inline 
void poly_addm(uint32_t *r, const uint32_t *a, const uint32_t *b, const unsigned int N, const uint32_t p) {
    __m512i q = _mm512_set1_epi32(p);
    for (int i = 0; i < N; i += 16) {
        __m512i va = _mm512_loadu_epi32((const void *)(a + i));
        __m512i vb = _mm512_loadu_epi32((const void *)(b + i));
        __m512i vr = _addm(va, vb, q);
        _mm512_storeu_epi32(r + i, vr);
    }
}
static inline 
void poly_subm(uint32_t *r, const uint32_t *a, const uint32_t *b, const unsigned int N, const uint32_t p) {
    __m512i q = _mm512_set1_epi32(p);
    for (int i = 0; i < N; i += 16) {
        __m512i va = _mm512_loadu_epi32((const void *)(a + i));
        __m512i vb = _mm512_loadu_epi32((const void *)(b + i));
        __m512i vr = _subm(va, vb, q);
        _mm512_storeu_epi32(r + i, vr);
    }
}
/*! \brief ротация полинома (b) по модулю (x^N + 1) и сложение с полином (a)
    \param r Output polynomial (array of N coefficients).
    \param a First  polynomial
    \param b Second polynomial
    \param N power of 2
    \param p prime
 */
static inline void poly_xtime_addm(uint32_t *r, const uint32_t *a, const uint32_t *b, const unsigned int N, const uint32_t p) {
    __m512i q = _mm512_set1_epi32(p);
    __m512i c = _mm512_loadu_epi32((const void *)(b + N-16)); // перенос
    c = _mm512_sub_epi32(q, c);         // c = q - c
    for (int i = 0; i < N; i += 16) {
        __m512i va = _mm512_loadu_epi32((const void *)(a + i));
        __m512i vb = _mm512_loadu_epi32((const void *)(b + i));
        c = _mm512_alignr_epi32 (vb,c, 15);
        va = _addm(va, c, q);
        _mm512_storeu_epi32(r + i, va);
        c = vb;
    }
}
/*! \brief нелинейная функция для Poseidon2 
    \param s вектор состояния
    \param c константы (round constants)
    \param e степень 3,5,7,11 -- выбирается в зависимости от q (модуля) gcd(e, q-1)=1
 */
static inline __m512i _sbox(__m512i s, __m512i c,  unsigned int e){
    __m512i q = _mm512_set1_epi32(Q_PRIME);
    __m512i u = _mm512_set1_epi64(U_BARRETT);
    s = _addm(s, c, q);
    __m512i r = s;
    while ((e>>=1)!=0) {
        s = _sqrm(s, q, u);
        if (e&1)
            r = _addm(r, s, q);
    }
    return r;
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
/*! \brief Операция умножения полинома на скаляр и сложение с результатом r = r*x + a*b
    \param a First polynomial (array of 256 uint32_t coefficients).
    \param b scalar (uint32_t).
    \param r Output polynomial (array of 256 uint32_t coefficients).
    \param N степень полинома (x^N + 1)
 */
static inline void poly_xtime_madd(uint32_t *r, const uint32_t *a, const uint32_t b,  const unsigned int N, uint32_t p) {
    const uint64_t Ur = INVL(p);
    __m512i q = _mm512_set1_epi32(p);
    __m512i q_= _mm512_set1_epi64(p);// это первое отличие
    __m512i u = _mm512_set1_epi64(Ur);
    __m512i vb= _mm512_set1_epi32(b);
    __m512i c = _mm512_loadu_epi32((const void *)(r + N-16)); // перенос
    c = _mm512_sub_epi32(q, c);// операция выполняется для полинома (x^N+1), для (x^N-1) не нужна
    for (int i = 0; i < N; i += 16) {// если N=256
        __m512i vr = _mm512_loadu_epi32((const void *)(r + i));
        __m512i va = _mm512_loadu_epi32((const void *)(a + i)); 
        c  = _mm512_alignr_epi32 (vr,c, 15);
        va = _maddm(va, vb, c, q_, u);
//        va = _mulm(va, vb, q_, u);
//        va = _addm(va, c, q);
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
    return _mm256_blend_epi32 (d0, _mm256_slli_epi64(d1,32), 0xaa);
}
/*! \brief Element-wise polynomial multiplication modulo q.
    \param r Output polynomial (array of N coefficients).
    \param a First  polynomial
    \param b Second polynomial
    
    \note Processes 8 coefficients at a time using AVX2, for polynomials in R_q = Z_q[x]/(x^{256} + 1).
 */
static inline void poly_mulm(uint32_t *result, const uint32_t *a, const uint32_t *b, unsigned int N, uint32_t p) {
    uint32_t Ur = INVL(p);
    __m256i q = _mm256_set1_epi32(p);
    __m256i q_= _mm256_set1_epi64x(p);
    __m256i u = _mm256_set1_epi64x(Ur);
    for (int i = 0; i < N; i += 8) {
        __m256i va = _mm256_loadu_si256((__m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((__m256i *)(b + i));
        __m256i vr = _mulm_avx2(va, vb, q_, u);
        _mm256_storeu_si256((__m256i *)(result + i), vr);
    }
}
static inline void poly_mulm_u(uint32_t *result, const uint32_t *a, const uint32_t b, unsigned int N, uint32_t p) {
    uint32_t Ur = INVL(p);
    __m256i q = _mm256_set1_epi32(p);
    __m256i q_= _mm256_set1_epi64x(p);
    __m256i u = _mm256_set1_epi64x(Ur);
    __m256i vb= _mm256_set1_epi32(b);
    for (int i = 0; i < N; i += 8) {
        __m256i va = _mm256_loadu_si256((__m256i *)(a + i));
        __m256i vr = _mulm_avx2(va, vb, q_, u);
        _mm256_storeu_si256((__m256i *)(result + i), vr);
    }
}
/*! редуцирование (b) и сложение полиномов x^N + 1
N = 256 
 */
static inline void poly_xtime_addm(uint32_t *r, const uint32_t *a, const uint32_t *b, unsigned int N, uint32_t p) {
    __m256i q = _mm256_set1_epi32(p);
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
void poly_xtime_madd(uint32_t *r, const uint32_t *a, const uint32_t b, unsigned int N, uint32_t p) {
    uint32_t Ur = INVL(p);
    __m256i q = _mm256_set1_epi32 (p);
    __m256i q_= _mm256_set1_epi64x(p);
    __m256i u = _mm256_set1_epi64x(Ur);
    __m256i vb= _mm256_set1_epi32 (b);
    __m256i c = _mm256_loadu_si256((__m256i *)(r + 256-8)); // перенос
    for (int i = 0; i < N; i += 8) {
        __m256i va = _mm256_loadu_si256((__m256i *)(a + i));
        __m256i vr = _mm256_loadu_si256((__m256i *)(r + i));
        va = _mulm_avx2(va, vb, q_, u);
        c  = _mm256_alignr_epi32 (vr,c, 7);
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
static inline void poly_addm(uint32_t *result, const uint32_t *a, const uint32_t *b, unsigned int N, uint32_t p) {
    __m256i q = _mm256_set1_epi32(p);
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
    \param p модуль простого числа
 */
static inline void poly_subm(uint32_t *result, const uint32_t *a, const uint32_t *b, unsigned int N, uint32_t p) {
    __m256i q = _mm256_set1_epi32(p);
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
/*! \brief Возведение в степень по модулю $b^a (\mod q)$ */
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
//!\}
/* Shoup modular multiplication. The most time-consuming primitive in NTT algorithms
is modular multiplication between the coefficients of a and the fixed (precomputed)
powers of ω. */
uint32_t soup_MULM(uint32_t a, uint64_t b, uint32_t p){
    uint64_t w = (double)(b<<32)/p;
    uint64_t q = (a*w)>>32;
    uint64_t r = a*b - q*p;
    return  (r-p< r)? r - p: r;// min(r, r-p)
}
uint32_t shoup_MULM(uint32_t a, uint32_t b, uint64_t w,  uint32_t p){
//    uint64_t w = (double)(b<<32)/p;
    uint64_t q = (a*(uint64_t)w)>>32;
    uint64_t r = a*(uint64_t)b - q*p;
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
/*! \brief Редукция Монтгомери 
    Работает на z<p, q<2^{31}
 */
int32_t  signed_mont_modm(int64_t z, int32_t q, int32_t p){
    int32_t m = z * p; // low product z_0 (1/q)
    z = (z + (int64_t)m*q)>>32; // high product
//    if (z>=q) z-=q;
    if (z<0) z+=q;
    return  (int32_t)z;
}

/*! \brief Функция для вычисления q^{-1} mod 2^{32} */
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


void vec_mulm_u(uint32_t *r, const uint32_t *a, const uint32_t b, const unsigned int N, const uint32_t q);
/*! \brief Cooley-Tukey batterfly векторный вариант "бабочки" */
static void NTT_CT_butterfly(uint32_t *a, uint32_t *b, uint32_t g, unsigned int n, uint32_t q);
/*! \brief Gentleman-Sande batterfly векторный вариант "бабочки" */
static void NTT_GS_butterfly(uint32_t *a, uint32_t *b, uint32_t g, unsigned int n, uint32_t q);
/*! \brief Чисто-теоретическое преобразование на кольце $\mathbb{Z}_q/\langle x^N + 1\rangle$ 
    \param a First polynomial (array of 256 uint32_t coefficients). return NTT(a) in bit-reversed order.
    \param gamma store powers of gamma in bit-reverse ordering
    \param N power of 2
    \param q prime modulus $q \equiv 1 \mod 2N$

    * [2012.01968](https://arxiv.org/pdf/2012.01968)
    * [2103.16400](https://arxiv.org/pdf/2103.16400) 
    * [2024/585](https://eprint.iacr.org/2024/585.pdf) 
*/
uint32_t* NTT(uint32_t *a, const uint32_t *gamma, unsigned int N, uint32_t q){
    unsigned int i, j, k, m, n;
    n = N/2;
    for (m = 1; m < N; m = 2*m, n = n/2) {
        for (i=0, k=0; i<m; i++, k+=2*n) {
            uint32_t w = gamma[m+i];
            NTT_CT_butterfly(a+k, a+k+n, w, n, q);
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
    unsigned int i, j, k, m, n;
    n = 1;
    for (m = N/2; m > 0; m = m/2, n=n*2) {
        for (i=0, k=0; i<m; i++, k+=2*n) {
            uint32_t w = gamma[m+i];
            NTT_GS_butterfly(a+k, a+k+n, w, n, q);
        }
    }
    uint32_t N_inv = INVM(N, q);
    vec_mulm_u(a, a, N_inv, N, q);
    return a;
}
/*! \brief Чисто-теоретическое преобразование на кольце $\mathbb{Z}_q/\langle x^N + 1\rangle$ 
    референсная реализация
    \param r результат преобразования
    \param a исходный вектор
    \param gamma N-th root of unity
    \param N power of 2
    \param q prime modulus $q \equiv 1 \mod 2N$
 */
void NTT_ref(uint32_t* r, const uint32_t *a, const uint32_t gamma, unsigned int N, uint32_t q)
{
    uint32_t d, g = 1;
    for (int i = 0; i<N; i++){
        uint32_t s = a[0];
        uint32_t w = g;
        for (int j = 1; j<N; j++){
            d = MULM(a[j], w, q);
            s = ADDM(s, d, q);
            w = MULM(w, g, q);
        }
        r[i] = s;
        g = MULM(g, gamma, q);
    }
}
/*! \brief Чисто-теоретическое преобразование на кольце $\mathbb{Z}_q/\langle x^N + 1\rangle$ 
    negative-wrap for negacyclic polynomial x^N + 1
    \param r результат преобразования
    \param a исходный вектор
    \param gamma 2N-th root of unity
    \param N power of 2
    \param q prime modulus $q \equiv 1 \mod 2N$
*/
void NTT_nw_ref(uint32_t* r, const uint32_t *a, const uint32_t gamma, unsigned int N, uint32_t q)
{
    uint32_t d, g = gamma;
    uint32_t g2 = SQRM(gamma, q);
    for (int i = 0; i<N; i++){
        uint32_t s = a[0];
        uint32_t w = g;
        for (int j = 1; j<N; j++){
            d = MULM(a[j], w, q);
            s = ADDM(s, d, q);
            w = MULM(w, g, q);
        }
        r[i] = s;
        g = MULM(g, g2, q);
    }
}
/*! \brief Домножить вектор на степени гаммы 
    Используется для коррекции полинома для применения NTT. 
    \param a вектор N-элементов, коэффициенты полинома
    \param gamma 2N-th root of unity
    \param N power of 2, степень полинома (x^N + 1)
    \param q prime modulus $q \equiv 1 \mod 2N$
 */ 
void poly_gamma(uint32_t *a, const uint32_t gamma, uint32_t N, uint32_t q){
    uint32_t g = 1;
    for(int i=1; i<N; i++){
        a[i] = MULM(a[i], g, q);
        g = MULM(g, gamma, q);
    }
}
/*! \brief Обратное Чисто-теоретическое преобразование на кольце $\mathbb{Z}_q/\langle x^N + 1\rangle$ 
    референсная реализация
    \param r результат преобразования
    \param a исходный вектор
    \param gamma обратная величина N-th root of unity gamma^{-1}
    \param N power of 2
    \param q prime modulus $q \equiv 1 \mod 2N$

 */
void invNTT_ref(uint32_t* r, const uint32_t *a, const uint32_t gamma, unsigned int N, uint32_t q)
{
    uint32_t N_inv = INVM(N, q);
    uint32_t d, g = 1;
    for (int i = 0; i<N; i++){
        uint32_t s = a[0];
        uint32_t w = g;
        for (int j = 1; j<N; j++){
            d = MULM(a[j], w, q);
            s = ADDM(s, d, q);
            w = MULM(w, g, q);
        }
        r[i] = MULM(s, N_inv, q);
        g = MULM(g, gamma, q);
    }
}
/*! \brief Обратное Чисто-теоретическое преобразование на кольце $\mathbb{Z}_q/\langle x^N + 1\rangle$ 
    референсная реализация для negative wrap for negacylic polynomial $x^N + 1$
    \param r результат преобразования
    \param a исходный вектор
    \param gamma обратная величина 2N-th root of unity gamma^{-1}
    \param N power of 2
    \param q prime modulus $q \equiv 1 \mod 2N$
 */
void invNTT_nw_ref(uint32_t* r, const uint32_t *a, const uint32_t gamma, unsigned int N, uint32_t q)
{
    uint32_t N_inv = INVM(N, q);
    uint32_t d, g = 1;
    uint32_t g2 = SQRM(gamma, q);
    for (int i = 0; i<N; i++){
        uint32_t s = a[0];
        uint32_t w = g;
        for (int j = 1; j<N; j++){
            d = MULM(a[j], w, q);
            s = ADDM(s, d, q);
            w = MULM(w, g, q);
        }
        r[i] = MULM(s, N_inv, q);
        N_inv= MULM(N_inv, gamma, q);
        g    = MULM(g, g2, q);
    }
}
/*! \brief Element-wise multiplication of two vectors modulo q 
    \param r результат умножения
    \param a первый вектор
    \param b второй вектор
    \param N размер векторов
    \param q prime modulus
*/
void vec_mulm(uint32_t *r, const uint32_t *a, const uint32_t *b, const unsigned int N, const uint32_t q){
    for (int i=0; i<N; i++)
        r[i] = MULM(a[i], b[i], q);
}
void vec_mulm_u_(uint32_t *r, const uint32_t *a, const uint32_t b, const unsigned int N, const uint32_t q){
    for (int i=0; i<N; i++)
        r[i] = MULM(a[i], b, q);
}
void vec_mulm_u(uint32_t *r, const uint32_t *a, const uint32_t b, const unsigned int N, const uint32_t q){
    uint64_t w = (double)((uint64_t)b<<32)/q;
    for (int i=0; i<N; i++)
        r[i] = shoup_MULM(a[i], b, w, q);
}
void vec_addm(uint32_t *r, const uint32_t *a, const uint32_t *b, const unsigned int N, const uint32_t q){
    for (int i=0; i<N; i++)
        r[i] = ADDM(a[i], b[i], q);
}
void vec_subm(uint32_t *r, const uint32_t *a, const uint32_t *b, const unsigned int N, const uint32_t q){
    for (int i=0; i<N; i++)
        r[i] = SUBM(a[i], b[i], q);
}
void vec_xtime_madd(uint32_t *r, const uint32_t *a, uint32_t b, const unsigned int N, const uint32_t q){
    uint32_t c = q-r[N-1];
    uint32_t t;
    for (int i=0; i<N; i++){
        t = r[i];
        r[i] = ADDM(MULM(a[i], b, q), c, q);
        c = t;
    }
}
void vec_xtime_madd_shoup(uint32_t *r, const uint32_t *a, uint32_t b, const unsigned int N, const uint32_t q){
    uint32_t c = q-r[N-1];
    uint32_t t;
    uint64_t w = (double)((uint64_t)b<<32)/q;
    for (int i=0; i<N; i++){
        t = r[i];
        r[i] = ADDM(shoup_MULM(a[i], b, w, q), c, q);
        c = t;
    }
}
void vec_xtime_addm(uint32_t *r, const uint32_t *a, const unsigned int N, const uint32_t q){
    uint32_t c = q-r[N-1];
    uint32_t t;
    for (int i=0; i<N; i++){
        t = r[i];
        r[i] = ADDM(a[i], c, q);
        c = t;
    }
}

/*! \brief умножение полиномов в кольце $\mathbb{Z}_q[x]/(x^N + 1)$ с использованием NTT 
    \param r результат умножения
    \param a первый полином, заменяется на результат NTT
    \param b второй полином, заменяется на результат NTT
    \param gamma store powers of $\gamma^i$ in bit-reverse ordering, \gamma - корень 2N степени из единицы
    \param r_gamma store powers of $\gamma^{-i}$ in bit-reverse ordering
    \param N power of 2
    \param q prime modulus satisfying $q \equiv 1 \mod 2N$

    \note gamma - корень степени 2N из единицы
 */
void NTT_mul(uint32_t *r, uint32_t *a, uint32_t *b, const uint32_t *gamma, const uint32_t *r_gamma, unsigned int N, uint32_t q)
{
    NTT(a, gamma, N, q);
    NTT(b, gamma, N, q);
    vec_mulm(r, a, b, N, q);
    invNTT(r, r_gamma, N, q);
}
/*! \brief умножение полиномов в кольце $\mathbb{Z}_q[x]/(x^N + 1)$ без использования NTT */
void poly_mul(uint32_t *r, const uint32_t *a, const uint32_t *b, unsigned int N, uint32_t q)
{
    uint32_t v[N];
    poly_mulm_u(r, a, b[N-1], N, q);
    for (int i = N-2; i>=0; i--){
        poly_xtime_madd(r, a, b[i], N, q);
        //vec_xtime_madd_shoup(r, a, b[i], N, q);
        //poly_mulm_u(v, a, b[i], N, q);
        //vec_xtime_addm(r, v, N, q);
    }
}
static
void NTT_CT_butterfly(uint32_t *a, uint32_t *b, uint32_t g, unsigned int n, uint32_t q) {
    uint32_t w[n];
    vec_mulm_u(w, b, g, n, q);
    vec_subm(b, a, w, n, q);
    vec_addm(a, a, w, n, q);
}
static 
void NTT_GS_butterfly(uint32_t *a, uint32_t *b, uint32_t g, unsigned int n, uint32_t q) {
    uint32_t w[n];
    vec_subm(w, a, b, n, q);
    vec_addm(a, a, b, n, q);
    vec_mulm_u(b, w, g, n, q);
}

/*! \brief Обратный битовый порядок для 8 битных чисел */
static uint8_t RevBits8(uint8_t x){
    x = ((x & 0x55) << 1) | ((x & 0xAA) >> 1);
    x = ((x & 0x33) << 2) | ((x & 0xCC) >> 2);
    x = ((x & 0x0F) << 4) | ((x & 0xF0) >> 4);
    return x;
}
/*! \brief Обратный битовый порядок для 32 битных чисел */
static uint32_t RevBits(uint32_t x){
    x = ((x & 0x55555555) << 1) | ((x & 0xAAAAAAAA) >> 1);
    x = ((x & 0x33333333) << 2) | ((x & 0xCCCCCCCC) >> 2);
    x = ((x & 0x0F0F0F0F) << 4) | ((x & 0xF0F0F0F0) >> 4);
    x = ((x & 0x00FF00FF) << 8) | ((x & 0xFF00FF00) >> 8);
    x = ((x & 0x0000FFFF) <<16) | ((x & 0xFFFF0000) >>16);
    return x;
}
/*! \brief Расчет вектора степеней корня степени N для применения в NTT
    \param r вектор степеней корня степени N, записывается в bit-reversed order
    \param gamma N-th root of unity
    \param N power of 2, степень полинома (x^N + 1)
    \param q prime modulus $q \equiv 1 \mod 2N$
 */ 
void ntt_precompute_rev(uint32_t* r, uint32_t gamma, uint32_t N, uint32_t q) {
    uint32_t g = gamma;
    int s = __builtin_clz(N-1);
    r[RevBits(0u)>>s] = 1;
    r[RevBits(1u)>>s] = g;
    for (uint32_t i=2; i<N; i++) {
        g = MULM(g, gamma, q);
        r[(RevBits(i)>>s)] = g;
    }
}
/*! \brief Расчет вектора степеней корня степени 2N для применения в NTT
    \param r вектор степеней корня степени N, записывается в порядке возрастания степени
    \param gamma 2N-th root of unity
    \param N power of 2, степень полинома (x^N + 1)
    \param q prime modulus $q \equiv 1 \mod 2N$
 */ 
void ntt_precompute(uint32_t* r, uint32_t gamma, unsigned int N, uint32_t q) {
    uint32_t g = gamma;
    r[0] = 1;
    r[1] = g;
    for (uint32_t i=2; i<N; i++) {
        g = MULM(g, gamma, q);
        r[i] = g;
    }
}

static int jacobi(uint64_t a, uint64_t m);
/*! \brief Поиск корня степени 2N= 2^s по модулю простого числа q
    \param N степень полинома (x^N + 1)
    \param q простое число, нечетное $q = 1 mod 2N$
    \return (r^2)^N = 1 mod q
 */
uint32_t ntt_root(uint32_t N, uint32_t q){
    uint32_t gen =  3;// выбор генератора - квадратичный не-вычет
    while (jacobi(gen,q)!=-1) gen++;
    uint32_t s = 1;
    while (POWM(gen, (q-1)>>s, q)==1) s++;
    if (s>1) N>>=s-1;
    return POWM(gen, (q-1)/(N+N), q);
/*
    uint32_t r = gen;
    do {
        while ( POWM(r, N+N, q)!=1)
            r = MULM(r, gen, q);
    } while (POWM(r, N, q)==1);
    return r; */
}

#define BIT(x,n) (((x)>>(n))&1)
#define SWAP(x,y) do {    \
   __typeof__(x) _x = x;  \
   __typeof__(y) _y = y;  \
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
static int jacobi(uint64_t a, uint64_t m) 
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
uint32_t mwc32u_next(uint32_t h, const uint32_t A){
    uint32_t r = (h&0xFFFFu)*A - (h>>16);
    if (r > (A<<16)) r+= (A<<16)+1;
    return r;
}
#if defined(TEST_NTT) || 1

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

/*! \brief выбор генератора мультипликативной группы  - квадратичный не-вычет 

    \see (https://eprint.iacr.org/2012/470.pdf) 
    An algorithm for generating a quadratic non-residue modulo
    
 */
uint32_t generate_quadratic_non_residue(uint32_t gen, uint32_t P){
    if ((P&3)==3) return P-1;
    if ((P&7)==5) return 2;// 2 примитивный корень для p = 3 mod 8 или p = 5 mod 8.

    while (jacobi(gen,P)!=-1) gen++;
    return gen;
}

/*! \brief Простые числа вида 2^W - A 2^N + 1 
    https://www.rieselprime.de/ziki/Proth_prime
    https://en.wikipedia.org/wiki/Proth_prime

    A Proth prime is not a true class of numbers, but primes in the form $k•2^n+1$ with $2^n > k$ 
    are often called _Proth primes_.
    The primality of a Proth number can be tested with Proth's theorem, which states that a 
    Proth number $p$ is prime if and only if there exists an integer $a$ for which
    $a^{\frac{p-1}{2}}\equiv -1 \pmod{p}$.
    This theorem can be used as a probabilistic test of primality, by checking for many random choices of 
    $a$ whether $a^{\frac{p-1}{2}}\equiv -1 \pmod{p}$. If this fails to hold for several random 
    $a$, then it is very likely that the number $p$ is composite.

*/
uint32_t primes[] = {
    (1u<<23) -(1u<<13)+1, // NIST
//            (1u<<31) -1, // Mersenne 31
    (1u<<31) -(1u<<27)+1,
    (1u<<31) -(1u<<25)+1,
            (1u<<31) -(1u<<24)+1,
    (1u<<31) -(1u<<19)+1,
    (1u<<31) -(1u<<17)+1,

    0x7fe7f001, 0x7fe01001, 0x7fd35001, 0x7fc5d001, 

    0xffffd001, 0xfffbb001, 0xfff16001, 0xffddb001, 
    0xffd4b001, 0xffbd7001, 0xffb5f001, 0xffb11001, 
    0xffae7001, 0xff83b001, 0xff5da001, 0xff502001, 
    0xff4ae001, 0xff382001,

//    (1u<<31) -(1u<< 9)+1,

    0x7ffd5601, 0x7ffd2601, 0x7ff8e201, 0x7ff83a01, 
    0x7ff82e01, 0x7ff04201, 0x7fee9201, 0x7fea4201,

    0xffffca01, 0xfffef201, 0xfffe2601, 0xfffce201, 
    0xfffa4e01, 0xfff55601, 0xfff30a01, 0xffefc201, 
    0xffea2201, 0xffe72e01, 0xffe6da01, 0xffe3c201, 

    (0x7bffu<<16) + 1,
    (0x7b27u<<16) + 1,
    (0x7a46u<<16) + 1,
    (0x79efu<<16) + 1,


    (0xff7bu<<16) + 1,
    (0xff03u<<16) + 1,
    (0xfe04u<<16) + 1,
    (0xfcf6u<<16) + 1,
    (0xfcd2u<<16) + 1,
    (0xfb13u<<16) + 1,
    (0xf9d5u<<16) + 1,
    (0xf960u<<16) + 1,
    (0xf8d6u<<16) + 1,
    (0xf8c7u<<16) + 1,
    (0xf804u<<16) + 1,

    (0xFFDFu<<16)+1,

    (0xFFACu<<16)+1,
    (0xFFA2u<<16)+1,
    (0xFF97u<<16)+1,
    (0xFF93u<<16)+1,
    (0xFF82u<<16)+1,
    (0xFF7Bu<<16)+1,

//    (0xFFF0u<<16)+1,// 2^32 - 2^20 +1
    (0xC000u<<16)+1,// 2^32 - 2^30 +1

    0x7f000001, 0x7e100001, 0x7e000001, 0x7d200001, 0x7c800001, 0x7bd00001,

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
/*! \brief проверка корректности вычислений 

    1. Проверка выбора простых чисел вида (2^W - a 2^N + 1) 
    2. Проверка операций преобразования вещественных чисел в модульную арифметику
    2. Проверка операций преобразования вещественных чисел в модульную арифметику
    3. Выбор простых чисел для операции MWC32 и MWC32s (модульная арифметика со знаком)
 */
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

    if (0) {// mwc32s
        printf("MWC32 test\n");
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
        if (0) for (int k = 0; k< sizeof(primes)/sizeof(primes[0]); k++) {
            const uint32_t p = primes[k];
            uint32_t h = 3;
            while (jacobi(h, p)!=-1 || gcd(h,p-1)!=1 || POWM(h, (p-1)/2, p)==1) h+=2;
            // h = generate_quadratic_non_residue(3, (uint32_t)p);
            uint32_t g = h;
            uint32_t a = p>>__builtin_ctz(p-1);
            for(uint32_t i=0; i<(uint32_t)p+1; i++){
                //h = (h*(uint64_t)(p>>16))%p;
                h = (h*(uint64_t)g)%p;
                if (h==g || h==1) {
                   //if (i>=p/2-2)
                        printf("mwc32 prime=%08x gen=%2d ord %08x=%d\n",p,g, i, (p-1)/(i+1));
                    break;
                }
            }
        }


        printf("MWC32s test\n");
        if (0) for (int k = 1; k< 0x3FFF; k++) {// MWC32s
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
            while (jacobi(h, p)!=-1 /* || gcd(h,p-1)!=1 || gcd(h,p+1)==1 */) h++;
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
                    if ((i&0x1FFE)==0x1FFE)
                        printf("mwc32s prime=%08x gen=%2d ord %08x\n",p,g, i);
                    break;
                }
            }
        }
        return 0;
    }
    if (0) {// montgomery test
        printf("Montogery test\n");
        uint32_t p = primes[0];
        uint32_t pi = -mod_inverse(p); 
        uint32_t qinv=4236238847;
        printf ("prime=%u pi = %u %x \n",p, pi, p*qinv);
        for (int k = 1; k< sizeof(primes)/sizeof(primes[0]); k++) {
            uint32_t p = primes[k];
            if ((p>>31)!=0) continue;
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
    if (1) {// Signed montgomery test
        printf("Signed Montogery test\n");
        for (int k = 1; k< sizeof(primes)/sizeof(primes[0]); k++) {
            int32_t p = primes[k];
            if ((p>>31)!=0) continue;
            int32_t pi = mod_inverse(p);
            printf ("p=%08x pi = %08x %08x \n",p, pi, p*(-pi));
            for (int32_t a =p; a>0; a--)
            {
                int32_t b  = ((int64_t)a<<32)%p;
                int64_t a2 = (int64_t)a*b;

                int32_t r  = signed_mont_modm(a2, p, -pi);
                int32_t r2 = ((int64_t)a*a)%p;
                if (r2!=r) {
                    printf ("r = %d %d p=%08x \n", r,r2,a);
                     break;
                }
            }
            //break;
        }
        return 0;

    }
    if (0) {// soup test
        printf("Soup modular reduction\n");
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
    if (1) {// NTT reference implementation
        printf("NTT test\n");
        uint32_t a[256];
        uint32_t b[256];
        uint32_t c[256];
        uint32_t t[256];
        uint32_t r[256];
        uint32_t wv[256]; // вектор корня степени N в bit-reversed order
        uint32_t vv[256]; // вектор корня степени N в bit-reversed order
        uint32_t uv[256];
        unsigned int n = 16;
        int res;
        int s = __builtin_clz(n-1);
        for (int k=0; k<sizeof(primes)/sizeof(primes[0]); k++) {
            for (int i=0; i<n; i++){// заполнение векторов
                a[i] = i*35789;
                c[i] = 0;
            }

            uint32_t p = primes[k];

            if (p%(n+n)!=1) continue;
            uint32_t g = ntt_root(n, p);
            uint32_t g_= INVM(g, p);// g^{-1}
            uint32_t w = SQRM(g, p);
            uint32_t w_= SQRM(g_, p);// w^{-1}

            ntt_precompute_rev(wv, g, n, p);// расчет вектора степеней корня степени N в bit-reversed order для применения в NTT
            ntt_precompute_rev(vv, g_, n, p);
            ntt_precompute(uv, g, n, p);
            if (0){
                for (int i=0; i<n; i++){// проверка результата
                    printf("%x ", wv[i]);
                }
                printf("\n");
            }
            printf("prime=%08x g=%8x w=%8x %4s ",p, g, w, 
                MULM(w, w_, p)==1 && POWM(w, n, p)==1 && POWM(w_, n, p)==1?"pass":"fail");
            poly_gamma(a, g, n, p);
            NTT_ref(c, a, w, n, p);
            if (0) {
                for (int i=0; i<n; i++){// проверка результата
                    printf("%x ", c[i]);
                }
                printf("\n");
            }
            invNTT_ref(a, c, w_, n, p);
            poly_gamma(a, g_, n, p);
            res = 1;
            for (int i=0; i<n; i++){// проверка результата
                res = res && (a[i]==i*35789);
            }
            printf("..%s ", res?"ok":"fail");
            NTT_nw_ref(c, a, g, n, p);
            for (int i=0; i<n; i++) b[i] = a[i];
            NTT(a, wv, n, p);
            res = 1;
            for (unsigned int i=0; i<n; i++){// проверка результата
                res = res && (a[RevBits(i)>>s]==c[i]);
            }
            printf("%4s ", res?"pass":"fail");
            invNTT(a, vv, n, p);
            
            //invNTT_nw_ref(a, c, g_, n, p);
            res = 1;
            for (int i=0; i<n; i++){// проверка результата
                res = res && (a[i]==i*35789);
            }
            printf("..%s\n", res?"ok":"fail");
        }
    }
    if (1) {// Умножение полиномов методом NTT
        int res;
        printf("poly mul\n"); 
        const unsigned int n = 256;
        uint32_t a[256];
        uint32_t b[256];
        uint32_t r[256];
        uint32_t c[256];
        uint32_t wv[256];
        uint32_t vv[256];
        uint32_t wu[256];
        uint32_t vu[256];
        for (int k=0; k<sizeof(primes)/sizeof(primes[0]); k++) {
            uint32_t p = primes[k];
            if ((p&(2*n-1))!=1) continue;
            
            for (int i=0; i<n; i++){// заполнение векторов
                b[i] = (i^7+2561894)%p;
                a[i] = i*3+1;
                c[i] = 0;
            }
            b[0] = 3;
            
            uint32_t g = ntt_root(n, p);
            uint32_t g_= INVM(g, p);// g^{-1}
            ntt_precompute_rev(wv, g, n, p);// расчет вектора степеней корня степени N в bit-reversed order для применения в NTT
            ntt_precompute_rev(vv, g_, n, p);
            ntt_precompute_rev(wu, SQRM(g,p), n, p);
            ntt_precompute_rev(vu, SQRM(g_,p), n, p);
            printf("prime=%08x g=%8x ",p, g);
            poly_mul(c, a, b, n, p); 
//            poly_gamma(a, g, n, p);
//            poly_gamma(b, g, n, p);
            NTT_mul(r, a, b, wv, vv, n, p);
//            poly_gamma(r, g_, n, p);
            res = 1;
            for (int i=0; i<n; i++){// проверка результата
                // printf("%d: %x %x\n", i, c[i], r[i]);
                res = res && (c[i]==r[i]);
            }
            printf("..%s\n", res?"ok":"fail");
        }
        return 0;
    }
    if (1) {// Поиск корней из единицы для простых чисел
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