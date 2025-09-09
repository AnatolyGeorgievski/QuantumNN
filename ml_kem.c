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

В реализации используется упаковка SIMD uint16x2_t, поскольку вычисления ведуться в сопряженных парах.

uint16x8_t, uint16x16_t, uint16x32_t для ускорения вычислений на современных процессорах.
Длина вектора определяется при компиляции и задается параметром VL=32,16,8.

Описание перестановок CT butterfly через индекс:
offs=4;
if (idx & 4u)  {// знак плюс
    a[idx] = a[idx] + a[idx+offs]*zeta;
} else {
    a[idx] = a[idx]*(-zeta) + a[idx-offs] 
}
offs=2;
if (idx & 2)  {// знак плюс
    a[idx] = a[idx] + a[idx+offs]*zeta;
} else {
    a[idx] = a[idx]*(-zeta) + a[idx-offs]
}
offs=1;
if (idx & 1)  {// знак плюс
    a[idx] = a[idx] + a[idx+offs]*zeta
} else {
    a[idx] = a[idx]*(-zeta) + a[idx-offs]
}
перестановки CT bufferfly:
zm = q - zp;
a = shuffle(f, {0,1,2,3, 0,1,2,3})
b = shuffle(f, {4,5,6,7, 4,5,6,7}
z = shuffle(zp,zm, {z4,z4,z4,z4, -z4,-z4,-z4,-z4})
f = a + VMULM(b,z)
a = shuffle(f, {0,1,0,1, 4,5,4,5})
b = shuffle(f, {2,3,2,3, 6,7,6,7})
z = shuffle(zp,zm, {z2,z2,-z2,-z2, z6,z6,-z6,-z6})
f = a + VMULM(b,z)
a = shuffle(f, {0,0,2,2, 4,4,6,6})
b = shuffle(f, {1,1,3,3, 5,5,7,7})
z = shuffle(zp,zm, {z1,-z1,z5,-z5, z3,-z3,z7,-z7})
f = a + VMULM(b,z)

*/

#include <stdint.h>
#include <stdio.h>
// Q_PRIME 12289 и N = 512 - другой вариант см. BLISS
// Q_PRIME  3329 и N = 256 - ML-KEM
#define N       256
#define Q_PRIME 3329// (13<<8) +1

#if Q_PRIME == 3329
#define ZETA    17  // корень из единицы N-й степени
#define N_INV   3303// ≡ 128^{−1} mod Q
#define Q_MONT  3327// -q^{-1} mod 2^{16} = 3327
#define U_BARRETT 40317 // 2^{27}/Q_PRIME
#elif Q_PRIME == 12289
#define ZETA    ??  // корень из единицы N-й степени
#define Q_MONT  53249
#endif

typedef uint16_t uint16x2_t  __attribute__((vector_size(4)));
typedef uint16_t uint16x16_t __attribute__((vector_size(32)));
typedef uint16_t uint16x32_t __attribute__((vector_size(64)));
typedef  int16_t  int16x2_t __attribute__((vector_size(4)));

// Алгоритм заменяет целочисленное деление на модуль на умножение (mul_hi) и сдвиг. 
uint32_t shoup_div(uint32_t b){
#if Q_PRIME == 3329
    return (b*0x9D7DBB41uLL)>>(43-16);
//  return (((b*0x3AFB7681uLL)>>16) +b)>>12;
#elif Q_PRIME == 12289
    return (b*0xAAA71C85uLL)>>(45-16);
#else
    return ((uint32_t)b<<16)/Q_PRIME;
#endif
}
uint16_t mod1(uint32_t x){
    uint32_t d = shoup_div(x);
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
#if 0//def __AVX512F__
#define VL 32 // vector length, epi16 elements per vector register
#define VSET1(x) {x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x}
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
    r0 = _mm512_sub_epi16(r0, r1);
    r1 = _mm512_add_epi16(r0, (__m512i)p);
    return (uint16x32_t)_mm512_min_epu16(r0, r1);
}
static inline uint16x32_t VMULM_mont(uint16x32_t a, uint16x32_t b, uint16x32_t qm, uint16x32_t q){
    __m512i z1 = _mm512_mulhi_epu16((__m512i)a, (__m512i)b);
    __m512i z0 = _mm512_mullo_epi16((__m512i)a, (__m512i)b);
    __m512i m  = _mm512_mullo_epi16(z0, (__m512i)qm);
    __m512i t1 = _mm512_mulhi_epu16(m,  (__m512i)q);
    __m512i r  = _mm512_add_epi16(z1, t1);
    return (uint16x32_t)_mm512_min_epu16(r, _mm512_sub_epi16(r, (__m512i)q));
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
#define VSET1(x) {x,x,x,x, x,x,x,x, x,x,x,x, x,x,x,x}
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
//    __m256i p = _mm256_set1_epi16(Q_PRIME);
    __m256i q = _mm256_mulhi_epu16((__m256i)a, (__m256i)w);
    __m256i r0= _mm256_mullo_epi16((__m256i)a, (__m256i)b);
    __m256i r1= _mm256_mullo_epi16(q, (__m256i)p);
    r0 = _mm256_sub_epi16(r0, r1);// тут могут быть отрицательные значения
    __m256i r = _mm256_min_epu16(r0, _mm256_add_epi16(r0, (__m256i)p));
    // r = _mm256_min_epu16(r, _mm256_sub_epi16(r, (__m256i)p));    
    return (uint16x16_t)r;
}
/*! Montogomery multiplication with precomputed qm = -q^{-1} mod 2^{16} 
    \return a*b\beta^{-1} mod q 
 */
static inline uint16x16_t VMULM_mont(uint16x16_t a, uint16x16_t b, uint16x16_t qm, uint16x16_t q){
    __m256i z1 = _mm256_mulhi_epu16((__m256i)a, (__m256i)b);
    __m256i z0 = _mm256_mullo_epi16((__m256i)a, (__m256i)b);
    __m256i m  = _mm256_mullo_epi16(z0, (__m256i)qm);
    __m256i t1 = _mm256_mulhi_epu16(m,  (__m256i)q);
    __m256i r  = _mm256_add_epi16(z1, t1);
    return (uint16x16_t)_mm256_min_epu16(r, _mm256_sub_epi16(r, (__m256i)q));
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
    __m256i Ur = _mm256_set1_epi16(U_BARRETT);
#if defined(__AVX512_VBMI2__) && defined(__AVX512VL__)// AVX10.1
    __m256i c1 = _mm256_shrdi_epi16(z0,z1, 11);
#else 
    __m256i c1 = _mm256_or_si256(_mm256_srli_epi16(z0, 11), _mm256_slli_epi16(z1, 16-11));
#endif
    __m256i c2 = _mm256_mulhi_epu16(Ur, c1);
    __m256i c4 = _mm256_mullo_epi16(c2,  (__m256i)q);
    __m256i r  = _mm256_sub_epi16(z0, c4);
    return (uint16x16_t)_mm256_min_epu16(r, _mm256_sub_epi16(r, (__m256i)q));
}

#else // Векторизация не используется
#define VL 2
#define VSET1(x) {x,x}
static inline uint16_t shoup_MULM(uint16_t a, uint16_t b, uint16_t w);
uint16x2_t VMULM(uint16x2_t a, uint16x2_t b, uint16x2_t w, uint16x2_t q){
    uint16x2_t r;
    r[0] = shoup_MULM(a[1], b[0], w[0]);
    r[1] = shoup_MULM(a[1], b[1], w[1]);
    return r;
}
#endif

/*! Редуцирование по модулю простого числа, с использованием Барретта 
Метод проверяем на всех числах 0<2^15
 */
uint32_t MODB(uint32_t a, uint32_t q, uint16_t U) {
const int Q = 11;
const int L = Q+16;
    uint16_t Ur = (1uL<<L)/q;
    uint32_t c2 = Ur*(a>>Q);// mul hi
     int16_t c4 = a - (int16_t)(q*(c2>>(L-Q)));
    if (c4>=q) c4 -= q;
    return c4;
}
/*! signed Montogomery multiplication with precomputed qm = -q^{-1} mod 2^{16} */
static inline int16_t  mont_MULM(int16_t a, int16_t b){

    int32_t z = (a*(int32_t)b);// signed high 16 bit
    int16_t z0=  a*(int32_t)b; // signed low 16 bit
    int32_t q = Q_PRIME;
    int16_t p = Q_MONT; // mod_inverse(Q_PRIME); // 1/q mod 2^{32}
    int16_t m = z0 * p; // low product z_0 (1/q)
    //printf("z=%x, mq=%x %x %x\n", z, m*q, (z>>16)-(m*q)>>16, (z - (m*q))>>16);
    z = (z + (m*q))>>16; // high product
    if (z<0) z+=q;
    return  (int16_t)z;
}
static inline uint16_t shoup_MULM(uint16_t a, uint16_t b, uint16_t w){
    uint16_t q = (a*(uint32_t)w)>>16;
    int16_t r0= (a*b);
    int16_t r1=  q*Q_PRIME;
    int16_t r = r0 - r1;
//    if (r<0) r+=Q_PRIME;
    if (r>=Q_PRIME) r-=Q_PRIME;
    return r;// (r+Q_PRIME<r)? r+Q_PRIME: r;
}
static inline void vec_mulm_u(uint16x2_t* r_, const uint16x2_t* a_, uint16_t b, unsigned int len){
    uint16x16_t* r = (uint16x16_t*)r_;
    uint16x16_t* a = (uint16x16_t*)a_;
    const uint16_t w = shoup_div(b);
    uint16x16_t wv = VSET1(w);
    uint16x16_t bv = VSET1(b);
    uint16x16_t prime = VSET1(Q_PRIME);
    for(int i=0; i<len/VL; i++){
        r[i] = VMULM(a[i], bv, wv, prime);
    }
}
static inline 
void vec_xtime_madd(uint16x16_t* r, uint16x16_t* a, uint16_t b, unsigned int len){
    uint16_t w  = shoup_div(b);
    uint16x16_t bv = VSET1(b);
    uint16x16_t wv = VSET1(w);
    uint16x16_t p  = VSET1(Q_PRIME);
    uint16x16_t c = p - r[len/VL-1];
    for (int i=0; i<len/VL; i++){
        uint16x16_t v = __builtin_shufflevector(r[i], c, 31, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14);
        c = r[i];
        r[i] = VADDM(v, VMULM(a[i],bv,wv, p), p);
    }
}
void poly_mul(uint16x2_t *r, const uint16x2_t *a, const uint16_t *b)
{
    vec_mulm_u(r, a, b[N-1], N);
    for (int i = N-2; i>=0; i--){
        vec_xtime_madd((uint16x16_t*)r, (uint16x16_t*)a, b[i], N);
    }
}
/*
> The Compress and Decompress algorithmssatisfy two important properties. First, decompression
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
// Decodes a byte array into an array of 𝑑-bit integers for 1 ≤ 𝑑 ≤ 12.
uint16_t* ByteDecode(uint8_t * s, unsigned int d){
    uint32_t mask = (1u<<d)-1;
    uint16_t* dst = (uint16_t*)s;
    int bits = 0;
    uint32_t v = 0;
    for(int i=0; i<N; i++){
        while (bits<d) {
            v |= (*s++)<<bits;
            bits+=8;
        }
        *dst++ = (uint16_t) v & mask;
        v>>=d;
        bits-=d;
    }
}
uint16_t* ByteEncode(uint8_t * s, unsigned int d){
}
#if 0
/*! \brief Algorithm 7 SampleNTT(𝐵)
Takes a 32-byte seed and two indices as input and outputs a pseudorandom element of 𝑇𝑞.*/
void SampleNTT(uint16_t *a, uint8_t *b){
    XOF_ctx_t ctx;
    XOF.init(&ctx);
    XOF.absorb(&ctx, b, 32);
    int j =0;
    uint32_t c=0; 
    uint16_t d0, d1;
    while (j<N){
        XOF.squeeze(&ctx, &c, 3);
        d0 = (c    ) & 0xFFF;
        d1 = (c>>12) & 0xFFF;
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
void NTT_GS_butterfly(uint16x2_t *f_, unsigned int len, uint16_t zeta){
#if 0
    uint16_t *f = (uint16_t *)f_;
    uint16_t *g = (uint16_t *)(f_+len/2);
    uint16_t t;
    for (int i=0; i<len; i++){
        t = f[i];
        f[i] = (g[i] + t)%Q_PRIME;
        g[i] = ((g[i] +Q_PRIME- t) * (uint32_t)zeta)%Q_PRIME;
    }
#else
if (len>=VL){
    uint16x16_t *f = (uint16x16_t*)(f_);
    uint16x16_t *g = (uint16x16_t*)(f_+len/2);
    uint16x16_t t;
    uint16x16_t z = VSET1(zeta);
    uint16x16_t p = VSET1(Q_PRIME);
    uint32_t w = shoup_div(zeta);
    uint16x16_t wv = VSET1(w);
    for (int i=0; i<len/VL; i++){
        t = f[i];
        f[i] = VADDM(g[i],t, p);
        g[i] = VMULM(VSUBM(g[i],t, p), z, wv, p); 
    }
} else {
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
void NTT_CT_butterfly(uint16x2_t *f_, unsigned int len, uint16_t zeta){
#if 0
    uint16_t *f = (uint16_t *)f_;
    uint16_t *g = (uint16_t *)(f_+len/2);
    uint16_t t;
    for (int i=0; i<len; i++){
        t = (g[i]* (uint32_t)zeta)%Q_PRIME;
        g[i] = (uint32_t)(f[i] +Q_PRIME- t)%Q_PRIME;
        f[i] = (f[i] + t)%Q_PRIME;
    }
#else
if (len>=VL){
    uint16x16_t *f = (uint16x16_t*)(f_);
    uint16x16_t *g = (uint16x16_t*)(f_+len/2);
    uint16x16_t t;
    uint16x16_t z = VSET1(zeta);
    uint16x16_t p = VSET1(Q_PRIME);
    uint32_t w = shoup_div(zeta);
    uint16x16_t wv = VSET1(w);
    for (int i=0; i<len/VL; i++){
        t = VMULM(g[i], z, wv, p);
        g[i] = f[i]+p-t;//VSUBM(f[i], t, p);
        f[i] = f[i]+t;//VADDM(f[i], t, p); 
    }
} else {
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
}
#endif

static uint16_t zeta2[] = {// степени zeta^{2 BitRev(i)+1} mod q
  17,  -17,2761,-2761, 583, -583,2649,-2649,
1637,-1637, 723, -723,2288,-2288,1100,-1100,
1409,-1409,2662,-2662,3281,-3281, 233, -233,
 756, -756,2156,-2156,3015,-3015,3050,-3050,
1703,-1703,1651,-1651,2789,-2789,1789,-1789,
1847,-1847, 952, -952,1461,-1461,2687,-2687,
 939, -939,2308,-2308,2437,-2437,2388,-2388,
 733, -733,2337,-2337, 268, -268, 641, -641,
1584,-1584,2298,-2298,2037,-2037,3220,-3220,
 375, -375,2549,-2549,2090,-2090,1645,-1645,
1063,-1063, 319, -319,2773,-2773, 757, -757,
2099,-2099, 561, -561,2466,-2466,2594,-2594,
2804,-2804,1092,-1092, 403, -403,1026,-1026,
1143,-1143,2150,-2150,2775,-2775, 886, -886,
1722,-1722,1212,-1212,1874,-1874,1029,-1029,
2110,-2110,2935,-2935, 885, -885,2154,-2154,
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
static uint16_t zeta [] = {// степени zeta^{k} mod q
   1,1729,2580,3289,2642, 630,1897, 848,
1062,1919, 193, 797,2786,3260, 569,1746,
 296,2447,1339,1476,3046,  56,2240,1333,
1426,2094, 535,2882,2393,2879,1974, 821,
#if 1// VL==2
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
#endif
};
/*! \brief Algorithm 9 NTT(𝑓) forward number theoretic transform 
Computes NTT representation 𝑓 of the given polynomial 𝑓 ∈ 𝑅_𝑞.

Алгоритм можно разделить на две части:
1. Прямое преобразование (CT) на регистрах с длиной 16 и более, когда используется одно значение zeta на вектор. 
2. Преобразование CT NTT_CT_butterfly_x8 на длины 8 и менее. 
Для этого используются перестановки значений zeta внутри вектора. 

*/
static const uint16x16_t xzetav[] = {// степени zeta^{BitRev(i)} mod q
{    1,  296,  289,  331,   17, 2761,  583, 2649,  3328, 3033, 3040, 2998, 3312,  568, 2746,  680,},
{ 1729, 2447, 3253, 1756, 1637,  723, 2288, 1100,  1600,  882,   76, 1573, 1692, 2606, 1041, 2229,},
{ 2580, 1339, 1197, 2304, 1409, 2662, 3281,  233,   749, 1990, 2132, 1025, 1920,  667,   48, 3096,},
{ 3289, 1476, 2277, 2055,  756, 2156, 3015, 3050,    40, 1853, 1052, 1274, 2573, 1173,  314,  279,},
{ 2642, 3046,  650, 1977, 1703, 1651, 2789, 1789,   687,  283, 2679, 1352, 1626, 1678,  540, 1540,},
{  630,   56, 2513,  632, 1847,  952, 1461, 2687,  2699, 3273,  816, 2697, 1482, 2377, 1868,  642,},
{ 1897, 2240, 2865,   33,  939, 2308, 2437, 2388,  1432, 1089,  464, 3296, 2390, 1021,  892,  941,},
{  848, 1333, 1320, 1915,  733, 2337,  268,  641,  2481, 1996, 2009, 1414, 2596,  992, 3061, 2688,},
{ 1062, 1426, 2319, 1435, 1584, 2298, 2037, 3220,  2267, 1903, 1010, 1894, 1745, 1031, 1292,  109,},
{ 1919, 2094,  807,  452,  375, 2549, 2090, 1645,  1410, 1235, 2522, 2877, 2954,  780, 1239, 1684,},
{  193,  535, 1438, 2868, 1063,  319, 2773,  757,  3136, 2794, 1891,  461, 2266, 3010,  556, 2572,},
{  797, 2882, 1534, 2402, 2099,  561, 2466, 2594,  2532,  447, 1795,  927, 1230, 2768,  863,  735,},
{ 2786, 2393, 2647, 2617, 2804, 1092,  403, 1026,   543,  936,  682,  712,  525, 2237, 2926, 2303,},
{ 3260, 2879, 1481,  648, 1143, 2150, 2775,  886,    69,  450, 1848, 2681, 2186, 1179,  554, 2443,},
{  569, 1974, 2474, 3110, 1722, 1212, 1874, 1029,  2760, 1355,  855,  219, 1607, 2117, 1455, 2300,},
{ 1746,  821, 1227,  910, 2110, 2935,  885, 2154,  1583, 2508, 2102, 2419, 1219,  394, 2444, 1175,},
};
static const uint16x16_t wzetav[] = {
{   19, 5827, 5689, 6516,  334,54354,11477,52149, 65516,59708,59846,59019,65201,11181,54058,13386,},
{34037,48172,64039,34569,32226,14233,45042,21655, 31498,17363, 1496,30966,33309,51302,20493,43880,},
{50790,26360,23564,45357,27738,52405,64591, 4586, 14745,39175,41971,20178,37797,13130,  944,60949,},
{64748,29057,44825,40455,14882,42443,59354,60043,   787,36478,20710,25080,50653,23092, 6181, 5492,},
{52011,59964,12796,38919,33525,32502,54905,35218, 13524, 5571,52739,26616,32010,33033,10630,30317,},
{12402, 1102,49471,12441,36360,18741,28761,52897, 53133,64433,16064,53094,29175,46794,36774,12638,},
{37345,44097,56401,  649,18485,45436,47975,47011, 28190,21438, 9134,64886,47050,20099,17560,18524,},
{16694,26241,25986,37699,14430,46007, 5275,12618, 48841,39294,39549,27836,51105,19528,60260,52917,},
{20906,28072,45652,28249,31183,45239,40101,63390, 44629,37463,19883,37286,34352,20296,25434, 2145,},
{37778,41223,15886, 8898, 7382,50180,41144,32384, 27757,24312,49649,56637,58153,15355,24391,33151,},
{ 3799,10532,28309,56460,20926, 6279,54590,14902, 61736,55003,37226, 9075,44609,59256,10945,50633,},
{15690,56736,30198,47286,41321,11044,48546,51066, 49845, 8799,35337,18249,24214,54491,16989,14469,},
{54846,47109,52109,51519,55200,21497, 7933,20198, 10689,18426,13426,14016,10335,44038,57602,45337,},
{64177,56677,29155,12756,22501,42325,54629,17442,  1358, 8858,36380,52779,43034,23210,10906,48093,},
{11201,38860,48704,61224,33899,23859,36892,20257, 54334,26675,16831, 4311,31636,41676,28643,45278,},
{34372,16162,24155,17914,41538,57779,17422,42404, 31163,49373,41380,47621,23997, 7756,48113,23131,},
};
static uint16x16_t NTT_CT_butterfly_x8(uint16x16_t f, uint16x16_t zv, uint16x16_t wv, uint16x16_t q){
    uint16x16_t a,b, z,w;
    z = __builtin_shuffle(zv,(uint16x16_t){1,1,1,1, 1,1,1,1, 9,9,9,9, 9,9,9,9});
    w = __builtin_shuffle(wv,(uint16x16_t){1,1,1,1, 1,1,1,1, 9,9,9,9, 9,9,9,9});
    a = __builtin_shuffle(f, (uint16x16_t){0,1, 2, 3, 4, 5, 6, 7, 0,1, 2, 3, 4, 5, 6, 7});
    b = __builtin_shuffle(f, (uint16x16_t){8,9,10,11,12,13,14,15, 8,9,10,11,12,13,14,15});
    f = VADDM(a, VMULM(b, z, w, q), q);
    z = __builtin_shuffle(zv,(uint16x16_t){2,2,2,2, 10,10,10,10, 3,3,3,3, 11,11,11,11});
    w = __builtin_shuffle(wv,(uint16x16_t){2,2,2,2, 10,10,10,10, 3,3,3,3, 11,11,11,11});
    a = __builtin_shuffle(f, (uint16x16_t){0,1,2,3, 0,1,2,3, 8, 9, 10,11,  8, 9,10,11});
    b = __builtin_shuffle(f, (uint16x16_t){4,5,6,7, 4,5,6,7, 12,13,14,15, 12,13,14,15});
    f = VADDM(a, VMULM(b, z, w, q), q);
    z = __builtin_shuffle(zv,(uint16x16_t){4,4,12,12, 5,5,13,13, 6,6,14,14, 7,7,15,15});
    w = __builtin_shuffle(wv,(uint16x16_t){4,4,12,12, 5,5,13,13, 6,6,14,14, 7,7,15,15});
    a = __builtin_shuffle(f, (uint16x16_t){0,1,0,1, 4,5,4,5, 8, 9, 8, 9,  12,13,12,13});
    b = __builtin_shuffle(f, (uint16x16_t){2,3,2,3, 6,7,6,7, 10,11,10,11, 14,15,14,15});
    f = VADDM(a, VMULM(b, z, w, q), q);
    return f;
}
// не сделано
static uint16x16_t NTT_GS_butterfly_x8(uint16x16_t f, uint16x16_t zv, uint16x16_t wv, uint16x16_t q){
    uint16x16_t a,b, z,w;
    z = __builtin_shuffle(zv,(uint16x16_t){4,4,12,12, 5,5,13,13, 6,6,14,14, 7,7,15,15});
    w = __builtin_shuffle(wv,(uint16x16_t){4,4,12,12, 5,5,13,13, 6,6,14,14, 7,7,15,15});
    a = __builtin_shuffle(f, (uint16x16_t){0,1,0,1, 4,5,4,5, 8, 9, 8, 9,  12,13,12,13});
    b = __builtin_shuffle(f, (uint16x16_t){2,3,2,3, 6,7,6,7, 10,11,10,11, 14,15,14,15});
    f = VMULM(VADDM(a, b, q), z, w, q);
    z = __builtin_shuffle(zv,(uint16x16_t){2,2,2,2, 10,10,10,10, 3,3,3,3, 11,11,11,11});
    w = __builtin_shuffle(wv,(uint16x16_t){2,2,2,2, 10,10,10,10, 3,3,3,3, 11,11,11,11});
    a = __builtin_shuffle(f, (uint16x16_t){0,1,2,3, 0,1,2,3, 8, 9, 10,11,  8, 9,10,11});
    b = __builtin_shuffle(f, (uint16x16_t){4,5,6,7, 4,5,6,7, 12,13,14,15, 12,13,14,15});
    f = VMULM(VADDM(a, b, q), z, w, q);
    z = __builtin_shuffle(zv,(uint16x16_t){1,1,1,1, 1,1,1,1, 9,9,9,9, 9,9,9,9});
    w = __builtin_shuffle(wv,(uint16x16_t){1,1,1,1, 1,1,1,1, 9,9,9,9, 9,9,9,9});
    a = __builtin_shuffle(f, (uint16x16_t){0,1, 2, 3, 4, 5, 6, 7, 0,1, 2, 3, 4, 5, 6, 7});
    b = __builtin_shuffle(f, (uint16x16_t){8,9,10,11,12,13,14,15, 8,9,10,11,12,13,14,15});
    f = VMULM(VADDM(a, b, q), z, w, q);
    return f;
}
uint16x2_t* NTT(uint16x2_t *f, const uint16_t *g){
    int i=1, len;
    for (len=N/2; len>=VL; len>>=1)
    for (int off=0; off<N; off+=2*len){
        uint16_t z = zeta[i];//g^{BitRev7(i)} 
        i++;
        NTT_CT_butterfly(f+off/2, len, z);
    }
#if VL>2 // len = {8,4,2} - три стадии для VL=16
    uint16x16_t q = VSET1(Q_PRIME);
    for (int off=0, j=0; off<N; off+=2*len, j++){
        uint16x16_t v = *(uint16x16_t*)(f+off/2);
        v = NTT_CT_butterfly_x8(v, xzetav[j], wzetav[j], q);
        *(uint16x16_t*)(f+off/2) = v;
    }
#endif
    return f;
}
/*! \brief Algorithm 10 iNTT(𝑓) Computes inverse NTT of the polynomial representation 𝑓 ∈ 𝑅_𝑞 that corresponds to the given NTT representation 𝑓 ∈ 𝑇_𝑞. */
uint16x2_t* iNTT(uint16x2_t *f, const uint16_t *g){
    int i=N/2-1;
    int len;
    for (len=2; len<=8; len<<=1)
    for (int off=0; off<N; off+=2*len){
        uint16_t z = zeta[i];//g[BitRev7(i)]; 
        i--;
        NTT_GS_butterfly(f+off/2, len, z);
    }
    for (; len<=N/2; len<<=1)
    for (int off=0; off<N; off+=2*len){
        uint16_t z = zeta[i];//g[BitRev7(i)]; 
        i--;
        NTT_GS_butterfly(f+off/2, len, z);
    }
    vec_mulm_u(f, f, N_INV, N);
    return f;
}
/*! \brief Algorithm 12 BaseCaseMultiply(𝑎0, 𝑎1, 𝑏0, 𝑏1, 𝛾) 

 - работает как умножение комплексных чисел, где 𝛾 - мнимая единица. */
uint16x2_t BaseCaseMultiply(uint16x2_t a, uint16x2_t b, uint16_t g){
    uint16x2_t c;
    c[0] = (a[0]*(uint32_t)b[0] + a[1]*(((uint32_t)b[1]*g)%Q_PRIME)) % Q_PRIME;
    c[1] = (a[1]*(uint32_t)b[0] + a[0]*(uint32_t)b[1]  ) % Q_PRIME;
    return c;
}
#if 0 // не сделано
uint16x16_t BaseCaseMultiply_x16(uint16x16_t a, uint16x16_t b, uint16x16_t z, uint16x16_t w, uint16x16_t p, uint16x16_t u){
    uint16x16_t b0 = __builtin_shufflevector(b, b, 0,0,2,2,4,4,6,6,8,8,10,10,12,12,14,14);
    uint16x16_t b1 = __builtin_shufflevector(b, b, 1,1,3,3,5,5,7,7,9,9,11,11,13,13,15,15);
    uint16x16_t a1 = __builtin_shufflevector(a, a, 1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14);
    r00 = VMULM_barrett(a0, b0, u, p);
    r11 = VMULM_barrett(a1, b1, u, p);
    r01 = VMULM_barrett(a0, b1, u, p);
    r10 = VMULM_barrett(a1, b0, u, p);
    r11 = VMULM(r11, z, w, p);
    r00 = VADDM(r00, r11, p);
    r11 = VADDM(r10, r01, p);
    return {r00, r11};
}
#endif
/*! Algorithm 11 MultiplyNTTs(𝑓,𝑔)
    Computes the product (in the ring 𝑇_𝑞) of two NTT representations. 
    \param lda линейное смещение при умножении
 */
void MultiplyNTTs(uint16x2_t *h, uint16x2_t *a, uint16x2_t *b, unsigned lda){

    for (int i=0; i<N/VL; i++){
        //BaseCaseMultiply_x16();
    }
    for (int i=0; i<N/2; i++)
        h[i] = BaseCaseMultiply(a[i*lda], b[i],  zeta2[i]);
}
// Dot product
void dot(uint16x2_t *h, uint16x2_t *a, uint16x2_t *b, unsigned int k){
    const unsigned int lda = 1;
    MultiplyNTTs(h, a, b, lda);
}
// Matrix mul vector
void mv_mul(uint16x2_t *h, uint16x2_t *a, uint16x2_t *b, unsigned int k){
    const unsigned int lda = N/2;
    for(int i=0;i<k; i++) {
        MultiplyNTTs(h, a, b, lda);
        h+=lda, a++, b+=lda;
    }
}
// Matrix^T mul vector MV с транспонированием матрицы A
void mv_mult(uint16x2_t *h, uint16x2_t *a, uint16x2_t *b, unsigned int k){
    const unsigned int lda = N/2;
    for(int i=0;i<k; i++){
        MultiplyNTTs(h, a, b, 1);
        h+=lda, a+=lda, b+=lda;
    }
}
#if 0
/*! 
*/
void K_PKE_Decrypt(uint8_t *ct, uint8_t *dk){

    uint32_t* u = Decompress(ct, du);//
    uint32_t* v = Decompress(ct+du*k, dv);
    unsigned int n = du;
    for (i = 0; i < k; i++){
        vec_subm(v, v, w, du, q);
        uint32_t* invNTT(s, NTT(u, gamma, du), )
    }
}
#endif
#include <stdio.h>
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
int main(int argc, char** argv)
{
    uint32_t qm = mod_inverse(Q_PRIME);
    printf ("Q_PRIME = %d, Q_MONT =%d %d Ur=%d\n", Q_PRIME, qm, (Q_PRIME*qm)&0xFFFF, U_BARRETT);
    int nd;
    uint32_t Q_DIV =div_c(Q_PRIME, &nd);
    int s = 8;
    //Q_DIV = (Q_DIV +(0u<<(s-1)))>>s; nd-=s;
    printf ("Q_DIV = x%04X>>%d\n", Q_DIV, nd);
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
        printf("%4d,%5d,", zeta2[i], zeta2[i+1]-Q_PRIME);
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
    NTT(a, zeta);
    NTT(b, zeta);
    MultiplyNTTs(a, a, b, 1);
    iNTT(a, zeta);
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
        uint16x16_t P = VSET1(Q_PRIME);
        uint16x16_t Ur = VSET1(U_BARRETT);
        uint16_t r;
        for (uint16_t a=0; a<p; a++){
            for (uint16_t b=0; b<p; b++){
                uint16x16_t av = VSET1(a);
                uint16x16_t bv = VSET1(b);
                uint16x16_t r = VMULM_barrett(av,bv, Ur, P);
                if (r[0] != (a*(uint32_t)b)%p) {
                    printf("fail mont mul %d * %d = %d != %d\n", a, b, r, (a*(uint32_t)b)%p);
                    //break;
                    return 0;
                }
            }
        }
        printf("Barrett OK\n");
    }    
    int i = 1;
    if (0) for (int len=N/2; len>=2; len>>=1){// стадия разложения длина
        printf("\nl=%d:\n", len);
        for (int offs=0; offs<N; offs+=2*len){// смещениеб число шагов N/2len
            printf("%d ", i);
            i++;
        }
    }
    if (1) {// расчет коэффициентов Shoup для операции NTT_CT_butterfly_x8
        uint16_t g,w;
        printf("xzeta[]=\n");
        int L=16, vl=8;
        for (int k=0; k<L; k++){
            int i = L+k;
//            printf("[%d]={", k);
            g = POWM(ZETA, BitRev7(k), Q_PRIME);
            w = ((uint32_t)(g) <<16)/Q_PRIME;
            printf("{%5d,", g);
            for (int len=vl, l=0; len>=2; len>>=1, l++){// стадия разложения длина
                for (int offs=0,j=0; offs<vl; offs+=len, j++){// смещение число шагов N/2len
                    //printf("%2d ", i+j);
                    g = POWM(ZETA, BitRev7(i+j), Q_PRIME);
                    w  = ((uint32_t)g <<16)/Q_PRIME;
                    printf("%5d,", g);
                }
                i += (L+k)<<l;
            }
            g = Q_PRIME-POWM(ZETA, BitRev7(k), Q_PRIME);
            w = ((uint32_t)(g) <<16)/Q_PRIME;
            printf(" %5d,", g);
            i=L+k;
            for (int len=vl, l=0; len>=2; len>>=1, l++){// стадия разложения длина
                for (int offs=0,j=0; offs<vl; offs+=len, j++){// смещение число шагов N/2len
                    //printf("-%2d ", i+j);
                    g = Q_PRIME-POWM(ZETA, BitRev7(i+j), Q_PRIME);
                    w = ((uint32_t)(g) <<16)/Q_PRIME;
                    printf("%5d,", g);
                }
                i += (L+k)<<l;
            }
            printf("},\n");
        }
    }


    if (1) {// проверка редукции Барретта
        uint32_t p = Q_PRIME;
        printf("Montgomery q=%d, qm=%d\n", p, Q_MONT);
        uint16x16_t P = VSET1(Q_PRIME);
        uint16x16_t qm = VSET1(-Q_MONT);
        uint16_t r;
        for (uint16_t a=0; a<p; a++){
            for (uint16_t b=0; b<p; b++){
                uint16x16_t av = VSET1(a);
                uint16x16_t bv = VSET1(b);
                uint16x16_t r = VMULM_mont(av,bv, qm, P);
                if (((uint32_t)r[0]<<16)%Q_PRIME != (a*(uint32_t)b)%p) {
                    printf("fail vmont mul %d * %d = %d != %d\n", a, b, r, (a*(uint32_t)b)%p);
                    //break;
                    return 0;
                }
            }
        }
        printf("Montgomery OK\n");
    }

    if (1) {// проверка умножения методом Монтгомери
        uint16_t r;
        for (uint16_t a=0; a<0x7FFF; a++){
            for (uint16_t b=0; b<0x7FFF; b++){
                r = mont_MULM(a, b);
                if (((uint32_t)r<<16)%Q_PRIME != (a*(uint32_t)b)%Q_PRIME) 
                    printf("fail mont mul %d * %d = %d != %d\n", a, b, r, (a*b)%Q_PRIME);
            }
        }            
    }
    if (1) {// проверка умножения методом Shoup
        for (uint16_t a=0; a<0xFFFE; a++){
            for (uint16_t b=0; b<Q_PRIME; b++){
                uint16_t w = ((uint32_t)b<<16)/Q_PRIME;
                uint16_t r = shoup_MULM(a, b, w);
                if (r != (a*(uint32_t)b)%Q_PRIME) 
                    printf("fail shoup mul %d * %d = %d != %d\n", a, b, r, (a*b)%Q_PRIME);
            }
        }            
    }


    return 0;
}