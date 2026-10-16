#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supported by OpenCL implementation on your device."
#endif

#ifdef cl_khr_subgroups
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#elif defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#error "Subgroup not supported on your device."
#endif

#define OVERLOADABLE __attribute__((overloadable))
#ifndef cl_khr_extended_bit_ops
static inline uchar16 OVERLOADABLE bitfield_extract_unsigned(uchar16 x, uchar offs, uchar mask){
    return (x>>offs) & (mask);
}
static inline uchar OVERLOADABLE bitfield_extract_unsigned(uchar x, uchar offs, uchar mask){
    return (x>>offs) & (mask);
}
#endif

#ifdef cl_intel_required_subgroup_size
// Always use subgroup size of 32 on Intel.
// reqd_sub_group_size()
//#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
// Always use subgroups size of 64 on Adreno.
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#else
// TODO: do not know how to choose subgroup size on other GPUs.
#warning "Selecting subgroup size is not supported on your device."
#endif

#if     defined(INTEL_GPU)
#define REQD_SUBGROUP_SIZE_DEFAULT REQD_SUBGROUP_SIZE_32
#elif   defined(ADRENO_GPU)
#define REQD_SUBGROUP_SIZE_DEFAULT REQD_SUBGROUP_SIZE_64
#else
#define REQD_SUBGROUP_SIZE_DEFAULT
#endif




/* Это определение эквивалентно intel_sub_group_block_read */
static inline half OVERLOADABLE _sub_group_block_read1(half* yy){
    int ix = get_sub_group_local_id();
    return yy[ix];
}
static inline float OVERLOADABLE _sub_group_block_read1(float* yy){
    int ix = get_sub_group_local_id();
    return yy[ix];
}
static inline half16 OVERLOADABLE _sub_group_block_read(half* yy){
    half16 y;
    int ix = get_sub_group_local_id();
    int sz = get_max_sub_group_size();
    for (int k=0; k<16; k++) y[k] = yy[k*sz+ix];
    return y;
}

static inline ushort16 OVERLOADABLE _sub_group_block_read(ushort* yy){
    ushort16 y;
    int ix = get_sub_group_local_id();
    int sz = get_max_sub_group_size();
    for (int k=0; k<16; k++) y[k] = yy[k*sz+ix];
    return y;
}
static inline float16 OVERLOADABLE _sub_group_block_read(float* yy){
    float16 y;
    int ix = get_sub_group_local_id();
    int sz = get_max_sub_group_size();
    for (int k=0; k<16; k++) y[k] = yy[k*sz+ix];
    return y;
}
static inline float8 OVERLOADABLE _sub_group_block_read(global float* yy){
    float8 y;
    int ix = get_sub_group_local_id();
    int sz = get_max_sub_group_size();
    for (int k=0; k<8; k++) y[k] = yy[k*sz+ix];
    return y;
}
static inline float4 OVERLOADABLE _sub_group_block_read4(global float* yy){
    float4 y;
    int ix = get_sub_group_local_id();
    int sz = get_max_sub_group_size();
    #pragma unroll
    for (int k=0; k<4; k++) y[k] = yy[k*sz+ix];
    return y;
}

static inline ushort OVERLOADABLE _sub_group_block_read_us1(global ushort* yy){
    int ix = get_sub_group_local_id();
    return yy[ix];
}
static inline ushort8 OVERLOADABLE _sub_group_block_read_us(global ushort* yy){
    ushort8 y;
    int ix = get_sub_group_local_id();
    int sz = get_max_sub_group_size();
    #pragma unroll
    for (int k=0; k<8; k++) y[k] = yy[k*sz+ix];
    return y;
}
static inline ushort4 OVERLOADABLE _sub_group_block_read_us4(global ushort* yy){
    ushort4 y;
    int ix = get_sub_group_local_id();
    int sz = get_max_sub_group_size();
    #pragma unroll
    for (int k=0; k<4; k++) y[k] = yy[k*sz+ix];
    return y;
}
static inline void OVERLOADABLE _sub_group_block_write(global float* yy, float8 y){
    int ix = get_sub_group_local_id();
    int sz = get_max_sub_group_size();
    #pragma unroll
    for (int k=0; k<8; k++) yy[k*sz+ix]=y[k];
}
static inline void OVERLOADABLE _sub_group_block_write4(global float* yy, float4 y){
    int ix = get_sub_group_local_id();
    int sz = get_max_sub_group_size();
    #pragma unroll
    for (int k=0; k<4; k++) yy[k*sz+ix]=y[k];
}


#if defined(__opencl_c_integer_dot_product_input_4x8bit)
// Это шаблон использования функции 
static int _matrix_mad_i8_i8_k32( int v, int8 b, int acc )
{
    // Note: 8 is the size of the subgroup.
    // As K is 32, and the size of the subgroup is 8, each
    // work item contributes 4 elements of the 1 x K vector.
    // as_char4() is used to reinterpret 32-bits of data
    // as four components of 8-bit data.

    int result = acc;

    result = dot_acc_sat( as_char4( sub_group_broadcast( v, 0 ) ), as_char4( b.s0 ), result );
    result = dot_acc_sat( as_char4( sub_group_broadcast( v, 1 ) ), as_char4( b.s1 ), result );
    result = dot_acc_sat( as_char4( sub_group_broadcast( v, 2 ) ), as_char4( b.s2 ), result );
    result = dot_acc_sat( as_char4( sub_group_broadcast( v, 3 ) ), as_char4( b.s3 ), result );

    result = dot_acc_sat( as_char4( sub_group_broadcast( v, 4 ) ), as_char4( b.s4 ), result );
    result = dot_acc_sat( as_char4( sub_group_broadcast( v, 5 ) ), as_char4( b.s5 ), result );
    result = dot_acc_sat( as_char4( sub_group_broadcast( v, 6 ) ), as_char4( b.s6 ), result );
    result = dot_acc_sat( as_char4( sub_group_broadcast( v, 7 ) ), as_char4( b.s7 ), result );

    return result;
}
// SIMD16 Intel B580 умножение в подгруппе, каждый элемент подгруппы держит свое значение v,
// полученное при групповом чтении
static float _matrix_mad_f16_f16_k16( half v, half16 b, float acc ){
#if defined(cl_intel_subgroup_matrix_multiply_accumulate)
    return as_float(intel_sub_group_f16_f16_matrix_mad_k16(as_short(v), as_int8(b), as_float(acc))); 
#else // эмуляция операции
    float result = acc;
    for (int ix=0; ix<16; ix++)
        result += sub_group_broadcast(v, ix)*b[ix];// fma
    return result;
#endif
}

// каждый элемент подгруппы считает свою строку
float16 mul_mv_f16(global half*a,  global half * b, float acc, int K){
    const int M=16;
    float16 sum=(0);
    for (int ib =0;  ib<K; ib++){// номер блока в строке
        for (int row=get_local_id(0); row<M; row+=get_local_size(0)){
            half v = _sub_group_block_read1(a);
            half16 u = vload16(ib, b);
            sum[row] = _matrix_mad_f16_f16_k16(v, u, sum[row]);
            a+= 16; 
        }
    }
    return sum;
}

static int _matrix_mad_i8_u8_k32( int v, int8 b, int acc ){
    int r = acc;
#if 1 // тут собраны функции для B580
//    return __builtin_IB_dp4a_uu(v,b,acc);
    return as_int(intel_sub_group_u8_u8_matrix_mad_k32(as_ushort2(v).x, as_uint8(b), as_int(acc))); 
    return as_int(intel_sub_group_f16_f16_matrix_mad_k16(as_short2(v).x, as_int8(b), as_float(acc))); 
    return as_int(intel_sub_group_bf16_bf16_matrix_mad_k16(as_short2(v).x, as_int8(b), as_float(acc))); 
//    return as_int(intel_sub_group_tf32_tf32_matrix_mad_k8(as_float(v), as_float8(b), as_float(acc))); 
    return (int)intel_sub_group_u8_u8_matrix_mad_k32(as_uint(v), as_uint8(b), acc); 
#else
    for (int k=0;k<8; ++k)
        r = dot_acc_sat( as_char4( sub_group_broadcast( v, k ) ), as_uchar4( b[k] ), r );
    return r;
#endif
}
#endif
static inline float OVERLOADABLE _matrix_mad_f16_f16_k16( int v, int8 b, float acc )
{
    float result = acc;
    result += dot(as_half2(sub_group_broadcast( v, 0 )), as_half2( b.s0 ));
    result += dot(as_half2(sub_group_broadcast( v, 1 )), as_half2( b.s1 ));
    result += dot(as_half2(sub_group_broadcast( v, 2 )), as_half2( b.s2 ));
    result += dot(as_half2(sub_group_broadcast( v, 3 )), as_half2( b.s3 ));
    result += dot(as_half2(sub_group_broadcast( v, 4 )), as_half2( b.s4 ));
    result += dot(as_half2(sub_group_broadcast( v, 5 )), as_half2( b.s5 ));
    result += dot(as_half2(sub_group_broadcast( v, 6 )), as_half2( b.s6 ));
    result += dot(as_half2(sub_group_broadcast( v, 7 )), as_half2( b.s7 ));
    return result;
}
static inline half OVERLOADABLE _matrix_mad_f16_f16_k16( int v, int8 b, half acc )
{
    float result = acc;
    for (int k=0;k<8; ++k)
        result += dot(as_half2(sub_group_broadcast( v, k )), as_half2( b[k] ));
    return result;
}
static inline float OVERLOADABLE _matrix_mad_f16_f16_k32( int v, int8 b, float acc )
{
    float result = acc;
    for (int k=0;k<16; ++k)
        result += dot(as_half2(sub_group_broadcast( v, k )), as_half2( b[k] ));
    return result;
}
//  Note каждый элемент подгруппы считает свою строку размерность local_work_size = может быть {m/8, 8, 1} {4,8,1}
static inline float OVERLOADABLE _matrix_mad_f32_f32_k8( int v, int8 b, float acc ) {
    float sum = acc;
    #pragma unroll
    for (int k=0;k<8; ++k)
        sum += as_float(sub_group_broadcast( v, k )) * as_float( b[k] );
    return sum;
}
static inline float OVERLOADABLE _matrix_mad_f32_f32_k16( int v, int16 b, float acc ) {
    float sum = acc;
    #pragma unroll
    for (int k=0;k<16; ++k)
        sum += as_float(sub_group_broadcast( v, k )) * as_float( b[k] );
    return sum;
}
// размер локальной группы {SG,1} {K/SG, Rows/RC} ASSERT(Rows%(SG*RC))
kernel void mul_vec_f16_f16_f32(global half* a, global half* b, global float* dst, uint ne){
    const int RC=8;
    float8 sum = (0);

    uint r0 = get_group_id(0)*RC;// номер строки
    global half *xx = a + (r0 + get_sub_group_local_id())*ne;
    global int  *yy = (global int*)b;

    for (int mb=0; mb<ne/8; mb++){
        half16 b = vload16(mb ,xx);// блоки по 8(16) 
        //dequantize_bf16_f32_block();
        for(int r=0; r<RC; ++r)
        {
            int v = yy[get_sub_group_local_id()];// групповая операция чтения
            sum[r] = _matrix_mad_f16_f16_k16(v, as_int8(b), sum[r]);
            yy += get_max_sub_group_size();
        }
    }
    for (int r=0; r<RC; ++r)
        sum[r] = sub_group_reduce_add(sum[r]);
    if (get_sub_group_local_id()==0){
        vstore8(sum, r0, (global float*)dst);
    }

}
#define QK4_0                   32
#define QR4_0                   2
#define QK4_1                   32
#define QR4_1                   2
#define QK5_0                   32
#define QR5_0                   2
#define QK5_1                   32
#define QR5_1                   2
#define QK8_0                   32
#define QR8_0                   1
#define QK8_1                   32
#define QR8_1                   1
#define QK_K                    256
#define K_QUANTS_PER_ITERATION  2

typedef char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;

//------------------------------------------------------------------------------
// block_q4_0
//------------------------------------------------------------------------------
struct block_q4_0
{
    half d;
    uint8_t qs[QK4_0 / 2];
};

//------------------------------------------------------------------------------
// block_q4_1
//------------------------------------------------------------------------------
struct block_q4_1
{
    half d;
    half m;
    uint8_t qs[QK4_1 / 2];
};

//------------------------------------------------------------------------------
// block_q5_0
//------------------------------------------------------------------------------
struct block_q5_0
{
    half d;
    uint32_t qh;
    uint8_t qs[QK5_0 / 2];
};

//------------------------------------------------------------------------------
// block_q5_1
//------------------------------------------------------------------------------
struct block_q5_1
{
    half d;
    half m;
    uint32_t qh;
    uint8_t qs[QK5_1 / 2];
};

//------------------------------------------------------------------------------
// block_q8_0
//------------------------------------------------------------------------------
struct block_q8_0
{
    half d;
    int8_t qs[QK8_0];
};
//------------------------------------------------------------------------------
// block_q8_1
//------------------------------------------------------------------------------
struct block_q8_1
{
    half d; // delta
    half s; // d * sum(qs[i])
    int8_t qs[QK8_1]; // quants
} ;
//------------------------------------------------------------------------------
// block_q2_K
//------------------------------------------------------------------------------
struct block_q2_K
{
    uint8_t scales[16];
    uint8_t qs[64];
    half d;
    half dmin;
};

//------------------------------------------------------------------------------
// block_q3_K
//------------------------------------------------------------------------------
struct block_q3_K
{
    uint8_t hmask[32];
    uint8_t qs[64];
    uint8_t scales[12];
    half d;
};

//------------------------------------------------------------------------------
// block_q4_K
//------------------------------------------------------------------------------
struct block_q4_K
{
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

//------------------------------------------------------------------------------
// block_q5_K
//------------------------------------------------------------------------------
struct block_q5_K
{
    half d;
    half dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

//------------------------------------------------------------------------------
// block_q6_K
//------------------------------------------------------------------------------
struct block_q6_K
{
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    half d;
};
//------------------------------------------------------------------------------
// Non-linear quants
//------------------------------------------------------------------------------
#define QK4_NL 32
struct block_iq4_nl {
    half d;
    uint8_t qs[QK4_NL/2];
};

struct block_iq4_xs {
    half d;
    uint16_t scales_h;
    uint8_t  scales_l[QK_K/64];
    uint8_t  qs[QK_K/2];
};
//constant char16 kvalues_iq4nl = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
//------------------------------------------------------------------------------
// block_q8_K
//------------------------------------------------------------------------------
// This is only used for intermediate quantization and dot products
struct block_q8_K {
    float   d;              // delta
    int8_t  qs[QK_K];       // quants
    int16_t bsums[QK_K/16]; // sum of quants in groups of 16
};
//------------------------------------------------------------------------------
// dequantize_q8_K_f32, dequantize_q8_K_f16 il = 0..15
//------------------------------------------------------------------------------
inline float16 dequantize_q8_K_f32(struct block_q8_K * x, uint il) {
    char16 q = vload16(il, x->qs);
    return (x->d)*convert_float16(q);
}
//------------------------------------------------------------------------------
// dequantize_q4_0_f32, dequantize_q4_0_f16, dequantize_q4_0_bf16
//------------------------------------------------------------------------------
inline float16 dequantize_q4_0_f32_(global struct block_q4_0 * xb, short il) {// удалить
    ushort8 qs = vload8(0, (global ushort *)xb->qs);
    float d1 = il ? (xb->d / 16.h) : xb->d;
    float d2 = d1 / 256.f;
    float md = -8.h * xb->d;
    ushort mask0 = il ? 0x00F0 : 0x000F;
    ushort mask1 = mask0 << 8;
    float16 r;
    r.even = d1 * convert_float8(qs & mask0) + md;
    r.odd  = d2 * convert_float8(qs & mask1) + md;
    return r;
}
inline float16 dequantize_q4_0_f32(global uchar * xb_qs, half d, short il) {
    ushort8 qs = vload8(0, (global ushort *)xb_qs);
    float d1 = il ? (d / 16.h) : d;
    float d2 = d1 / 256.f;
    float md = -8.h * d;
    ushort mask0 = il ? 0x00F0 : 0x000F;
    ushort mask1 = mask0 << 8;
    float16 r;
    r.s02468ace = d1 * convert_float8(qs & mask0) + md;
    r.s13579bdf = d2 * convert_float8(qs & mask1) + md;
    return r;
}

inline half16 dequantize_q4_0_f16(global struct block_q4_0 * xb, ushort il) {
    ushort8 qs = vload8(0,(global ushort *)xb->qs);
    half d1 = il ? (xb->d / 16.h) : xb->d;
    half d2 = d1 / 256.h;
    half md = -8.h * xb->d;
    ushort mask0 = il ? 0x00F0 : 0x000F;
    ushort mask1 = mask0 << 8;
    half16 r;
    r.s02468ace = d1 * convert_half8(qs & mask0) + md;
    r.s13579bdf = d2 * convert_half8(qs & mask1) + md;
    return r;
}
#ifdef cl_intel_bfloat16_conversions
inline ushort16 dequantize_q4_0_bf16(global struct block_q4_0 * xb, short il) {
    ushort8 qs = vload8(0, (global ushort *)xb->qs);
    float d1 = il ? (xb->d / 16.h) : xb->d;
    float d2 = d1 / 256.f;
    float md = -8.h * xb->d;
    ushort mask0 = il ? 0x00F0 : 0x000F;
    ushort mask1 = mask0 << 8;
    ushort16 r;
    r.even = intel_convert_bfloat168_as_ushort8(d1 * convert_float8(qs & mask0) + md);
    r.odd  = intel_convert_bfloat168_as_ushort8(d2 * convert_float8(qs & mask1) + md);
    return r;
}
#endif
//------------------------------------------------------------------------------
// dequantize_q4_1_f32, dequantize_q4_1_f16
//------------------------------------------------------------------------------
inline half16 dequantize_q4_1_f16(global struct block_q4_1 * xb, ushort il) {
    ushort8 qs = vload8(0,(global ushort *)xb->qs);
    half d1 = il ? (xb->d / 16.h) : xb->d;
    half d2 = d1 / 256.h;
    half md = xb->m;
    ushort mask0 = il ? 0x00F0 : 0x000F;
    ushort mask1 = mask0 << 8;
    half16 r;
    r.s02468ace = d1 * convert_half8(qs & mask0) + md;
    r.s13579bdf = d2 * convert_half8(qs & mask1) + md;
    return r;
}
/* Interleaving il  = 0..1 */
inline float16 dequantize_q4_1_f32(global struct block_q4_1 * xb, ushort il) {
    ushort8 qs = vload8(0,(global ushort *)xb->qs);
    half d1 = il ? (xb->d / 16.h) : xb->d;
    half d2 = d1 / 256.h;
    half md = xb->m;
    ushort mask0 = il ? 0x00F0 : 0x000F;
    ushort mask1 = mask0 << 8;
    float16 r;
    r.s02468ace = d1 * convert_float8(qs & mask0) + md;
    r.s13579bdf = d2 * convert_float8(qs & mask1) + md;
    return r;
}
//------------------------------------------------------------------------------
// dequantize_q5_0_f32, dequantize_q5_0_f16
//------------------------------------------------------------------------------
#if 0
/* Interleaving il  = 0..15 */
inline float16 void dequantize_q5_0_f32(global struct block_q5_0 * x, uint il){
    half d = x->d;
    uint qh = *(uint*)(x->qh);

    const int xh_0 = ((qh >> (il +  0)) <<4) & 0x10;
    const int xh_1 = ((qh >> (il + 16)) <<4) & 0x10;

    v.x = d*(((x->qs[il] & 0xf) | xh_0)-16);
    v.y = d*(((x->qs[il] >>  4) | xh_1)-16);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}
#endif
//------------------------------------------------------------------------------
// dequantize_q8_0_f32, dequantize_q8_0_f16
//------------------------------------------------------------------------------
/* Interleaving il  = 0..1 */
inline half16 dequantize_q8_0_f16(global struct block_q8_0 * xb, ushort il) {
    half d = xb->d;
    char16 ql = vload16(il, xb->qs);
    return d*convert_half16(ql);
}
/* Interleaving il  = 0..1 */
inline float16 dequantize_q8_0_f32(global char * qs, half d, ushort il) {
    char16 ql = vload16(il, qs);
    return convert_float16(d*convert_half16(ql));
}
//------------------------------------------------------------------------------
// K-quants
//------------------------------------------------------------------------------
inline void get_scale_min_k4(int j, global const uint8_t * q, private uint8_t * d, private uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63; 
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
/* Interleaving il  = 0..15 */
inline float16 dequantize_q4_k_f32(global struct block_q4_K * xb, ushort il) {
    half d = xb->d;
    half dmin = xb->dmin;
    uchar mask = 0x0F;
    uint8_t sc, m;
    get_scale_min_k4(il&7, xb->scales, &sc, &m);
    const float d1 = d * sc; 
    const float m1 = dmin * m;
    uchar16 ql = vload16(il&7, xb->qs);// 128 bytes
    ql = il&8? ql>>4 : ql & mask;
    return d1*convert_float16(ql)-m1;
}
/* Interleaving il  = 0..15 */
inline float16 dequantize_q5_k_f32(global struct block_q5_K * xb, ushort il) {
    half d = xb->d;
    uchar mask = 0x0F;
    uint8_t sc, m;
    get_scale_min_k4(il&7, xb->scales, &sc, &m);
    uchar16 ql = vload16(il&7, xb->qs);// 128 bytes
    uchar16 qh = vload16(il&1, xb->qh);// 32 bytes
    ql = il>=8? ql>>4 : ql & mask;
    return d*sc*convert_float16(ql|((qh>>(il/2)&(uchar)1)<<4))-m;
}
/* Interleaving il  = 0..15 */
inline half16 dequantize_q6_k_f16(global struct block_q6_K * xb, ushort il) {
    half d = xb->d;
    int8_t sc = xb->scales[il];
    uchar mask = 0x0F;
    uchar16 ql = vload16(il&7, xb->ql);// 128 bytes
    uchar16 qh = vload16(il&3, xb->qh);// 64 bytes
    ql = il>=8? ql>>4 : ql & mask;
    return d*sc*convert_half16((ql)|(qh>>(2*(il>>2))<<4) - (uchar)32);
}
/* Interleaving il  = 0..15 */
inline float16 dequantize_q6_k_f32(global struct block_q6_K * xb, ushort il) {
    half d = xb->d;
    int8_t sc = xb->scales[il];
    uchar mask = 0x0F;
    uchar16 ql = vload16(il&7, xb->ql);// 128 bytes
    uchar16 qh = vload16(il&3, xb->qh);// 64 bytes
    ql = bitfield_extract_unsigned(ql,(il>>3)*4, 4);
    qh = bitfield_extract_unsigned(qh,(il>>2)*2, 2);

    return d*sc*convert_float16((ql|(qh<<4))- (uchar)32);// есть операция MADW
}

#ifdef cl_intel_bfloat16_conversions
/* Interleaving il  = 0.. - выбор индекса */
inline float16 dequantize_bf16_f32(global ushort * xb, uint il) {
    ushort16 q = vload16(il, xb);
    return intel_convert_as_bfloat1616_float16(q);
}
inline ushort16 quantize_f32_bf16(global float * xb, uint il) {
    float16 q = vload16(il, xb);
    return as_ushort16(intel_convert_f32_to_bf16(q));
}
#endif
inline float OVERLOADABLE vec_dot(struct block_q8_0 *x, struct block_q8_0 * y){
    int sumi = 0;
#if defined(__opencl_c_integer_dot_product_input_4x8bit)
    for (int j = 0; j < QK8_0/4; j++) // DP4A or dot_4x8packed
        sumi = dot_acc_sat(vload4(j,x->qs),vload4(j,y->qs),sumi);
#else
    for (int j = 0; j < QK8_0; j++)
        sumi += x->qs[j]*y->qs[j];
#endif
    return sumi*((float)x->d*(float)y->d);
}
inline float OVERLOADABLE vec_dot(struct block_q8_1 *x, struct block_q8_1 * y){
    int sumi = 0;
#if defined(__opencl_c_integer_dot_product_input_4x8bit)
    for (int j = 0; j < QK8_1/4; j++) // DP4A or dot_4x8packed
        sumi = dot_acc_sat(vload4(j,x->qs),vload4(j,y->qs),sumi);
#else
    for (int j = 0; j < QK8_0; j++)
        sumi += x->qs[j]*y->qs[j];
#endif
    return sumi*((float)x->d*(float)y->d)+ ((float)x->s*(float)y->s);
}
// assuming SIMD group size is 32
inline void OVERLOADABLE quantize_q8_1(global float * x, local struct block_q8_1 *y) {
    uint ix = get_sub_group_local_id();
    float v = x[ix];//as_float(intel_sub_group_block_read((global uint*)x));// 
    float sum = sub_group_reduce_add(v);
    float amax = sub_group_reduce_max(fabs(v));
    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : rint(v / d );// проверить способ округления
    y->qs[ix] = q;
    if (ix == 0){
        y->d = d;
        y->s = sum;
    }
}
// assuming SIMD group size is 32
inline float OVERLOADABLE quantize_q8_0(float * x, char *qs) {
    uint ix = get_sub_group_local_id();
    float v = x[ix];
    float amax = sub_group_reduce_max(fabs(v));
    const float d = amax / 127;
    const float id = d!=0.f ? 1.0f/d : 0.0f;
    qs[ix] = rint(v*id);
    return d;
}
inline half2 OVERLOADABLE quantize_2f16(float * x) {
    uint ix = get_sub_group_local_id();
    float v = x[ix];
    return (half2){v, v-(half)(v)};
}
#if 0
inline void quantize_q8_K(float * x, struct block_q8_K *y) {
    uint ix = get_sub_group_local_id();
    for (uint i=; i<QK_K; i+=get_max_sub_group_size()){
        float xi = x[ix];
        float sum[get_sub_group_id()] = sub_group_reduce_add(xi);

        float amax = sub_group_reduce_max(fabs(xi));
    }
    const float d = amax / 127;
    const int8_t q = amax == 0.0f ? 0 : rint(xi / d );// проверить способ округления
    if (ix == 0){
        y->d = d;
        y->s = sum;
    }
    y->qs[ix] = q;
}
#endif
//------------------------------------------------------------------------------
// add
//------------------------------------------------------------------------------

// assumption: src1 is a row
// broadcast src1 into src0
kernel void kernel_add_row(
        global float8 * src0,
        global float8 * src1,
        global float8 * dst,
        uint ne
) {
    uint gid = get_global_id(0);
    gid = sub_group_broadcast(gid,0);
    uint idx1 = gid - (gid/ne)*ne; // get_global_id(0) % ne
//    idx1= sub_group_broadcast(idx1,0);
    float8 a = _sub_group_block_read((global float*)(src0+gid));
    float8 b = _sub_group_block_read((global float*)(src1+idx1));
    _sub_group_block_write((global float*)(dst+gid), (a+b));
}
// assumption: src1 is a row
// broadcast src1 into src0
kernel void kernel_mul_row(
        global float8 * src0,
        global float8 * src1,
        global float8 * dst,
        uint ne
) {
    uint gid = get_global_id(0);
    uint idx1 = gid - (gid/ne)*ne; // get_global_id(0) % ne
    gid = sub_group_broadcast(gid,0);
    idx1= sub_group_broadcast(idx1,0);
    float8 a = _sub_group_block_read((global float*)(src0+gid));
    float8 b = _sub_group_block_read((global float*)(src1+idx1));
    _sub_group_block_write((global float*)(dst+gid), (a*b));
}

//------------------------------------------------------------------------------
// gelu
//------------------------------------------------------------------------------
#define GELU_QUICK_COEF -1.702f
#define GELU_COEF_A     0.044715f
#define SQRT_2_OVER_PI  0.79788456080286535587989211986876f

//------------------------------------------------------------------------------
// silu, gelu, elu, sigmoid
//------------------------------------------------------------------------------
inline float  OVERLOADABLE silu(float  x){  return x / (1.f + exp(-x)); }
inline float8 OVERLOADABLE silu(float8 x){  return x / (1.f + exp(-x)); }
inline float  OVERLOADABLE gelu_quick(float x) { return x * (1.f / (1.f + exp(GELU_QUICK_COEF * x))); }
inline float  OVERLOADABLE gelu(float  x){  return 0.5f*x*(1.f + tanh(SQRT_2_OVER_PI*x*(1.f + GELU_COEF_A*x*x))); }
inline float8 OVERLOADABLE gelu(float8 x){  return 0.5f*x*(1.f + tanh(SQRT_2_OVER_PI*x*(1.f + GELU_COEF_A*x*x))); }
inline float  OVERLOADABLE relu(float  x){  return fmax(0.f, x); }
inline float8 OVERLOADABLE relu(float8 x){  return fmax(0.f, x); }
inline float  OVERLOADABLE  elu(float  x){  return (x > 0.f) ? x : expm1(x); }

inline half  relu_f16(half  x){  return fmax(0.h, x); }
inline half  gelu_f16(half  xh){  
    float x = xh;
    return convert_half_rte(0.5f*x*(1.f + tanh(SQRT_2_OVER_PI*x*(1.f + GELU_COEF_A*x*x)))); }
inline half  gelu_quick_f16(half xh) { 
    float x = xh;
    return convert_half_rte(x * (1.f / (1.f + exp(GELU_QUICK_COEF * x)))); }
inline half   elu_f16(half  x){  return (x > 0.h) ? x : half_exp(x)-1.h; }
inline half  silu_f16(half  x){  return x / (1.h + half_exp(-x)); }
inline half   sgn_f16(half  x){  return sign(x); }
inline half   exp_f16(half  x){  return half_exp(x); }
inline half  tanh_f16(half  x){  return tanh(x); }
inline half   neg_f16(half  x){  return -x; }
inline half   sqr_f16(half  x){  return x*x; }
inline half  sqrt_f16(half  x){  return half_sqrt(x); }
inline half  fabs_f16(half  x){  return fabs(x); }
inline half  sigmoid_f16(half x) { return native_recip((1.h + half_exp(-x))); }
inline half step_f16(half  x)   { return  x > 0.h; }
inline half hardsigmoid_f16(half x){ return fmin(1.h, fmax(0.h, (x + 3.h) / 6.h)); }
inline half hardswish_f16 (half x) { return x * fmin(1.h, fmax(0.h, (x + 3.h) / 6.h)); }

inline float sigmoid   (float x) { return native_recip((1.f + exp(-x))); }
inline float hardsigmoid(float x){ return fmin(1.0f, fmax(0.0f, (x + 3.0f) / 6.0f)); }
inline float hardswish (float x) { return x * fmin(1.0f, fmax(0.0f, (x + 3.0f) / 6.0f)); }
inline float op_sgn   (float x)  { return sign(x)/* (x > 0.f ? 1.f : ((x < 0.f ? -1.f : 0.f))) */; }
inline float op_neg  (float x)   { return -x; }
inline float op_step (float x)   { return  x > 0.f; }
inline float op_clamp(float x, const float min, const float max){ return  x < min ? min : (x > max ? max : x); }
inline float op_leaky_relu(float x, const float negative_slope) { return fmax(x, 0) + fmin(x, 0.0f) * negative_slope; }
inline float op_sqr(float  x){  return x*x; }

// синтез унарных операций по шаблону для заданного типа данных
#define KERNEL_UNARY(op, type)      \
kernel void kernel_##op(            \
        global type * src0,         \
        global type * dst           \
) {                                 \
    uint gid = get_global_id(0);    \
    type x = src0[gid];             \
    dst[gid] = op(x);               \
}
KERNEL_UNARY(silu_f16, half);
KERNEL_UNARY(gelu_f16, half);
KERNEL_UNARY(gelu_quick_f16, half);
KERNEL_UNARY(relu_f16, half);
KERNEL_UNARY(tanh_f16, half);
KERNEL_UNARY(exp_f16,  half);
KERNEL_UNARY(elu_f16,  half);
KERNEL_UNARY(sigmoid_f16,   half);
KERNEL_UNARY(hardsigmoid_f16,   half);
KERNEL_UNARY(hardswish_f16,   half);
KERNEL_UNARY(sgn_f16,    half);// sign
KERNEL_UNARY(neg_f16,    half);
KERNEL_UNARY(sqr_f16,    half);
KERNEL_UNARY(sqrt_f16,   half);
KERNEL_UNARY(step_f16,   half);


KERNEL_UNARY(silu, float);
KERNEL_UNARY(gelu, float);
KERNEL_UNARY(gelu_quick,float);
KERNEL_UNARY(relu, float);
KERNEL_UNARY(elu,  float);
KERNEL_UNARY(hardswish, float);
KERNEL_UNARY(hardsigmoid, float);
KERNEL_UNARY(sigmoid,   float);
KERNEL_UNARY(op_sgn,    float);// sign
KERNEL_UNARY(op_neg,    float);
KERNEL_UNARY(op_step,   float);
KERNEL_UNARY(op_sqr,   float);
// __builtin_elementwise_*
KERNEL_UNARY(fabs, float);
KERNEL_UNARY(sign, float);
KERNEL_UNARY(erf,  float);
KERNEL_UNARY(erfc, float);
KERNEL_UNARY(exp,  float);
KERNEL_UNARY(exp2, float);
KERNEL_UNARY(exp10,float);
KERNEL_UNARY(expm1,float);
KERNEL_UNARY(log,  float);
KERNEL_UNARY(log2, float);
KERNEL_UNARY(log10,float);
KERNEL_UNARY(sqrt, float);
KERNEL_UNARY(rsqrt,float);
KERNEL_UNARY(sin,  float);
KERNEL_UNARY(cos,  float);
KERNEL_UNARY(tan,  float);
KERNEL_UNARY(asin, float);
KERNEL_UNARY(acos, float);
KERNEL_UNARY(atan, float);
KERNEL_UNARY(sinh, float);
KERNEL_UNARY(cosh, float);
KERNEL_UNARY(tanh, float);
KERNEL_UNARY(ceil, float);
KERNEL_UNARY(floor,float);
KERNEL_UNARY(trunc,float);
KERNEL_UNARY(round,float);
KERNEL_UNARY(rint, float);

#undef KERNEL_UNARY
inline float8 OVERLOADABLE add(float8 u, float8 v){  return u+v; }
inline float8 OVERLOADABLE sub(float8 u, float8 v){  return u-v; }
inline float8 OVERLOADABLE mul(float8 u, float8 v){  return u*v; }
inline float8 OVERLOADABLE div(float8 u, float8 v){  return u/v; }
// синтез бинарных операций: flat - совсем плоское представление, row - одна строка, дублирование строк
#define KERNEL_BINARY(op, type)                                     \
kernel void kernel_##op##_flat(                                            \
        global type##8 * src0,                                      \
        global type##8 * src1,                                      \
        global type##8 * dst,                                       \
        uint ne                                                     \
) {                                                                 \
    uint gid = sub_group_broadcast(get_global_id(0),0);             \
    type##8 a = _sub_group_block_read((global type*)(src0+gid));    \
    type##8 b = _sub_group_block_read((global type*)(src1+gid));    \
    _sub_group_block_write((global float*)(dst+gid), (op(a,b)));    \
}                                                                   
#define KERNEL_BINARY_ROW(op, type)                                     \
kernel void kernel_##op##_row(                                      \
        global type##8 * src0,                                      \
        global type##8 * src1,                                      \
        global type##8 * dst,                                       \
        uint ne, uint ne01                                          \
) {                                                                 \
    uint gid = sub_group_broadcast(get_global_id(0),0);             \
    type##8 a = _sub_group_block_read((global type*)(src0+gid));    \
    type##8 b = _sub_group_block_read((global type*)(src1+(gid%ne01)));    \
    _sub_group_block_write((global float*)(dst+gid), (op(a,b)));    \
}

KERNEL_BINARY(add, float);
KERNEL_BINARY(sub, float);
KERNEL_BINARY(mul, float);
KERNEL_BINARY(div, float);
KERNEL_BINARY(fmod, float);
KERNEL_BINARY(fmax, float);
KERNEL_BINARY(fmin, float);
KERNEL_BINARY(atan2, float);
KERNEL_BINARY(pow, float);
KERNEL_BINARY(copysign, float);

#undef KERNEL_BINARY

inline float  OVERLOADABLE lerp(float  u, float  v, float mu){  return mix(u,v,mu); }
inline float2 OVERLOADABLE lerp(float2 u, float2 v, float mu){  return mix(u,v,mu); }
inline float4 OVERLOADABLE lerp(float4 u, float4 v, float mu){  return mix(u,v,mu); }
inline float8 OVERLOADABLE lerp(float8 u, float8 v, float mu){  return mix(u,v,mu); }

#if defined(cl_intel_bfloat16_conversions)
static inline float   OVERLOADABLE convert_bf16_to_f32(ushort   u) { return intel_convert_as_bfloat16_float(u); }
static inline float4  OVERLOADABLE convert_bf16_to_f32(ushort4  u) { return intel_convert_as_bfloat164_float4(u); }
static inline float8  OVERLOADABLE convert_bf16_to_f32(ushort8  u) { return intel_convert_as_bfloat168_float8(u); }
static inline float16 OVERLOADABLE convert_bf16_to_f32(ushort16 u) { return intel_convert_as_bfloat1616_float16(u); }
static inline ushort  OVERLOADABLE convert_f32_to_bf16(float    x) { return intel_convert_bfloat16_as_ushort(x); }
#else
// эмуляция конвертации типа BF16 (E8M7)
static inline float   OVERLOADABLE convert_bf16_to_f32(ushort   u) { return as_float(convert_uint(u)<<16); }
static inline float4  OVERLOADABLE convert_bf16_to_f32(ushort4  u) { return as_float4(convert_uint4(u)<<16); }
static inline float8  OVERLOADABLE convert_bf16_to_f32(ushort8  u) { return as_float8(convert_uint8(u)<<16); }
static inline float16 OVERLOADABLE convert_bf16_to_f32(ushort16 u) { return as_float16(convert_uint16(u)<<16); }
static inline ushort convert_f32_to_bf16(float x) { 
#if __has_builtin(__builtin_elementwise_roundeven)

#endif
    if ((as_uint(x) & 0x7fffffff) > 0x7f800000)  /* nan */
        return (as_uint(x) >> 16) | 64; /* force to quiet */
    /* RNE round */
    return (as_uint(x) + (0x7fffu + ((as_uint(x) >> 16) & 1))) >> 16;
}
#endif
static inline ushort convert_f32_to_bf16_rtn(float x) { 
    return (as_uint(x) + 0x8000u) >> 16;
}

static inline half   OVERLOADABLE convert_bf8_to_f16(uchar   u) { return as_half(convert_ushort(convert_ushort(u)<<8u)); }
static inline half8  OVERLOADABLE convert_bf8_to_f16(uchar8  u) { return as_half8(as_ushort8(convert_ushort8(u)<<8u)); }
static inline half16 OVERLOADABLE convert_bf8_to_f16(uchar16 u) { return as_half16(as_ushort16(convert_ushort16(u)<<8u)); }
// эмуляция конвертации типа half в BF8 (E5M2)
static inline uchar convert_f16_to_bf8(half x) { 
    if ((as_ushort(x)&0x7FFF) >0x7C00)// nan
        return (as_ushort(x)>>8) | 2;
    /* RNE round */
    return (as_ushort(x) + (0x7fu + ((as_ushort(x)>> 8) & 1))) >> 8;
}

// Квантизация с компенсацией ошибки округления
kernel void kernel_quant_f16(global float * src0, global float * rem, global half *dst)
{
    uint gid = get_global_id(0);
    float x = src0[gid];
    half q = convert_half_rte(x+rem[gid]);
    dst[gid] = q;
    rem[gid] = x - convert_float(q);
}
typedef ushort __bf16_t;
typedef ushort8 __bf16x8_t ;
typedef ushort16 __bf16x16_t;
kernel void kernel_quant_bf16(global float * src0, global float * rem, global __bf16_t *dst)
{
    uint gid = get_global_id(0);
    float x = src0[gid];
    __bf16_t q = convert_f32_to_bf16_rtn(x+rem[gid]);
    dst[gid] = q;
    rem[gid] = x - convert_bf16_to_f32(q);
}
// квантизация такого вида может выполняться аппаратно при загрузке half => CL_SNORM_INT8 write_imageh read_imageh
// float → CL_SNORM_INT8 (8-bit signed integer)
static inline half quantize_block_q8_0(float x, char* q){
    uint tid = get_sub_group_local_id();
    float vmax = sub_group_reduce_max(fabs(x))/127.0f;
    *q = convert_char_sat_rte(x/vmax);
    return convert_half_rte(vmax);
}
// float → CL_UNORM_INT8 (8-bit signed integer)
static inline half2 quantize_block_q8_1(float x, uchar* q){
    uint tid = get_sub_group_local_id();
    float vmin =  sub_group_reduce_min(x);
    float vmax = (sub_group_reduce_max(x)-vmin)/127.h;
    *q = convert_char_sat_rte((x-vmin)/vmax);
    return (half2)(convert_half_rte(vmax),convert_half_rte(vmax));
}
// субгруппа 32, d = d[tid/32]
static inline float dequantize_block_q8_0(global char* x, half dmax){
    uint tid = get_sub_group_local_id();
    float v = x[tid]*dmax;
    return v;
}
static inline float dequantize_block_q8_1(global uchar* x, half2 dm){
    uint tid = get_sub_group_local_id();
    float v = x[tid]*dm.s0+dm.s1;
    return v;
}
// субгруппа 16, d = d[tid/16]
static inline half2 dequantize_block_q4_0(global uchar* x, half d){
    uint tid = get_sub_group_local_id();
    uchar q = x[tid];
    half2 v;
    v.lo = convert_half(bitfield_extract_unsigned(q,0,4)-(char)8)*d;
    v.hi = convert_half(bitfield_extract_unsigned(q,4,4)-(char)8)*d;
    return v;
}
static inline half2 dequantize_block_q4_1(global uchar* x, half2 dm){
    uint tid = get_sub_group_local_id();
    uchar q = x[tid];
    half2 v;
    v.lo = bitfield_extract_unsigned(q,0,4)*dm.s0+dm.s1;
    v.hi = bitfield_extract_unsigned(q,4,4)*dm.s0+dm.s1;
    return v;
}
// выделить методы BLAS в отдельный файл
// L1 BLAS COPY ( Y ← X )
void blas_copy(float* X, float *Y, uint N, uint offx, uint incx, uint offy, uint incy){
    for(uint i = get_sub_group_local_id(); i<N; i+=get_max_sub_group_size())
        Y[offy+i*incy] = X[offx+i*incx];
}
void blas_swap(float* X, float *Y, uint N, uint offx, uint incx, uint offy, uint incy){
    for(uint i = get_sub_group_local_id(); i<N; i+=get_max_sub_group_size()) {
        float xi = X[offx+i*incx];
        float yi = Y[offy+i*incy];
        X[offx+i*incx] = yi;
        Y[offy+i*incy] = xi;
    }
}
// L1 BLAS SCAL ( X ← α X )
void blas_scal(float* X, uint N, uint offx,uint incx, float alpha){
    for(uint i = get_sub_group_local_id(); i<N; i+=get_max_sub_group_size())
        X[offx + i*incx] = alpha* X[offx + i*incx];
}
// L1 BLAS AXPY ( Y ← α X + Y )
void blas_axpy(float* X, float* Y, uint N, uint offx,uint incx, uint offy,uint incy, float alpha){
    for(uint i = get_sub_group_local_id(); i<N; i+=get_max_sub_group_size())
        Y[offy + i*incy] = alpha* X[offx + i*incx] + Y[offy + i*incy];
}
// L1 BLAS DOT product
float blas_dot(float* X, float* Y, uint N, uint offx,uint incx, uint offy,uint incy, float alpha){
    float sumf;
    for(uint i = get_sub_group_local_id(); i<N; i+=get_max_sub_group_size())
        sumf += Y[offy + i*incy]*X[offx + i*incx];
    return sub_group_reduce_add(sumf);
}
// L1 BLAS NRM2 euclidean norm^2
float blas_nrm2(float* X, float* Y, uint N, uint offx,uint incx, uint offy,uint incy, float alpha){
    float sumf=0;
    for(uint i = get_sub_group_local_id(); i<N; i+=get_max_sub_group_size()){
        float v = *(X + mad24(i, incx, offx));
        sumf += v*v;
    }
    return sub_group_reduce_add(sumf);
}
// L1 BLAS ASUM sum of absolute values
float blas_asum(float* X, uint N, uint offx, uint incx){
    float sumf=0;
    for(uint i = get_sub_group_local_id(); i<N; i+=get_max_sub_group_size()){
        float v = *(float*)(X + mad24(i, incx, offx));
        sumf += fabs(v);
    }
    return sub_group_reduce_add(sumf);
}
// L2 BLAS GER - General matrix rank 1 operation ( A ← α X Y^T+ A )
void blas_ger(float* X, float* Y, float* A, uint M, uint N, uint 
    offx, uint incx, uint offy, uint incy, uint offa, uint lda, float alpha 
){
    uint r = get_group_id(0)* get_num_sub_groups();
    A += mad24(lda, r, offa);
    // for (uint j=0; j<M; ++j)
    float y = *(Y + mad24(r, incy, offy));
    for (uint i=get_local_id(0); i<N; i+=get_local_size(0))
        A[i+ offa] += X[mad24(i, incx, offx)]*y;
}

#define LORA 4096
kernel void lora(global float* x, global float* y, constant float* A, constant float* B, local float * slm, uint M, uint N, uint R, 
    uint offx, uint incx, 
    uint offy, uint incy, 
    uint offa, uint lda, float alpha 
){
    uint r = get_group_id(0)* get_num_sub_groups();
    uint rb = get_group_id(1);
    A += mad24(lda, r, offa);

    float sumf=0;
    for(uint i = get_sub_group_local_id(); i<N; i+=get_max_sub_group_size())
        sumf += A[i]*x[i];
    slm[rb] = sub_group_reduce_add(sumf);

    sumf=0;
    for(uint k = get_sub_group_local_id(); k<R; k+=get_max_sub_group_size())
        sumf += B[k]*slm[k];
    sumf = sub_group_reduce_add(sumf);
    y[r] = alpha*sumf + y[r];
}
// L2 BLAS GEMV -- ( y ← α A x + β y ) ( y ← α A^Tx + β y )
// order (clblasOrder [in]) – Row/column order.
void gemv(
    global char* A, global char* X, global char* Y, 
    uint M, uint N, 
    uint offa, uint lda, 
    uint offx, uint nb01, 
    uint offy, uint nb11, 
    float alpha, float beta
){
    uint row = get_group_id(0)* get_num_sub_groups();
    global float *a = (global float *)(A + mad24(row, lda,  offa));
    global float *x = (global float *)(X + mad24(row, nb01, offx));
    global float *y = (global float *)(Y + mad24(row, nb11, offy));
    float sumf=0;
    for(int i=get_sub_group_local_id(); i < N; i+=get_max_sub_group_size()){
        sumf+=a[i]*x[i];
    }
    sumf = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id()==0)
        y[get_sub_group_id()] = alpha*sumf + beta*y[get_sub_group_id()];
}
// L3 BLAS GEMM 
kernel void gemm(
    global char* A, global char* X, global char* Y, uint M, uint N, 
    uint offa, uint lda, 
    uint offx, uint nb01, 
    uint offy, uint nb11, 
    float alpha, float beta
){
    const uint SD = 16;// размер плитки SD строк
    uint row = get_group_id(0)* get_num_sub_groups();

    global float *a = (global float *)(A + mad24(row, lda*SD,  offa));
    global float *x = (global float *)(X + mad24(row, nb01, offx));
    global float *y = (global float *)(Y + mad24(row, nb11, offy));

    float sumf=0;
    for(uint i=get_sub_group_local_id(); i < N; i+=get_max_sub_group_size()){
        float v = x[i];
        for (uint k = 0; k< SD; ++k)
            sumf+=a[k]*sub_group_broadcast(v, k);
        a+=lda;
    }
    sumf = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id()==0)
        y[get_sub_group_id()] = alpha*sumf + y[get_sub_group_id()]*beta;
}
// L2 BLAS TRSV - метод нужен для QR - разложения upper, 
kernel void trsv(){

}
// !blas


kernel void kernel_quant_q8_0(global float * src0, global float * rem, global char *dst)
{
    uint gid = get_global_id(0);
    float x = src0[gid];
    char q;
    half dmax = quantize_block_q8_0(x+rem[gid], &q);
    dst[gid] = q;
    rem[gid] = x - q*dmax;// остаток 
}

// todo квантизация типов TF32 (E8M10)

// операции для разложения матриц
static inline
float givens(float a, float b, float*c, float* s){
	float r = sqrt(a*a + b*b);
	float d = 1.0f/r;
	*c =  a*d; *s = b*d;
	return r;
}

// Average absolute deviation or Mean absolute difference.
// считает средне-квдратичное отклонение (ошибку)
kernel void kernel_rmse(global float * src0, global float * src1, global float *dst, uint ne00, uint ne01, uint ne11)
{
    uint row = get_group_id(0);
    src0 += ne00*(row%ne01);
    src1 += ne00*(row%ne11);
    float sum=0.f;
    for(int i=get_local_id(0); i<ne00; i+=get_global_size(0)){
        float v = src0[i] - src1[i];
        sum += v*v;
    }
    sum = sub_group_reduce_add(sum);
    if (get_local_id(0)==0) dst[row] = sqrt(sum)/ne00;
}
kernel void kernel_rms(global float * src0, global float *dst, uint ne00)
{
    uint row = get_group_id(0);
    src0 += ne00*row;
    float sum=0.f;
    for(int i=get_local_id(0); i<ne00; i+=get_global_size(0)){
        float v = src0[i];
        sum += v*v;
    }
    sum = sub_group_reduce_add(sum);
    if (get_local_id(0)==0) dst[row] = sqrt(sum)/ne00;
}
inline half   OVERLOADABLE spline(half   b0, half   b1, half  t){  return mix(b0, b1, t); }
inline half2  OVERLOADABLE spline(half2  b0, half2  b1, half  t){  return mix(b0, b1, t); }
inline half16 OVERLOADABLE spline(half16 b0, half16 b1, half  t){  return mix(b0, b1, t); }
inline float  OVERLOADABLE spline(float  b0, float  b1, float t){  return mix(b0, b1, t); }
inline float2 OVERLOADABLE spline(float2 b0, float2 b1, float t){  return mix(b0, b1, t); }
inline float8 OVERLOADABLE spline(float8 b0, float8 b1, float t){  return mix(b0, b1, t); }
// квадратичный сплайн определяется через линейные
float qspline(float b0, float b1, float b2, float t){
    b0 = spline(b0, b1, t);
    b2 = spline(b1, b2, t);
    b0 = spline(b0, b2, t);
    return b0;
}
// кубический сплайн определяется через квадратичные
float cspline(float b0, float b1, float b2, float b3, float t){
    b0 = qspline(b0, b1, b2, t);
    b3 = qspline(b1, b2, b3, t);
    b0 =  spline(b0, b3, t);
    return b0;
}
// Расчет полиномов и степенных рядов
#if defined(cl_khr_subgroup_non_uniform_arithmetic)
// Факториал k!
inline float factorial(const float* a, float x){
    int tid = get_sub_group_local_id();
    // returns [I, a0, (a0 op a1), … (a0 op a1 op … op an-2)].
    return sub_group_non_uniform_scan_exclusive_mul(tid);
}
// биномиальные коэффициенты, k - номер треда в WARP группе
inline float binom(float n){
    uint k = get_sub_group_local_id();
    uint kf=sub_group_non_uniform_scan_exclusive_mul(k);
    return sub_group_non_uniform_scan_exclusive_mul(n-k)/kf;
}
// функция выдает сумму степенного ряда до степени 16
inline float v_poly_eval(const float* a, float x){
    int tid = get_sub_group_local_id();
    float v = sub_group_non_uniform_scan_inclusive_mul(x);
    return sub_group_reduce_add(a[tid]*v);
}
// полиномы Бернштейна
inline float b_poly_eval(const float* a, float x){
    int tid = get_sub_group_local_id();
    float p = sub_group_non_uniform_scan_exclusive_mul(x);
    float q = sub_group_non_uniform_scan_exclusive_mul(1.f-x);
    q = sub_group_shuffle_xor(q,~0u);
    return sub_group_reduce_add(a[tid]*q*p);
}
// интерполяция Лагранжа
inline float l_poly_eval(const float* a, float x){
    return 0;
}
#endif
//------------------------------------------------------------------------------
// clamp
//------------------------------------------------------------------------------
kernel void kernel_clamp(
        global float * src0,
        global float * dst,
        uint ne00,
        uint nb01,
        float min,
        float max
) {
    uint r0 = get_group_id(0);
    global float * x = (global float *) ((global char *) src0 + mul24(r0,nb01));
    for (int i = get_local_id(0); i < ne00; i += get_local_size(0)) 
        dst[i] = op_clamp(x[i], min, max);
}

//------------------------------------------------------------------------------
// level 1 kernels: _asum, _iamax, _nrm2, _dot, _sddot
// level 1 kernels: _axpy, _scal, _swap, _copy
// level 1 kernels: _rotg, _rot, _rotm, _rotmg
// CblasNoTrans, CblasTrans, CblasConjTrans.
// CblasUnit, CblasNonUnit, 
// CblasUpper, CblasLower,
// CblasLeft, CblasRight,
// level 2 gemv: post_op = A, A^T, A^H
// level 2 trmv: post_op = A, A^T, A^H
// level 2 trsv: post_op = A, A^T, A^H
// level 2 symv: CblasUpper, CblasLower
// level 2 hemv: CblasUpper, CblasLower
// level 2 ger :
// level 2 syr : CblasUpper, CblasLower
// level 2 her : CblasUpper, CblasLower
// level 2 syr2: CblasUpper, CblasLower
// level 2 her2: CblasUpper, CblasLower
// level 3 gemm: CblasNoTrans, CblasTrans, CblasConjTrans.
// level 3 symm: CblasUpper, CblasLower
// level 3 hemm: CblasUpper, CblasLower
// level 3 trmm: 
// level 3 trsm: 
// level 3 syrk: 
// level 3 syr2k: 
// level 3 her2k: 
//------------------------------------------------------------------------------
#define N_EMBED 4096 // размерность нейросети
struct _view {
    size_t offs; // смещение от начала массива данных
    size_t size; // смещение от начала массива данных
    uint4 ne;// число элементов тензора
    uint4 nb;// приращения выражены в байтах
};
typedef struct _view view_t;
// 1.1 Рзабить массив на строки
uint4 view_array    (uint M, uint N, uint pitch){
    uint4 nb;
    nb[0] = pitch;
    nb[1] = N*pitch;
    nb[2] = M*N*pitch;
    nb[3] = nb[2];
    return nb;
}
uint4 view_array_transposed(uint M, uint N, uint pitch){
    uint4 nb;
    nb[0] = N*pitch;
    nb[1] = pitch;
    nb[2] = M*N*pitch;
    nb[3] = nb[2];
    return nb;
}
// транспонирование
uint4 view_transpose(uint4 nb){
    nb.s01 = nb.s10;
    return nb;
}
// Appendix D: Checking for Memory Copy Overlap
view_t view_submatrix(uint4 nb, uint2 origin, uint2 range){
    view_t t;
    t.offs = nb[0]*origin[0] + nb[1]*origin[1];
    t.size = sizeof(float) + (range[0]-1)*nb[0] + (range[1]-1)*nb[1];
    t.nb   = nb;
    t.ne.s01 = range;
    t.ne.s23 = 1u;
    return t;
}
// разбивка матрицы на полосы - подстроки 
void view_permute(uint* nb, uint rb){
    nb[2] = nb[2]/rb;
}
// сшивает обратно переставленные строки
void view_cont(uint* nb, uint rb){
    nb[2] = nb[2]*rb;
}
bool is_permuted   (uint4 ne, uint4 nb) {
    return ne[2]*nb[2] == nb[1];
}
bool is_transposed (uint4 ne, uint4 nb) {
    return ne[1]*nb[1] == nb[0];
}
bool is_contiguous (uint4 ne, uint4 nb) {
    return ne[0]*nb[0] == nb[1]
        && ne[1]*nb[1] == nb[2]
        && ne[2]*nb[2] == nb[3];
}

kernel void kernel_mul(
        global float * src0,
        global float * src1,
        global float * dst,
        uint ne00,
        uint nb01,
        uint nb11
) {
    uint r0 = get_group_id(0);
    uint row= mad24(r0,get_num_sub_groups(), get_sub_group_id());
    global float * x = (global float *) ((global char *) src0 + mul24(row,nb01));
    global float * y = (global float *) ((global char *) src1 + mul24(row,nb11));
    global float * d = (global float *) dst  + mul24(row,ne00);// вектор сплошной
    
    for (uint i = get_sub_group_local_id(); i < ne00; i += get_max_sub_group_size())
        d[i] = x[i]*y[i];
}
kernel void kernel_scal(
        global float * src0,
        global float * dst,
        uint ne00,
        uint nb01,
        float alpha
) {
    uint r0 = get_group_id(0);
    uint row= mad24(r0,get_num_sub_groups(), get_sub_group_id());
    global float * x = (global float *) ((global char *) src0 + mul24(row,nb01));
    global float * d = (global float *) dst  + mul24(row,ne00);// вектор сплошной
    for (uint i = get_sub_group_local_id(); i < ne00; i += get_max_sub_group_size())
        d[i] =alpha*x[i];
}
kernel void kernel_copy(
        global float * src0,
        global float * dst,
        uint ne00,
        uint nb01
) {
    uint r0 = get_group_id(0);
    uint row= mad24(r0,get_num_sub_groups(), get_sub_group_id());
    global float * x = (global float *) ((global char *) src0 + mul24(row,nb01));
    global float * d = (global float *) dst  + mul24(row,ne00);// вектор сплошной
    for (uint i = get_sub_group_local_id(); i < ne00; i += get_max_sub_group_size())
        d[i] =(x[i]);// деквантизовать
}
kernel void kernel_swap(
        global float * src0,
        global float * dst,
        uint ne00,
        uint nb01
) {
    uint r0 = get_group_id(0);
    uint row= mad24(r0,get_num_sub_groups(), get_sub_group_id());
    global float * x = (global float *) ((global char *) src0 + mul24(row,nb01));
    global float * d = dst + mul24(row, ne00);
    for (uint i = get_sub_group_local_id(); i < ne00; i += get_max_sub_group_size()){
        float xi =(x[i]);// деквантизовать
        float di =(d[i]);// шаблон
        d[i] = xi;
        x[i] = di;
    }
}
kernel void kernel_axpy(
        global float * src0,
        global float * src1,
        global float * dst,
        uint ne00,
        uint nb01,
        uint nb11,
        uint nb1,
        float alpha
) {
    uint r0 = get_group_id(0);
    uint row= mad24(r0,get_num_sub_groups(), get_sub_group_id());
    global float * x = (global float *) ((global char *) src0 + mul24(row,nb01));
    global float * y = (global float *) ((global char *) src1 + mul24(row,nb11));
    dst = dst + mul24(row, nb1);
    for (uint i = get_sub_group_local_id(); i < ne00; i += get_max_sub_group_size())
        dst[i] = fma(alpha,x[i],y[i]);
}
kernel void kernel_rot(
        global float * src0,
        global float * src1,
        uint ne00,
        uint nb01,
        uint nb11,
        float c, float s
) {
    uint r0 = get_group_id(0);
    uint row= mad24(r0,get_num_sub_groups(), get_sub_group_id());
    global float * x = (global float *) ((global char *) src0 + mul24(row,nb01));
    global float * y = (global float *) ((global char *) src1 + mul24(row,nb11));
    for (uint i = get_sub_group_local_id(); i < ne00; i += get_max_sub_group_size()){
        float2 v;
        v.x = x[i];
        v.y = y[i];
        x[i] = c*v.x + s*v.y;
        y[i] = c*v.y - s*v.x;
    }
}
kernel void kernel_mix(
        global float * src0,
        global float * src1,
        global float * dst,
        uint ne00,
        uint nb01,
        uint nb11,
        uint nb1,
        float alpha// \in [0,1]
) {
    uint r0 = get_group_id(0);
    uint row= mad24(r0,get_num_sub_groups(), get_sub_group_id());
    global float * x = (global float *) ((global char *) src0 + mul24(row,nb01));
    global float * y = (global float *) ((global char *) src1 + mul24(row,nb11));
//    global float * dst = (global float *) ((global char *) dst + mul24(rb,nb1));
    dst = dst + mul24(row, nb1);
    for (uint i = get_local_id(0); i < ne00; i += get_local_size(0)) 
        dst[i] = mix(x[i],y[i],alpha);
}
kernel void kernel_asum(
        global float * src0,
        global float * dst,
        uint ne00,
        uint nb01
){
    uint r0 = get_group_id(0);
    uint row= mad24(r0,get_num_sub_groups(), get_sub_group_id());
    global float * x = (global float *) ((global char *) src0 + mul24(row,nb01));

    float sumf = 0.0f;
    for (uint i = get_sub_group_local_id(); i < ne00; i += get_max_sub_group_size())
        sumf += fabs(x[i]);
    sumf = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id()==0)
        dst[row] = sumf;
}
kernel void kernel_nrm2(
        global float * src0,
        global float * dst,
        uint ne00,
        uint nb01
){
    uint r0 = get_group_id(0);
    uint row= mad24(r0,get_num_sub_groups(), get_sub_group_id());
    global float * x = (global float *) ((global char *) src0 + mul24(row,nb01));

    float sumf = 0.0f;
    for (uint i = get_sub_group_local_id(); i < ne00; i += get_max_sub_group_size())
        sumf += x[i]*x[i];
    sumf = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id()==0)
        dst[row] = sqrt(sumf);
    // добавить редуцирование по рабочей группе
}
kernel void kernel_dot(
        global float * src0,
        global float * src1,
        global float * dst,
        uint ne00,
        uint nb01,
        uint nb11
){
    uint r0 = get_group_id(0);
    uint row= mad24(r0,get_num_sub_groups(), get_sub_group_id());
    global float * x = (global float *) ((global char *) src0 + mul24(row,nb01));
    global float * y = (global float *) ((global char *) src1 + mul24(row,nb11));

    float sumf = 0.0f;
    for (uint i = get_sub_group_local_id(); i < ne00; i += get_max_sub_group_size())
        sumf += x[i]*y[i];
    sumf = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id()==0)
        dst[row] = sumf;
}
kernel void kernel_ffn(
        global float * src0,
        global float * src1,
        global float * srcb,
        global float * srcw,
        global float * dst,
        local  float * slm,
        uint ne00,
        uint nb01,
        uint nb11,
        uint nb21
){
    uint r0 = get_group_id(0);
    uint row= mad24(r0,get_num_sub_groups(), get_sub_group_id());
    global float * x = (global float *) ((global char *) src0 + mul24(row,nb01));
    global float * y = (global float *) ((global char *) src1 + mul24(row,nb11));
    global float * b = (global float *) ((global char *) srcb + mul24(row,nb21));
    global float * w = (global float *) ((global char *) srcw + mul24(row,nb21));
    float norm =0.f;
    float2 sumf = (0.f);
    for (uint i = get_sub_group_local_id(); i < ne00; i += get_max_sub_group_size()){
        float v = b[i];
        norm  += v*v;
        v = v*w[i];
        sumf.s0 += x[i]*v;
        sumf.s1 += y[i]*v;
    }
    norm    = sub_group_reduce_add(norm);
    sumf.s0 = sub_group_reduce_add(sumf.s0);
    sumf.s1 = sub_group_reduce_add(sumf.s1);
    // b = NORM MUL(c); MM SILU MM MUL
    if (get_sub_group_local_id()==0){// пост операция может быть друая, наприммер норма
        sumf*=rsqrt(norm);
        dst[row] = silu(sumf.s0)*sumf.s1;
    }
}
// генератор псевдо случайных чисел, используется для начальных значений скорости.
enum{ MWC64X_A = 4294883355U };
static inline uint mwc64x( uint2* state, uint Prime) {
    state[0] = as_uint2((ulong)Prime*state[0].s0 + state[0].s1);
    return state[0].s0;
}
static inline uint dtree_hash(uint hash, uchar data, uint Prime) {
    return rotate(hash, 6u) ^ data;
}
static inline uint4 xxh32_hash( uint4 state, uint* data, uint len, uint Prime1, uint Prime2) {
//    uint4 state = {Prime1 + Prime2, Prime2, 0, -Prime1};
//    state += hash;
    uint blocks = len>>2;
    for (int i=0; i<blocks; i++) {// вектор 128 бит
        uint4 block = vload4(i, data);
        state = rotate(state + block*Prime2, 13) * Prime1;
    }
    return state;
//  hash += rotate(state.s0,  1) + rotate(state.s1,  7) + rotate(state.s2, 12) + rotate(state.s3, 18);
}
static inline uint2 fnv_hash(uint2 hash, uint8_t data) {
//    const uint64_t fnv_prime = 0x100000001b3ULL;
//    uint64_t hash = 0xcbf29ce484222325ULL;
    hash.s0 ^= data;
//    hash *= fnv_prime;
    hash.s0 = hash.s0*0x1b3u;
    hash.s1 = hash.s0*0x100u + mul_hi(hash.s0, 0x1b3u) + hash.s1*0x1b3u;
    return hash;
}

static inline uint2 mwc_hash( uint2 state, uint data, uint Prime) {
    return as_uint2((ulong)Prime*(state.s0 ^ data) + state.s1);
}
kernel void bloom(global uint2 *state, global uint* data, constant uint* Primes, uint ne00)
{
    uint tid = get_local_id(0);// номер треда
    const uint Prime = Primes[tid];// у каждого треда свой
    uint2 s = state[tid];
    for (int i=0; i<ne00; ++i)
        s = mwc_hash(s, data[i], Prime);
    state[tid] = s;
}
// заполнить константы снаружи
#define Sigma2 1.0f // сигма в квадрате
#define DIM_X 128
#define DIM_Y 128
#define DIM_Z 128
#define Scale_X 1.0f
#define Scale_Y 1.0f
#define Scale_Z 1.0f
#define FIN UINT_MAX // инициализация

#define IDX(x,y,z) (x + y*(DIM_X) + z*(DIM_X*DIM_Y))
constant float3 Scale = {Scale_X,Scale_Y*(DIM_X),Scale_Z*(DIM_X*DIM_Y)};
static inline uint vec3d_hash(float3 p){
	return convert_uint_sat_rtz(dot(p,Scale));
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
// аргументом является функция сравнения и функция хэш
float3 ht_lookup(float3 p, local float3* pp, local uint* cells, local uint*next){
    uint tid = get_local_id(0);
    float3 fs = (0.f);
    uint h = vec3d_hash(p);
    uint i = cells[h];// хеш таблица
    while (i != FIN){
        if(i != tid){// проверка условия взаимодействия
            fs += v_force(pp[i], p);
        }
        i = next[i];
    }
    return fs;
}
void ht_update(local uint* cells, local uint*next, float3 p){
	uint idx = vec3d_hash(p); // преобразовать вектор в hash
    uint tid = get_local_id(0);
	next[tid]= atomic_xchg(&cells[idx], tid);// список
}
// однородное распрелеление [0,1)
float uniform (uint2 *state) {
	uint x = mwc64x(state, MWC64X_A);
	return x * 0x1.0p-24f;// можно использовать ldexp(x,-24)
}
// Box-Muller transform 
float2 gaussian(uint2 *state)
{
	float u1 = 1.0f-uniform(state);	// распределение (0,1]
	float u2 = uniform(state);
    return sqrt(-2*log(u1))*(float2)(cospi(u2), sinpi(u2));
}
kernel void kernel_uniform(global long *states, global float * dst, uint ne)
{
    uint gid = get_global_id(0);
    uint2 state = as_uint2(states[gid]);
    for (int k=0; k<ne; k++)
        dst[gid*ne+k] = uniform(&state);
}
// расчет контрольной суммы для матриц
kernel void kernel_crc32(global uchar *data, global uint *dst, uint ne){

const uint16 CRC32B_Lookup4={// пеерсчитать с учетом Folding 32
    0x00000000, 0x1DB71064, 0x3B6E20C8, 0x26D930AC,
    0x76DC4190, 0x6B6B51F4, 0x4DB26158, 0x5005713C,
    0xEDB88320, 0xF00F9344, 0xD6D6A3E8, 0xCB61B38C,
    0x9B64C2B0, 0x86D3D2D4, 0xA00AE278, 0xBDBDF21C
};
    uint crc = 0;
    int tid = get_sub_group_local_id();
//    uint2 mask = k12 & (1<<tid);

    for (int k=get_sub_group_local_id(); k<ne; k+=get_max_sub_group_size()) {// fold
		crc^= data[k];
//        crc = sub_group_clustered_reduce_xor(crc.lo*mask.lo,16)// операция умножения без переноса
//            ^ sub_group_clustered_reduce_xor(crc.hi*mask.hi,16);
		crc = (crc>>4) ^ CRC32B_Lookup4[crc & 0xF];
		crc = (crc>>4) ^ CRC32B_Lookup4[crc & 0xF];
	}
    dst[get_sub_group_local_id()] = crc;
}
