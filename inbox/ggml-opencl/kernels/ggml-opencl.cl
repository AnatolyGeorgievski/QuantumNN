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
#error "Selecting subgroup size is not supported on your device."
#endif

#if     defined(INTEL_GPU)
#define REQD_SUBGROUP_SIZE_DEFAULT REQD_SUBGROUP_SIZE_32
#elif   defined(ADRENO_GPU)
#define REQD_SUBGROUP_SIZE_DEFAULT REQD_SUBGROUP_SIZE_64
#else
#define REQD_SUBGROUP_SIZE_DEFAULT
#endif

#define OVERLOADABLE __attribute__((overloadable))


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
    for (int k=0; k<4; k++) y[k] = yy[k*sz+ix];
    return y;
}

static inline ushort OVERLOADABLE _sub_group_block_read_us1(global ushort* yy){
    int ix = get_sub_group_local_id();
    return yy[ix];
}
static inline ushort8 OVERLOADABLE _sub_group_block_read_us(global ushort* yy){
    ushort8 y;
    uint ix = get_sub_group_local_id();
    uint sz = get_max_sub_group_size();
    #pragma unroll
    for (uint k=0; k<8; k++) y[k] = yy[k*sz+ix];
    return y;
}
static inline ushort4 OVERLOADABLE _sub_group_block_read_us4(global ushort* yy){
    ushort4 y;
    uint ix = get_sub_group_local_id();
    uint sz = get_max_sub_group_size();
    #pragma unroll
    for (uint k=0; k<4; k++) y[k] = yy[k*sz+ix];
    return y;
}
static inline void OVERLOADABLE _sub_group_block_write(global float* yy, float8 y){
    uint ix = get_sub_group_local_id();
    uint sz = get_max_sub_group_size();
    #pragma unroll
    for (uint k=0; k<8; k++) yy[k*sz+ix]=y[k];
}
static inline void OVERLOADABLE _sub_group_block_write4(global float* yy, float4 y){
    uint ix = get_sub_group_local_id();
    uint sz = get_max_sub_group_size();
    #pragma unroll
    for (uint k=0; k<4; k++) yy[k*sz+ix]=y[k];
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
static const char16 kvalues_iq4nl = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
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

// general-purpose kernel for addition of two tensors
// pros: works for non-contiguous tensors, supports broadcast across dims 1, 2 and 3
// cons: not very efficient
kernel void kernel_add(
        global char * src0,
        global char * src1,
        global char * dst,
        uint   ne00,
        uint   ne01,
        uint   ne02,
        uint   ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        uint  ne10,
        uint   ne11,
        uint   ne12,
        uint   ne13,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        uint   ne0,
        uint   ne1,
        uint   ne2,
        uint   ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    uint i03 = get_group_id(2);
    uint i02 = get_group_id(1);
    uint i01 = get_group_id(0);

    uint i13 = i03 % ne13;
    uint i12 = i02 % ne12;
    uint i11 = i01 % ne11;

    global char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
    global char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
    global char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1;

    for (uint i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const uint i10 = i0 % ne10;
        *((global float *)(dst_ptr + i0*nb0)) = *((global float *)(src0_ptr + i0*nb00)) + *((global float *)(src1_ptr + i10*nb10));
    }
}

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

REQD_SUBGROUP_SIZE_16
kernel void kernel_add_flat_f16(
        global half16 * src0,
        global half16 * src1,
        global half16 * dst,
        uint ne
) {
    uint gid = get_global_id(0);
if(ne%get_max_sub_group_size()==0){
    gid = sub_group_broadcast(gid,0);
    half16 a = as_half16(intel_sub_group_block_read8((global uint*)(src0+gid)))
             + as_half16(intel_sub_group_block_read8((global uint*)(src1+gid)));
    intel_sub_group_block_write8((global uint*)(dst+gid), as_uint8(a));
} else 
    dst[gid] = src0[gid] + src1[gid];
}

//------------------------------------------------------------------------------
// mul
//------------------------------------------------------------------------------
kernel void kernel_mul(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global char * dst,
        ulong offsetd,
        uint ne00,
        uint ne01,
        uint ne02,
        uint ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        uint ne10,
        uint ne11,
        uint ne12,
        uint ne13,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        uint ne0,
        uint ne1,
        uint ne2,
        uint ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst + offsetd;

    uint i03 = get_group_id(2);
    uint i02 = get_group_id(1);
    uint i01 = get_group_id(0);

    uint i13 = i03 % ne13;
    uint i12 = i02 % ne12;
    uint i11 = i01 % ne11;

    global char * src0_ptr = src0 + i03*nb03 + i02*nb02 + i01*nb01;
    global char * src1_ptr = src1 + i13*nb13 + i12*nb12 + i11*nb11;
    global char * dst_ptr  = dst  + i03*nb3  + i02*nb2  + i01*nb1;
if(nb00==sizeof(float) && nb10==sizeof(float) && nb0==sizeof(float) && ne10%(get_max_sub_group_size())==0){// сделать отдельную функцию flat
    for (uint i0 = sub_group_broadcast(get_local_id(0),0); i0 < ne0; i0 += get_local_size(0)) {
        const int i10 = i0 % ne10;
        float a = as_float(intel_sub_group_block_read((global uint*)(src0_ptr + i0*nb00)));
        float b = as_float(intel_sub_group_block_read((global uint*)(src1_ptr + i10*nb10)));
        intel_sub_group_block_write((global uint*)(dst_ptr + i0*nb0), as_uint(a*b));
    }
} else
    for (uint i0 = get_local_id(0); i0 < ne0; i0 += get_local_size(0)) {
        const int i10 = i0 % ne10;
        *((global float *)(dst_ptr + i0*nb0)) = *((global float *)(src0_ptr + i0*nb00)) * *((global float *)(src1_ptr + i10*nb10));
    }
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
// scale
//------------------------------------------------------------------------------
kernel void kernel_scale(
        global float8 * src0,
        global float8 * dst,
        float scale
) {
    uint gid = get_global_id(0);
if (get_local_size(0)%get_max_sub_group_size()==0){
    gid = sub_group_broadcast(gid,0);
    float8 a = as_float8(intel_sub_group_block_read8((global uint*)(src0+gid)));
    intel_sub_group_block_write8((global uint*)(dst+gid), as_uint8(a*scale));
} else
    dst[get_global_id(0)] = src0[get_global_id(0)] * scale;
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
inline float swish   (float x)   { return x * native_recip((1.f + exp(-x))); }
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
KERNEL_UNARY(fabs_f16, half);
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
KERNEL_UNARY(relu, float);
KERNEL_UNARY(elu,  float);
KERNEL_UNARY(hardswish, float);
KERNEL_UNARY(hardsigmoid, float);
KERNEL_UNARY(gelu_quick,float);
KERNEL_UNARY(swish,     float);
KERNEL_UNARY(sigmoid,   float);
KERNEL_UNARY(op_sgn,    float);// sign
KERNEL_UNARY(op_neg,    float);
KERNEL_UNARY(op_step,   float);
KERNEL_UNARY(op_sqr,    float);


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
// синтез бинарных векторных операций: flat - совсем плоское представление, 
// row - одна строка, повтор строк второго аргумента
#define KERNEL_BINARY(op, type)                                     \
kernel void kernel_##op##_flat(                                     \
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

KERNEL_BINARY(add, float);
KERNEL_BINARY(sub, float);
KERNEL_BINARY(mul, float);
KERNEL_BINARY(div, float);
KERNEL_BINARY(fmod,float);
KERNEL_BINARY(fmax,float);
KERNEL_BINARY(fmin,float);
KERNEL_BINARY(atan2, float);
KERNEL_BINARY(pow,   float);
KERNEL_BINARY(copysign, float);

#undef KERNEL_BINARY

inline float  OVERLOADABLE lerp(float  u, float  v, float mu){  return mix(u,v,mu); }
inline float2 OVERLOADABLE lerp(float2 u, float2 v, float mu){  return mix(u,v,mu); }
inline float4 OVERLOADABLE lerp(float4 u, float4 v, float mu){  return mix(u,v,mu); }
inline float8 OVERLOADABLE lerp(float8 u, float8 v, float mu){  return mix(u,v,mu); }

REQD_SUBGROUP_SIZE_16
kernel void kernel_silu_4(
        global float8 * src0,
        global float8 * dst
) {
    uint gid = get_global_id(0);
    if(get_local_size(0)%(8*get_max_sub_group_size())==0){
        gid = sub_group_broadcast(gid,0);
        float8 x = as_float8(intel_sub_group_block_read8((global uint*)(src0+gid)));
        float8 y = silu(x);
        intel_sub_group_block_write8((global uint*)(dst+gid), as_uint8(y));
    } else {
        float8 x = src0[gid];
        dst[gid] = silu(x);
    }
}
#if defined(cl_intel_bfloat16_conversions)
static inline float   OVERLOADABLE convert_bf16_to_f32(ushort   u) { return intel_convert_as_bfloat16_float(u); }
static inline float4  OVERLOADABLE convert_bf16_to_f32(ushort4  u) { return intel_convert_as_bfloat164_float4(u); }
static inline float8  OVERLOADABLE convert_bf16_to_f32(ushort8  u) { return intel_convert_as_bfloat168_float8(u); }
static inline float16 OVERLOADABLE convert_bf16_to_f32(ushort16 u) { return intel_convert_as_bfloat1616_float16(u); }
static inline ushort  OVERLOADABLE convert_f32_to_bf16(float    x) { return intel_convert_bfloat16_as_ushort(x); }
#else
// эмуляция конвертации типа BF16 (E8M7)
static inline float   OVERLOADABLE convert_bf16_to_f32(ushort   u) { return as_float(convert_uint(u)<<16); }
static inline float4  OVERLOADABLE convert_bf16_to_f32(ushort4  u) { return as_float(convert_uint4(u)<<16); }
static inline float8  OVERLOADABLE convert_bf16_to_f32(ushort8  u) { return as_float(convert_uint8(u)<<16); }
static inline float16 OVERLOADABLE convert_bf16_to_f32(ushort16 u) { return as_float(convert_uint16(u)<<16); }
static inline ushort convert_f32_to_bf16(float x) { 
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
typedef ushort8 __bf16x8_t;
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
    float vmax = sub_group_reduce_max(fabs(x))/127.0;
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
    v.lo = (bitfield_extract_unsigned(q,0,4)-8)*d;
    v.hi = (bitfield_extract_unsigned(q,4,4)-8)*d;
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
REQD_SUBGROUP_SIZE_DEFAULT
kernel void kernel_get_rows_q8_0(
        global void * src0,
        global uint * rows,
        global float* dst,
        uint ne00,
        uint nb01,
        uint nb02,
        uint nb10,
        uint nb11,
        uint nb1,
        uint nb2
) {
    uint row = get_group_id(0);//*get_local_size(0);// число элементов в строке/размер группы
    uint i11 = get_group_id(1);

    uint r = rows[i11]; //((global int32_t *) ((global char *) src1 + i11*nb11 + i10*nb10))[0];

//    uint i02 = i11;
    global struct block_q8_0 *blk = (global struct block_q8_0 *)(src0 + mul24(row,nb01));

//    global struct block_q8_0 * blk = (global struct block_q8_0 *) ((global char *) src0 + r*nb01 + i02*nb02);
    global float *y = (global float *) ((global char *) dst + mul24(row,nb1));//i11*nb2 + i10*nb1);
    for (uint i = get_sub_group_local_id(); i < ne00; i += get_max_sub_group_size()) 
        y[i] = dequantize_block_q8_0(blk[i/QK8_0].qs, blk[i/QK8_0].d);
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
    float sumf;
    for(uint i = get_sub_group_local_id(); i<N; i+=get_max_sub_group_size()){
        float v = *(X + mad24(i, incx, offx));
        sumf += v*v;
    }
    return sub_group_reduce_add(sumf);
}
// L1 BLAS ASUM sum of absolute values
float blas_asum(float* X, uint N, uint offx, uint incx){
    float sumf;
    for(uint i = get_sub_group_local_id(); i<N; i+=get_max_sub_group_size()){
        float v = *(float*)(X + mad24(i, incx, offx));
        sumf += fabs(v);
    }
    return sub_group_reduce_add(sumf);
}
// L2 BLAS GER - General matrix rank 1 operation ( A ← α X Y^T+ A )
void ger(float* X, float* Y, float* A, uint M, uint N, uint 
    offx, uint incx, uint offy, uint incy, uint offa, uint lda, float alpha 
){
    uint r = get_group_id(0)* get_num_sub_groups();
    A += mad24(lda, r, offa);
    // for (uint j=0; j<M; ++j)
    float y = *(Y + mad24(r, incy, offy));
    for (uint i=get_local_id(0); i<N; i+=get_local_size(0))
        A[i] += X[mad24(i, incx, offx)]*y;
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
        y[get_sub_group_id()] = alpha*sumf + y[get_sub_group_id()]*beta;
}
// L3 BLAS GEMM 
kernel void gemm(
    global char* A, global char* X, global char* Y, uint M, uint N, 
    uint offa, uint lda, 
    uint offx, uint nb01, 
    uint offy, uint nb11, 
    float alpha, float beta
){
    const uint SD = 16;
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
        float min,
        float max
) {
    uint gid = get_global_id(0);
    dst[gid] = op_clamp(src0[gid], min, max);
}

//------------------------------------------------------------------------------
// norm
//------------------------------------------------------------------------------
kernel void kernel_norm(
        global void * src0,
        global float * dst,
        int ne00,
        ulong nb01,
        float eps,
        local float * sum
) {
    global float * x = (global float *) ((global char *) src0 + get_group_id(0)*nb01);

    // MEAN
    // parallel sum
    sum[get_local_id(0)] = 0.0f;
    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        sum[get_local_id(0)] += x[i00];
    }
    // reduce
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i = get_local_size(0)/2; i > 0; i /= 2) {
        if (get_local_id(0) < i) {
            sum[get_local_id(0)] += sum[get_local_id(0) + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float mean  = sum[0] / ne00;

    // recenter and VARIANCE
    barrier(CLK_LOCAL_MEM_FENCE);
    global float * y = dst + get_group_id(0)*ne00;
    sum[get_local_id(0)] = 0.0f;
    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        y[i00] = x[i00] - mean;
        sum[get_local_id(0)] += y[i00] * y[i00];
    }

    // reduce
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint i = get_local_size(0)/2; i > 0; i /= 2) {
        if (get_local_id(0) < i) {
            sum[get_local_id(0)] += sum[get_local_id(0) + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float variance = sum[0] / ne00;

    float scale = 1.0f/sqrt(variance + eps);
    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        y[i00] = y[i00] * scale;
    }
}

//------------------------------------------------------------------------------
// rms_norm
//------------------------------------------------------------------------------
kernel void kernel_rms_norm(
        global void * src0,
        global float* dst,
        uint ne00,// число элементов
        uint nb01,
        float eps
){
    uint lid = get_local_id(0);
    uint r0  = get_group_id(0);
    global float4 * x = (global float4 *) ((global char *) src0 + r0*nb01);

    float4 sumf = 0;
    for (uint i = sub_group_broadcast(lid,0); i < ne00/4; i += get_local_size(0)) {
        float4 xi =as_float4(intel_sub_group_block_read4((global uint*)(x+i)));
        sumf += xi * xi;
    }
    float all_sum = sumf.s0 + sumf.s1 + sumf.s2 + sumf.s3;
    all_sum = work_group_reduce_add(all_sum);
    float mean = all_sum/ne00;
    float scale = rsqrt(mean + eps);

    global float4 * y = (global float4 *) (dst + r0*ne00);
    for (uint i = sub_group_broadcast(lid,0); i < ne00/4; i += get_local_size(0)) {
        float4 xi =as_float4(intel_sub_group_block_read4((global uint*)(x+i)));
        float4 yi = xi * scale;
        intel_sub_group_block_write4((global uint*)(y+i), as_uint4(yi));
    }
}
// Объединяет операцию rms_norm  c пост-операцией mul, нормуль. Операция mul с броадкастом, с повтором строки 
// предполагается выравнивание данных на рабочую группу 16/32 и размер 4/8 float = 64-256 элементов, 
kernel void kernel_rms_norm_mul(
        global void * src0,
        global float * src1,
        global float * dst,
        uint ne00,// число элементов
        uint nb01,
        uint ne11,
        float eps
){
    uint r0  = get_group_id(0);
    global float4 * x = (global float4 *) ((global char   *) src0 + mul24(r0,nb01));

    uint lid = get_local_id(0);
    float4 sumf = 0;
    for (uint i = sub_group_broadcast(lid,0); i < ne00/4; i += get_local_size(0)) {
        float4 xi =as_float4(_sub_group_block_read4((global float*)(x+i)));
        sumf += xi * xi;
    }
    float all_sum = sumf.s0 + sumf.s1 + sumf.s2 + sumf.s3;
    all_sum = work_group_reduce_add(all_sum);
    float mean = all_sum/ne00;
    float scale = rsqrt(mean + eps);

    global float4 * w = (global float4 *) (src1 + (r0%ne11)*ne00);// может быть бродкаст
    global float4 * y = (global float4 *) (dst  + (r0     )*ne00);
    for (uint i = sub_group_broadcast(lid,0); i < ne00/4; i += get_local_size(0)) {
        float4 xi =as_float4(_sub_group_block_read4((global float*)(x+i)));
        float4 wi =as_float4(_sub_group_block_read4((global float*)(w+i)));
        float4 yi = xi * wi * scale;
        _sub_group_block_write4((global float*)(y+i), as_float4(yi));
    }
}
kernel void kernel_rms_norm_(
        global void * src0,
        global float * dst,
        uint ne00,// число элементов
        uint nb01,
        float eps,
        local float * sum // Note, the size depends on number of subgroups
) {
    global float4 * x = (global float4 *) ((global char *) src0 + get_group_id(0)*nb01);
    global float * x_scalar = (global float *) x;
    float4 sumf = 0;
    float all_sum = 0;
    uint lid =  get_local_id(0);
    // parallel sum
    for (uint i = sub_group_broadcast(lid,0); i < ne00/4; i += get_local_size(0)) {
        float4 xi =as_float4(intel_sub_group_block_read4((global uint*)(x+i)));
        sumf += xi*xi;
    }
    all_sum = sumf.s0 + sumf.s1 + sumf.s2 + sumf.s3;
    all_sum = sub_group_reduce_add(all_sum);
    if (get_sub_group_local_id() == 0) {
        sum[get_sub_group_id()] = all_sum;
    }

    work_group_barrier(CLK_LOCAL_MEM_FENCE);
    // broadcast
    for (uint i = get_local_size(0) / get_max_sub_group_size() / 2; i > 0; i /= 2) {
       if (lid < i) 
           sum[lid] += sum[lid + i];
    }
    float scale;
    if (lid == 0) {
        float s = sum[0];
        for (uint i = (ne00&~3); i < ne00; i++) 
            s += x_scalar[i];
        float mean = s/ne00;
        scale = rsqrt(mean + eps);
    }
    scale  = work_group_broadcast(scale,0);

    global float4 * y = (global float4 *) (dst + get_group_id(0)*ne00);
    global float * y_scalar = (global float *) y;
    for (uint i = sub_group_broadcast(lid,0); i < ne00/4; i += get_local_size(0)) {
        float4 xi =as_float4(intel_sub_group_block_read4((global uint*)(x+i)));
        float4 yi = xi * scale;
        intel_sub_group_block_write4((global uint*)(y+i), as_uint4(yi));
    }
    if (get_local_id(0) == 0) {
        for (int i00 = 4 * (ne00 / 4); i00 < ne00; i00++) {
            y_scalar[i00] = x_scalar[i00] * scale;
        }
    }
}

//------------------------------------------------------------------------------
// diag_mask_inf kernels
//------------------------------------------------------------------------------
kernel void kernel_diag_mask_inf(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        uint ne00,
        uint ne01,
        uint n_past
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    uint i02 = get_global_id(2);
    uint i01 = get_global_id(1);
    uint i00 = get_global_id(0);
    uint offs = mad24(mad24(i02,ne01,i01),ne00,i00);

    dst[offs] = (i00 > n_past + i01)? -INFINITY: src0[offs];
}

kernel void kernel_diag_mask_inf_8(
        global float4 * src0,
        ulong offset0,
        global float4 * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int n_past
) {
    src0 = (global float4*)((global char*)src0 + offset0);
    dst = (global float4*)((global char*)dst + offsetd);

    int i = 2*get_global_id(0);

    dst[i+0] = src0[i+0];
    dst[i+1] = src0[i+1];
    int i4 = 4*i;
    int i02 = i4/(ne00*ne01); i4 -= i02*ne00*ne01;
    int i01 = i4/(ne00);      i4 -= i01*ne00;
    int i00 = i4;
    for (int k = 3; k >= 0; --k) {
        if (i00 + 4 + k <= n_past + i01) {
            break;
        }
        (&dst[i+1])[k] = -INFINITY;
        if (i00 + k > n_past + i01) {
            (&dst[i])[k] = -INFINITY;
        }
    }
}
// генератор псевдо случайных чисел, используется для начальных значений скорости.
enum{ MWC64X_A = 4294883355U };
uint mwc64x(uint2 *state)
{
    uint x=(*state).x, c=(*state).y;
    uint res=x^c;
    x=x*MWC64X_A+c;
    c=mad_hi(x,MWC64X_A, (uint)(x<c));
    *state=(uint2)(x,c);
    return res;
}
static inline uint dtree_hash(uint hash, uchar data, uint Prime) {
    return rotate(hash, 6u) ^ data;
}
static inline ulong mwc_hash( ulong state, uint data, uint Prime) {
    return (ulong)Prime*(uint)(state ^ data) + (state>>32u);
}
kernel void bloom(global ulong *state, global uint* data, constant uint* Primes, uint ne00)
{
    uint tid = get_local_id(0);// номер треда
    const uint Prime = Primes[tid];// у каждого треда свой
    ulong s = state[tid];
    for (int i=0; i<ne00; ++i)
        s = mwc_hash(s, data[i], Prime);
    state[tid] = s;
}
// однородное распрелеление [0,1)
float uniform (uint2 *state) {
	uint x = mwc64x(state);
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
//------------------------------------------------------------------------------
// softmax
//------------------------------------------------------------------------------
#ifndef CACHE_SIZE
#define CACHE_SIZE 4096
#endif

REQD_SUBGROUP_SIZE_DEFAULT
kernel void kernel_soft_max(
        global float * src0,
        global float * src1,
        global float * dst,
        uint ne00,
//        ((global float*)(dst + rb*nb1 + im*nb2))[r0] = sumf;
//        uint nb1, uint nb2,
        uint ne01, uint ne02,
        float scale,
        float slope
) {
  local float _dst[CACHE_SIZE];// размер уточнить при компиляции
    size_t im = get_group_id(2);
    size_t rb = get_group_id(1);// число голов
    size_t r0 = get_group_id(0);

//    size_t offs = im*nb2 + rb*nb1;
    size_t offs = mad24(im,ne02,rb)*ne01*ne00;
    global float * psrc0 = (global float *)(src0 + offs) + r0*ne00;
    global float * pmask = (global float *)(src1       ) + r0*ne00;
    global float * pdst  = (global float *)(dst  + offs) + r0*ne00;
    float lmax = -INFINITY;
    for (uint i = get_sub_group_local_id(); i < ne00; i += get_max_sub_group_size()) {
        float v = psrc0[i]*scale + slope*pmask[i];
        lmax = fmax(lmax, v);
        _dst[i] = v;
    }
    float max = sub_group_reduce_max(lmax);

    float lsum = 0.f;
    for (uint i = get_sub_group_local_id(); i < ne00; i += get_max_sub_group_size()) {
        float exp_v = exp(_dst[i] - max);
        lsum += exp_v;
        _dst[i] = exp_v;
    }
    const float sum = native_recip(sub_group_reduce_add(lsum));
    for (uint i = get_sub_group_local_id(); i < ne00; i += get_max_sub_group_size())
        pdst[i] = _dst[i]*sum;
}
REQD_SUBGROUP_SIZE_DEFAULT
kernel void kernel_soft_max_bf16(
        global float * src0,
        global ushort* src1,
        global float * dst,
        uint ne00,
//        uint nb1, uint nb2,
        uint ne01, uint ne02,
        float scale,
        float slope
) {
  local float _dst[CACHE_SIZE];// размер уточнить при компиляции
    size_t im = get_group_id(2);
    size_t rb = get_group_id(1);// число голов
    size_t r0 = get_group_id(0);

//    size_t offs = im*nb2 + rb*nb1;
    size_t offs = im*ne02*ne01*ne00 + rb*ne01*ne00;

//    size_t offs = i03*ne02*ne01*ne00 + i02*ne01*ne00;
    global float * psrc0 = (global float *)(src0 + offs) + r0*ne00;
    global ushort* pmask = (global ushort*)(src1       ) + r0*ne00;
    global float * pdst  = (global float *)(dst  + offs) + r0*ne00;
    float lmax = -INFINITY;
    for (uint i = get_sub_group_local_id(); i < ne00; i += get_max_sub_group_size()) {
        float v = psrc0[i]*scale + slope*convert_bf16_to_f32(pmask[i]);
        lmax = fmax(lmax, v);
        _dst[i] = v;
    }
    float max = sub_group_reduce_max(lmax);

    float lsum = 0.f;
    for (uint i = get_sub_group_local_id(); i < ne00; i += get_max_sub_group_size()) {
        float exp_v = exp(_dst[i] - max);
        lsum += exp_v;
        _dst[i] = exp_v;
    }
    const float sum = native_recip(sub_group_reduce_add(lsum));
    for (uint i = get_sub_group_local_id(); i < ne00; i += get_max_sub_group_size())
        pdst[i] = _dst[i]*sum;
}


REQD_SUBGROUP_SIZE_DEFAULT
kernel void kernel_soft_max_4(
        global float * src0,
        global float * src1,
        global float * dst,
        uint ne00,
        uint ne01,
        uint ne02,
        float scale,
        float slope
) {
 local float4 _dst[CACHE_SIZE/4];
    size_t i03 = get_group_id(2);
    size_t i02 = get_group_id(1);
    size_t i01 = get_group_id(0);

    global float4 * psrc4 = (global float4 *)(src0 + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);
    global float4 * pmask = (global float4 *)(src1 +                                      i01*ne00);// src1 != src0 ? (global float4 *)(src1 + i01*ne00) : 0;
    global float4 * pdst4 = (global float4 *)(dst  + i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00);

    float4 lmax4 = -INFINITY;
    for (int i = get_sub_group_local_id(); i < ne00/4; i += get_max_sub_group_size()) {
        float4 v = psrc4[i]*scale + slope*pmask[i];// (pmask ? slope*pmask[i00] : 0.0f));
        lmax4 = fmax(lmax4, v);
        _dst[i] = v;
    }
    float lmax = fmax(fmax(lmax4.s0, lmax4.s1), fmax(lmax4.s2, lmax4.s3));

    const float max = sub_group_reduce_max(lmax);

    float4 lsum4 = (0.f);
    for (int i = get_sub_group_local_id(); i < ne00/4; i += get_max_sub_group_size()) {
        const float4 exp_v = exp(_dst[i] - max);
        lsum4 += exp_v;
        _dst[i] = exp_v;
    }
    float lsum = lsum4.s0 + lsum4.s1 + lsum4.s2 + lsum4.s3;

    const float sum = native_recip(sub_group_reduce_add(lsum));
    for (int i = get_sub_group_local_id(); i < ne00/4; i += get_max_sub_group_size()) {
        pdst4[i] = _dst[i]*sum;
    }
}

REQD_SUBGROUP_SIZE_DEFAULT
kernel void kernel_soft_max_f16(
        global float * src0,
        global half  * src1,
        global float * dst,
        uint ne00,
//        uint nb1, uint nb2,
        uint ne01, uint ne02,
        float scale,
        float slope
) {
    size_t im = get_group_id(2);
    size_t rb = get_group_id(1);// число голов
    size_t r0 = get_group_id(0);
    
//    size_t offs = im*nb2 + rb*nb1;
    size_t offs = im*ne02*ne01*ne00 + rb*ne01*ne00;

    global float * psrc0 = (global float *)(src0 + offs) + r0*ne00;
    global half  * pmask = (global half  *)(src1       ) + r0*ne00;
    global float * pdst  = (global float *)(dst  + offs) + r0*ne00;

    float lmax = -INFINITY;
    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        lmax = fmax(lmax, psrc0[i00]*scale + (pmask ? slope*pmask[i00] : 0.0f));
    }
    float max = sub_group_reduce_max(lmax);

    float lsum = 0.0f;
    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        float exp_psrc0 = exp((psrc0[i00]*scale + slope*pmask[i00] - max));
        lsum += exp_psrc0;
        pdst[i00] = exp_psrc0;
    }

    const float sum = native_recip(sub_group_reduce_add(lsum));

    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        pdst[i00] *= sum;
    }
}

// не отлажено
kernel void kernel_soft_max_back(global float * grad, global float * dstf, global float * dst, 
        const int ncols, const float scale) {
    const int tid  = get_sub_group_local_id();
    const int rowx = get_group_id(0);

    grad += (rowx)*ncols;
    dstf += (rowx)*ncols;
    dst  += (rowx)*ncols;

    float dgf_dot = 0.f;

    for (int col = get_sub_group_local_id(); col < ncols; col += get_max_sub_group_size()) {
        dgf_dot += dstf[col]*grad[col];
    }

    dgf_dot = sub_group_reduce_add(dgf_dot);
    for (int col = get_sub_group_local_id(); col < ncols; col += get_max_sub_group_size()) {
        dst[col] = scale * (grad[col] - dgf_dot) * dstf[col];
    }
}
//------------------------------------------------------------------------------
// kernel_rope
//------------------------------------------------------------------------------
float rope_yarn_ramp(float low, float high, int i0) {
    const float y = (i0 / 2 - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// YaRN algorithm based on LlamaYaRNScaledRotaryEmbedding.py from https://github.com/jquesnelle/yarn
// MIT licensed. Copyright (c) 2023 Jeffrey Quesnelle and Bowen Peng.
float2 rope_yarn(
    float theta_extrap, float freq_scale, float2 corr_dims, int i0, float ext_factor, float mscale
) {
    // Get n-d rotational scaling corrected for extrapolation
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        float ramp_mix = rope_yarn_ramp(corr_dims.s0, corr_dims.s1, i0) * ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale *= 1.0f + 0.1f * log(1.0f / freq_scale);
    }
    return (float2)(cos(theta) * mscale, sin(theta) * mscale);
}

// Apparently solving `n_rot = 2pi * x * base^((2 * max_pos_emb) / n_dims)` for x, we get
// `corr_fac(n_rot) = n_dims * log(max_pos_emb / (n_rot * 2pi)) / (2 * log(base))`
float rope_yarn_corr_factor(int n_dims, int n_ctx_orig, float n_rot, float base) {
    return n_dims * log(n_ctx_orig / (n_rot * 2 * M_PI_F)) / (2 * log(base));
}

float2 rope_yarn_corr_dims(
    int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow
) {
    // start and end correction dims
    return (float2)(
        max(0.0f,         floor(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_fast, freq_base))),
        min(n_dims - 1.0f, ceil(rope_yarn_corr_factor(n_dims, n_ctx_orig, beta_slow, freq_base)))
    );
}

kernel void kernel_rope_norm_f32(
        global void  * src0,
        global int   * src1,
        global float * src2,
        global float * dst,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow
) {
/*    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    src2 = (global float*)((global char*)src2 + offset2);
    dst = (global float*)((global char*)dst + offsetd);
*/
    int i3 = get_group_id(2);
    int i2 = get_group_id(1);
    int i1 = get_group_id(0);

    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);

    global int * pos = src1;

    float theta_base = (float) pos[i2];
    float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        if (i0 < n_dims) {
            int ic = i0/2;

            float theta = theta_base * pow(freq_base, inv_ndims*i0);

            float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

            global float * src       = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            global float * dst_data  = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            float x0 = src[0];
            float x1 = src[1];

            dst_data[0] = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
            dst_data[1] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
        } else {
            global float * src      = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            global float * dst_data = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

kernel void kernel_rope_norm_f16(
        global void * src0,
        global int * src1,
        global float * src2,
        global float * dst,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow
) {
    int i3 = get_group_id(2);
    int i2 = get_group_id(1);
    int i1 = get_group_id(0);

    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);

    global int * pos = src1;

    float theta_base = (float) pos[i2];
    float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        if (i0 < n_dims) {
            int ic = i0/2;

            float theta = theta_base * pow(freq_base, inv_ndims*i0);

            float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

            global half * src       = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            global half * dst_data  = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            float x0 = src[0];
            float x1 = src[1];

            dst_data[0] = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
            dst_data[1] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
        } else {
            global half * src      = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            global half * dst_data = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

kernel void kernel_rope_neox_f32(
        global void * src0,
        global int * src1,
        global float * src2,
        global float * dst,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow
) {
    int i3 = get_group_id(2);
    int i2 = get_group_id(1);
    int i1 = get_group_id(0);

    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);

    global int * pos = src1;

    float theta_base = (float) pos[i2];
    float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        if (i0 < n_dims) {
            int ic = i0/2;

            const float theta = theta_base * pow(freq_base, inv_ndims*i0);

            const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

            global float * src      = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
            global float * dst_data = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

            const float x0 = src[0];
            const float x1 = src[n_dims/2];

            dst_data[0]        = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
            dst_data[n_dims/2] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
        } else {
            global float * const src = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            global float * dst_data  = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

kernel void kernel_rope_neox_f16(
        global void * src0,
        global int * src1,
        global float * src2,
        global float * dst,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow
) {
    int i3 = get_group_id(2);
    int i2 = get_group_id(1);
    int i1 = get_group_id(0);

    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);

    global int * pos = src1;

    float theta_base = (float) pos[i2];
    float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        if (i0 < n_dims) {
            int ic = i0/2;

            const float theta = theta_base * pow(freq_base, inv_ndims*i0);

            const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

            global half * src       = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
            global half * dst_data  = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

            const float x0 = src[0];
            const float x1 = src[n_dims/2];

            dst_data[0]        = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
            dst_data[n_dims/2] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
        } else {
            global half * const src = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            global half * dst_data  = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

kernel void kernel_rope_multi_f32(
        global void * src0,
        global int * src1,
        global float * src2,
        global float * dst,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        int4 sections
) {
    int i3 = get_group_id(2);
    int i2 = get_group_id(1);
    int i1 = get_group_id(0);

    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);

    global int * pos = src1;

    const int sect_dims = sections.s0 + sections.s1 + sections.s2 + sections.s3;
    const int sec_w = sections.s1 + sections.s0;

    float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        if (i0 < n_dims) {
            int ic = i0/2;

            const int sector = (i0 / 2) % sect_dims;
            float theta_base = 0.0f;

            if (sector < sections.s0) {
                theta_base = pos[i2];
            }
            else if (sector >= sections.s0 && sector < sec_w) {
                theta_base = pos[i2 + ne2 * 1];
            }
            else if (sector >= sec_w && sector < sec_w + sections.s2) {
                theta_base = pos[i2 + ne2 * 2];
            }
            else if (sector >= sec_w + sections.s2) {
                theta_base = pos[i2 + ne2 * 3];
            }

            const float theta = theta_base * pow(freq_base, inv_ndims*i0);

            const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

            global float * src      = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
            global float * dst_data = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

            const float x0 = src[0];
            const float x1 = src[n_dims/2];

            dst_data[0]        = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
            dst_data[n_dims/2] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
        } else {
            global float * const src = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            global float * dst_data  = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

kernel void kernel_rope_multi_f16(
        global void * src0,
        global int * src1,
        global float * src2,
        global half * dst,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        int4 sections
) {
    int i3 = get_group_id(2);
    int i2 = get_group_id(1);
    int i1 = get_group_id(0);

    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);

    global int * pos = src1;

    const int sect_dims = sections.s0 + sections.s1 + sections.s2 + sections.s3;
    const int sec_w = sections.s1 + sections.s0;

    float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        if (i0 < n_dims) {
            int ic = i0/2;

            const int sector = (i0 / 2) % sect_dims;
            float theta_base = 0.0f;

            if (sector < sections.s0) {
                theta_base = pos[i2];
            }
            else if (sector >= sections.s0 && sector < sec_w) {
                theta_base = pos[i2 + ne2 * 1];
            }
            else if (sector >= sec_w && sector < sec_w + sections.s2) {
                theta_base = pos[i2 + ne2 * 2];
            }
            else if (sector >= sec_w + sections.s2) {
                theta_base = pos[i2 + ne2 * 3];
            }

            const float theta = theta_base * pow(freq_base, inv_ndims*i0);

            const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

            float2 cos_sin_theta = rope_yarn(theta/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

            global half * src      = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
            global half * dst_data = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

            const float x0 = src[0];
            const float x1 = src[n_dims/2];

            dst_data[0]        = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
            dst_data[n_dims/2] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
        } else {
            global half * const src = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
            global half * dst_data  = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0);

            dst_data[0] = src[0];
            dst_data[1] = src[1];
        }
    }
}

kernel void kernel_rope_vision_f32(
        global void * src0,
        global int * src1,
        global float * src2,
        global float * dst,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        int4 sections
) {
    int i3 = get_group_id(2);
    int i2 = get_group_id(1);
    int i1 = get_group_id(0);

    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);

    global int * pos = src1;

    const int sect_dims = sections.s0 + sections.s1;
    const int sec_w = sections.s1 + sections.s0;

    float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        int ic = i0/2;

        const int sector = (i0/2) % sect_dims;
        float theta_base = 0.0f;

        if (sector < sections.s0) {
            const int p = sector;
            theta_base = pos[i2] * pow(freq_base, inv_ndims*2.0f*p);
        } else if (sector >= sections.s0 && sector < sec_w) {
            const int p = sector - sections.s0;
            theta_base = pos[i2 + ne2] * pow(freq_base, inv_ndims*2.0f*p);
        }

        const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

        float2 cos_sin_theta = rope_yarn(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

        global float * src      = (global float *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
        global float * dst_data = (global float *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

        const float x0 = src[0];
        const float x1 = src[n_dims];

        dst_data[0]      = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
        dst_data[n_dims] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
    }
}

kernel void kernel_rope_vision_f16(
        global void * src0,
        global int * src1,
        global float * src2,
        global half * dst,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int n_past,
        int n_dims,
        int n_ctx_orig,
        float freq_base,
        float freq_scale,
        float ext_factor,
        float attn_factor,
        float beta_fast,
        float beta_slow,
        int4 sections
) {
    int i3 = get_group_id(2);
    int i2 = get_group_id(1);
    int i1 = get_group_id(0);

    float2 corr_dims = rope_yarn_corr_dims(n_dims, n_ctx_orig, freq_base, beta_fast, beta_slow);

    global int * pos = src1;

    const int sect_dims = sections.s0 + sections.s1;
    const int sec_w = sections.s1 + sections.s0;

    float inv_ndims = -1.f/n_dims;

    for (int i0 = 2*get_local_id(0); i0 < ne0; i0 += 2*get_local_size(0)) {
        int ic = i0/2;

        const int sector = (i0/2) % sect_dims;
        float theta_base = 0.0f;

        if (sector < sections.s0) {
            const int p = sector;
            theta_base = pos[i2] * pow(freq_base, inv_ndims*2.0f*p);
        } else if (sector >= sections.s0 && sector < sec_w) {
            const int p = sector - sections.s0;
            theta_base = pos[i2 + ne2] * pow(freq_base, inv_ndims*2.0f*p);
        }

        const float freq_factor = src2 != src0 ? src2[ic] : 1.0f;

        float2 cos_sin_theta = rope_yarn(theta_base/freq_factor, freq_scale, corr_dims, i0, ext_factor, attn_factor);

        global half * src      = (global half *)((global char *) src0 + i3*nb03 + i2*nb02 + i1*nb01 + ic*nb00);
        global half * dst_data = (global half *)((global char *)  dst + i3*nb3  + i2*nb2  + i1*nb1  + ic*nb0);

        const float x0 = src[0];
        const float x1 = src[n_dims];

        dst_data[0]      = x0*cos_sin_theta.s0 - x1*cos_sin_theta.s1;
        dst_data[n_dims] = x0*cos_sin_theta.s1 + x1*cos_sin_theta.s0;
    }
}

//------------------------------------------------------------------------------
// cpy
//------------------------------------------------------------------------------

kernel void kernel_cpy_f16_f16(
        global half * src0,
        ulong offset0,
        global half * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global half*)((global char*)src0 + offset0);
    dst = (global half*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    int i3 = n / (ne2*ne1*ne0);
    int i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    global half * dst_data = (global half *) ((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        global const half * src = (global half *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        dst_data[i00] = src[0];
    }
}

kernel void kernel_cpy_f16_f32(
        global half * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {

    src0 = (global half*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    int i3 = n / (ne2*ne1*ne0);
    int i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    global float * dst_data = (global float *) ((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
if (nb00==sizeof(half) && ne00%8==0){// выделить в отдельную функцию
    global half* src_data = (global half*)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01);
    for (uint i = sub_group_broadcast(get_local_id(0),0); i < ne00/8; i += get_local_size(0)) {
        half8 tmp = as_half8(intel_sub_group_block_read_us8((global ushort*)src0+i*8));
        //half16 tmp = vload16(i, src_data);
        //vstore16(convert_float16(tmp), i, dst_data);
        intel_sub_group_block_write8((global uint*)dst_data+i*8, as_uint8(convert_float8(tmp)));
    }
} else
    for (int i = get_local_id(0); i < ne00; i += get_local_size(0)) {
        global half * src = (global half *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i*nb00);
        dst_data[i] = src[0];
    }
}
// Эта чаще запускается
kernel void kernel_cpy_f32_f16(
        global float * src0,
        ulong offset0,
        global half * dst,
        ulong offsetd,
        uint ne00,
        uint ne01,
        uint ne02,
        uint ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        uint ne0,
        uint ne1,
        uint ne2,
        uint ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global half*)((global char*)dst + offsetd);

    uint i03 = get_group_id(2);
    uint i02 = get_group_id(1);
    uint i01 = get_group_id(0);

    uint n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    uint i3 = n / (ne2*ne1*ne0);
    uint i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    uint i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    uint i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    global half * dst_data = (global half *)((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
/*
if (nb00==sizeof(float) && ne00%8==0){// выделить в отдельную функцию
    global float* src_data = (global float*)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01);
    for (uint i = sub_group_broadcast(get_local_id(0),0); i < ne00/8; i += get_local_size(0)) {
        float8 tmp = as_float8(intel_sub_group_block_read8((global uint*)src_data+i*8));
        intel_sub_group_block_write_us8((global ushort*)dst_data+i*8, as_ushort8(convert_half8(tmp)));
    }
} else */
    for (uint i = get_local_id(0); i < ne00; i += get_local_size(0)) {
        global const float * src = (global float *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i*nb00);
        dst_data[i] = convert_half_rte(src[0]);
    }
}
#if defined(cl_intel_bfloat16_conversions)
kernel void kernel_cpy_bf16_f32(
        global half * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {

    src0 = (global half*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    int i3 = n / (ne2*ne1*ne0);
    int i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    global float * dst_data = (global float *) ((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
/*    if (nb00==sizeof(ushort) && ne00%16==0){// выделить в отдельную функцию
        global ushort* src_data = (global ushort*)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01);
        for (uint i = get_local_id(0); i < ne00/16; i += get_local_size(0)) {
            ushort16 tmp = vload16(i, src_data);
            vstore16(intel_convert_as_bfloat1616_float16(tmp), i, dst_data);
        }
    } else */
        src0 = (global half*)((global char*)src0 + i03*nb03 + i02*nb02 + i01*nb01);
        uint stride = nb00/2;
        for (int i = get_local_id(0); i < ne00; i += get_local_size(0)) {
            dst_data[i] = convert_bf16_to_f32(src0[i*stride]);
        }
}
kernel void kernel_cpy_f32_bf16(
        global float * src0,
        ulong offset0,
        global ushort * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global half*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    int i3 = n / (ne2*ne1*ne0);
    int i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    global ushort * dst_data = (global ushort *)((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
/*
if (nb00==sizeof(float) && ne00%16==0){// выделить в отдельную функцию
    global float* src_data = (global float*)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01);
    if (get_local_size(0)%get_max_sub_group_size()==0) {// работает!
        for (uint i = sub_group_broadcast(get_local_id(0),0); i < ne00/8; i += get_local_size(0)) {
            const float8 t0 = as_float8(intel_sub_group_block_read8((global uint*)(src_data+i*8)));
            intel_sub_group_block_write_us8((global ushort*)dst_data+i*8, intel_convert_bfloat168_as_ushort8(t0));
        }
    } else 
        for (uint i = get_local_id(0); i < ne00/16; i += get_local_size(0)) {
            float16 tmp = vload16(i, src_data);
    #if defined(cl_intel_subgroup_matrix_multiply_accumulate)// можно по два значения сохранять
            vstore16(intel_convert_f32_to_bf16(tmp), i, (global short *)dst_data);
    #else
            vstore16(intel_convert_bfloat1616_as_ushort16(tmp), i, dst_data);
    #endif
        }
} else */
    src0 = (global float *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01);
    uint stride = nb00/4;
    for (uint i = get_local_id(0); i < ne00; i += get_local_size(0)) {
        dst_data[i] = convert_f32_to_bf16(src0[i*stride]);//convert_f32_to_bf16_rtn
    }
}
#endif
#if defined(cl_intel_subgroup_matrix_multiply_accumulate_tf32)
kernel void kernel_cpy_f32_tf32(
        global float * src0,
        ulong offset0,
        global float * dst,// tf32_as_float
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    int i3 = n / (ne2*ne1*ne0);
    int i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    global float * dst_data = (global float *)((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
/*
if (nb00==sizeof(float) && ne00 % 16 ==0){// выделить в отдельную функцию
    global float* src_data = (global float*)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01);
    for (uint i = get_local_id(0); i < ne00/16; i += get_local_size(0)) {
        float16 tmp = vload16(i, src_data);
        vstore16(intel_convert_tfloat3216_as_float16(tmp), i, dst_data);
    }
} else */
    for (uint i = get_local_id(0); i < ne00; i += get_local_size(0)) {
        global const float * src = (global float *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i*nb00);
        dst_data[i] = intel_convert_tfloat32_as_float(src[0]);
    }
}
#endif

kernel void kernel_cpy_f32_f32(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    int i3 = n / (ne2*ne1*ne0);
    int i2 = (n - i3*ne2*ne1*ne0) /  (ne1*ne0);
    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) /   ne0;
    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    global float* dst_data = (global float *)((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
    global float* src_data = (global float *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01);
/*
if (nb00==sizeof(float) && ne00%get_max_sub_group_size()==0){// выделить в отдельную функцию
    for (uint i = sub_group_broadcast(get_local_id(0),0); i < ne00; i += get_local_size(0)) {
        uint tmp = intel_sub_group_block_read((global uint*)(src_data+i));
        intel_sub_group_block_write((global uint*)(dst_data+i), tmp);
    }
} else */
    uint stride = nb00/4;
    for (uint i = get_local_id(0); i < ne00; i += get_local_size(0)) {
        dst_data[i] = src_data[i*stride];
    }
}

//------------------------------------------------------------------------------
// get_rows по шаблону для данной квантизации
//------------------------------------------------------------------------------
REQD_SUBGROUP_SIZE_DEFAULT
kernel void kernel_get_rows_f32(
        global void * src0,
        global uint * rows,
        global float* dst,
        uint ne00,
        uint nb01,
        uint nb02,
        uint nb10,
        uint nb11,
        uint nb1,
        uint nb2
) {
    uint i10 = get_group_id(0);
    uint i11 = get_group_id(1);

    uint r = rows[i11];

    global float *x = (global float *) ((global char *) src0 + mad24(r,nb01,mul24(i11,nb02)));
    global float *y = (global float *) ((global char *) dst  + mad24(i10,nb1,mul24(i11,nb2)));

    for (uint ind = get_sub_group_local_id(); ind < ne00; ind += get_max_sub_group_size())
        y[ind] = x[ind];
}

kernel void kernel_get_rows_f16(
        global void * src0,
        global uint * rows,
        global float* dst,
        uint ne00,
        uint nb01,
        uint nb02,
        uint nb10,
        uint nb11,
        uint nb1,
        uint nb2
) {
    uint i10 = get_group_id(0)*get_local_size(0);
    uint i11 = get_group_id(1);

    uint r = rows[i11]; //((global int32_t *) ((global char *) src1 + i11*nb11 + i10*nb10))[0];

    uint i02 = i11;
    global half8 *x = (global half8 *) ((global char *) src0 + r*nb01 + i02*nb02);
    global float8 *y = (global float8 *) ((global char *) dst + i11*nb2 + i10*nb1);

    for (uint ind = sub_group_broadcast(get_local_id(0),0); ind < ne00/8; ind += get_local_size(0)) {
        half8 tmp = as_half8(intel_sub_group_block_read_us8((global ushort*)(x+ind)));
        intel_sub_group_block_write8((global uint*)(y+ind), as_uint8(convert_float8(tmp)));        
       // y[ind] = convert_float16(x[ind]);
    }
    // дописать остаток
}

kernel void kernel_get_rows_bf16(
        global void * src0,
        global uint * rows,
        global float * dst,
        uint ne00,
        uint nb01,
        uint nb02,
        uint nb10,
        uint nb11,
        uint nb1,
        uint nb2
) {
    uint i10 = get_group_id(0)*get_local_size(0);
    uint i11 = get_group_id(1);

    uint r = rows[i11];

    uint i02 = i11;
    global ushort16 *x = (global short16 *) ((global char *) src0 + r*nb01 + i02*nb02);
    global float16  *y = (global float16 *) ((global char *) dst + i11*nb2 + i10*nb1);

    for (uint ind = get_local_id(0); ind < ne00/16; ind += get_local_size(0)) {
        y[ind] = convert_bf16_to_f32(x[ind]);
    }
}

kernel void kernel_get_rows_q4_0(
        global void * src0,
        global uint * rows,
        global float* dst,
        uint ne00,
        uint nb01,
        uint nb02,
        uint nb10,
        uint nb11,
        uint nb1,
        uint nb2
) {
    const int NL = 2;

    uint i10 = get_group_id(0)*get_local_size(0);
    uint i11 = get_group_id(1);

    uint r = rows[i11];

    uint i02 = i11;
    global struct block_q4_0 * blk = (global struct block_q4_0 *) ((global char *) src0 + r*nb01 + i02*nb02);
    global float16 *y = (global float16 *) ((global char *) dst + i11*nb2 + i10*nb1);
    for (uint ind = get_local_id(0); ind < ne00/16; ind += get_local_size(0)) {
        float16  tmp = dequantize_q4_0_f32(blk[ind/NL].qs, blk[ind/NL].d, ind%NL);
        y[ind] = tmp;
    }
}
kernel void kernel_get_rows_q4_0_flat(
        global void * src0,
        global void * src0_d,
        global uint * rows,
        global float* dst,
        uint ne00,
        ulong nb01,
        ulong nb02,
        uint ne10,
        ulong nb10,
        ulong nb11,
        ulong nb1,
        ulong nb2
) {
    const int NL = 2;

    uint i10 = get_group_id(0)*get_local_size(0);
    uint i11 = get_group_id(1);

    uint r = rows[i11]; //((global int32_t *) ((global char *) src1 + i11*nb11 + i10*nb10))[0];

    uint i02 = i11;
    global uchar * qs = (global uchar *) ((global uchar *) src0 + r*nb01 + i02*nb02);
    global half  * d  = (global half  *) ((global uchar *) src0_d + r*nb01 + i02*nb02);// проверить ошибка
    global float16 *y = (global float16 *) ((global char *) dst + i11*nb2 + i10*nb1);
    
    for (uint ind = get_local_id(0); ind < ne00/16; ind += get_local_size(0)) {
        float16  tmp = dequantize_q4_0_f32(qs + ind/NL, d[ind/NL], ind%NL);
        y[ind] = tmp;
    }
}
kernel void kernel_get_rows_q4_1(
        global void * src0,
        global uint * rows,
        global float* dst,
        uint ne00,
        uint nb01,
        uint nb02,
        uint nb10,
        uint nb11,
        uint nb1,
        uint nb2
) {
    const int NL = 2;

    int i10 = get_group_id(0)*get_local_size(0);
    int i11 = get_group_id(1);

    int r = rows[i11];

    int i02 = i11;
    global struct block_q4_1 * blk = (global struct block_q4_1 *) ((global char *) src0 + r*nb01 + i02*nb02);
    global float16 *y = (global float16 *) ((global char *) dst + i11*nb2 + i10*nb1);
    for (uint ind = get_local_id(0); ind < ne00/16; ind += get_local_size(0)) {
        float16  tmp = dequantize_q4_1_f32(blk + ind/NL, ind%NL);
        y[ind] = tmp;
    }
}

kernel void kernel_get_rows_q8_0_(
        global void * src0,
        global uint * rows,
        global float* dst,
        uint ne00,
        uint nb01,
        uint nb02,
        uint nb10,
        uint nb11,
        uint nb1,
        uint nb2
) {
    const uint NL = 2;// число итераций наблок

    uint i10 = get_group_id(0)*get_local_size(0);// число элементов в строке/размер группы
    uint i11 = get_group_id(1);

    uint r = rows[i11]; //((global int32_t *) ((global char *) src1 + i11*nb11 + i10*nb10))[0];

    uint i02 = i11;
    global struct block_q8_0 * blk = (global struct block_q8_0 *) ((global char *) src0 + r*nb01 + i02*nb02);
    global float16 *y = (global float16 *) ((global char *) dst + i11*nb2 + i10*nb1);
    for (uint ind = get_local_id(0); ind < ne00/16; ind += get_local_size(0)) 
    {
        float16  tmp = dequantize_q8_0_f32(blk[ind/NL].qs, blk[ind/NL].d, ind%NL);
        y[ind] = tmp;
    }
}

kernel void kernel_get_rows_q6_K(
        global void * src0,
        global uint * rows,
        global float * dst,
        uint ne00,
        uint nb01,
        uint nb02,
        uint nb10,
        uint nb11,
        uint nb1,
        uint nb2
) {
    const uint NL = 16;

    uint i10 = get_group_id(0)*get_local_size(0);
    uint i11 = get_group_id(1);

    uint r = rows[i11];

    uint i02 = i11;
    global struct block_q6_K * blk = (global struct block_q6_K *) ((global char *) src0 + r*nb01 + i02*nb02);
    global float16 *y = (global float16 *) ((global char *) dst + i11*nb2 + i10*nb1);

    for (uint ind = get_local_id(0); ind < ne00/16; ind += get_local_size(0)) {
        float16  tmp = dequantize_q6_k_f32(blk + ind/NL, ind%NL);
        y[ind] = tmp;
    }
}
// local {nb, 1,1} nb кратно 16
kernel void kernel_get_rows_q4_K(
        global void * src0,
        global uint * rows,
        global float* dst,
        uint ne00,
        uint nb01,
        uint nb02,
        uint nb10,
        uint nb11,
        uint nb1,
        uint nb2
) {
    const uint NL = 16;

    uint i10 = get_group_id(0)*get_local_size(0);
    uint i11 = get_group_id(1);// ne11

    uint r = rows[i11];

    uint i02 = i11;
    global struct block_q4_K * blk = (global struct block_q4_K *) ((global char *) src0 + r*nb01 + i02*nb02);
    global float16 *y = (global float16 *) ((global char *) dst + i11*nb2 + i10*nb1);

    for (uint ind = get_local_id(0); ind < ne00/16; ind += get_local_size(0)) {
        float16  tmp = dequantize_q4_k_f32(blk + ind/NL, ind%NL);
        y[ind] = tmp;
    }
}

//------------------------------------------------------------------------------
// mul_mat_f32_f32
//------------------------------------------------------------------------------
#define N_F32_F32 4

kernel void kernel_mul_mat_f32_f32(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        uint ne00,
        uint ne01,
        uint ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        uint ne10,
        uint ne11,
        uint ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        uint ne0,
        uint ne1,
        uint r2,// кратность r2 = ne12/ne02;
        uint r3// = ne13/ne03;
) {
//    src0 = (global char*)((global char*)src0 + offset0);
//    src1 = (global char*)((global char*)src1 + offset1);
//    dst = (global float*)((global char*)dst + offsetd);

    int r0 = get_group_id(0);
    int rb = get_group_id(1)*N_F32_F32;
    int im = get_group_id(2);

    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset_src0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;

    global float * x = (global float *) (src0 + offset_src0);

    if (ne00 < 128) {
        for (uint row = 0; row < N_F32_F32; ++row) {
            uint r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            ulong offset_src1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

            global float * y = (global float *) (src1 + offset_src1);

            float sumf = 0;
            for (uint i = get_sub_group_local_id(); i < ne00; i += get_max_sub_group_size()) {
                sumf += x[i] * y[i];
            }

            float all_sum = sub_group_reduce_add(sumf);
            if (get_sub_group_local_id() == 0) {
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
        }
    } else {
        for (int row = 0; row < N_F32_F32; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            ulong offset_src1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

            global float  * y  = (global float  *) (src1 + offset_src1);
            global float4 * x4 = (global float4 *) x;
            global float4 * y4 = (global float4 *) y;
            float sumf = 0;
            for (int i = get_sub_group_local_id(); i < ne00/4; i += get_max_sub_group_size()) {
#if 1
                sumf += dot(x4[i], y4[i]);
#else
                sumf += (float) x4[i].s0 * y4[i].s0;
                sumf += (float) x4[i].s1 * y4[i].s1;
                sumf += (float) x4[i].s2 * y4[i].s2;
                sumf += (float) x4[i].s3 * y4[i].s3;
#endif
            }

            float all_sum = sub_group_reduce_add(sumf);
            if (get_sub_group_local_id() == 0) {
                for (int i = 4*(ne00/4); i < ne00; ++i) {
                    all_sum += (float) x[i] * y[i];
                }
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
        }
    }
}

//------------------------------------------------------------------------------
// mul_mat_f16_f16
//------------------------------------------------------------------------------
#define N_F16_F16 4
REQD_SUBGROUP_SIZE_DEFAULT
kernel void kernel_mul_mat_f16_f16(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3)
{
//    src0 = (global char*)((global char*)src0 + offset0);
//    src1 = (global char*)((global char*)src1 + offset1);
//    dst = (global float*)((global char*)dst + offsetd);

    int r0 = get_group_id(0);
    int rb = get_group_id(1)*N_F16_F16;
    int im = get_group_id(2);

    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset_src0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;

    global half * x = (global half *) (src0 + offset_src0);

    if (ne00 < 128) {
        for (int row = 0; row < N_F16_F16; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            ulong offset_src1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

            global half * y = (global half *) (src1 + offset_src1);

            float sumf = 0;
            for (uint i = get_sub_group_local_id(); i < ne00/4; i += get_max_sub_group_size()) {
                sumf += (half) x[i] * (half) y[i];
            }

            float all_sum = sub_group_reduce_add(sumf);
            if (get_sub_group_local_id() == 0) {
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
        }
    } else {
        global half4 * x4 = (global half4 *)x;
        for (int row = 0; row < N_F16_F16; ++row) {
            int r1 = rb + row;
            if (r1 >= ne11) {
                break;
            }

            ulong offset_src1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

            global half  * y  = (global half  *) (src1 + offset_src1);
            global half4 * y4 = (global half4 *) y;

            float sumf = 0;
            for (int i = get_sub_group_local_id(); i < ne00/4; i += get_max_sub_group_size()) {
#if 1
                sumf += dot(x4[i], y4[i]);
#else
                sumf += (half) x4[i].s0 * y4[i].s0;
                sumf += (half) x4[i].s1 * y4[i].s1;
                sumf += (half) x4[i].s2 * y4[i].s2;
                sumf += (half) x4[i].s3 * y4[i].s3;
#endif
            }

            float all_sum = sub_group_reduce_add(sumf);
            if (get_sub_group_local_id() == 0) {
                for (int i = 4*(ne00/4); i < ne00; ++i) {
                    all_sum += (half) x[i] * y[i];
                }
                dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
            }
        }
    }
}

//------------------------------------------------------------------------------
// mul_mat_f16_f32_1row
//------------------------------------------------------------------------------
REQD_SUBGROUP_SIZE_DEFAULT
kernel void kernel_mul_mat_f16_f32_1row(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
//    src0 = (global char*)((global char*)src0 + offset0);
//    src1 = (global char*)((global char*)src1 + offset1);
//    dst = (global float*)((global char*)dst + offsetd);

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset_src0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    ulong offset_src1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

    global half  * x = (global half  *) (src0 + offset_src0);
    global float * y = (global float *) (src1 + offset_src1);

    float sumf = 0;
    for (int i = get_sub_group_local_id(); i < ne00; i += get_max_sub_group_size()) {
        sumf += x[i] * y[i];
    }
    float all_sum = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id() == 0) {
        dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
    }
}

//------------------------------------------------------------------------------
// mul_mat_f16_f32
//------------------------------------------------------------------------------
#define N_F16_F32 4

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined(ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
//    src0 = (global char*)((global char*)src0 + offset0);
//    src1 = (global char*)((global char*)src1 + offset1);
//    dst = (global float*)((global char*)dst + offsetd);

    int r0 = get_group_id(0);
    int rb = get_group_id(1)*N_F16_F32;
    int im = get_group_id(2);
#if 0
    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset_src0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    ulong offset_src1 = rb*nb11 + /*row*nb11 +*/ (i12   )*nb12 + (i13   )*nb13;

    global half * x = (global half *) (src0 + offset_src0);
    global float* y = (global float*) (src1 + offset_src1);
#endif
    global ushort * x = (global ushort*) (src0 + r0*nb01 + (im/r2)*nb02);
    global float  * y = (global float *) (src1 + rb*nb11 + (im   )*nb12);


    float16 sumf = (0);

    //if (ne00 % 8==0){// можно сделать отдельную функцию
    for (uint i = 0; i < ne00/8; i += get_max_sub_group_size()){
        float8 v = convert_float8(as_half8(intel_sub_group_block_read_us8((global ushort*)x+i*8)));
        for (uint row = 0; row < N_F16_F32; ++row) {
            float8 u = as_float8(intel_sub_group_block_read8((global uint*)y+i*8+row*nb11/4));// x16 8x4=256
            sumf[row] += dot(v.lo, u.lo);
            sumf[row] += dot(v.hi, u.hi);
        }
    }
    for (uint i = ne00&~7; i < ne00; i += get_max_sub_group_size()){
        float v = convert_float(as_half(intel_sub_group_block_read_us((global ushort*)x+i)));
        for (uint row = 0; row < N_F16_F32; ++row) {
            float u = as_float(intel_sub_group_block_read((global uint*)y+i+row*nb11/4));// x16 8x4=256
            sumf[row] += v * u;
        }
    }
    for (uint row = 0; row < N_F16_F32; ++row){
        if (rb + row >= ne11) break;
        sumf[row] = sub_group_reduce_add(sumf[row]);
        if (get_sub_group_local_id() == 0)
            dst[im*ne1*ne0 + rb*ne0 + row*ne0 + r0] = sumf[row];
    }
}
//------------------------------------------------------------------------------
// mul_mat_bf16_f32
//------------------------------------------------------------------------------
#define N_BF16_F32 1

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined(ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_bf16_f32(
        global char * src0,
        global char * src1,
        global float * dst,
        uint ne00,
//        uint ne01,
//        uint ne02,
//        uint nb00,
        uint nb01,
        uint nb02,
//        uint ne10,  //==ne00
//        uint ne11,
//        uint ne12,  //
//        uint nb10,
        uint nb11,
        uint nb12,
        uint ne0,
        uint ne1,
        uint r2    //==ne02/ne12
//        uint r3     //==ne03/ne13==1
) {
//    src0 = (global char*)((global char*)src0 + offset0);
//    src1 = (global char*)((global char*)src1 + offset1);
//    dst = (global float*)((global char*)dst + offsetd);

    uint r0 = get_group_id(0);// число строк - номер строки
    uint rb = get_group_id(1);
    uint im = get_group_id(2);// batch, permute
/*
    uint i12 = im%ne12;
    uint i13 = im/ne12;

    uint offset_src0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;
    uint offset_src1 = rb*nb11 + (i12   )*nb12 + (i13   )*nb13;
*/
    global ushort * x = (global ushort*) (src0 + mad24(im/r2,nb02, mul24(r0,nb01)));
    global float  * y = (global float *) (src1 + mad24(im   ,nb12, mul24(rb,nb11)));

    float sumf = 0.f;
    for (uint i = 0; i < ne00/4; i += get_max_sub_group_size()){
        float4 v = convert_bf16_to_f32(as_ushort4(_sub_group_block_read_us4((global ushort*)x+i*4)));
        float4 u = as_float4(_sub_group_block_read4((global float*)y+i*4));
        sumf += dot(v, u);
    }
    for (uint i = ne00&~3; i < ne00; i += get_max_sub_group_size()){
        float v = convert_bf16_to_f32(as_short(_sub_group_block_read_us1((global ushort*)x+i)));            
        float u = _sub_group_block_read1((global float*)y+i);// x16 8x4=256
        sumf += v * u;
    }

    sumf = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id() == 0)
        dst[im*ne1*ne0 + rb*ne0 + r0] = sumf;
}

#if defined( cl_intel_bfloat16_conversions )
// Assumes row size (ne00) is a multiple of 8
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined(ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_bf16_f32_l4(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src0 = (global char*)((global char*)src0 + offset0);
    src1 = (global char*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int nrows = ne11;
    int r0 = get_group_id(0);
    int im = get_group_id(2);

    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset_src0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;

    global half8 * x4 = (global half8 *) (src0 + offset_src0);

    for (int r1 = 0; r1 < nrows; ++r1) {
        ulong offset_src1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

        global float8 * y4 = (global float8 *) (src1 + offset_src1);

        float sumf = 0;
        for (int i = 0/*get_sub_group_local_id()*/; i < ne00/8; i += get_max_sub_group_size()) {

            float8 v = convert_bf16_to_f32(_sub_group_block_read_us((global ushort*)(x4 + i)));
            float8 u = as_float8(_sub_group_block_read((global float*)(y4 + i)));
			sumf += dot(v.lo,u.lo);
			sumf += dot(v.hi,u.hi);
        }

        float all_sum = sub_group_reduce_add(sumf);
        if (get_sub_group_local_id() == 0) {
            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
        }
    }
}
#endif

//------------------------------------------------------------------------------
// mul_mat_f16_f32_l4
//------------------------------------------------------------------------------
// Assumes row size (ne00) is a multiple of 8
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined(ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        int ne11,
        int ne12,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
//    src0 = (global char*)((global char*)src0 + offset0);
//    src1 = (global char*)((global char*)src1 + offset1);
//    dst = (global float*)((global char*)dst + offsetd);

    int nrows = ne11;
    int r0 = get_group_id(0);
    int im = get_group_id(2);

    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset_src0 = r0*nb01 + (i12/r2)*nb02 + (i13/r3)*nb03;

    global half8 * x4 = (global half8 *) (src0 + offset_src0);

    for (int r1 = 0; r1 < nrows; ++r1) {
        ulong offset_src1 = r1*nb11 + (i12   )*nb12 + (i13   )*nb13;

        global float8 * y4 = (global float8 *) (src1 + offset_src1);

        float sumf = 0;
        for (int i = 0/*get_sub_group_local_id()*/; i < ne00/8; i += get_max_sub_group_size()) {

            float8 v = convert_float8(as_half8(intel_sub_group_block_read_us8((global ushort*)(x4 + i))));
            float8 u = as_float8(intel_sub_group_block_read8((global uint*)(y4 + i)));
			sumf += dot((v.lo),u.lo);
			sumf += dot((v.hi),u.hi);
        }

        float all_sum = sub_group_reduce_add(sumf);
        if (get_sub_group_local_id() == 0) {
            dst[im*ne1*ne0 + r1*ne0 + r0] = all_sum;
        }
    }
}

//------------------------------------------------------------------------------
// mul_vec_q_n_f32
//------------------------------------------------------------------------------
// function for calculate inner product between half a q4_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_4_0_dot_y(
        global uchar* xb_qs, half d,
        float16 yl, float16 yh
) {
    uchar16 qs = vload16(0, xb_qs);
    const uchar mask = 0xF;
    float4 acc = 0.f;
    char16 ql = as_char16((qs & mask) - (char)(8));
    char16 qh = as_char16((qs >>4   ) - (char)(8));
	#pragma unroll
    for (uint i = 0; i < 16; i++) {
        acc[i/4]  += yl[i] * ql[i];
        acc[i/4]  += yh[i] * qh[i];
    }
    return d* (acc.s0+acc.s1+acc.s2+acc.s3);
}
// считает блок
inline float block_q_4_0_dot_y_warp(
        global uchar* qb_qs,
        global half * qb_d,
        global float* yb
) {
    half d = *qb_d;
    uchar   q = intel_sub_group_block_read_uc((global uchar*)qb_qs);
    float2 y2 = as_float2(intel_sub_group_block_read2((global uint*)yb));// можно вынести из функции и из цикла
    float sum = y2.s0*(q&0x0F)+y2.s1*(q>>4);
    return d*sub_group_reduce_add(sum);
}
inline float block_q_4_0_dot_y_flat(
        global uchar * qb_qs,
        half    d,
        float16 yl,
        float16 yh
) {
    uchar16 qs = vload16(0,((global uchar*)qb_qs));
    const uchar mask = 0xF;
    float4 acc = 0.f;
    char16 ql = as_char16((qs & mask) - (uchar)(8));
    char16 qh = as_char16((qs >>4   ) - (uchar)(8));
	#pragma unroll
    for (uint i = 0; i < 16; i++) {
        acc[i/4]  += yl[i] * ql[i];
        acc[i/4]  += yh[i] * qh[i];
    }
    return d * (acc.s0+acc.s1+acc.s2+acc.s3);
}

inline float block_q_8_0_dot_y(
        global struct block_q8_0 * qb_curr,
        float* y
) {
    float4 acc = (0.f);
    half d = qb_curr->d;
    #pragma unroll    
    for (uint i = 0; i < 32/16; i++) 
    {
        char16 ql = vload16(i,(global char*)qb_curr->qs);
        float16 yl = vload16(i, y);
        #pragma unroll
        for (uint k = 0; k < 16; k++)
            acc[k/4] += yl[k]*ql[k];
    }
    return d*(acc.s0+acc.s1+acc.s2+acc.s3);
}
inline float block_q_4_1_dot_y(
        global struct block_q4_1 * qb_curr,
        global float* yb
) {
    float d = qb_curr->d;
    float m = qb_curr->m;
    uchar16 qs = vload16(0,((global uchar*)qb_curr->qs));
    float16 yl = vload16(0, yb);
    float16 yh = vload16(1, yb);
    const uchar mask = 0xF;
    float4 acc = 0.f;
    uchar16 ql = (qs & mask);
    uchar16 qh = (qs >>4   );
	#pragma unroll
    for (uint i = 0; i < 16; i++) {
        acc[i/4]  += yl[i] * (d*ql[i] + m);
        acc[i/4]  += yh[i] * (d*qh[i] + m);
    }
    return (acc.s0+acc.s1+acc.s2+acc.s3);
}
#if 0
inline float block_q_4_k_dot_y(
        global struct block_q4_k * qb_curr,
        global float* yb
) {
    float d = qb_curr->d;
    float dm = qb_curr->m;

    uchar16 qs = vload16(0,((global uchar*)qb_curr->qs));
    const uchar mask = 0xF;
    float4 acc = 0.f;
    uchar16 ql = (qs & mask);
    uchar16 qh = (qs >>4   );

    float16 yl = vload16(0, yb);
    float16 yh = vload16(4, yb);// 64+
    const float d1 = d*sc;
    const float m1 = dm*m;
    const float d2 = d*sc;
    const float m2 = dm*m;
	#pragma unroll
    for (uint i = 0; i < 16; i++) {
        acc[i/4]  += yl[i] * (d1*ql[i] - m1);
        acc[i/4]  += yh[i] * (d2*qh[i] - m2);
    }
    return (acc.s0+acc.s1+acc.s2+acc.s3);
}
#endif

#ifdef INTEL_GPU
#define N_DST 1 // each SIMD group works on 4 rows
#define N_SIMDGROUP 1 // number of SIMD groups in a thread group
#elif defined (ADRENO_GPU)
#define N_DST 4
#define N_SIMDGROUP 1
#endif
inline void mul_vec_q4_0_f32_(
        global void * src0,
        global float * src1,
        global float * dst,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3
) {

    const ulong nb = ne00/QK4_0;
// ne12*ne13==1
    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = 0;//get_group_id(2); -- число матриц в задании

    // (r0 * N_SIMDGROUP + get_sub_group_id()) is essenatially the linear global
    // id of a SIMD group in the grid.
    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset0 = (first_row + ((i12/r2) + (i13/r3)*ne02)*ne01) * nb;

    global struct block_q4_0 * x = (global struct block_q4_0 *) src0 + offset0;
    global float             * y = (global float             *) src1 + r1*ne10 + im*ne00*ne1;

    float sumf[N_DST]={0.f};

    // each thread in a SIMD group deals with half a block.
    const uint stride = get_max_sub_group_size();
    for (uint ib = get_sub_group_local_id(); ib < nb; ib += stride) {
        float16 yl = vload16(0, y+QK4_0*ib);
        float16 yh = vload16(1, y+QK4_0*ib);
        #pragma unroll
        for (uint row = 0; row < N_DST; row++)
            sumf[row] += block_q_4_0_dot_y(x[ib+row*nb].qs, x[ib+row*nb].d, yl,yh);// каждый тред делает свой блок, надо чтобы делал со общим y
    }

    for (uint row = 0; row < N_DST; ++row) {
        sumf[row] = sub_group_reduce_add(sumf[row]);
        if (get_sub_group_local_id() == 0 && first_row + row < ne01) 
            dst[(r1 + im*ne1)*ne0 + first_row + row] = sumf[row];
    }
}
inline void mul_vec_q4_0_f32(
        global void * src0,
        global void * src0_d,
        global float * src1,
        global float * dst,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    const ulong nb = ne00/QK4_0;
// ne12*ne13==1
    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = 0;//get_group_id(2); -- число матриц в задании

    // (r0 * N_SIMDGROUP + get_sub_group_id()) is essenatially the linear global
    // id of a SIMD group in the grid.
    const int RC = 16;
    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * RC;

    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset0 = (first_row + ((i12/r2) + (i13/r3)*ne02)*ne01) * nb;

    global uchar * x = (global uchar *) src0   + offset0*(QK4_0/2);
    global half  * d = (global half  *) src0_d + offset0;
    global float * y = (global float *) src1 + r1*ne10 + im*ne00*ne1;

    float sumf =0.f;
    for (uint ib = 0; ib < nb; ib ++) {
        for (uint row = 0; row < RC; row++){
            float sum = block_q_4_0_dot_y_warp(x+(ib+row*nb)*(QK4_0/2), d+(ib+row*nb), y+QK4_0*ib);
            if (get_sub_group_local_id()==row)
                sumf += sum;
        }
    }
    intel_sub_group_block_write((global uint*)&dst[(r1 + im*ne1)*ne0 + first_row], as_uint(sumf));
}
inline void mul_vec_q4_1_f32(
        global void * src0,
        global float * src1,
        global float * dst,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3
) {

    const ulong nb = ne00/QK4_1;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = 0;//get_group_id(2);

    // (r0 * N_SIMDGROUP + get_sub_group_id()) is essenatially the linear global
    // id of a SIMD group in the grid.
    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset0 = (first_row + ((i12/r2) + (i13/r3)*ne02)*ne01) * nb;

    global struct block_q4_1 * x = (global struct block_q4_1 *) src0 + offset0;
    global float             * y = (global float             *) src1 + r1*ne10 + im*ne00*ne1;

    float sumf[N_DST]={0.f};

    // each thread in a SIMD group deals with half a block.
    const uint stride = get_max_sub_group_size();
    for (uint ib = get_sub_group_local_id(); ib < nb; ib += stride) {
        for (uint row = 0; row < N_DST; row++)
            sumf[row] += block_q_4_1_dot_y(x+ib+row*nb, y+QK4_1*ib);
    }
    for (uint row = 0; row < N_DST; ++row) {
        sumf[row] = sub_group_reduce_add(sumf[row]);
        if (get_sub_group_local_id() == 0 && first_row + row < ne01) 
            dst[(r1 + im*ne1)*ne0 + first_row + row] = sumf[row];
    }
}
#undef N_DST
#undef N_SIMDGROUP
#define N_SIMDGROUP 1
#define N_DST 1
inline void mul_vec_q8_0_f32(
        global void * src0,
        global float * src1,
        global float * dst,
        uint ne00,
        uint ne01,
        uint ne02,
        uint ne10,
        uint ne12,
        int ne0,
        int ne1,
        uint r2,
        uint r3
) {

    const uint nb = ne00/QK8_0;

    uint r0 = get_group_id(0);
    uint r1 = get_group_id(1);// делится на число элементов подгруппы
    uint im = 0;//get_group_id(2);

    // (r0 * N_SIMDGROUP + get_sub_group_id()) is essenatially the linear global
    // id of a SIMD group in the grid.
    uint first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    int i12 = im%ne12;
    int i13 = im/ne12;

    ulong offset0 = (first_row + ((i12/r2) + (i13/r3)*ne02)*ne01) * nb;

    global struct block_q8_0 * x = (global struct block_q8_0 *) src0 + offset0;
    global float             * y = (global float             *) src1 + r1*ne10 + im*ne00*ne1;

    const uint stride = get_max_sub_group_size();
    const uint ix = get_sub_group_local_id();
    float sumf[N_DST]={0.f};
    for (uint ib = ix; ib < nb; ib += stride) {
        for (uint row = 0; row < N_DST; row++){// все субгруппы одновременно читают y[ib]
            sumf[row] += block_q_8_0_dot_y(x+ib+row*nb, y+QK8_0*ib);
        }
    }
    for (uint row = 0; row < N_DST; ++row) {
        sumf[row] = sub_group_reduce_add(sumf[row]);
        if (get_sub_group_local_id() == 0 && first_row + row < ne01) 
            dst[(r1 + im*ne1)*ne0 + first_row + row] = sumf[row];
    }
}

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q8_0_f32(
        global void * src0,
        global float * src1,
        global float * dst,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1
//        int r2,
//        int r3
) {
/*    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global float*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);
*/
    mul_vec_q8_0_f32(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, 1, 1);
}
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q4_0_f32(
        global void * src0,
        global float * src1,
        global float * dst,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1
) {
/*
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global float*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);
*/
    mul_vec_q4_0_f32_(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, 1, 1);
}
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q4_1_f32(
        global void * src0,
        global float * src1,
        global float * dst,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1
) {
/*
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global float*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);
*/
    mul_vec_q4_1_f32(src0, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, 1, 1);
}

//------------------------------------------------------------------------------
// kernel_convert_block_q4_0
// Convert the block_q4_0 format to 2 separate arrays (AOS -> SOA).
// This kernel does not deshuffle the bits.
//------------------------------------------------------------------------------
kernel void kernel_convert_block_q4_0(
    global struct block_q4_0 * src0,
    global uchar * dst_q,
    global half  * dst_d
) {
    global struct block_q4_0 * b = (global struct block_q4_0 *) src0 + get_global_id(0);
    global uchar * q = (global uchar *) dst_q + QK4_0/2*get_global_id(0);
    global half  * d = (global half *) dst_d + get_global_id(0);

    *d = b->d;

    for (int i = 0; i < QK4_0/2; ++i) {
        q[i] = b->qs[i];
    }
}

kernel void kernel_restore_block_q4_0(
    global uchar * src_q,
    global half  * src_d,
    global struct block_q4_0 * dst
) {
    global struct block_q4_0 * b = (global struct block_q4_0 *) dst + get_global_id(0);
    global uchar * q = (global uchar *) src_q + QK4_0/2*get_global_id(0);
    global half  * d = (global half *) src_d + get_global_id(0);

    b->d = *d;
    for (int i = 0; i < QK4_0/2; ++i) {
        b->qs[i] = q[i];
    }
}

//------------------------------------------------------------------------------
// mul_vec_q_n_f32_flat
//
// This variation uses flat arrays (struct of arrays, SOA) representation for
// quant tensors.
//------------------------------------------------------------------------------
#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

#ifdef INTEL_GPU
#define N_DST 1 // each SIMD group works on 4 rows
#define N_SIMDGROUP 1 // number of SIMD groups in a thread group
#define N_SIMDWIDTH 16 // assuming SIMD group size is 32
#elif defined (ADRENO_GPU)
#define N_DST 4
#define N_SIMDGROUP 1
#define N_SIMDWIDTH 64
#endif

inline void mul_vec_q_n_f32_flat(
        global uchar * src0_q,
        global half  * src0_d,
        global float * src1,
        global float * dst,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    const ulong nb = ne00/QK4_0;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = 0;//get_group_id(2);

    // (r0 * N_SIMDGROUP + get_sub_group_id()) is the linear global id of
    // a SIMD group in the grid. Each SIMD group produces N_DST values in the
    // result, hence uses nb blocks, i.e., the offset becomes first_row*nb.
    // Currently with llama2 7B, im is always 0.
    // TODO: how to handle im/gqa*(nb*ne0)?
    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    int i12 = im%ne12;
    int i13 = im/ne12;

    // The number of scales is the same as the number of blocks.
    ulong offset0_d = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    // Each block contains QK4_0/2 uchars, hence offset for qs is as follows.
    ulong offset0_q = (first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02)) * QK4_0/2;

    global uchar * x = (global uchar *) src0_q + offset0_q;
    global half  * d = (global half  *) src0_d + offset0_d;
    global float * y = (global float *) src1   + r1*ne10 + im*ne00*ne1;

    float sumf[N_DST] ={0.f};

    int ix = get_sub_group_local_id();
    global float * yb = y + ix*QK4_0;

    const uint stride = get_max_sub_group_size();
    for (uint ib = ix; ib < nb; ib += stride) {// 16*(QK4_0/2) - линия кеша 256
        float16 yl = vload16(0, yb);
        float16 yh = vload16(1, yb);
        #pragma unroll
        for (uint row = 0; row < N_DST; row++)
            sumf[row] += block_q_4_0_dot_y_flat(x + (ib + row*nb)*(QK4_0/2), d[ib + row*nb], yl, yh);
        yb += QK4_0 * (N_SIMDWIDTH);
    }
    for (uint row = 0; row < N_DST; ++row) {
        sumf[row] = sub_group_reduce_add(sumf[row]);
        if (get_sub_group_local_id() == 0/* && first_row + row < ne01*/) 
            dst[(r1 + im*ne1)*ne0 + first_row + row] = sumf[row];
    }
}

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q4_0_f32_flat(
        global uchar * src0_q,
        global half  * src0_d,
        global float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src1 = (global float*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    mul_vec_q_n_f32_flat(src0_q, src0_d, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3);
}

//
// This variant outputs 8 values.
//
#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

#ifdef INTEL_GPU
#define N_DST 8 // each SIMD group works on 8 rows
#define N_SIMDGROUP 1 // number of SIMD groups in a thread group
#define N_SIMDWIDTH 16 // assuming SIMD group size is 32
#elif defined (ADRENO_GPU)
#define N_DST 8
#define N_SIMDGROUP 1
#define N_SIMDWIDTH 64
#endif

inline void mul_vec_q_n_f32_8x_flat(
        global uchar * src0_q,
        global half  * src0_d,
        global float * src1,
        global float * dst,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    const ulong nb = ne00/QK4_0;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    // (r0 * N_SIMDGROUP + get_sub_group_id()) is the linear global id of
    // a SIMD group in the grid. Each SIMD group produces N_DST values in the
    // result, hence uses nb blocks, i.e., the offset becomes first_row*nb.
    // Currently with llama2 7B, im is always 0.
    // TODO: how to handle im/gqa*(nb*ne0)?
    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    int i12 = im%ne12;
    int i13 = im/ne12;

    // The number of scales is the same as the number of blocks.
    ulong offset0_d = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    // Each block contains QK4_0/2 uchars, hence offset for qs is as follows.
    ulong offset0_q = (first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02)) * (QK4_0/2);

    global uchar * x = (global uchar *) src0_q + offset0_q;
    global half  * d = (global half  *) src0_d + offset0_d;
    global float * y = (global float *) src1   + r1*ne10 + im*ne00*ne1;

    float16 yl;
    float8 sumf = 0.f;

    int ix = get_sub_group_local_id();

    global float * yb = y + ix*QK4_0;

    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH) {
        float16 yl = vload16(0, yb);
        float16 yh = vload16(1, yb);
        for (uint row = 0; row < N_DST; row++)
            sumf[row] += block_q_4_0_dot_y_flat(x + (ib + row*nb)*(QK4_0/2), d[ib + row*nb], yl, yh);
        yb += QK4_0 * (N_SIMDWIDTH);
    }

    float8 tot = (float8)(
        sub_group_reduce_add(sumf.s0), sub_group_reduce_add(sumf.s1),
        sub_group_reduce_add(sumf.s2), sub_group_reduce_add(sumf.s3),
        sub_group_reduce_add(sumf.s4), sub_group_reduce_add(sumf.s5),
        sub_group_reduce_add(sumf.s6), sub_group_reduce_add(sumf.s7)
    );

    if (get_sub_group_local_id() == 0) {
        if (first_row + 0 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 0] = tot.s0;
        }
        if (first_row + 1 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 1] = tot.s1;
        }
        if (first_row + 2 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 2] = tot.s2;
        }
        if (first_row + 3 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 3] = tot.s3;
        }

        if (first_row + 4 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 4] = tot.s4;
        }
        if (first_row + 5 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 5] = tot.s5;
        }
        if (first_row + 6 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 6] = tot.s6;
        }
        if (first_row + 7 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 7] = tot.s7;
        }
    }
}

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q4_0_f32_8x_flat(
        global uchar * src0_q,
        global half  * src0_d,
        global float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    src1 = (global float*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    mul_vec_q_n_f32_8x_flat(src0_q, src0_d, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3);
}
