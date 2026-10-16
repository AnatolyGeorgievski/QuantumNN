//------------------------------------------------------------------------------
// This file is contains additional mulmat kernels
// (and potentially other kernels).
//------------------------------------------------------------------------------
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#elif defined(cl_amd_fp16)
#pragma OPENCL EXTENSION cl_amd_fp16 : enable
#else
#error "Half precision floating point not supportedby OpenCL implementation on your device."
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
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
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
// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 6.5625 bits per weight
typedef struct {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    half d;             // super-block scale
} block_q6_K;

#define OVERLOADABLE __attribute__((overloadable))

/* Это определение эквивалентно intel_sub_group_block_read */
static inline float16 OVERLOADABLE _sub_group_block_read(float* yy){
    float16 y;
    int ix = get_sub_group_local_id();
    int sz = get_max_sub_group_size();
    for (int k=0; k<16; k++) y[k] = yy[k*sz+ix];
    return y;
}

//------------------------------------------------------------------------------
// These are the variant for matmatmul, based on the matvecmul kernel with
// flattened block_q4_0.
//------------------------------------------------------------------------------
inline float block_q_4_0_dot_y_warp( global uchar * x,  global half  * dh,  global float* y) {
    uchar   q = intel_sub_group_block_read_uc((global uchar*)x);// x16
    float2 y2 = as_float2(intel_sub_group_block_read2((global uint*)y));// можно вынести из функции и из цикла
    float sum = y2.s0*(q&0x0F)+y2.s1*(q>>4);
    float  d  = *dh;
    return d*sub_group_reduce_add(sum);
}
// Common dot prod.
inline float mm_block_q_4_0_dot_y_flat(
        global uchar * x,
        global half  * dh,
        float16 yl,
        int il
) {
    float           d   = *dh;
    global uchar * qs  = ((global uchar *)x + il);// 0 +8
    float4          acc = (0.f);

    acc.s0 += yl.s0 * ((qs[0] & 0x0F)-8);
    acc.s0 += yl.s1 * ((qs[1] & 0x0F)-8);
    acc.s0 += yl.s2 * ((qs[2] & 0x0F)-8);
    acc.s0 += yl.s3 * ((qs[3] & 0x0F)-8);

    acc.s1 += yl.s4 * ((qs[4] & 0x0F)-8);
    acc.s1 += yl.s5 * ((qs[5] & 0x0F)-8);
    acc.s1 += yl.s6 * ((qs[6] & 0x0F)-8);
    acc.s1 += yl.s7 * ((qs[7] & 0x0F)-8);

    acc.s2 += yl.s8 * ((qs[0] >>4)-8);
    acc.s2 += yl.s9 * ((qs[1] >>4)-8);
    acc.s2 += yl.sa * ((qs[2] >>4)-8);
    acc.s2 += yl.sb * ((qs[3] >>4)-8);

    acc.s3 += yl.sc * ((qs[4] >>4)-8);
    acc.s3 += yl.sd * ((qs[5] >>4)-8);
    acc.s3 += yl.se * ((qs[6] >>4)-8);
    acc.s3 += yl.sf * ((qs[7] >>4)-8);

    return d * (acc.s0+acc.s1+acc.s2+acc.s3);
}
inline float mm_block_q_4_0_dot_y_flat_(
        global uchar * qb_qs,
        global half * qb_d,
        global float* yb
) {
    half d = *qb_d;
    uchar16 qs = vload16(0,((global uchar*)qb_qs));
    float16 yl = vload16(0, yb);
    float16 yh = vload16(1, yb);
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
#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

#ifdef INTEL_GPU
#define N_DST 8 // each SIMD group works on 8 rows (in weights matrix)
#define N_SIMDGROUP 1 // number of SIMD groups in a thread group
#define N_SIMDWIDTH 16 // assuming SIMD group size is 16
#elif defined (ADRENO_GPU)
#define N_DST 8
#define N_SIMDGROUP 1
#define N_SIMDWIDTH 64
#endif
//
// This variant performs 1d blocking with 8x output.
// Eeach simdgroup outputs 8 values on `n0` dim (row in the output matrix).
//
inline void mul_mat_q_n_f32_1d_8x_flat(
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
    const int nb = ne00/QK4_0;

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

    float16 yl;
    float8 sumf = (0.f);

    int ix = get_sub_group_local_id()/2;
    int il = 8*(get_sub_group_local_id()%2);

    global float * yb = y + ix*QK4_0 + il;

    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/2) {
        yl.lo = vload8(0, yb);
        yl.hi = vload8(2, yb);
        for (uint r=0; r<8; ++r)
            sumf[r] += mm_block_q_4_0_dot_y_flat(x + ib*QK4_0/2 +  r*nb*QK4_0/2, d + ib +  r*nb, yl, il);
        yb += QK4_0 * (N_SIMDWIDTH/2);
    }
    for (int i = 0; i<8 && first_row + i < ne01; i++){
        float tot = sub_group_reduce_add(sumf[i]);
        if (get_sub_group_local_id() == 0)
            dst[r1*ne0 + im*ne0*ne1 + first_row + i] = tot;
    }
}

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q4_0_f32_1d_8x_flat(
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

    mul_mat_q_n_f32_1d_8x_flat(src0_q, src0_d, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3);
}

#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

#ifdef INTEL_GPU
#define N_DST 16 // each SIMD group works on 8 rows (in weights matrix)
#define N_SIMDGROUP 1 // number of SIMD groups in a thread group
#define N_SIMDWIDTH 16 // assuming SIMD group size is 16
#elif defined (ADRENO_GPU)
#define N_DST 16
#define N_SIMDGROUP 1
#define N_SIMDWIDTH 64
#endif
//
// This variant performs 1d blocking with 16x output.
// Eeach simdgroup outputs 16 values on `n0` dim (row in the output matrix).
//
inline void mul_mat_q_n_f32_1d_16x_flat(
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
    const int nb = ne00/QK4_0;

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

    float16 yl;
    float16 sumf = (0.f);

    int ix = get_sub_group_local_id()/2;
    int il = 8*(get_sub_group_local_id()%2);

    global float * yb = y + ix*QK4_0 + il;

    for (uint ib = ix; ib < nb; ib += N_SIMDWIDTH/2) {
        yl.lo = vload8(0, yb);
        yl.hi = vload8(2, yb);
        // все 32 значения распределены между участниками подгруппы
        float2 y2= as_float2(intel_sub_group_block_read2((global uint*)yb));

        for (uint r=0; r<16; ++r)
            sumf[r] += mm_block_q_4_0_dot_y_flat(x + (ib+r*nb)*QK4_0/2, d + (ib +  r*nb), yl, il);
        yb += QK4_0 * (N_SIMDWIDTH/2);
    }
    for (uint r=0; r<16; ++r) {
        float tot = sub_group_reduce_add(sumf[r]);
        if(get_sub_group_local_id() == 0)
            dst[r1*ne0 + im*ne0*ne1 + first_row + r] = tot;
    }
}

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q4_0_f32_1d_16x_flat(
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

    mul_mat_q_n_f32_1d_16x_flat(src0_q, src0_d, src1, dst, ne00, ne01, ne02, ne10, ne12, ne0, ne1, r2, r3);
}

//------------------------------------------------------------------------------
// kernel_mul_mat_q4_0_f32_flat_v0
//------------------------------------------------------------------------------
inline float block_q_4_0_dot_y_flat_v2(
    half   x,
    half   d,
    float  sumy,
    float4 yl
) {
    uchar2 q = as_uchar2(x);
    float acc = 0.0f;

    acc += (q.s0 & 0x0F) * yl.s0;
    acc += (q.s1 & 0x0F) * yl.s1;

    float ac1 =0.f;
    ac1 += (q.s0 >> 4) * yl.s2;
    ac1 += (q.s1 >> 4) * yl.s3;

    return d * (sumy * -8.f + acc+ac1);
}

inline float block_q_4_0_dot_y_flat_v4(
    float  x,
    half   d,
    float  sumy,
    float8 yl
) {
    uchar4 q = as_uchar4(x);
    float acc = 0.0f;

    uchar4 p = q <<4>>4;
    acc += (p.s0) * yl.s0;
    acc += (p.s1) * yl.s1;
    acc += (p.s2) * yl.s2;
    acc += (p.s3) * yl.s3;

    q = (q >> 4);
    float ac1 = 0.f;
    ac1 += (q.s0) * yl.s4;
    ac1 += (q.s1) * yl.s5;
    ac1 += (q.s2) * yl.s6;
    ac1 += (q.s3) * yl.s7;

    return d * (sumy * -8.f + acc + ac1);
}

inline float block_q_4_0_dot_y_flat_v8(
    float2  x,
    half    d,
    float   sumy,
    float16 yl
) {
    uchar8 q = as_uchar8(x);
    float acc = 0.0f;
    uchar8 p = q <<4>>4;
    acc += (p.s1) * yl.s1;
    acc += (p.s2) * yl.s2;
    acc += (p.s3) * yl.s3;
    acc += (p.s4) * yl.s4;
    acc += (p.s5) * yl.s5;
    acc += (p.s6) * yl.s6;
    acc += (p.s7) * yl.s7;

    q = q >> 4;
    float acc1 = 0.f;
    acc1 += (q.s0) * yl.s8;
    acc1 += (q.s1) * yl.s9;
    acc1 += (q.s2) * yl.sa;
    acc1 += (q.s3) * yl.sb;
    acc1 += (q.s4) * yl.sc;
    acc1 += (q.s5) * yl.sd;
    acc1 += (q.s6) * yl.se;
    acc1 += (q.s7) * yl.sf;

    return d * (sumy * -8.f + acc + acc1);
}

#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

#ifdef INTEL_GPU
#define THREADS_PER_BLK 4   // Number of threads per block, or each thread process 1/THREADS_PER_BLK of a block
#define N_DST           4
#define N_SIMDGROUP     1
#define N_SIMDWIDTH     16
#elif defined (ADRENO_GPU)
#define THREADS_PER_BLK 4
#define N_DST           4
#define N_SIMDGROUP     1
#define N_SIMDWIDTH     64
#endif

#if THREADS_PER_BLK == 2                // Each thread processes 1/2 block
#   define ACT_TY                       float16
#   define Q_BLK_LD_TY                  float2
#   define block_q_4_0_dot_y_flat       block_q_4_0_dot_y_flat_v8
#elif THREADS_PER_BLK == 4              // Each thread processes 1/4 block
#   define ACT_TY                       float8
#   define Q_BLK_LD_TY                  float
#   define block_q_4_0_dot_y_flat       block_q_4_0_dot_y_flat_v4
#elif THREADS_PER_BLK == 8              // Each thread processes 1/8 block
#   define ACT_TY                       float4
#   define Q_BLK_LD_TY                  half
#   define block_q_4_0_dot_y_flat       block_q_4_0_dot_y_flat_v2
#endif

#define BTYES_PER_THREAD_IN_BLK         (QK4_0/2/THREADS_PER_BLK)

#if N_DST == 2
#   define  SUM_TY                      float2
#elif N_DST == 4
#   define  SUM_TY                      float4
#elif N_DST == 8
#   define  SUM_TY                      float8
#elif N_DST == 16
#   define  SUM_TY                      float16
#endif

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q4_0_f32_flat_v0(
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

    const int nb = ne00/QK4_0;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

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

    int ix = get_sub_group_local_id()/THREADS_PER_BLK;
    int il = get_sub_group_local_id()%THREADS_PER_BLK;

    global float * yb = y + ix*QK4_0 + BTYES_PER_THREAD_IN_BLK*il;

    // Registers for caching activation
    ACT_TY yl = 0.f;

    // Registers for caching quants
    Q_BLK_LD_TY q_blk_0 = 0, q_blk_1 = 0;
#if N_DST == 4 || N_DST == 8 || N_DST == 16
    Q_BLK_LD_TY q_blk_2 = 0, q_blk_3 = 0;
#endif
#if N_DST == 8 || N_DST == 16
    Q_BLK_LD_TY q_blk_4 = 0, q_blk_5 = 0, q_blk_6 = 0, q_blk_7 = 0;
#endif

    // Partial sum
    SUM_TY sumf = 0.f;

    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/THREADS_PER_BLK) {
        float sumy = 0.f;

        q_blk_0 = *(global Q_BLK_LD_TY*)(x + ib*QK4_0/2 + BTYES_PER_THREAD_IN_BLK*il + 0*nb*QK4_0/2);
        q_blk_1 = *(global Q_BLK_LD_TY*)(x + ib*QK4_0/2 + BTYES_PER_THREAD_IN_BLK*il + 1*nb*QK4_0/2);
#if N_DST == 4 || N_DST == 8 || N_DST == 16
        q_blk_2 = *(global Q_BLK_LD_TY*)(x + ib*QK4_0/2 + BTYES_PER_THREAD_IN_BLK*il + 2*nb*QK4_0/2);
        q_blk_3 = *(global Q_BLK_LD_TY*)(x + ib*QK4_0/2 + BTYES_PER_THREAD_IN_BLK*il + 3*nb*QK4_0/2);
#endif
#if N_DST == 8 || N_DST == 16
        q_blk_4 = (*(global Q_BLK_LD_TY*)(x + ib*QK4_0/2 + BTYES_PER_THREAD_IN_BLK*il + 4*nb*QK4_0/2));
        q_blk_5 = (*(global Q_BLK_LD_TY*)(x + ib*QK4_0/2 + BTYES_PER_THREAD_IN_BLK*il + 5*nb*QK4_0/2));
        q_blk_6 = (*(global Q_BLK_LD_TY*)(x + ib*QK4_0/2 + BTYES_PER_THREAD_IN_BLK*il + 6*nb*QK4_0/2));
        q_blk_7 = (*(global Q_BLK_LD_TY*)(x + ib*QK4_0/2 + BTYES_PER_THREAD_IN_BLK*il + 7*nb*QK4_0/2));
#endif

        // Load activation
#if THREADS_PER_BLK == 2    // Each thread processes 1/2 block
        yl.s01234567 = *(global float8 *)(yb);
        yl.s89abcdef = *(global float8 *)(yb + 16);

        sumy += yl.s0;
        sumy += yl.s1;
        sumy += yl.s2;
        sumy += yl.s3;
        sumy += yl.s4;
        sumy += yl.s5;
        sumy += yl.s6;
        sumy += yl.s7;
        sumy += yl.s8; 
        sumy += yl.s9; 
        sumy += yl.sa; 
        sumy += yl.sb; 
        sumy += yl.sc; 
        sumy += yl.sd; 
        sumy += yl.se; 
        sumy += yl.sf; 
#elif THREADS_PER_BLK == 4  // Each thread processes 1/4 block
        yl.s0123 = *(global float4 *)(yb);
        yl.s4567 = *(global float4 *)(yb + 16);

        sumy += yl.s0;
        sumy += yl.s1;
        sumy += yl.s2;
        sumy += yl.s3;
        sumy += yl.s4; 
        sumy += yl.s5; 
        sumy += yl.s6; 
        sumy += yl.s7; 
#elif THREADS_PER_BLK == 8  // Each thread processes 1/8 block
        yl.s01 = *(global float2 *)(yb);
        yl.s23 = *(global float2 *)(yb + 16);

        sumy += yl.s0;
        sumy += yl.s1;
        sumy += yl.s2; 
        sumy += yl.s3; 
#endif

        sumf.s0 += block_q_4_0_dot_y_flat(q_blk_0, *(d + ib + 0*nb), sumy, yl);
        sumf.s1 += block_q_4_0_dot_y_flat(q_blk_1, *(d + ib + 1*nb), sumy, yl);
#if N_DST == 4 || N_DST == 8 || N_DST == 16
        sumf.s2 += block_q_4_0_dot_y_flat(q_blk_2, *(d + ib + 2*nb), sumy, yl);
        sumf.s3 += block_q_4_0_dot_y_flat(q_blk_3, *(d + ib + 3*nb), sumy, yl);
#endif
#if N_DST == 8 || N_DST == 16
        sumf.s4 += block_q_4_0_dot_y_flat(q_blk_4, *(d + ib + 4*nb), sumy, yl);
        sumf.s5 += block_q_4_0_dot_y_flat(q_blk_5, *(d + ib + 5*nb), sumy, yl);
        sumf.s6 += block_q_4_0_dot_y_flat(q_blk_6, *(d + ib + 6*nb), sumy, yl);
        sumf.s7 += block_q_4_0_dot_y_flat(q_blk_7, *(d + ib + 7*nb), sumy, yl);
#endif

        yb += QK4_0 * (N_SIMDWIDTH/THREADS_PER_BLK);
    }

    SUM_TY tot = (SUM_TY)(
          sub_group_reduce_add(sumf.s0), sub_group_reduce_add(sumf.s1)
#if N_DST == 4 || N_DST == 8 || N_DST == 16
        , sub_group_reduce_add(sumf.s2), sub_group_reduce_add(sumf.s3)
#endif
#if N_DST == 8 || N_DST == 16
        , sub_group_reduce_add(sumf.s4), sub_group_reduce_add(sumf.s5)
        , sub_group_reduce_add(sumf.s6), sub_group_reduce_add(sumf.s7)
#endif
    );

    if (get_sub_group_local_id() == 0) {// Матрицы всегда выравнивают
        if (1 || first_row + 0 < ne01) 
            dst[r1*ne0 + im*ne0*ne1 + first_row + 0] = tot.s0;
        if (1 || first_row + 1 < ne01)
            dst[r1*ne0 + im*ne0*ne1 + first_row + 1] = tot.s1;
#if N_DST == 4 || N_DST == 8 || N_DST == 16
        if (1 || first_row + 2 < ne01) 
            dst[r1*ne0 + im*ne0*ne1 + first_row + 2] = tot.s2;
        if (1 || first_row + 3 < ne01) 
            dst[r1*ne0 + im*ne0*ne1 + first_row + 3] = tot.s3;
#endif
#if N_DST == 8 || N_DST == 16
        if (1 || first_row + 4 < ne01) 
            dst[r1*ne0 + im*ne0*ne1 + first_row + 4] = tot.s4;
        if (1 || first_row + 5 < ne01) 
            dst[r1*ne0 + im*ne0*ne1 + first_row + 5] = tot.s5;
        if (1 || first_row + 6 < ne01) 
            dst[r1*ne0 + im*ne0*ne1 + first_row + 6] = tot.s6;
        if (1 || first_row + 7 < ne01) 
            dst[r1*ne0 + im*ne0*ne1 + first_row + 7] = tot.s7;
#endif
    }
}

//------------------------------------------------------------------------------
// Using image1d_buffer_t

#if defined(cl_qcom_subgroup_shuffle)
#pragma OPENCL EXTENSION cl_qcom_subgroup_shuffle : enable
float qcom_sub_group_reduce_add(float sum) {
    sum += qcom_sub_group_shuffle_down(sum, 32, CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM, 0.f);
    sum += qcom_sub_group_shuffle_down(sum, 16, CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM, 0.f);
    sum += qcom_sub_group_shuffle_down(sum,  8, CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM, 0.f);
    sum += qcom_sub_group_shuffle_down(sum,  4, CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM, 0.f);
    sum += qcom_sub_group_shuffle_down(sum,  2, CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM, 0.f);
    sum += qcom_sub_group_shuffle_down(sum,  1, CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM, 0.f);
    return sum;
}
#define sub_group_reduce_add qcom_sub_group_reduce_add
#else
#define sub_group_reduce_add sub_group_reduce_add
#endif

#undef THREADS_PER_BLK
#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

#ifdef INTEL_GPU
#define THREADS_PER_BLK 4   // Number of threads per block, or each thread process 1/THREADS_PER_BLK of a block
#define N_DST           4
#define N_SIMDGROUP     1
#define N_SIMDWIDTH     16
#elif defined (ADRENO_GPU)
#define THREADS_PER_BLK 4
#define N_DST           4
#define N_SIMDGROUP     1
#define N_SIMDWIDTH     64
#endif

#if THREADS_PER_BLK == 2                // Each thread processes 1/2 block
#   define ACT_TY                       float16
#   define Q_BLK_LD_TY                  float2
#   define EXTRACT_BLK_DATA(tmp, part)  *((float2*)&tmp + part)
#   define block_q_4_0_dot_y_flat       block_q_4_0_dot_y_flat_v8
#elif THREADS_PER_BLK == 4              // Each thread processes 1/4 block
#   define ACT_TY                       float8
#   define Q_BLK_LD_TY                  float
#   define EXTRACT_BLK_DATA(tmp, part)  *((float*)&tmp + part)
#   define block_q_4_0_dot_y_flat       block_q_4_0_dot_y_flat_v4
#elif THREADS_PER_BLK == 8              // Each thread processes 1/8 block
#   define ACT_TY                       float4
#   define Q_BLK_LD_TY                  half
#   define EXTRACT_BLK_DATA(tmp, part)  *((half*)&tmp + part)
#   define block_q_4_0_dot_y_flat       block_q_4_0_dot_y_flat_v2
#endif

#define BTYES_PER_THREAD_IN_BLK         (QK4_0/2/THREADS_PER_BLK)

#if N_DST == 2
#   define  SUM_TY                      float2
#elif N_DST == 4
#   define  SUM_TY                      float4
#elif N_DST == 8
#   define  SUM_TY                      float8
#elif N_DST == 16
#   define  SUM_TY                      float16
#endif

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_q4_0_f32_flat_img_v0(
        read_only image1d_buffer_t src0_q,
        read_only image1d_buffer_t src0_d,
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

    const int nb = ne00/QK4_0;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    int i12 = im%ne12;
    int i13 = im/ne12;

    // The number of scales is the same as the number of blocks.
    ulong offset0_d = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);
    // Each block contains QK4_0/2 uchars, hence offset for qs is as follows.
    ulong offset0_q = first_row * nb + (i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    global float * y = (global float *) src1   + r1*ne10 + im*ne00*ne1;

    int ix = get_sub_group_local_id()/THREADS_PER_BLK;
    int il = get_sub_group_local_id()%THREADS_PER_BLK;

    global float * yb = y + ix*QK4_0 + BTYES_PER_THREAD_IN_BLK*il;

    // Registers for caching activation
    ACT_TY yl = 0.f;

    // Registers for caching quants
    Q_BLK_LD_TY q_blk_0 = 0, q_blk_1 = 0;
#if N_DST == 4 || N_DST == 8 || N_DST == 16
    Q_BLK_LD_TY q_blk_2 = 0, q_blk_3 = 0;
#endif
#if N_DST == 8 || N_DST == 16
    Q_BLK_LD_TY q_blk_4 = 0, q_blk_5 = 0, q_blk_6 = 0, q_blk_7 = 0;
#endif

    // Partial sum
    SUM_TY sumf = 0.f;

    for (int ib = ix; ib < nb; ib += N_SIMDWIDTH/THREADS_PER_BLK) {
        float sumy = 0.f;;

        float4 tmp;
        tmp = read_imagef(src0_q, offset0_q + ib + 0*nb);
        q_blk_0 = EXTRACT_BLK_DATA(tmp, il);
        tmp = read_imagef(src0_q, offset0_q + ib + 1*nb);
        q_blk_1 = EXTRACT_BLK_DATA(tmp, il);
#if N_DST == 4 || N_DST == 8 || N_DST == 16
        tmp = read_imagef(src0_q, offset0_q + ib + 2*nb);
        q_blk_2 = EXTRACT_BLK_DATA(tmp, il);
        tmp = read_imagef(src0_q, offset0_q + ib + 3*nb);
        q_blk_3 = EXTRACT_BLK_DATA(tmp, il);
#endif
#if N_DST == 8 || N_DST == 16
        tmp = read_imagef(src0_q, offset0_q + ib + 4*nb);
        q_blk_4 = EXTRACT_BLK_DATA(tmp, il);
        tmp = read_imagef(src0_q, offset0_q + ib + 5*nb);
        q_blk_5 = EXTRACT_BLK_DATA(tmp, il);
        tmp = read_imagef(src0_q, offset0_q + ib + 6*nb);
        q_blk_6 = EXTRACT_BLK_DATA(tmp, il);
        tmp = read_imagef(src0_q, offset0_q + ib + 7*nb);
        q_blk_7 = EXTRACT_BLK_DATA(tmp, il);
#endif

        // Load activation
#if THREADS_PER_BLK == 2    // Each thread processes 1/2 block
        yl.s01234567 = *(global float8 *)(yb);
        yl.s89abcdef = *(global float8 *)(yb + 16);

        sumy += yl.s0;
        sumy += yl.s1;
        sumy += yl.s2;
        sumy += yl.s3;
        sumy += yl.s4;
        sumy += yl.s5;
        sumy += yl.s6;
        sumy += yl.s7;
        sumy += yl.s8;
        sumy += yl.s9;
        sumy += yl.sa;
        sumy += yl.sb;
        sumy += yl.sc;
        sumy += yl.sd;
        sumy += yl.se;
        sumy += yl.sf;
#elif THREADS_PER_BLK == 4  // Each thread processes 1/4 block
        yl.s0123 = *(global float4 *)(yb);
        yl.s4567 = *(global float4 *)(yb + 16);

        sumy += yl.s0;
        sumy += yl.s1;
        sumy += yl.s2;
        sumy += yl.s3;
        sumy += yl.s4;
        sumy += yl.s5;
        sumy += yl.s6;
        sumy += yl.s7;
#elif THREADS_PER_BLK == 8  // Each thread processes 1/8 block
        yl.s01 = *(global float2 *)(yb);
        yl.s23 = *(global float2 *)(yb + 16);

        sumy += yl.s0;
        sumy += yl.s1;
        sumy += yl.s2;
        sumy += yl.s3;
#endif

        sumf.s0 += block_q_4_0_dot_y_flat(q_blk_0, read_imageh(src0_d, offset0_d + ib + 0*nb).s0, sumy, yl);
        sumf.s1 += block_q_4_0_dot_y_flat(q_blk_1, read_imageh(src0_d, offset0_d + ib + 1*nb).s0, sumy, yl);
#if N_DST == 4 || N_DST == 8 || N_DST == 16
        sumf.s2 += block_q_4_0_dot_y_flat(q_blk_2, read_imageh(src0_d, offset0_d + ib + 2*nb).s0, sumy, yl);
        sumf.s3 += block_q_4_0_dot_y_flat(q_blk_3, read_imageh(src0_d, offset0_d + ib + 3*nb).s0, sumy, yl);
#endif
#if N_DST == 8 || N_DST == 16
        sumf.s4 += block_q_4_0_dot_y_flat(q_blk_4, read_imageh(src0_d, offset0_d + ib + 4*nb).s0, sumy, yl);
        sumf.s5 += block_q_4_0_dot_y_flat(q_blk_5, read_imageh(src0_d, offset0_d + ib + 5*nb).s0, sumy, yl);
        sumf.s6 += block_q_4_0_dot_y_flat(q_blk_6, read_imageh(src0_d, offset0_d + ib + 6*nb).s0, sumy, yl);
        sumf.s7 += block_q_4_0_dot_y_flat(q_blk_7, read_imageh(src0_d, offset0_d + ib + 7*nb).s0, sumy, yl);
#endif

        yb += QK4_0 * (N_SIMDWIDTH/THREADS_PER_BLK);
    }

    SUM_TY tot = (SUM_TY)(
          sub_group_reduce_add(sumf.s0), sub_group_reduce_add(sumf.s1)
#if N_DST == 4 || N_DST == 8 || N_DST == 16
        , sub_group_reduce_add(sumf.s2), sub_group_reduce_add(sumf.s3)
#endif
#if N_DST == 8 || N_DST == 16
        , sub_group_reduce_add(sumf.s4), sub_group_reduce_add(sumf.s5)
        , sub_group_reduce_add(sumf.s6), sub_group_reduce_add(sumf.s7)
#endif
    );

    if (get_sub_group_local_id() == 0) {
        if (first_row + 0 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 0] = tot.s0;
        }
        if (first_row + 1 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 1] = tot.s1;
        }
#if N_DST == 4 || N_DST == 8 || N_DST == 16
        if (first_row + 2 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 2] = tot.s2;
        }
        if (first_row + 3 < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + 3] = tot.s3;
        }
#endif
#if N_DST == 8 || N_DST == 16
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
#endif
    }
}

//------------------------------------------------------------------------------
// kernel_mul_mv_q4_0_f32
//------------------------------------------------------------------------------
#undef N_DST
#ifdef INTEL_GPU
#define N_DST 1 // number of rows each SIMD group works on
#define N_SIMDGROUP 1 // number of SIMD groups in a thread group
#elif defined (ADRENO_GPU)
#define N_DST 1
#define N_SIMDGROUP 2
#endif

#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q4_0_f32(
        global void * src0,
        ulong offset0,
        global float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        uint ne00,
        uint ne01,
        uint ne02,
        uint ne10,
        uint ne12,
        uint ne0,
        uint ne1
//        uint r2,
//        uint r3
) {
    src0 = (global void *)((global char*)src0 + offset0);
    src1 = (global float*)((global char*)src1 + offset1);
    dst  = (global float*)((global char*)dst  + offsetd);

    uint nb = ne00/QK4_0; // число блоков в строке

    uint r0 = get_group_id(0);// число строк/N_SIMDGROUP
    uint r1 = get_group_id(1);
    uint im = 0;//get_group_id(2);

    uint row = N_SIMDGROUP * r0 + get_sub_group_id();

    uint i12 = 0;//im%ne12;
    uint i13 = 0;//im/ne12;

    const ulong offset_src0 = 0;//(i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    global struct block_q4_0 * x = (global struct block_q4_0 *) src0 + row*nb + offset_src0;
    global float      * yy = (global float     *) src1 + r1*ne10 + im*ne00*ne1;

    float sumf = 0;
    uint stride = get_max_sub_group_size();
    for (uint i = get_sub_group_local_id(); i < nb; i += stride) {// максимальное число субгрупп =128, каждая субгруппа считает о
        half d = x[i].d;
        uchar16 q  = vload16(0, x[i].qs);
//        float16 y0 = vload16(0, yy + i * QK4_0);
//        float16 y1 = vload16(1, yy + i * QK4_0);
        global float * y = yy + i * QK4_0;
        uchar16 ql = q & (uchar)0xF;
        uchar16 qh = q >> 4;
        float4 sums = (0.f);
        #pragma unroll
        for (uint k = 0; k<16; k++){// k < QK4_0/2
            sums[k/4] += y[k   ] * (ql[k] -(char)8);
            sums[k/4] += y[k+16] * (qh[k] -(char)8);
        }
        sumf += d * (sums.s0  + sums.s1 + sums.s2  + sums.s3);
    }
    float tot = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id() == 0) {
        dst[(r1 + im*ne1)*ne0 + row] = tot;
    }
}


//------------------------------------------------------------------------------
// kernel_mul_mv_q6_K_f32
//------------------------------------------------------------------------------

#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

#ifdef INTEL_GPU
#define N_DST 1 // number of rows each SIMD group works on
#define N_SIMDGROUP 1 // number of SIMD groups in a thread group
#define N_SIMDWIDTH 16 // SIMD group size
#elif defined (ADRENO_GPU)
#define N_DST 1
#define N_SIMDGROUP 2
#define N_SIMDWIDTH 64
#endif

#define BLOCK_STRIDE (N_SIMDWIDTH/16) // number of blocks each subgroup processes

inline float dot_block_q6_K(global block_q6_K *x, float16 y){
    uint tid  = get_sub_group_local_id();
        float  d   = x->d;
        int8_t sca = x->scales[tid];
        global uint8_t * q1 = x->ql + tid;
        global uint8_t * qh = x->qh + tid;
        float4 sums = (0.f);
        float8 a;
        {
            a[0] = sub_group_broadcast(sca,0)* ((char)(bitfield_extract_unsigned(q1[ 0 ], 0, 4) | (bitfield_extract_unsigned(qh[0], 0, 2) << 4)) - 32);
            a[1] = sub_group_broadcast(sca,1)* ((char)(bitfield_extract_unsigned(q1[ 16], 0, 4) | (bitfield_extract_unsigned(qh[16], 0, 2) << 4)) - 32);
            a[2] = sub_group_broadcast(sca,2)* ((char)(bitfield_extract_unsigned(q1[ 32], 0, 4) | (bitfield_extract_unsigned(qh[0], 2, 2) << 4)) - 32);
            a[3] = sub_group_broadcast(sca,3)* ((char)(bitfield_extract_unsigned(q1[ 48], 0, 4) | (bitfield_extract_unsigned(qh[16], 2, 2) << 4)) - 32);
            sums.s0 += dot(y.lo.lo, (a.lo));
            a[4] = sub_group_broadcast(sca,4)* ((char)(bitfield_extract_unsigned(q1[ 0 ], 4, 4) | (bitfield_extract_unsigned(qh[0], 4, 2) << 4)) - 32);
            a[5] = sub_group_broadcast(sca,5)* ((char)(bitfield_extract_unsigned(q1[ 16], 4, 4) | (bitfield_extract_unsigned(qh[16], 4, 2) << 4)) - 32);
            a[6] = sub_group_broadcast(sca,6)* ((char)(bitfield_extract_unsigned(q1[ 32], 4, 4) | (bitfield_extract_unsigned(qh[0], 6, 2) << 4)) - 32);
            a[7] = sub_group_broadcast(sca,7)* ((char)(bitfield_extract_unsigned(q1[ 48], 4, 4) | (bitfield_extract_unsigned(qh[16], 6, 2) << 4)) - 32);
            sums.s1 += dot(y.lo.hi, (a.hi));
#if N_SIMDGROUP==1
            q1+=64; qh+=32;
            a[0] = sub_group_broadcast(sca, 8)* ((char)(bitfield_extract_unsigned(q1[ 0 ], 0, 4) | (bitfield_extract_unsigned(qh[0], 0, 2) << 4)) - 32);
            a[1] = sub_group_broadcast(sca, 9)* ((char)(bitfield_extract_unsigned(q1[ 16], 0, 4) | (bitfield_extract_unsigned(qh[16], 0, 2) << 4)) - 32);
            a[2] = sub_group_broadcast(sca,10)* ((char)(bitfield_extract_unsigned(q1[ 32], 0, 4) | (bitfield_extract_unsigned(qh[0], 2, 2) << 4)) - 32);
            a[3] = sub_group_broadcast(sca,11)* ((char)(bitfield_extract_unsigned(q1[ 48], 0, 4) | (bitfield_extract_unsigned(qh[16], 2, 2) << 4)) - 32);
            sums.s2 += dot(y.hi.lo, (a.lo));
            a[4] = sub_group_broadcast(sca,12)* ((char)(bitfield_extract_unsigned(q1[ 0 ], 4, 4) | (bitfield_extract_unsigned(qh[0], 4, 2) << 4)) - 32);
            a[5] = sub_group_broadcast(sca,13)* ((char)(bitfield_extract_unsigned(q1[ 16], 4, 4) | (bitfield_extract_unsigned(qh[16], 4, 2) << 4)) - 32);
            a[6] = sub_group_broadcast(sca,14)* ((char)(bitfield_extract_unsigned(q1[ 32], 4, 4) | (bitfield_extract_unsigned(qh[0], 6, 2) << 4)) - 32);
            a[7] = sub_group_broadcast(sca,15)* ((char)(bitfield_extract_unsigned(q1[ 48], 4, 4) | (bitfield_extract_unsigned(qh[16], 6, 2) << 4)) - 32);
            sums.s3 += dot(y.hi.hi, (a.hi));
#endif
        }
    return d * (sums.s0  + sums.s1 + sums.s2  + sums.s3);
}
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q6_K_f32(
        global void * src0,
        global float * src1,
        global float * dst,
        uint ne00,// K
        uint ne01,// M
        uint ne02,//== 1
        uint ne10,// K
        uint ne12,//== 1
        uint ne0,// == ne01
        uint ne1 // N == ne11
) {
//    src0 = (global void *)((global char*)src0 + offset0);
//    src1 = (global float*)((global char*)src1 + offset1);
//    dst  = (global float*)((global char*)dst + offsetd);

    int nb = ne00/QK_K;

    int r0 = get_group_id(0);// 0..N/{N_SIMDGROUP}
    int r1 = get_group_id(1);// 0..N-1 col
    int im = 0;//get_group_id(2);
// 0..M-1
    int row = get_num_sub_groups() * N_DST*r0 + get_sub_group_id();

//    int i12 = im%ne12;
//    int i13 = im/ne12;

    const ulong offset_src0 = 0;//(i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);



    float sumf[N_DST] = {0};
    uint tid  = get_sub_group_local_id();//%16;

    global float      * yy = (global float     *) src1 + r1*ne10 + im*ne00*ne1;
    for (uint i = tid/16; i < nb; i += get_max_sub_group_size()/16) {// nb=K/QK
        float16 y;
        y.lo = as_float8(intel_sub_group_block_read8((global uint*)yy + i * QK_K));
        y.hi = as_float8(intel_sub_group_block_read8((global uint*)yy + i * QK_K+QK_K/2));
//        half16 yh = convert_half16(y);
//        half16 yl = convert_half16(y-convert_float16(yh));
        for (uint r = 0; r < N_DST; r ++) {
            global block_q6_K * x = (global block_q6_K *) src0 + row*nb + r*nb + offset_src0;
            sumf[r] += dot_block_q6_K(x+i, y);
        }
    }
    for(int r=0; r<N_DST; r++) {
        float tot = sub_group_reduce_add(sumf[r]);
        if (get_sub_group_local_id() == 0) {
            dst[(r1 + im*ne1)*ne0 + row +r] = tot;
        }
    }
}
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mva_q6_K_f32(
        global void * src0,
        global float * src1,
        global float * src2,
        global float * dst,
        uint ne00,// K
        uint ne01,// M
        uint ne10,// K
        uint ne0,// == ne01
        uint ne1,// N == ne11
        float alpha
) {

    uint nb = ne00/QK_K;// число блоков

    uint r0 = get_group_id(0);// 0..N/{N_SIMDGROUP}
    uint r1 = get_group_id(1);// 0..N-1 col
// 0..M-1
    uint row = get_num_sub_groups()*N_DST*r0 + get_sub_group_id();

    float sumf[N_DST] = {0};
    uint tid  = get_sub_group_local_id();//%16;

    global float      * yy = (global float     *) src1 + mul24(r1,ne10);
    for (uint i = tid/16; i < nb; i +=get_max_sub_group_size()/16) {// nb=K/QK
        float16 y;
        y.lo = as_float8(intel_sub_group_block_read8((global uint*)yy + i * QK_K));
        y.hi = as_float8(intel_sub_group_block_read8((global uint*)yy + i * QK_K+QK_K/2));
        for (uint r = 0; r < N_DST; r ++) {
            global block_q6_K * x = (global block_q6_K *) src0 + row*nb + r*nb;
            sumf[r] += dot_block_q6_K(x+i, y);
        }
    }
    dst +=mad24(r1,     ne0, row);
    src2+=mad24(r1%ne1, ne0, row);
    for(int r=0; r<N_DST; r++) {
        float tot = sub_group_reduce_add(sumf[r]);
        if (get_sub_group_local_id() == 0) {
            dst[r] = alpha*tot + src2[r];// чтобы получился broadcast нужнo ne1==0
//            dst[(r1 + im*ne1)*ne0 + row +r] = tot;
        }
    }
}

inline float silu(float x){  return x *native_recip(1.f + exp(-x)); }
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_16
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_ffn_q6_K_f32(
        global void * src0,
        global float * src1,
        global float * src2,
        global float * dst,
        uint ne00,// K
        uint ne01,// M
        uint ne10,// K
        uint ne0,// == ne01
        uint ne1 // N == ne11
) {

    uint nb = ne00/QK_K;

    uint r0 = get_group_id(0);// 0..N/{N_SIMDGROUP}
    uint r1 = get_group_id(1);// 0..N-1 col
// 0..M-1
    uint row = get_num_sub_groups()*N_DST*r0 + get_sub_group_id();

    float2 sumf[N_DST] = {(0)};
    uint tid  = get_sub_group_local_id();//%16;

    global float      * yy = (global float     *) src2 + mul24(r1, ne10);
    for (uint i = tid/16; i < nb; i +=get_max_sub_group_size()/16) {// nb=K/QK
        float16 y;
        y.lo = as_float8(intel_sub_group_block_read8((global uint*)yy + i * QK_K));
        y.hi = as_float8(intel_sub_group_block_read8((global uint*)yy + i * QK_K+QK_K/2));
        for (uint r = 0; r < N_DST; r ++) {
            global block_q6_K * x0 = (global block_q6_K *) src0 + row*nb + r*nb;
            global block_q6_K * x1 = (global block_q6_K *) src1 + row*nb + r*nb;
            sumf[r].s0 += dot_block_q6_K(x0+i, y);
            sumf[r].s1 += dot_block_q6_K(x1+i, y);
        }
    }
    dst+=mad24(r1,ne0, row);
    for(int r=0; r<N_DST; r++) {
        float s0 = sub_group_reduce_add(sumf[r].s0);
        float s1 = sub_group_reduce_add(sumf[r].s1);
        if (get_sub_group_local_id() == 0) {
            dst[r] = silu(s0)*s1;
        }
    }
}

#undef N_DST
#undef N_SIMDGROUP
#undef N_SIMDWIDTH

#ifdef INTEL_GPU
#define N_DST 1 // number of rows each SIMD group works on
#elif defined (ADRENO_GPU)
#define N_DST 1
#endif

inline void get_scale_min_k4(int j, global const uint8_t * q, private uint8_t * d, private uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63; 
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
inline void get_scale_min_k4_2(int j, global const uchar2 * q, private uchar2 * d, private uchar2 * m) {
    if (j <2) {
        *d = q[j] & (uchar)63; 
        *m = q[j + 2] & (uchar)63;
    } else {
        *d = (q[j+2] & (uchar)0xF) | ((q[j-2] >> 6) << 4);
        *m = (q[j+2] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
inline float  dot_block_q4_K(global struct block_q4_K *x, global float* yy)
{
    const uint tid = get_sub_group_local_id();
    uint n    = 4;
    uint il  = tid/8;
    uint ir  = tid%8;
    uint q_offset = 32*il + n*ir;
    uint y_offset = 64*il + n*ir;

    const half dall = x->d;
    const half dmin = x->dmin;
    global float* y = yy + y_offset;
    uchar2 sc, m;
    get_scale_min_k4_2(il, (global uchar2*)x->scales, &sc, &m);
    const half2 d2 = dall * convert_half2(sc); 
    const half2 m2 = dmin * convert_half2(m);

    uchar4 q = vload4(0,x->qs + q_offset);
    uchar4 q0= as_uchar4((as_uint(q)>>0)&0x0f0f0f0f);
    uchar4 q1= as_uchar4((as_uint(q)>>4)&0x0f0f0f0f);
    float2 sums=(0.f);
    #pragma unroll
    for (uint k = 0; k<n; k++) {
        sums.s0 += y[k+ 0] * (d2.s0*q0[k] - m2.s0);
        sums.s1 += y[k+32] * (d2.s1*q1[k] - m2.s1);
    }
    return sums.s0 + sums.s1;
}
inline float  dot_block_q5_K(global struct block_q5_K *x, global float* y)
{
    const uint tid = get_sub_group_local_id();
    uint n    = 4;
    uint il  = tid/8;
    uint ir  = tid%8;
    uint q_offset = 32*il + n*ir;
    uint q_offset_h = n*ir;
    const half dall = x->d;
    const half dmin = x->dmin;

    uchar2 sc, m;
    get_scale_min_k4_2(il, (global uchar2*)x->scales, &sc, &m);
    const half2 d2 = dall * convert_half2(sc); 
    const half2 m2 = dmin * convert_half2(m);

    uchar4 q = vload4(0,x->qs + q_offset);
    uchar4 q0= as_uchar4((as_uint(q)>>0)&0x0f0f0f0f);
    uchar4 q1= as_uchar4((as_uint(q)>>4)&0x0f0f0f0f);
    uchar4 qh = vload4(0, x->qh + q_offset_h);
    float2 sums=(0.f);
    #pragma unroll
    for (uint k = 0; k<n; k++) {
        uint8_t h1 = 1<<(2*il);
        sums.s0 += y[k+ 0] * (d2.s0*(q0[k]|(qh[k] & h1?16:0)) - m2.s0);
        uint8_t h2 = 2<<(2*il);
        sums.s1 += y[k+32] * (d2.s1*(q1[k]|(qh[k] & h2?16:0))  - m2.s1);
    }
    return sums.s0 + sums.s1;
}


#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q4_K_f32(
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
    int nb = ne00/QK_K;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = 0;//get_group_id(2);

    int row = get_num_sub_groups()* r0 + get_sub_group_id();

    int i12 = 0;//im%ne12;
    int i13 = 0;//im/ne12;

    const ulong offset_src0 = 0;//(i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    global struct block_q4_K * x = (global struct block_q4_K *) src0 + mul24(row,nb) + offset_src0;
    global float      * yy = (global float     *) src1 + mul24(r1,ne10) + im*ne00*ne1;

    float sumf = 0;
    for (uint i = 0; i < nb; i++) {
        global float * y = yy + i * QK_K;
        sumf += dot_block_q4_K(x+i, y);
    }
    sumf = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id() == 0) {
//        sumf = POST_OP2(POST_OP(sa),sb);
        dst[mad24(r1,ne0,row) + im*ne0*ne1] = sumf;
    }
}
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mva_q4_K_f32(
        global void * src0,
        global float * src1,
        global float * src2,
        global float * dst,
        uint ne00,
        uint ne01,
        uint ne10,
        uint ne0,
        uint ne1,
        float alpha
) {
    uint nb = ne00/QK_K;

    uint r0 = get_group_id(0);
    uint r1 = get_group_id(1);// batch
    uint im = 0;//get_group_id(2);

    uint row = get_num_sub_groups()* r0 + get_sub_group_id();

    const ulong offset_src0 = 0;//(i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    global struct block_q4_K * x = (global struct block_q4_K *) src0 + mul24(row,nb);
    global float      * yy = (global float     *) src1 + mul24(r1,ne10);

    float sumf = 0;
    for (uint i = 0; i < nb; i++) {
        global float * y = yy + i * QK_K;
        sumf += dot_block_q4_K(x+i, y);
    }
    sumf = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id() == 0) {
        dst[mad24(r1,ne0,row)] = alpha*sumf + src2[mad24((r1%ne1),ne0, row)];
    }
}
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_ffn_q4_K_f32(
        global void  * src0,
        global float * src1,
        global float * src2,
        global float * dst,
        uint ne00,
        uint ne01,
        uint ne10,
        uint ne0,
        uint ne1
) {
    uint nb = ne00/QK_K;

    uint r0 = get_group_id(0);
    uint r1 = get_group_id(1);// batch
    uint im = 0;//get_group_id(2);

    uint row = get_num_sub_groups() * r0 + get_sub_group_id();

    global struct block_q4_K * x1 = (global struct block_q4_K *) src0 + mul24(row,nb);
    global struct block_q4_K * x2 = (global struct block_q4_K *) src1 + mul24(row,nb);
    global float      * yy = (global float     *) src2 + r1*ne10 + im*ne00*ne1;

    float2 sumf = 0;
    for (uint i = 0; i < nb; i++) {
        global float * y = yy + i * QK_K;
        sumf.s0 += dot_block_q4_K(x1+i, y);
        sumf.s1 += dot_block_q4_K(x2+i, y);
    }
    sumf.s0 = sub_group_reduce_add(sumf.s0);
    sumf.s1 = sub_group_reduce_add(sumf.s1);
    if (get_sub_group_local_id() == 0)
        dst[mad24(r1,ne0,row)] = silu(sumf.s0) * sumf.s1;
}
#if 0// пережиток
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q5_K_f32(
        global void * src0,
        global float * src1,
        global float * dst,
        uint ne00,
        uint ne01,
        uint ne02,
        uint ne10,
        uint ne12,
        uint ne0,
        uint ne1
) {
    int nb = ne00/QK_K;

    int r0 = get_group_id(0);// 1.32.1
    int r1 = get_group_id(1);
    int im = 0;//get_group_id(2);

    int row = get_num_sub_groups() * r0 + get_sub_group_id();

    int i12 = 0;//im%ne12;
    int i13 = 0;//im/ne12;

    const ulong offset_src0 = 0;//(i12/r2)*(nb*ne01) + (i13/r3)*(nb*ne01*ne02);

    global struct block_q5_K * x = (global struct block_q5_K *) src0 + row*nb + offset_src0;
    global float      * yy = (global float     *) src1 + r1*ne10 + im*ne00*ne1;

    float sumf = 0;

    uint tid = get_sub_group_local_id(); // first block_stride groups have tid=0
    uint il  = tid/8;   // first or second half of (super) block (0 or 1)
    uint ir  = tid%8;   // each half has 8 parts, one per scale
    uint n    = 32/8;       // 2 scales at a time (and 2 sums)

    uint y_offset   = 64*il + n*ir;
    uint q_offset_l = 32*il + n*ir;
    uint q_offset_h = n*ir;
    for (uint i = 0; i < nb; i++) {

        uchar4 ql = vload4(0, x[i].qs + q_offset_l);
        uchar4 qh = vload4(0, x[i].qh + q_offset_h);
        const half dall = x[i].d;
        const half dmin = x[i].dmin;

        uchar2 sc, m;
        get_scale_min_k4_2(il, (global uchar2*)x[i].scales, &sc, &m);
        const half2 d2 = dall * convert_half2(sc); 
        const half2 m2 = dmin * convert_half2(m);

        global float * y = yy + i * QK_K + y_offset;
        float2 sums = (0.f);
        for (int k = 0; k<n; k++) {
            uint8_t h1 = 1<<(2*il);
            sums.s0 += y[k+ 0] * (d2.s0*(bitfield_extract_unsigned(ql[k],0,4)|(qh[k] & h1?16:0)) - m2.s0);
            uint8_t h2 = 2<<(2*il);
            sums.s1 += y[k+32] * (d2.s1*(bitfield_extract_unsigned(ql[k],4,4)|(qh[k] & h2?16:0))  - m2.s1);
        }
        sumf += sums.s0 + sums.s1;
    }

    sumf = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id() == 0) {
        dst[r1*ne0 + im*ne0*ne1 + row] = sumf;
    }
}
#endif
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mv_q5_K_f32(
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
    int nb = ne00/QK_K;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);// batch

    int row = get_num_sub_groups() * r0 + get_sub_group_id();

    global struct block_q5_K * x = (global struct block_q5_K *) src0 + mul24(row,nb);
    global float      * yy = (global float     *) src1 + mul24(r1,ne10);


    const uint tid = get_sub_group_local_id();
    uint il  = tid/8;
    uint ir  = tid%8;
    uint n    = 4;

    uint y_offset = 64*il + n*ir;
    float sumf = 0;
    for (uint i = 0; i < nb; i++) {
        global float * y = yy + i * QK_K + y_offset;
        sumf += dot_block_q5_K(x+i, y);
    }
    sumf = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id() == 0) {
        dst[mad24(r1,ne0, row)] = sumf;// + src2[(r1%ne1)*ne0 + row];
    }
}
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mva_q5_K_f32(
        global void * src0,
        global float * src1,
        global float * src2,
        global float * dst,
        uint ne00,
        uint ne01,
        uint ne10,
        uint ne0,
        uint ne1,
        float alpha
) {
    uint nb = ne00/QK_K;

    uint r0 = get_group_id(0);
    uint r1 = get_group_id(1);// batch

    uint row = get_num_sub_groups() * r0 + get_sub_group_id();

    global struct block_q5_K * x = (global struct block_q5_K *) src0 + mul24(row,nb);
    global float      * yy = (global float     *) src1 + mul24(r1,ne10);


    const uint tid = get_sub_group_local_id();
    uint il  = tid/8;
    uint ir  = tid%8;
    uint n    = 4;

    uint y_offset = 64*il + n*ir;
    float sumf = 0;
    for (uint i = 0; i < nb; i++) {
        global float * y = yy + i * QK_K + y_offset;
        sumf += dot_block_q5_K(x+i, y);
    }
    sumf = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id() == 0) {
        dst[mad24(r1,ne0,row)] = alpha*sumf + src2[mad24((r1%ne1),ne0, row)];
    }
}
#ifdef INTEL_GPU
REQD_SUBGROUP_SIZE_32
#elif defined (ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_ffn_q5_K_f32(
        global void  * src0,
        global void  * src1,
        global float * src2,
        global float * dst,
        int ne00,
        int ne01,
        int ne10,
        int ne0,
        int ne1
) {
    int nb = ne00/QK_K;

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);// batch
    int im = 0;//get_group_id(2);

    int row = get_num_sub_groups() * r0 + get_sub_group_id();

    global struct block_q5_K * x1 = (global struct block_q5_K *) src0 + row*nb;
    global struct block_q5_K * x2 = (global struct block_q5_K *) src1 + row*nb;
    global float      * yy = (global float     *) src2 + r1*ne10 + im*ne00*ne1;

    float2 sumf = 0;

    const uint tid = get_sub_group_local_id(); // first block_stride groups have tid=0
    uint il  = tid/8;
    uint ir  = tid%8;
    uint n    = 4;

    uint y_offset = 64*il + n*ir;
    for (uint i = 0; i < nb; i++) {
        global float * y = yy + i * QK_K + y_offset;
        sumf.s0 += dot_block_q5_K(x1+i, y);
        sumf.s1 += dot_block_q5_K(x2+i, y);
    }
    sumf.s0 = sub_group_reduce_add(sumf.s0);
    sumf.s1 = sub_group_reduce_add(sumf.s1);
    if (get_sub_group_local_id() == 0) {
        dst[r1*ne0 + row] = silu(sumf.s0) * sumf.s1;
    }
}

#if 0
kernel void kernel_ffn_bf16_f32(
        global ushort * src0,
        global ushort * src1,
        global float * src2,
        global float * dst,
        uint ne00,// K
        uint ne01,// M
        uint ne10,// K
        uint ne0,// == ne01
        uint ne1 // N == ne11
) {

    int nb = ne00/QK_K;

    int r0 = get_group_id(0);// 0..N/{N_SIMDGROUP}
    int r1 = get_group_id(1);// 0..N-1 col
// 0..M-1
    int row = get_num_sub_groups() * N_DST*r0 + get_sub_group_id();

    float2 sumf[N_DST] = {(0)};
    uint tid  = get_sub_group_local_id();//%16;

    global float      * yy = (global float     *) src2 + r1*ne10;
    for (uint i = 0; i < nb; i ++) {// nb=K/QK
        float16 y;
        y.lo = as_float8(intel_sub_group_block_read8((global uint*)yy + i * QK_K));
        y.hi = as_float8(intel_sub_group_block_read8((global uint*)yy + i * QK_K+QK_K/2));
        for (uint r = 0; r < N_DST; r ++) {
            ushort8 x0 = intel_sub_group_block_read_us8(src0 + row*nb + r*nb);
            ushort8 x1 = intel_sub_group_block_read_us8(src1 + row*nb + r*nb);
            sumf[r].s0 += dot_block_q6_K(x0+i, y);
            sumf[r].s1 += dot_block_q6_K(x1+i, y);
        }
    }
    for(int r=0; r<N_DST; r++) {
        float s0 = sub_group_reduce_add(sumf[r].s0);
        float s1 = sub_group_reduce_add(sumf[r].s1);
        if (get_sub_group_local_id() == 0) {
            dst[r1*ne0 + row+r] = silu(s0)*s1;
        }
    }
}
// на выходе - квантователь
kernel void kernel_ffn_bf16_f16(global ushort * src0, global ushort * src1, global float * src2, global half *dst, 
    global float * q
){
    for(int r=0; r<N_DST; r++) {
        float s0 = sub_group_reduce_add(sumf[r].s0);
        float s1 = sub_group_reduce_add(sumf[r].s1);
        if (get_sub_group_local_id() == 0) {
            float q = q[r1*ne0 + row+r];// остаток
            float y = fma(silu(s0),s1, q);
            half  yh = convert_half(y);
            dst[r1*ne0 + row+r] = yh;
              q[r1*ne0 + row+r] = y - yh;
        }
    }
}
#endif