#pragma OPENCL EXTENSION cl_khr_fp16 : enable
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
#endif
#define OVERLOADABLE __attribute__((overloadable))

// BFloat16(E8M7)
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
// Кванты порождают шаблоны. 
#define uint8_t uchar
#define  int8_t  char
#define QK8_0 32
#define QK8_1 32
#define QK_K  256
#define QR6_K 2

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
    half2 ds; // delta, s = d* sum(qs[i])
    int8_t qs[QK8_1]; // quants
} ;
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
#if !defined __opencl_c_integer_dot_product_input_4x8bit
int dot_sat_acc(char4 a, char4 b, int c){
    int d;
    d = c + a.x * b.x + a.y * b.y + a.z * b.z + a.w + b.w;
    return d;
}
#endif
// __opencl_c_integer_dot_product_input_4x8bit_packed
static inline int __dp4a(a, b, c){
#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(RDNA2)
    c = __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(RDNA3)
    c = __builtin_amdgcn_sudot4( true, a, true, b, c, false);
#endif
    return c;
}
int dp4a(const char4 a, const char4 b, const int c) { // 4-wide byte dot product and accumulate
#if defined(cl_nv_compute_capability) && cl_nv_compute_capability>=61 // use hardware-supported dp4a on Nvidia Pascal or newer GPUs with inline PTX assembly
	int d;
    asm("dp4a.s32.s32\t%0,%1,%2,%3;":"=r"(d):"r"(as_int(a)),"r"(as_int(b)),"r"(c));
    return d;
#elif defined(__opencl_c_integer_dot_product_input_4x8bit) // use hardware-supported dp4a on some Intel GPUs
	return  c+dot(a,b);// dot_acc_sat(a, b, c); -- is slow
#elif __has_builtin(__builtin_amdgcn_sdot4) // use hardware-supported dp4a on some AMD GPUs
    return __builtin_amdgcn_sdot4(as_int(a), as_int(b), c, false);
#elif defined(cl_arm_integer_dot_product_accumulate_int8) // use hardware-supported dp4a on some ARM GPUs
    return arm_dot_acc(a, b, c);
#else // fallback emulation (compilers will turn this into hardware-supported dp4a instruction if available)
	return c+a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;
#endif
}
static inline int __vsubss4(int a, int b){
    return as_int(as_char4(a) - as_char4(b));
}
#define VDR (32/4)
static inline half vec_dot_q8_0_q8_1(const int * v, const int * u, half d8_0, half d8_1) {
    int sumi = 0;
    #pragma unroll
    for (int i = 0; i < VDR; ++i) 
        sumi = __dp4a(v[i], u[i], sumi);
    return d8_0*d8_1 * (float) sumi;
}

#if 0
static inline float vec_dot_q8_1_q8_1(const int * v, const int * u, half2 dm8, const half2 ds8) {
    int sum = 0;
    #pragma unroll
    for (int i = 0; i < VDR; ++i)
        sum = __dp4a(v[i], u[i], sum);
    const float2 tmp = convert_float2(dm8 * ds8);
    const float d8d8 = tmp.x;
    const float m8s8 = tmp.y;
    return sum*d8d8 + m8s8 / (QI8_1 / VDR);
}
#endif
static inline float vec_dot_q6_K_q8_1(
    int vl, int vh, const int * u, const int8_t * scales, float d, const float * d8) {
    float sumf = 0.0f;
    #pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4*i];
        const int vil =  (vl >> (4*i))       & 0x0F0F0F0F;
        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;
        const int vi = __vsubss4((vil | vih), 0x20202020); // vi = (vil | vih) - 32
        sumf += d8[i] * (__dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }
    return d*sumf;
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


// ---------------------------------------------
// MV and MVA
// ---------------------------------------------
inline float silu(float x){  return x *native_recip(1.f + exp(-x)); }
#define QK4_0 32
#define QK8_0 32

REQD_SUBGROUP_SIZE_DEFAULT
kernel void mv_q8_0_f32(global char* src, global float *src1, global float *dst, uint ne00, uint nb01)
{
    uint row = get_group_id(0);
    global struct block_q8_0 *blk = (global struct block_q8_0 *)(src + mul24(row,nb01));
    float sumf=0;
    for(uint i=get_sub_group_local_id(); i < ne00; i+=get_max_sub_group_size()){
        float y = (src1[i]);// деквантизовать вектор
        sumf+=dequantize_block_q8_0(blk[i/QK8_0].qs, blk[i/QK8_0].d)*y;
    }
    sumf = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id()==0)
        dst[row] = sumf;
}
REQD_SUBGROUP_SIZE_DEFAULT
kernel void mva_q8_0_f32(global char* src, global float *src1, global float *src2, global float *dst, 
    uint ne00, 
    uint ne01,
    uint ne10,
    uint ne0,
    uint ne1,
    float alpha
)
{
    uint r0 = get_group_id(0);
    uint r1 = get_group_id(1);// batch

    uint row = get_num_sub_groups()*r0 + get_sub_group_id();
    global struct block_q8_0 *blk = (global struct block_q8_0 *)src + mul24(row,ne00/QK8_0);
    src1 = (global float     *) src1 + mul24(r1,ne10);
    float sumf=0;
    for(uint i=get_sub_group_local_id(); i < ne00; i+=get_max_sub_group_size()){
        float y = (src1[i]);
        sumf+=dequantize_block_q8_0(blk[i/QK8_0].qs, blk[i/QK8_0].d)*y;
    }
    sumf = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id()==0)
        dst[mad24(r1,ne0,row)] = alpha*sumf + src2[mad24((r1%ne1),ne0, row)];
}

REQD_SUBGROUP_SIZE_DEFAULT
kernel void ffn_q8_0_f32(global char* src0, global float *src1, global float *src2, global float *dst, 
    uint ne00, 
    uint ne01,
    uint ne10,
    uint ne0,
    uint ne1
)
{
    uint r0 = get_group_id(0);
    uint r1 = get_group_id(1);// batch

    uint row = get_num_sub_groups()*r0 + get_sub_group_id();
    global struct block_q8_0 *blk0 = (global struct block_q8_0 *)src0 + mul24(row,ne00/QK8_0);
    global struct block_q8_0 *blk1 = (global struct block_q8_0 *)src1 + mul24(row,ne00/QK8_0);
    src2 = (global float     *) src2 + mul24(r1,ne10);
    float2 sumf=(0);
    for(uint i=get_sub_group_local_id(); i < ne00; i+=get_max_sub_group_size()){
        float y = (src2[i]);
        sumf.s0+=dequantize_block_q8_0(blk0[i/QK8_0].qs, blk0[i/QK8_0].d)*y;
        sumf.s1+=dequantize_block_q8_0(blk1[i/QK8_0].qs, blk1[i/QK8_0].d)*y;
    }
    sumf.s0 = sub_group_reduce_add(sumf.s0);
    sumf.s1 = sub_group_reduce_add(sumf.s1);
    if (get_sub_group_local_id()==0)
        dst[mad24(r1,ne0,row)] = silu(sumf.s0)*sumf.s1;
}
// не рабочий вариант
REQD_SUBGROUP_SIZE_16
kernel void mva_q6_k_f32(global char* src, global float *src1, global float *src2, global float *dst, 
    uint ne00, 
    uint ne01,
    uint ne10,
    uint ne0,// == ne01
    uint ne1,
    float alpha
)
{
    const uint NL=16;// interleaving
    uint r0 = get_group_id(0);
    uint r1 = get_group_id(1);// batch

    uint row = get_num_sub_groups()*r0 + get_sub_group_id();
    global struct block_q6_K *blk = (global struct block_q6_K *)src + mul24(row,ne00/QK_K);
    src1 = (global float     *) src1 + mul24(r1,ne10);
    uint il = get_sub_group_local_id()%NL; // спрятать
    float sumf=0;
    for(uint i=get_sub_group_local_id()/NL; i < ne00/QK_K; i+=get_max_sub_group_size()/NL){
        float16 y = vload16(il, src1 + i*QK_K);
        float16 x = dequantize_q6_k_f32(blk + i, il);
        #pragma unroll
        for (uint k=0; k<16; ++k)
            sumf += x[k] * y[k];
    }
    sumf = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id()==0)
        dst[mad24(r1,ne0,row)] = alpha*sumf + src2[mad24((r1%ne1),ne0, row)];
}

// функция работает с nb01 != ne00*nb00 c с подматрицами
kernel void mv_bf16_f32(global char* src, global float *src1, global float *dst, uint ne00, uint nb01)
{
    uint row = get_group_id(0)* get_num_sub_groups() + get_sub_group_id();
    global ushort *blk = (global ushort *)(src + row*nb01);
    float sumf=0;
    for(int i=get_sub_group_local_id(); i < ne00; i+=get_max_sub_group_size()){
        float y = (src1[i]);// деквантизовать вектор
        sumf+=convert_bf16_to_f32(blk[i])*y;
    }
    sumf = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id()==0){
        dst[row] = sumf;
    }
}
kernel void mv_f16_f32(global char* src, global float *src1, global float *dst, uint ne00, uint nb01)
{
    uint row = get_group_id(0)* get_num_sub_groups() + get_sub_group_id();
    global half *blk = (global half *)(src + row*nb01);
    float sumf=0;
    for(int i=get_sub_group_local_id(); i < ne00; i+=get_max_sub_group_size()){
        float y = (src1[i]);
        sumf+=(blk[i])*y;
    }
    sumf = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id()==0){
        dst[row] = sumf;
    }
}
kernel void mv_f32_f32(global char* src, global float *src1, global float *dst, uint ne00, uint nb01)
{
    const uint row = get_group_id(0)* get_num_sub_groups() + get_sub_group_id();

    global float *blk = (global float *)(src + mul24(row,nb01));
    float sumf=0;
    for(int i=get_sub_group_local_id(); i < ne00; i+=get_max_sub_group_size()){
        sumf+=(blk[i])*(src1[i]);
    }
    sumf = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id()==0){
        dst[row] = sumf;
    }
}
#if defined(__opencl_c_fp64)
kernel void mv_f32_f64(global char* src, global double *src1, global double *dst, uint ne, uint nb01)
{
    uint row= get_group_id(0)*get_num_sub_groups() + get_sub_group_id();
    uint rb = get_group_id(0);
    global float *blk = (global float *)(src + row*nb01);
    double sumf=0;
    for(int i=get_sub_group_local_id(); i < ne; i+=get_max_sub_group_size()){
        float y = (src1[i]);
        sumf+=(blk[i])*y;
    }
    sumf = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id()==0){
        dst[row] = (sumf);
    }
}
#endif
// осмыслить
kernel void mv_permuted_bf16_f32(global char* src0, global char *src1, global char *dst, 
    uint ne00, uint nb01, uint nb02,
    uint nb11, uint nb12,
    uint nb1,  uint nb2,
    uint r2
){
    uint r0 = get_group_id(0);
    uint rb = get_group_id(1);
    uint im = get_group_id(2);

    uint row = get_num_sub_groups()*r0 + get_sub_group_id();

    global ushort * x = (global ushort*) (src0 + mul24(row,nb01) + (im/r2)*nb02);
    global float  * y = (global float *) (src1 + mul24(rb ,nb11) + (im   )*nb12);

    float sumf=0;
    for(int i=get_sub_group_local_id(); i < ne00; i+=get_max_sub_group_size()){
        float y = (src1[i]);
        sumf+=convert_bf16_to_f32(x[i])*y;
    }
    sumf = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id()==0){
        ((global float*)(dst + mul24(rb,nb1) + im*nb2))[row] = sumf;
    }
}
REQD_SUBGROUP_SIZE_DEFAULT
kernel void mva_bf16_f32(global char* src, global float *src1, global float *srcb,  global float *dst, 
    uint ne00, 
    uint ne01,
    uint ne10,
    uint ne0,
    uint ne1,//ne11%ne1==0 - кратно
    float alpha
)
{
    uint r0 = get_group_id(0);// row
    uint r1 = get_group_id(1);// batch
// N_SIMDGROUP = get_num_sub_groups()
    uint row = get_num_sub_groups()*r0 + get_sub_group_id();
    global ushort *blk = (global ushort *)(src) + mul24(row,ne00);
    src1 = src1 + mul24(r1,ne10);
    float sumf=0;
    for(int i=get_sub_group_local_id(); i < ne00; i+=get_max_sub_group_size()){
        sumf+=convert_bf16_to_f32(blk[i])*src1[i];
    }
    sumf = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id()==0){
        dst[mad24(r1,ne0,row)] = alpha*sumf + srcb[mad24((r1%ne1),ne0, row)];
    }
}
REQD_SUBGROUP_SIZE_DEFAULT
kernel void mva_f16_f32(global char* src, global float *src1, global float *srcb,  global float *dst, 
    uint ne00, 
    uint ne01,
    uint ne10,
    uint ne0,
    uint ne1,//ne11%ne1==0 - кратно
    float alpha
)
{
    uint r0 = get_group_id(0);// row
    uint r1 = get_group_id(1);// batch

    uint row= get_num_sub_groups()*r0 + get_sub_group_id();
    global half *blk = (global half *)src + mul24(row,ne00);
    src1 = src1 + mul24(r1,ne10);
    float sumf=0;
    for(int i=get_sub_group_local_id(); i < ne00; i+=get_max_sub_group_size()){
        sumf+=blk[i] * src1[i];
    }
    sumf = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id()==0)
        dst[mad24(r1,ne0,row)] = alpha*sumf + srcb[mad24((r1%ne1),ne0, row)];
}

REQD_SUBGROUP_SIZE_DEFAULT
kernel void ffn_f16_f32(global char* src0, global char *src1, global float *src2,  global float *dst, 
    uint ne00, 
    uint ne01,
    uint ne10,
    uint ne0,
    uint ne1//ne11%ne1==0 - кратно
)
{
    uint r0 = get_group_id(0);// row
    uint r1 = get_group_id(1);// batch

    uint row= get_num_sub_groups()*r0 + get_sub_group_id();
    global half *blk0 = (global half *)src0 + mul24(row,ne00);
    global half *blk1 = (global half *)src1 + mul24(row,ne00);
    src2 = src2 + mul24(r1,ne10);
    float2 sumf=(0);
    for(int i=get_sub_group_local_id(); i < ne00; i+=get_max_sub_group_size()){
        float y = src2[i];
        sumf.s0+=blk0[i] * y;
        sumf.s1+=blk1[i] * y;
    }
    sumf.s0 = sub_group_reduce_add(sumf.s0);
    sumf.s1 = sub_group_reduce_add(sumf.s1);
    if (get_sub_group_local_id()==0)
        dst[mad24(r1,ne0,row)] = silu(sumf.s0)*sumf.s1;
}
// этот вариант с квантизацией и компенсацией остатка округления предназначен для операций на KV кэше
kernel void mv_bf16_bf16q(global char* src0, global char *src1, global char *dst,
    uint ne00, uint r2, 
    uint nb01, uint nb02,
    uint nb11, uint nb12,
    uint nb1,  uint nb2
){
    uint r0 = get_group_id(0);
    uint rb = get_group_id(1);
    uint im = get_group_id(2);

    global ushort * x = (global ushort*) (src0 + r0*nb01 + (im/r2)*nb02);
    global float  * y = (global float *) (src1 + rb*nb11 + (im   )*nb12);

    float sumf=0;
    float nrm2=0;// norma_2
    for(int i=get_sub_group_local_id(); i < ne00; i+=get_max_sub_group_size()){
        float y = (src1[i]);
        float v = convert_bf16_to_f32(x[i]);
        sumf+= v*y;
        nrm2+= v*v;// _mul_bf16
    }
    nrm2 = sub_group_reduce_add(nrm2);
    sumf = sub_group_reduce_add(sumf);
    if (get_sub_group_local_id()==0){
        global half2  * rq = (global half2 *) (dst + rb*nb1 + (im   )*nb2);
        half2 z;
        sumf = sumf*rsqrt(nrm2) + rq[r0].s0;// компенсация ошибки, time-shift gradient
        z.s1 = convert_f32_to_bf16_rtn(sumf);
        z.s0 = (sumf - convert_bf16_to_f32(z.s1));// вычисляем градиент
        rq[r0] = z;
    }
}
