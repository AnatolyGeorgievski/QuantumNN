/*! \brief Quantum NN */
#ifndef _QNN_H
#define _QNN_H

#include <glib.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include "quarks.h"
// Это определение надо расширить на случай библиотеки
#define GGML_API extern
// Проверки для отладки моделей
#define GGML_ASSERT(x) 

struct _gguf_ctx {
	struct gguf_header {
		char magic[4];
		uint32_t version;
		uint64_t n_tensors;
		uint64_t n_kv;
	} header;
	struct gguf_kv* kv;
	struct gguf_tensor_info* infos;
	void* data;
	uint64_t offset; // data segment 

	size_t size;
	unsigned alignment;
};
struct gguf_str {
    uint64_t n;  // GGUFv2
    char * data;
};
static inline 
int gguf_strcmp(const struct gguf_str* str, const char* key){
    size_t n = __builtin_strlen(key);
    if (str->n != n) return str->n - n;
    return __builtin_strncmp(str->data, key, n);
}
enum gguf_type {
	GGUF_TYPE_UINT8   = 0,
	GGUF_TYPE_INT8    = 1,
	GGUF_TYPE_UINT16  = 2,
	GGUF_TYPE_INT16   = 3,
	GGUF_TYPE_UINT32  = 4,
	GGUF_TYPE_INT32   = 5,
	GGUF_TYPE_FLOAT32 = 6,
	GGUF_TYPE_BOOL    = 7,
	GGUF_TYPE_STRING  = 8,
	GGUF_TYPE_ARRAY   = 9,
	GGUF_TYPE_UINT64  = 10,
	GGUF_TYPE_INT64   = 11,
	GGUF_TYPE_FLOAT64 = 12,
	GGUF_TYPE_COUNT,       // marks the end of the enum
};
enum ggml_type {
        GGML_TYPE_F32     = 0,
        GGML_TYPE_F16     = 1,
        GGML_TYPE_Q4_0    = 2,
        GGML_TYPE_Q4_1    = 3,
        // GGML_TYPE_Q4_2 = 4, support has been removed
        // GGML_TYPE_Q4_3 = 5, support has been removed
        GGML_TYPE_Q5_0    = 6,
        GGML_TYPE_Q5_1    = 7,
        GGML_TYPE_Q8_0    = 8,
        GGML_TYPE_Q8_1    = 9,
        GGML_TYPE_Q2_K    = 10,
        GGML_TYPE_Q3_K    = 11,
        GGML_TYPE_Q4_K    = 12,
        GGML_TYPE_Q5_K    = 13,
        GGML_TYPE_Q6_K    = 14,
        GGML_TYPE_Q8_K    = 15,
        GGML_TYPE_IQ2_XXS = 16,
        GGML_TYPE_IQ2_XS  = 17,
        GGML_TYPE_IQ3_XXS = 18,
        GGML_TYPE_IQ1_S   = 19,
        GGML_TYPE_IQ4_NL  = 20,
        GGML_TYPE_IQ3_S   = 21,
        GGML_TYPE_IQ2_S   = 22,
        GGML_TYPE_IQ4_XS  = 23,
        GGML_TYPE_I8      = 24,
        GGML_TYPE_I16     = 25,
        GGML_TYPE_I32     = 26,
        GGML_TYPE_I64     = 27,
        GGML_TYPE_F64     = 28,
        GGML_TYPE_IQ1_M   = 29,
        GGML_TYPE_BF16    = 30,
        GGML_TYPE_BF8     = 31,// E5M2 -- AG добавил в QNN
        GGML_TYPE_HF8     = 32,
        GGML_TYPE_E4M3FN  = 33,
        // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
        // GGML_TYPE_Q4_0_4_8 = 32,
        // GGML_TYPE_Q4_0_8_8 = 33,
        GGML_TYPE_TQ1_0   = 34,
        GGML_TYPE_TQ2_0   = 35,
        GGML_TYPE_I2_S    = 36,// BitNet
        GGML_TYPE_I8_S    = 37,
        GGML_TYPE_TL1     = 38,
        GGML_TYPE_TL2     = 39,
        // GGML_TYPE_IQ4_NL_4_4 = 36,
        // GGML_TYPE_IQ4_NL_4_8 = 37,
        // GGML_TYPE_IQ4_NL_8_8 = 38,
        GGML_TYPE_COUNT   = 40,
};

typedef struct { uint16_t bits; } ggml_bf16_t;// E8M7
typedef _Float16 ggml_fp16_t;
typedef struct { uint8_t  bits; } ggml_fp8_t;// E4M3FN
typedef struct { uint8_t  bits; } ggml_bf8_t;// E5M2
typedef _Float16 ggml_half;
#define QK8_0 32
typedef struct _block_q8_0 {
    ggml_half d;       // delta
    int8_t  qs[QK8_0]; // quants
} block_q8_0;
#define QK4_0 32
typedef struct _block_q4_0 {
    ggml_half d;           // delta
    uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;

#define QK_K 256
#define K_SCALE_SIZE 12
// 4-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 4.5 bits per weight
typedef struct _block_q4_K {
    struct {
		ggml_half d;    // super-block scale for quantized scales
		ggml_half dmin; // super-block scale for quantized mins
    };
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K/2];           // 4--bit quants
} block_q4_K;
// 5-bit quantization
// 8 blocks of 32 elements each
// weight is represented as x = a * q + b
// Effectively 5.5 bits per weight
typedef struct _block_q5_K {
	struct {
		ggml_half d;    // super-block scale for quantized scales
		ggml_half dmin; // super-block scale for quantized mins
	};
    uint8_t scales[K_SCALE_SIZE]; // scales and mins, quantized with 6 bits
    uint8_t qh[QK_K/8];           // quants, high bit
    uint8_t qs[QK_K/2];           // quants, low 4 bits
} block_q5_K;
// 6-bit quantization
// weight is represented as x = a * q
// 16 blocks of 16 elements each
// Effectively 6.5625 bits per weight
typedef struct _block_q6_K {
    uint8_t ql[QK_K/2];      // quants, lower 4 bits
    uint8_t qh[QK_K/4];      // quants, upper 2 bits
    int8_t  scales[QK_K/16]; // scales, quantized with 8 bits
    ggml_half d;             // super-block scale
} block_q6_K;
//static_assert(sizeof(block_q5_K) == 2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2 + QK_K/8, "wrong q5_K block size/padding");
// This is only used for intermediate quantization and dot products
#define Q8K_K 32
typedef struct _block_q8_K {
    float   d;              // delta
	int8_t  ex;
    int16_t  qs[Q8K_K];       // quants
    int16_t bsums[Q8K_K/16]; // sum of quants in groups of 16
} block_q8_K;

extern
const char*  GGML_TYPE_NAME[GGML_TYPE_COUNT];
extern
const size_t GGUF_TYPE_SIZE[GGUF_TYPE_COUNT];
static inline size_t gguf_type_size(enum gguf_type type) {
    //GGML_ASSERT(0 <= type && type < GGUF_TYPE_COUNT);
    return GGUF_TYPE_SIZE[type];
}
#if defined(__F16C__)
#include <x86intrin.h>
	#define GGML_FP16_TO_FP32(x) _cvtsh_ss(*(uint16_t*)&(x))
/*
    (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
    (_MM_FROUND_TO_NEG_INF |_MM_FROUND_NO_EXC)     // round down, and suppress exceptions
    (_MM_FROUND_TO_POS_INF |_MM_FROUND_NO_EXC)     // round up, and suppress exceptions
    (_MM_FROUND_TO_ZERO |_MM_FROUND_NO_EXC)        // truncate, and suppress exceptions
    _MM_FROUND_CUR_DIRECTION                       // use MXCSR.RC; see _MM_SET_ROUNDING_MODE
*/
    #define GGML_FP32_TO_FP16(x) _cvtss_sh(x,  (_MM_FROUND_TO_NEAREST_INT))
#else
	#define GGML_FP16_TO_FP32(x) (_Float16)(x)
#endif

/**
 * Converts brain16 to float32
 *
 * The bfloat16 floating point format has the following structure:
 *
 *       ┌sign
 *       │   ┌exponent
 *       │   │      ┌mantissa
 *       │┌──┴───┐┌─┴───┐
 *     0b0000000000000000 brain16
 *
 * Since bf16 has the same number of exponent bits as a 32bit float,
 * encoding and decoding numbers becomes relatively straightforward.
 *
 *       ┌sign
 *       │   ┌exponent
 *       │   │      ┌mantissa
 *       │┌──┴───┐┌─┴───────────────────┐
 *     0b00000000000000000000000000000000 IEEE binary32
 *
 * For comparison, the standard fp16 format has fewer exponent bits.
 *
 *       ┌sign
 *       │  ┌exponent
 *       │  │    ┌mantissa
 *       │┌─┴─┐┌─┴──────┐
 *     0b0000000000000000 IEEE binary16
 *
 * @see IEEE 754-2008
 */
static inline float ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
    union {
        float f;
        uint32_t i;
    } u;
// h.bits = ((h.bits & 0x7f80) == 0x0? (uint16_t)(h.bits & 0x8000) : h.bits);
// see libxsmm libxsmm_convert_bf16_to_f32
    u.i = (uint32_t)h.bits << 16;
    return u.f;
}
/*! \brief Converts brain8 to float16
 *
 * The bfloat8 (E5M2) floating point format has the following structure:
 *
 *       ┌sign
 *       │  ┌exponent
 *       │  │   ┌mantissa
 *       │┌─┴─┐┌┤
 *     0b00000000 brain8
 */
static inline _Float16 ggml_compute_bf8_to_fp16(ggml_bf8_t h) {
    union {
        _Float16 f;
        uint16_t i;
    } u;
    u.i = (uint32_t)h.bits << 8;
    return u.f;
}
static inline float ggml_convert_bf8_to_f32(ggml_bf8_t in)
{
    _Float16 tmp = ggml_compute_bf8_to_fp16(in);
    return GGML_FP16_TO_FP32(tmp);
}
// typedef __bf16 bfloat16_t; ARM 8.2+bf16 
// typedef __fp16  float16_t; ARM 8.2+fp16 
// see [ACLE-2024Q4](https://github.com/ARM-software/acle/releases)

/**
 * Converts float32 to brain16.
 *
 * This is binary identical with Google Brain float conversion.
 * Floats shall round to nearest even, and NANs shall be quiet.
 * Subnormals aren't flushed to zero, except perhaps when used.
 * This code should vectorize nicely if using modern compilers.
 */
static inline ggml_bf16_t ggml_compute_fp32_to_bf16(float s) {
    ggml_bf16_t h;
    union {
        float f;
        uint32_t i;
    } u;
    u.f = s;
    if ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        h.bits = (u.i >> 16) | 64; /* force to quiet */
        return h;
    }
    /* RNE round */
    h.bits = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    return h;
}
static inline ggml_bf16_t ggml_compute_fp32_to_bf16_rnaz(float s) {
    ggml_bf16_t h;
    union {
        float f;
        uint32_t i;
    } u;
    u.f = s;
    if ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        h.bits = (u.i >> 16) | 64; /* force to quiet */
        return h;
    }
    /* RNAZ round (+0.5) */
    h.bits = (u.i + 0x8000) >> 16;
    return h;
}
/* bf8 (E5M2) s.11111.00 = INF, s.11111.xx = NaN 
	\see <https://arxiv.org/pdf/2209.05433>
    \see <https://patents.google.com/patent/US20240248720A1/en>
 */
static inline ggml_bf8_t ggml_compute_fp16_to_bf8(float s) {
    ggml_bf8_t h;
    union {
        _Float16 f;
        uint16_t i;
    } u;
    u.f = s;
    if ((u.i & 0x7fff) > 0x7c00) { /* nan */
        h.bits = (u.i >> 8) | 2; /* force to quiet */
        return h;
    }
/*    if ((u.i & 0x7fff) == 0x7c00) {// infty
        h.bits = (u.i >> 8);
        return h;
    } */
    /* RNE round */
    unsigned fixup = ((u.i >> 8) & 1);
    h.bits = (u.i + 0x7fu + fixup) >> 8;
    return h;
}
#if 0// https://github.com/libxsmm/libxsmm/blob/main/src/libxsmm_math.c
LIBXSMM_API libxsmm_hfloat8 libxsmm_convert_f16_to_hf8_rne(libxsmm_float16 in)
{
  unsigned int f16_bias = 15;
  unsigned int f8_bias = 7;
  libxsmm_hfloat8 res = 0;
  unsigned short s, e, m, e_f16, m_f16;
  unsigned int fixup;

  s = (in & 0x8000) >> 8;
  e_f16 = (in & 0x7c00) >> 10;
  m_f16 = (in & 0x03ff);

  /* special value --> make it NaN */
  if (e_f16 == 0x1f) {
    e = 0xf;
    m = 0x7;
    /* overflow --> make it NaN */
  }
  else if ((e_f16 >  (f16_bias - f8_bias + 15)) ||
          ((e_f16 == (f16_bias - f8_bias + 15)) && (m_f16 > 0x0340)))
  {
    e = 0xf;
    m = 0x7;
    /* smaller than denormal f8 + eps */
  }
  else if (e_f16 < f16_bias - f8_bias - 3) {
    e = 0x0;
    m = 0x0;
    /* denormal */
  }
  else if (e_f16 <= f16_bias - f8_bias) {
    /* RNE */
    /* denormalized mantissa */
    m = m_f16 | 0x0400;
    /* additionally subnormal shift */
    m = m >> ((f16_bias - f8_bias) + 1 - e_f16);
    /* preserve sticky bit (some sticky bits are lost when denormalizing) */
    m |= (((m_f16 & 0x007f) + 0x007f) >> 7);
    /* RNE Round */
    fixup = (m >> 7) & 0x1;
    m = m + LIBXSMM_CAST_USHORT(0x003f + fixup);
    m = m >> 7;
    e = 0x0;
    /* normal */
  }
  else {
    /* RNE round */
    fixup = (m_f16 >> 7) & 0x1;
    in = in + LIBXSMM_CAST_USHORT(0x003f + fixup);
    e = (in & 0x7c00) >> 10;
    m = (in & 0x03ff);
    LIBXSMM_ASSERT(e >= LIBXSMM_CAST_USHORT(f16_bias - f8_bias));
    e -= LIBXSMM_CAST_USHORT(f16_bias - f8_bias);
    m = m >> 7;
  }

  /* set result to 0 */
  res = 0x0;
  /* set exponent and mantissa */
  res |= e << 3;
  res |= m;
  /* sign it */
  res |= s;

  return res;
}
#endif
/*


*/
static inline uint8_t ggml_compute_fp16_to_fp8_e4m3fn(_Float16 h) {
    const int f16_bias = 15;
    const int fp8_bias =  7;
    union {
        _Float16 h;
        uint16_t bits;
    } v;
    v.h = h;
    uint8_t  s= (v.bits&0x8000u)>> 8;
    uint8_t ex= (v.bits&0x7c00u)>>10;
    uint8_t  m;

    if (ex==0x1f){// inf or nan, 
        return s|((v.bits&0x03FFu)==0? 0x7E: 0x7F);
    } else 
    if (ex> (f16_bias+fp8_bias)){// overflow
        return s|0x7E;// MAX
    } else
    if (ex< (f16_bias-fp8_bias-3)){ // underflow
        return s|0;
    } else
    if (ex<=(f16_bias-fp8_bias)){ // denormal
// описать ... 
        uint16_t fixup = (m>>7)&1;
        m = (m + (uint16_t)(0x3f+fixup))>>7;
        ex = 0;
    } else {// normal
        uint16_t fixup = (v.bits>>7)&1;
        v.bits = v.bits + (uint16_t)(0x3fu+fixup);
        ex=((v.bits&0x7c00)>>10) - (f16_bias-fp8_bias);
        m = (v.bits&0x3FF)>>7;
    }

    return s|(ex<<3)|m;
    
}
static inline uint8_t ggml_compute_fp16_to_fp8(_Float16 s) {
    const int f16_bias = 15;
    const int fp8_bias =  7;
    union {
        _Float16 h;
        uint16_t bits;
    } v;
    v.h = s;
    uint8_t si= (v.bits&0x8000u)>> 8;
    uint8_t ex= (v.bits&0x7c00u)>>10;
    uint8_t  m= 0;
    uint16_t ma= (v.bits&0x03ffu);

    if (ex==0x1f){// inf or nan, 
        ex = 0xf;
        m  = ma==0? 0: (ma>>7)|4;
        // если e4m3fn то NaN => s.7F, INF= s.7E (MAX)
        // m  = ma==0? 0x7E: 0x7F;
    } else 
    if (ex> (f16_bias+fp8_bias)){// overflow
        ex = 0xf; m=0;
        // если e4m3fn то INF= s.7E (MAX)
    } else
    if (ex< (f16_bias-fp8_bias-3)){ // underflow
        ex=0, m = 0;
        // если e4m3fn то INF= s.0 (ZERO)
    } else 
    if (ex<=(f16_bias-fp8_bias)){ // denormal
        m = ma|0x0400;
        m = m>>((f16_bias-fp8_bias)+1 - ex);
        m|= ((ma&0x007F)+0x007F)>>7;
        /* RNE Round */
        unsigned fixup = (m>>7)&1;
        m = (m + (uint16_t)(0x3f+fixup))>>7;
        ex = 0;
    } else {// normal
        uint16_t fixup = (ma>>7)&1;
        v.bits = v.bits + (uint16_t)(0x3fu+fixup);
        ex=((v.bits&0x7c00)>>10) - (f16_bias-fp8_bias);
        m = (v.bits&0x3FF)>>7;
    }
    return si|(ex<<3)|m;
}
// https://onnx.ai/onnx/technical/float8.html
static inline uint8_t ggml_compute_fp32_to_bf8(float s) {
    const int f32_bias =127;
    const int bf8_bias = 15;// может использовать смещение 16
    union {
        float f;
        uint32_t bits;
    } v;
    v.f = s;
    uint8_t si = (v.bits&0x80000000u)>>24;
    uint8_t ex = (v.bits&0x7F800000u)>>23;
    uint8_t  m;
    uint32_t ma = (v.bits&0x007FFFFFu);
    if (ex==0xFF){// inf & nan
        m  = ma==0? 0: (ma>>21)|2;
    } else
    if (ex> (f32_bias+bf8_bias)) {// overflow
        ex = 0x1f; m = 0;
    } else
    if (ex< (f32_bias-bf8_bias-2)) {// underflow
        ex = 0; m = 0;
    } else
    if (ex<=(f32_bias-bf8_bias)) {// denornal
        m = ma | 0x00800000u;
        m = m>>((f32_bias-bf8_bias)+1 - ex);
        m|= (((ma&0x1ffffu)+0x1ffffu)>>21);
    } else 
    {// normal
        m = (v.bits + 0x7ffff + ((ma >> 20) & 1)) >> 20;
    }
    return si|(ex<<3)|m;
}
// https://patentimages.storage.googleapis.com/0d/96/d3/1ab325d9d67232/EP4318224A1.pdf

#define GGML_FP32_TO_BF16(x) ggml_compute_fp32_to_bf16(x)
#define GGML_BF16_TO_FP32(x) ggml_compute_bf16_to_fp32(x)
#define GGML_BF16_TO_FP8(x)  ggml_compute_bf16_to_fp8(x)
#define GGML_FP8_TO_BF16(x)  ggml_compute_fp8_to_bf16(x)

#define GGML_FP16_TO_BF8(x)  ggml_compute_fp16_to_bf8(x)
#define GGML_BF8_TO_FP16(x)  ggml_compute_bf8_to_fp16(x)
#define GGML_FP4_TO_FP8(x)   ggml_compute_fp4_to_fp8(x)
#define GGML_FP8_TO_FP4(x)   ggml_compute_fp8_to_fp4(x)
/* Продолжения не будет Из FP8 не сделать BF4 (E4M0) останется только экспонента */


union gguf_value {
    uint8_t  uint8;
    int8_t   int8;
    uint16_t uint16;
    int16_t  int16;
    uint32_t uint32;
    int32_t  int32;
    float    float32;
    uint64_t uint64;
    int64_t  int64;
    double   float64;
    bool     bool_;

    struct gguf_str str;

    struct {
        enum gguf_type type;

        uint64_t n;  // GGUFv2
        void * data;
    } arr;
};
struct gguf_kv {
    struct gguf_str key;

    enum  gguf_type  type;
    union gguf_value value;
};

// GGML
#include <stdbool.h>
#include <stdint.h>
#define GGML_MAX_DIMS       4 // не хотим 4
#define GGML_MAX_OP_PARAMS  64// снизить
#define GGML_MAX_SRC        2 // 4
#define GGML_MAX_NAME       64

// this tensor...
enum ggml_tensor_flag {
    GGML_TENSOR_FLAG_INPUT  =  1, // ...is an input for the GGML compute graph
    GGML_TENSOR_FLAG_OUTPUT =  2, // ...is an output for the GGML compute graph
    GGML_TENSOR_FLAG_PARAM  =  4, // ...contains trainable parameters
    GGML_TENSOR_FLAG_LOSS   =  8, // ...defines loss for numerical optimization (multiple loss tensors add up)
    GGML_TENSOR_FLAG_SHARED,
    GGML_TENSOR_FLAG_CACHE,
};
// available tensor operations:
enum ggml_op {
        GGML_OP_NONE = 0,

        GGML_OP_DUP,
        GGML_OP_ADD,
        GGML_OP_ADD1,
        GGML_OP_ACC,
        GGML_OP_SUB,
        GGML_OP_MUL,
        GGML_OP_DIV,
        GGML_OP_SQR,
        GGML_OP_SQRT,
        GGML_OP_LOG,
        GGML_OP_SIN,
        GGML_OP_COS,
        GGML_OP_SUM,
        GGML_OP_SUM_ROWS,
        GGML_OP_MEAN,
        GGML_OP_ARGMAX,
        GGML_OP_COUNT_EQUAL,
        GGML_OP_REPEAT,
        GGML_OP_REPEAT_BACK,
        GGML_OP_CONCAT,
        GGML_OP_SILU_BACK,
        GGML_OP_NORM, // normalize
        GGML_OP_RMS_NORM,
        GGML_OP_RMS_NORM_BACK,
        GGML_OP_GROUP_NORM,
        GGML_OP_L2_NORM,

        GGML_OP_MUL_MAT,
        GGML_OP_MUL_MAT_ID,
        GGML_OP_OUT_PROD,   //!< outer product

        GGML_OP_SCALE,
        GGML_OP_SET,
        GGML_OP_CPY,
        GGML_OP_CONT,
        GGML_OP_RESHAPE,
        GGML_OP_VIEW,
        GGML_OP_PERMUTE,
        GGML_OP_TRANSPOSE,
        GGML_OP_GET_ROWS,
        GGML_OP_GET_ROWS_BACK,
        GGML_OP_DIAG,
        GGML_OP_DIAG_MASK_INF,
        GGML_OP_DIAG_MASK_ZERO,
        GGML_OP_SOFT_MAX,   //!< Softmax
        GGML_OP_SOFT_MAX_BACK,
        GGML_OP_ROPE,   //!< 
        GGML_OP_ROPE_BACK,
        GGML_OP_CLAMP,
        GGML_OP_CONV_TRANSPOSE_1D,
        GGML_OP_IM2COL,     //!< 
        GGML_OP_IM2COL_BACK,
        GGML_OP_CONV_TRANSPOSE_2D,
        GGML_OP_POOL_1D,
        GGML_OP_POOL_2D,    //!< 2D pooling
        GGML_OP_POOL_2D_BACK,
        GGML_OP_UPSCALE, // nearest interpolate
        GGML_OP_PAD,
        GGML_OP_PAD_REFLECT_1D,
        GGML_OP_ARANGE,
        GGML_OP_TIMESTEP_EMBEDDING,
        GGML_OP_ARGSORT,
        GGML_OP_LEAKY_RELU,

        GGML_OP_FLASH_ATTN_EXT, //!< Flash attention
        GGML_OP_FLASH_ATTN_BACK,
        GGML_OP_SSM_CONV,
        GGML_OP_SSM_SCAN,
        GGML_OP_WIN_PART,
        GGML_OP_WIN_UNPART,
        GGML_OP_GET_REL_POS,
        GGML_OP_ADD_REL_POS,
        GGML_OP_RWKV_WKV6,  // 
        GGML_OP_GATED_LINEAR_ATTN,  // 
        GGML_OP_RWKV_WKV7,  // 

        GGML_OP_UNARY,

        GGML_OP_MAP_UNARY,
        GGML_OP_MAP_BINARY,

        GGML_OP_MAP_CUSTOM1_F32,
        GGML_OP_MAP_CUSTOM2_F32,
        GGML_OP_MAP_CUSTOM3_F32,

        GGML_OP_MAP_CUSTOM1,
        GGML_OP_MAP_CUSTOM2,
        GGML_OP_MAP_CUSTOM3,

        GGML_OP_CROSS_ENTROPY_LOSS,
        GGML_OP_CROSS_ENTROPY_LOSS_BACK,
        GGML_OP_OPT_STEP_ADAMW,

        GGML_OP_COUNT,
    };

enum ggml_unary_op {// unary operations
    GGML_UNARY_OP_ABS,
    GGML_UNARY_OP_SGN,
    GGML_UNARY_OP_NEG,
    GGML_UNARY_OP_STEP,
    GGML_UNARY_OP_TANH,
    GGML_UNARY_OP_ELU,
    GGML_UNARY_OP_RELU,
    GGML_UNARY_OP_SIGMOID,
    GGML_UNARY_OP_GELU,
    GGML_UNARY_OP_GELU_QUICK,
    GGML_UNARY_OP_SILU,
    GGML_UNARY_OP_HARDSWISH,
    GGML_UNARY_OP_HARDSIGMOID,
    GGML_UNARY_OP_EXP,

    GGML_UNARY_OP_COUNT,
};

//!< Тензоры и операции над тензорами
//!< n-dimensional tensor
typedef struct ggml_tensor tensor_t;
typedef struct _weight_tensor tensor_weight_t;
struct _weight_tensor {
    uint64_t sdnv;      //!< Self-Delimiting Variable-Length Integer, local object identifier, OID = {type, cname, idx, suffix|op } имя восстанавливается по хеш таблице quarks
    enum ggml_type type;//!< tensor  type: F32, F16, BF16, HF8, BF8, quantized: Q4_0, ... Q8_0, Q4_K, ... Q8_K, ...
    enum ggml_op   op;  //!< содержит размерность тензора n_dim при загрузке из файла GGUF, OP_NONE
    size_t ne[GGML_MAX_DIMS];   // number of elements ne[i]==1 для i>=n_dim
    uint64_t offset;    //!< смещение в байтах от начала файла
    void*    data;      //!< указатель на данные в памяти
    size_t   size;      //!< размер в байтах
};

struct ggml_tensor {
    uint64_t sdnv;      //!< Self-Delimiting Variable-Length Integer, local object identifier, OID = {type, sdnv}
    enum ggml_type type;//!< tensor  type: F32, F16, BF16, HF8, BF8, quantized: Q4_0, ... Q8_0, Q4_K, ... Q8_K, ...
    enum ggml_op op;    //!< compute opcode: mul, add, mul_mat, norm, softmax, ...
    size_t ne[GGML_MAX_DIMS];   // number of elements
    size_t nb[GGML_MAX_DIMS];   // stride in bytes:
                                // nb[0] = ggml_type_size(type)
                                // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                                // nb[i] = nb[i-1] * ne[i-1]
    int32_t flags;      //!< GGML_TENSOR_FLAG_INPUT, GGML_TENSOR_FLAG_OUTPUT, GGML_TENSOR_FLAG_CONST ...
    int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];// op params - allocated as int32_t for alignment
    struct ggml_tensor * src[GGML_MAX_SRC];//!< source tensors

    // source tensor and offset for views

    void * data;

    // const char* name; //!< tensor name, reference to dynsym (\see quarks dynsym)
    // ссылка на tensor_info или на dynsym

//        char padding[8];
};
struct ggml_context {
    size_t mem_size;    //!< total memory size
    size_t mem_offs;    //!< offset of the next object
    void * mem_buffer;  //!< pointer to the beginning of the memory buffer
    int    n_objects;   //!< number of objects in the memory buffer
    QTable_t* quarks;   //!< hesh table, хранит имена тензоров
};

/*! \brief кодирование элемента SDNV 
	\param sdnv - указатель на начало SDNV
	\param value - значение 0..2^32-1
	\return указатель на конец SDNV
 */
static inline uint8_t *  _sdnv_encode(uint8_t* sdnv, uint32_t value){
	do {
		uint8_t data = value & 0x7F;
		value>>=7;
		*sdnv++ = value? data|0x80: data;
	} while (value);
	return sdnv;
}
/*! \brief сохранение имени тензора в хеш таблицу и назначение SDNV идентификатора
    \param ctx - указатель на контекст
    \param a   - указатель на тензор
    \param cname - имя тензора, должно быть уникальным
 */
static inline void ggml_set_name (struct ggml_context *ctx, struct ggml_tensor  * a, const char* cname){
    uint32_t id = _quark_insert(ctx->quarks, cname); // добавить в таблицу имен quarks
    //printf("set name cname=%s id=%d\n", cname, id);
    uint64_t sdnv = 0;
    _sdnv_encode((uint8_t*)&sdnv, id);
    a->sdnv = sdnv;
}

static inline void ggml_set_op_params(struct ggml_tensor  * a, void * data, size_t size){
    __builtin_memcpy(a->op_params, data, size);
}

enum _name_suffix {
    _SUFFIX_NONE,
    _SUFFIX_TRANSPOSED,
    _SUFFIX_RESHAPED,
    _SUFFIX_VIEW,
    _SUFFIX_PERMUTED,
    _SUFFIX_CONT,
};
extern const char *name_suffix[];
extern void ggml_format_name(struct ggml_tensor *tensor,enum _name_suffix suffix, const struct ggml_tensor *ref);

static inline bool ggml_is_transposed(const struct ggml_tensor * tensor);
static inline bool ggml_is_permuted  (const struct ggml_tensor * tensor);
static inline bool ggml_is_empty     (const struct ggml_tensor * tensor);
static inline bool ggml_is_scalar    (const struct ggml_tensor * tensor);
static inline bool ggml_is_vector    (const struct ggml_tensor * tensor);
static inline bool ggml_is_matrix    (const struct ggml_tensor * tensor);
static inline int  ggml_n_dims       (const struct ggml_tensor * tensor); // returns 1 for scalars

static inline bool ggml_is_contiguous  (const struct ggml_tensor * tensor);
static inline bool ggml_is_contiguous_0(const struct ggml_tensor * tensor); // same as ggml_is_contiguous()
static inline bool ggml_is_contiguous_1(const struct ggml_tensor * tensor); // contiguous for dims >= 1
static inline bool ggml_is_contiguous_2(const struct ggml_tensor * tensor); // contiguous for dims >= 2
static inline int64_t ggml_nelements (const struct ggml_tensor * tensor);
static inline int64_t ggml_nrows     (const struct ggml_tensor * tensor);
static inline size_t  ggml_nbytes    (const struct ggml_tensor * tensor);
static inline size_t  ggml_nbytes_pad(const struct ggml_tensor * tensor); // same as ggml_nbytes() but padded to GGML_MEM_ALIGN

struct _type_traits {
    short type_size;
    short blck_size;
};
extern
struct _type_traits type_traits[GGML_TYPE_COUNT];

static inline int64_t ggml_blck_size(enum ggml_type type) {
    return type_traits[type].blck_size;
}

static inline size_t ggml_type_size(enum ggml_type type) {
    return type_traits[type].type_size;
}
static inline
const char * ggml_type_name(enum ggml_type type) {
    return type < GGML_TYPE_COUNT ? GGML_TYPE_NAME[type] /* type_traits[type].type_name */ : "NONE";
}

static inline
void * _slice_alloc(struct ggml_context* ctx, size_t size){
    void * data = ctx->mem_buffer + ctx->mem_offs;
    ctx->mem_offs += size;
    ctx->n_objects++;
    return data;
}
static inline 
struct ggml_tensor * ggml_tensor_new(
    struct ggml_context * ctx,
    enum   ggml_type type,
    const size_t ne[GGML_MAX_DIMS])
{
    // TODO memory_pool
    struct ggml_tensor * tensor = g_slice_new0(struct ggml_tensor);
    // struct ggml_tensor * tensor = (struct ggml_tensor *)_slice_alloc(ctx, sizeof(struct ggml_tensor));
    // __builtin_bzero(tensor, sizeof(struct ggml_tensor));
    // заполняем поля отличные от нуля, op=GGML_OP_NONE
    tensor->type = type;
    for (int i = 0; i < GGML_MAX_DIMS; i++)
        tensor->ne[i] = ne[i];
    // is_contiguous 
    tensor->nb[0] = ggml_type_size(type);
    tensor->nb[1] = tensor->nb[0]*(tensor->ne[0]/ggml_blck_size(type));
    for (int i = 2; i < GGML_MAX_DIMS; i++) {
        tensor->nb[i] = tensor->nb[i - 1]*tensor->ne[i - 1];
    }
    return tensor;
}

static inline 
void ggml_tensor_free(struct ggml_context * ctx, struct ggml_tensor * tensor) {
    for(int i=0; i<GGML_MAX_SRC; ++i){
        if (tensor->src[i]!=NULL) 
            ggml_tensor_free(ctx, tensor->src[i]);
    }
    g_slice_free(struct ggml_tensor, tensor);
}
static inline 
struct ggml_tensor * ggml_tensor_dup(struct ggml_context * ctx, const struct ggml_tensor * src) {
    return ggml_tensor_new(ctx, src->type, src->ne);
}
static inline 
struct ggml_tensor * ggml_tensor_view(struct ggml_context * ctx,const struct ggml_tensor  * src) 
{
    struct ggml_tensor * tensor = ggml_tensor_new(ctx, src->type, src->ne);
    ggml_format_name(tensor, _SUFFIX_VIEW, src);
    return tensor;
}

static inline
bool ggml_is_transposed(const struct ggml_tensor * tensor) {
    return tensor->nb[0] > tensor->nb[1];
}
static inline
bool ggml_is_permuted(const struct ggml_tensor * tensor) {
//    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");
    return /* tensor->nb[0] > tensor->nb[1] || */tensor->nb[1] > tensor->nb[2]/* || tensor->nb[2] > tensor->nb[3] */;
}
static inline
bool ggml_is_empty(const struct ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] == 0) // empty if any dimension has no elements
            return true;
    }
    return false;
}
static inline
bool ggml_is_scalar(const struct ggml_tensor * tensor) {
    return tensor->ne[0] == 1 && tensor->ne[1] == 1 && tensor->ne[2] == 1 /* && tensor->ne[3] == 1 */;
}
static inline
bool ggml_is_vector(const struct ggml_tensor * tensor) {
    return tensor->ne[1] == 1 && tensor->ne[2] == 1/* && tensor->ne[3] == 1 */;
}
static inline
bool ggml_is_matrix(const struct ggml_tensor * tensor) {
    return tensor->ne[2] == 1/* && tensor->ne[3] == 1 */;
}
// плоское представление, матрица уложена по строкам без зазоров
static inline
bool ggml_is_flat(const struct ggml_tensor * tensor) {
    //    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");
        return tensor->nb[0] == ggml_type_size(tensor->type) && tensor->nb[1] == tensor->nb[0]*tensor->ne[0] && tensor->nb[2] == tensor->nb[1]*tensor->ne[1];
                                   // nb[0] = ggml_type_size(type)
                                   // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
                                   // nb[i] = nb[i-1] * ne[i-1]
}
static bool ggml_is_contiguous_n(const struct ggml_tensor * tensor, int n) {
    size_t next_nb = ggml_type_size(tensor->type);
//    if (tensor->ne[0] != ggml_blck_size(tensor->type) && tensor->nb[0] != next_nb) return false;
    if (tensor->nb[0] != next_nb) return false;
    next_nb *= tensor->ne[0]/ggml_blck_size(tensor->type);
    if (tensor->nb[1] != next_nb) return false;
    next_nb *= tensor->ne[1];
    if (tensor->nb[2] != next_nb) return false;
    return true;
}
static bool ggml_is_contiguous_1(const struct ggml_tensor * tensor) {
    size_t next_nb = ggml_type_size(tensor->type);
//    if (tensor->ne[0] != ggml_blck_size(tensor->type) && tensor->nb[0] != next_nb) return false;
    if (tensor->nb[0] != next_nb) return false;
    next_nb *= tensor->ne[0]/ggml_blck_size(tensor->type);
    //if (tensor->nb[1] != next_nb) return false;
    next_nb *= tensor->ne[1];
    if (tensor->nb[2] != next_nb) return false;
    return true;
}
static inline
int ggml_n_dims(const struct ggml_tensor * tensor) {
    for (int i = GGML_MAX_DIMS - 1; i >= 1; --i)
        if (tensor->ne[i] > 1) return i + 1;
    return 1;
}

//static const size_t GGML_TENSOR_SIZE = sizeof(struct ggml_tensor);



// --- Тензорные операции --- более компактный и систематизированный способ описания операций
static inline
struct ggml_tensor * ggml_transpose(
    struct ggml_context * ctx,
    struct ggml_tensor  * a) 
{
    struct ggml_tensor * tensor = ggml_tensor_view(ctx, a);
    ggml_format_name(tensor, _SUFFIX_TRANSPOSED, a);

    tensor->ne[0] = a->ne[1];
    tensor->ne[1] = a->ne[0];

    tensor->nb[0] = a->nb[1];
    tensor->nb[1] = a->nb[0];

    tensor->op     = GGML_OP_TRANSPOSE;
    tensor->src[0] = a;

    return tensor;
}
static inline
struct ggml_tensor * ggml_binary_impl(
    struct ggml_context * ctx,
    enum   ggml_op op,
    struct ggml_tensor  * a,
    struct ggml_tensor  * b,
    bool                  inplace) 
{
    GGML_ASSERT(ggml_can_repeat(b, a));
    struct ggml_tensor * tensor = inplace ? ggml_tensor_view(ctx, a) : ggml_tensor_dup(ctx, a);

    tensor->op     = op;
    tensor->src[0] = a;
    tensor->src[1] = b;

    return tensor;
}
static inline
struct ggml_tensor * ggml_unary_impl(
    struct ggml_context * ctx,
    enum   ggml_op op,
    struct ggml_tensor  * a,
    bool                  inplace) 
{
    struct ggml_tensor * tensor = inplace ? ggml_tensor_view(ctx, a) : ggml_tensor_dup(ctx, a);

    tensor->op     = op;
    tensor->src[0] = a;

    return tensor;
}

static inline
struct ggml_tensor * ggml_unary_param(
    struct ggml_context * ctx,
    enum   ggml_op op,
    struct ggml_tensor  * a,
    float                 eps) 
{
    struct ggml_tensor * tensor = ggml_tensor_dup(ctx, a);

    tensor->op     = op;
    tensor->src[0] = a;

    *(float*)(tensor->op_params) = eps;
    return tensor;
}

static inline
struct ggml_tensor * ggml_unary_op_impl(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    enum ggml_unary_op    op,
    bool                  inplace) 
{
    GGML_ASSERT(ggml_is_contiguous_1(a));

    struct ggml_tensor * tensor = inplace ? ggml_tensor_view(ctx, a) : ggml_tensor_dup(ctx, a);

    tensor->op_params[0] = (int32_t) op;

    tensor->op     = GGML_OP_UNARY;
    tensor->src[0] = a;

    return tensor;
}
static inline 
struct ggml_tensor * ggml_mul_mat(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    struct ggml_tensor  * b) 
{
    GGML_ASSERT(ggml_can_mul_mat(a, b));
    GGML_ASSERT(!ggml_is_transposed(a));

    const size_t ne[4] = { a->ne[1], b->ne[1], b->ne[2], b->ne[3] };
    struct ggml_tensor * tensor = ggml_tensor_new(ctx, GGML_TYPE_F32, ne);

    tensor->op     = GGML_OP_MUL_MAT;
    tensor->src[0] = a;
    tensor->src[1] = b;

    return tensor;
}
#define ggml_dup(ctx, a)    ggml_unary_impl(ctx, GGML_OP_DUP,  a, false)
#define ggml_sqr(ctx, a)    ggml_unary_impl(ctx, GGML_OP_SQR,  a, false)
#define ggml_sqrt(ctx, a)   ggml_unary_impl(ctx, GGML_OP_SQRT, a, false)
#define ggml_log(ctx, a)    ggml_unary_impl(ctx, GGML_OP_LOG,  a, false)
#define ggml_sin(ctx, a)    ggml_unary_impl(ctx, GGML_OP_SIN,  a, false)
#define ggml_cos(ctx, a)    ggml_unary_impl(ctx, GGML_OP_COS,  a, false)
#define ggml_norm(ctx, a, eps) ggml_unary_param(ctx, GGML_OP_NORM,  a, eps)
#define ggml_l2_norm(ctx, a, eps) ggml_unary_param(ctx, GGML_OP_L2_NORM,  a, eps)
#define ggml_rms_norm(ctx, a, eps) ggml_unary_param(ctx, GGML_OP_RMS_NORM,  a, eps)
#define ggml_scale(ctx, a, scale) ggml_unary_param(ctx, GGML_OP_SCALE,  a, scale)
#define ggml_abs(ctx, a)    ggml_unary_op_impl(ctx, a, GGML_UNARY_OP_ABS,  false)
#define ggml_exp(ctx, a)    ggml_unary_op_impl(ctx, a, GGML_UNARY_OP_EXP,  false)
#define ggml_neg(ctx, a)    ggml_unary_op_impl(ctx, a, GGML_UNARY_OP_NEG,  false)
#define ggml_sgn(ctx, a)    ggml_unary_op_impl(ctx, a, GGML_UNARY_OP_SGN,  false)
#define ggml_elu(ctx, a)    ggml_unary_op_impl(ctx, a, GGML_UNARY_OP_ELU,  false)
#define ggml_gelu(ctx, a)   ggml_unary_op_impl(ctx, a, GGML_UNARY_OP_GELU, false)
#define ggml_relu(ctx, a)   ggml_unary_op_impl(ctx, a, GGML_UNARY_OP_RELU, false)
#define ggml_silu(ctx, a)   ggml_unary_op_impl(ctx, a, GGML_UNARY_OP_SILU, false)
#define ggml_step(ctx, a)   ggml_unary_op_impl(ctx, a, GGML_UNARY_OP_STEP, false)
#define ggml_sigmoid(ctx,a) ggml_unary_op_impl(ctx, a, GGML_UNARY_OP_SIGMOID, false)
#define ggml_tanh(ctx, a)   ggml_unary_op_impl(ctx, a, GGML_UNARY_OP_TANH, false)
#define ggml_gelu_quick(ctx, a)   ggml_unary_op_impl(ctx, a, GGML_UNARY_OP_GELU_QUICK, false)
#define ggml_gelu_hardswish(ctx, a)   ggml_unary_op_impl(ctx, a, GGML_UNARY_OP_HARDSWISH, false)
#define ggml_gelu_hardsigmoid(ctx, a)   ggml_unary_op_impl(ctx, a, GGML_UNARY_OP_HARDSIGMOID, false)

#define ggml_add(ctx, a,b)  ggml_binary_impl(ctx, GGML_OP_ADD, a, b, false)
#define ggml_sub(ctx, a,b)  ggml_binary_impl(ctx, GGML_OP_SUB, a, b, false)
#define ggml_mul(ctx, a,b)  ggml_binary_impl(ctx, GGML_OP_MUL, a, b, false)
#define ggml_div(ctx, a,b)  ggml_binary_impl(ctx, GGML_OP_DIV, a, b, false)

static inline
struct ggml_tensor * ggml_reshape_impl(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    const  size_t         ne[4]) 
{
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne[0]*ne[1]*ne[2]*ne[3]);

    struct ggml_tensor * tensor = ggml_tensor_new(ctx, a->type, ne);
    ggml_format_name(tensor, _SUFFIX_RESHAPED, a);

    tensor->op     = GGML_OP_RESHAPE;
    tensor->src[0] = a;

    return tensor;
}
static inline
struct ggml_tensor * ggml_view_impl(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    const size_t          ne[4],
    size_t                offset) 
{
    struct ggml_tensor * tensor = ggml_tensor_new(ctx, a->type, ne);
    ggml_format_name(tensor, _SUFFIX_VIEW, a);

    ggml_set_op_params(tensor, &offset, sizeof(offset));

    tensor->op     = GGML_OP_VIEW;
    tensor->src[0] = a;

    return tensor;
}
static inline
struct ggml_tensor * ggml_view_1d(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    size_t                ne0,
    size_t                offset) 
{
    const size_t ne[4] = {ne0,1,1,1};
    struct ggml_tensor * tensor = ggml_view_impl(ctx, a, ne, offset);

    return tensor;
}
static inline
struct ggml_tensor * ggml_view_2d(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    size_t                ne0,
    size_t                ne1,
    size_t                nb1,
    size_t                offset) 
{
    const size_t ne[4] = {ne0,ne1,1,1};
    struct ggml_tensor * tensor = ggml_view_impl(ctx, a, ne, offset);

    tensor->nb[1] = nb1;
    tensor->nb[2] = tensor->nb[1]*ne1;
    tensor->nb[3] = tensor->nb[2];
    return tensor;
}
static inline
struct ggml_tensor * ggml_view_3d(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    size_t                ne0,
    size_t                ne1,
    size_t                ne2,
    size_t                nb1,
    size_t                nb2,
    size_t                offset) 
{
    const size_t ne[4] = {ne0,ne1,ne2,1};
    struct ggml_tensor * tensor = ggml_view_impl(ctx, a, ne, offset);

    tensor->nb[1] = nb1;
    tensor->nb[2] = nb2;
    tensor->nb[3] = tensor->nb[2]*ne2;
    return tensor;
}
extern 
struct ggml_tensor * ggml_permute(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   axis0,
        int                   axis1,
        int                   axis2,
        int                   axis3);
extern 
struct ggml_tensor * ggml_get_rows(
        struct ggml_context * ctx,
        struct ggml_tensor  * a, struct ggml_tensor  * b);

static inline
struct ggml_tensor * ggml_reshape_2d(struct ggml_context * ctx, struct ggml_tensor  * a, size_t ne0, size_t ne1) 
{
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1);

    const size_t ne[4] = { ne0, ne1, 1, 1};
    struct ggml_tensor * tensor = ggml_tensor_new(ctx, a->type, ne);
    ggml_format_name(tensor, _SUFFIX_RESHAPED, a);

    tensor->op     = GGML_OP_RESHAPE;
    tensor->src[0] = a;

    return tensor;
}
static inline
struct ggml_tensor * ggml_reshape_3d(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    size_t               ne0,
    size_t               ne1,
    size_t               ne2) 
{
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2);

    const size_t ne[4] = { ne0, ne1, ne2, 1 };
    struct ggml_tensor * tensor = ggml_tensor_new(ctx, a->type, ne);
    ggml_format_name(tensor, _SUFFIX_RESHAPED, a);

    tensor->op     = GGML_OP_RESHAPE;
    tensor->src[0] = a;

    return tensor;
}
static inline
struct ggml_tensor * ggml_reshape_4d(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    size_t               ne0,
    size_t               ne1,
    size_t               ne2,
    size_t               ne3) 
{
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2*ne3);

    const size_t ne[4] = { ne0, ne1, ne2, ne3 };
    struct ggml_tensor * tensor = ggml_tensor_new(ctx, a->type, ne);
    ggml_format_name(tensor, _SUFFIX_RESHAPED, a);

    tensor->op     = GGML_OP_RESHAPE;
    tensor->src[0] = a;

    return tensor;
}

static inline 
struct ggml_tensor * ggml_cont_impl(
    struct ggml_context * ctx,
    struct ggml_tensor  * a) 
{
    struct ggml_tensor * tensor = ggml_tensor_dup(ctx, a);
    ggml_format_name(tensor, _SUFFIX_CONT, a);

    tensor->op     = GGML_OP_CONT;
    tensor->src[0] = a;

    return tensor;
}
static inline
struct ggml_tensor * ggml_cont(struct ggml_context * ctx, struct ggml_tensor * a) {
    return ggml_cont_impl(ctx, a);
}
static inline 
struct ggml_tensor * ggml_cont_4d(struct ggml_context * ctx, struct ggml_tensor  * a, const size_t ne[4]) 
{
    GGML_ASSERT(ggml_nelements(a) == (ne[0]*ne[1]*ne[2]*ne[3]));

    struct ggml_tensor * tensor = ggml_tensor_new(ctx, a->type, ne);
    ggml_format_name(tensor, _SUFFIX_CONT, a);

    tensor->op     = GGML_OP_CONT;
    tensor->src[0] = a;

    return tensor;
}

static inline struct ggml_tensor * ggml_cont_2d(struct ggml_context * ctx, struct ggml_tensor  * a, size_t ne0, size_t ne1) {
    const size_t ne[4] = {ne0,ne1,1,1};
    return ggml_cont_4d(ctx, a, ne);
}
static inline struct ggml_tensor * ggml_cont_3d(struct ggml_context * ctx, struct ggml_tensor  * a, size_t ne0, size_t ne1, size_t ne2) {
    const size_t ne[4] = {ne0,ne1,ne2,1};
    return ggml_cont_4d(ctx, a, ne);
}
enum ggml_op_pool {
    GGML_OP_POOL_MAX =0,
    GGML_OP_POOL_AVG =1
};


static inline
size_t ggml_calc_pool_output_size(size_t ins, int ks, unsigned int s, float p) {
    return (ins + 2 * p - ks) / s + 1;
}
static inline
struct ggml_tensor * ggml_pool_2d(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    enum ggml_op_pool     op,
    int                   k0,
    int                   k1,
    int                   s0,
    int                   s1,
    float                 p0,
    float                 p1) 
{

    const size_t ne[4] = {
        ggml_calc_pool_output_size(a->ne[0], k0, s0, p0),
        ggml_calc_pool_output_size(a->ne[1], k1, s1, p1),
        a->ne[2],
        a->ne[3],
    };
    struct ggml_tensor * tensor = ggml_tensor_new(ctx, GGML_TYPE_F32, ne);

    int32_t params[] = { op, k0, k1, s0, s1, p0, p1 };
    ggml_set_op_params(tensor, params, sizeof(params));

    tensor->op     = GGML_OP_POOL_2D;
    tensor->src[0] = a;

    return tensor;
}
static inline
struct ggml_tensor * ggml_soft_max_impl(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    struct ggml_tensor  * mask,
    float                 scale,
    float                 max_bias,
    bool                  inplace) 
{
    GGML_ASSERT(ggml_is_contiguous(a));

    if (mask) {
        GGML_ASSERT(mask->type == GGML_TYPE_F16 || mask->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(mask));
        GGML_ASSERT(ggml_is_matrix(mask));
        GGML_ASSERT(mask->ne[0] == a->ne[0]);
        GGML_ASSERT(mask->ne[1] >= a->ne[1]);
    }

    if (max_bias > 0.0f) {
        GGML_ASSERT(mask);
    }

    struct ggml_tensor * tensor = inplace ? ggml_tensor_view(ctx, a) : ggml_tensor_dup(ctx, a);

    float params[] = { scale, max_bias };
    ggml_set_op_params(tensor, params, sizeof(params));

    tensor->op     = GGML_OP_SOFT_MAX;
    tensor->src[0] = a;
    tensor->src[1] = mask;

    return tensor;
}
static inline
struct ggml_tensor * ggml_soft_max( struct ggml_context * ctx, struct ggml_tensor  * a) {
    return ggml_soft_max_impl(ctx, a, NULL, 1.0f, 0.0f, false);
}
static inline
struct ggml_tensor * ggml_soft_max_ext( struct ggml_context * ctx, struct ggml_tensor  * a, 
        struct ggml_tensor  * mask, float scale, float max_bias) {
    return ggml_soft_max_impl(ctx, a, mask, scale, max_bias, false);
}

GGML_API struct ggml_tensor * ggml_conv_2d(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,   // convolution kernel
    struct ggml_tensor  * b,   // data
    int                   s0,  // stride dimension 0
    int                   s1,  // stride dimension 1
    int                   p0,  // padding dimension 0
    int                   p1,  // padding dimension 1
    int                   d0,  // dilation dimension 0
    int                   d1); // dilation dimension 1
GGML_API struct ggml_tensor * ggml_im2col(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,  // convolution kernel
        struct ggml_tensor  * b,  // data
        int                   s0, // stride dimension 0
        int                   s1, // stride dimension 1
        int                   p0, // padding dimension 0
        int                   p1, // padding dimension 1
        int                   d0, // dilation dimension 0
        int                   d1, // dilation dimension 1
        bool                  is_2D,
        enum ggml_type        dst_type);
// ----------------------------
static inline
bool ggml_are_same_shape(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
//    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");
    bool same=true;
    for (int i=0; i<GGML_MAX_DIMS; ++i)
        same = same && (t0->ne[i] == t1->ne[i]);
    return same;
}
static inline
bool _are_same_stride(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
//    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");
    bool same=true;
    for (int i=0; i<GGML_MAX_DIMS; ++i)
        same = same && (t0->nb[i] == t1->nb[i]);
    return same;
}

// check if t1 can be represented as a repetition of t0
static inline
bool _can_repeat(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
//    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");
    bool same=true;
    for (int i=0; i<GGML_MAX_DIMS; ++i)
        same = same && (t1->ne[i]%t0->ne[i]==0);
    return same;
}
/*
#define GGUF_MAGIC 		"GGUF"
#define GGUF_VERSION 	3
#define GGUF_DEFAULT_ALIGNMENT 32
#define GGUF_MAX_DIMS 	4// максимальный размер в файле может отличаться от макс. размерности при вычислениях GGUF_MAX_DIMS>=GGML_MAX_DIMS

struct gguf_tensor_info {
    struct gguf_str name;

    enum ggml_type type;
    uint32_t n_dims;    // 1..4
    uint32_t ne[GGUF_MAX_DIMS];


    uint64_t offset; // offset from start of `data`, must be a multiple of `ALIGNMENT`

    // for writing API
    const void * data;
    size_t size;
};*/
typedef struct gguf_context gguf_cxt_t;
typedef struct _HTable HTable_t;

extern int gguf_find_key(const struct gguf_context * ctx, const char * key);
struct gguf_context {
    struct gguf_header header;

    struct gguf_kv          * kv;
    struct gguf_tensor_info * infos;

    size_t alignment;
    size_t offset;    // offset of `data` from beginning of file
    size_t size;      // size of `data` in bytes

    //uint8_t * padding;
    void * data;
    HTable_t * htable;
};

extern uint64_t xxh64(uint64_t hash, uint8_t* data, size_t data_len);
extern gguf_cxt_t * gguf_init_empty(void);
extern gguf_cxt_t * gguf_init_from_file(const char * fname, uint32_t params);
extern void         gguf_print_header(gguf_cxt_t * ctx);
extern void         gguf_free(gguf_cxt_t* ctx);
extern int          gguf_find_key (const gguf_cxt_t *ctx, const char *key);

extern int          gguf_get_arr_n   (const struct gguf_context * ctx, int key_id);
extern void*        gguf_get_arr_data(const struct gguf_context * ctx, int key_id);


static inline
 int32_t gguf_get_val_i32(const struct gguf_context * ctx, int key_id) {
    return ctx->kv[key_id].value.int32;
}
static inline
uint32_t gguf_get_val_u32(const struct gguf_context * ctx, int key_id) {
    return ctx->kv[key_id].value.uint32;
}
static inline
uint32_t gguf_get_val_f32(const struct gguf_context * ctx, int key_id) {
    return ctx->kv[key_id].value.float32;
}
static inline
bool     gguf_get_val_bool(const struct gguf_context * ctx, int key_id) {
    return ctx->kv[key_id].value.bool_;
}
static inline
const struct gguf_str * gguf_get_val_str(const struct gguf_context * ctx, int key_id) {
    return &ctx->kv[key_id].value.str;
}
/*
static inline
const char * gguf_get_key(const struct gguf_context * ctx, int key_id) {
    return ctx->kv[key_id].key.data;
}
*/
struct qnn_cgraph {
//    int size;    // maximum number of nodes/leafs/grads/grad_accs
    int n_nodes; // number of nodes currently in use
//    int n_leafs; // number of leafs currently in use

    struct ggml_tensor ** nodes;     // tensors with data that can change if the graph is evaluated
};

extern struct qnn_cgraph * qnn_graph_new(struct ggml_context * );

extern struct gguf_tensor_info * gguf_tensor_info(const gguf_cxt_t *ctx, const char *cname, int idx);


extern struct ggml_context* ggml_init(void * data, size_t size);
extern void ggml_free(struct ggml_context*);
#endif//_QNN_H