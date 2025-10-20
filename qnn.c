#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "qnn.h"
/*! \note Проект изначально был создан под GGML
    Добавлен ряд идей, которые отличаются от оригинальной реализации.
    1. Хеширование имен тензора (Quarks)
    2. Назначение уникальных идентификаторов тензорам (SDNV)
    3. Компактный способ определения тензорных операций - шаблоны унарных и бинарных операций
    4. Сворачивание графа тензорных операций
    5. "Тензорный ассемблер" - текстовая отладка фрагментов графа тензорных операций
    6. Реализация на языке C
    7. Неблокирующие примитивы синхронизации тредов
    8. Собственная концепция RPC-протокола, включает кодирование CBOR и идентификаторы тензоров SDNV
 */
/// @brief 
struct ggml_tensor * ggml_permute(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   axis0,
        int                   axis1,
        int                   axis2,
        int                   axis3) {
    GGML_ASSERT(axis0 >= 0 && axis0 < GGML_MAX_DIMS);
    GGML_ASSERT(axis1 >= 0 && axis1 < GGML_MAX_DIMS);
    GGML_ASSERT(axis2 >= 0 && axis2 < GGML_MAX_DIMS);
    GGML_ASSERT(axis3 >= 0 && axis3 < GGML_MAX_DIMS);

    GGML_ASSERT(axis0 != axis1);
    GGML_ASSERT(axis0 != axis2);
    GGML_ASSERT(axis0 != axis3);
    GGML_ASSERT(axis1 != axis2);
    GGML_ASSERT(axis1 != axis3);
    GGML_ASSERT(axis2 != axis3);

    struct ggml_tensor * tensor = ggml_tensor_view(ctx, a);
    ggml_format_name(tensor, _SUFFIX_PERMUTED, a);

    size_t ne[GGML_MAX_DIMS];
    size_t nb[GGML_MAX_DIMS];

    ne[axis0] = a->ne[0];
    ne[axis1] = a->ne[1];
    ne[axis2] = a->ne[2];
    ne[axis3] = a->ne[3];

    nb[axis0] = a->nb[0];
    nb[axis1] = a->nb[1];
    nb[axis2] = a->nb[2];
    nb[axis3] = a->nb[3];

    tensor->ne[0] = ne[0];
    tensor->ne[1] = ne[1];
    tensor->ne[2] = ne[2];
    tensor->ne[3] = ne[3];

    tensor->nb[0] = nb[0];
    tensor->nb[1] = nb[1];
    tensor->nb[2] = nb[2];
    tensor->nb[3] = nb[3];

    tensor->op     = GGML_OP_PERMUTE;
    tensor->src[0] = a;

    int32_t params[] = { axis0, axis1, axis2, axis3 };
    ggml_set_op_params(tensor, params, sizeof(params));

    return tensor;
}
/*! ggml_conv_2d

// операции reshape переставляют порядок но не меняют массив данных, 
    операция im2col призвана подготовить массив данных для использования  матричного умножения. 
    Надо совместить операции reshape и im2col
    a: [OC，IC, KH, KW]
    b: [N, IC, IH, IW]
    result: [N, OC, OH, OW] 
 */
struct ggml_tensor * ggml_conv_2d(
    struct ggml_context * ctx,
    struct ggml_tensor  * a,
    struct ggml_tensor  * b,
    int                   s0,
    int                   s1,
    int                   p0,
    int                   p1,
    int                   d0,
    int                   d1) {
    struct ggml_tensor * im2col = ggml_im2col(ctx, a, b, s0, s1, p0, p1, d0, d1, true, a->type); // [N, OH, OW, IC * KH * KW]

    struct ggml_tensor * result =
    ggml_mul_mat(ctx,
            ggml_reshape_2d(ctx, im2col, im2col->ne[0],  im2col->ne[3] * im2col->ne[2] * im2col->ne[1]), // [N, OH, OW, IC * KH * KW] => [N*OH*OW, IC * KH * KW]
            ggml_reshape_2d(ctx, a, (a->ne[0] * a->ne[1] * a->ne[2]),  a->ne[3]));                       // [OC，IC, KH, KW] => [OC, IC * KH * KW]

    result = ggml_reshape_4d(ctx, result, im2col->ne[1], im2col->ne[2], im2col->ne[3], a->ne[3]); // [OC, N, OH, OW]
    result = ggml_cont(ctx, ggml_permute(ctx, result, 0, 1, 3, 2)); // [N, OC, OH, OW]
    return result;
}
static int64_t ggml_calc_conv_output_size(int64_t ins, int64_t ks, int s, int p, int d) {
    return (ins + 2 * p - d * (ks - 1) - 1) / s + 1;
}
// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
// a: [OC，IC, KH, KW]
// b: [N, IC, IH, IW]
// result: [N, OH, OW, IC*KH*KW]
struct ggml_tensor * ggml_im2col(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        int                   s0,
        int                   s1,
        int                   p0,
        int                   p1,
        int                   d0,
        int                   d1,
        bool                  is_2D,
        enum ggml_type        dst_type) {
    if (is_2D) {
        GGML_ASSERT(a->ne[2] == b->ne[2]);
    } else {
        //GGML_ASSERT(b->ne[1] % a->ne[1] == 0);
        GGML_ASSERT(b->ne[1] == a->ne[1]);
        GGML_ASSERT(b->ne[3] == 1);
    }

    const size_t OH = is_2D ? ggml_calc_conv_output_size(b->ne[1], a->ne[1], s1, p1, d1) : 0;
    const size_t OW =         ggml_calc_conv_output_size(b->ne[0], a->ne[0], s0, p0, d0);

    GGML_ASSERT((!is_2D || OH > 0) && "b too small compared to a");
    GGML_ASSERT((OW > 0)           && "b too small compared to a");

    const size_t ne[4] = {
        is_2D ? (a->ne[2] * a->ne[1] * a->ne[0]) : a->ne[1] * a->ne[0],
        OW,
        is_2D ? OH : b->ne[2],
        is_2D ?      b->ne[3] : 1,
    };

    struct ggml_tensor * tensor = ggml_tensor_new(ctx, dst_type, ne);
    int32_t params[] = { s0, s1, p0, p1, d0, d1, (is_2D ? 1 : 0) };
    ggml_set_op_params(tensor, params, sizeof(params));

    tensor->op     = GGML_OP_IM2COL;
    tensor->src[0] = a;
    tensor->src[1] = b;

    return tensor;
}



struct ggml_context* ggml_init(void* shm, size_t size){
    struct ggml_context* ctx = g_new0(struct ggml_context,1);
    size_t alloc_size = 1024*sizeof(tensor_t);
    ctx->mem_buffer = _aligned_malloc(alloc_size, 1024);
    ctx->mem_offs = 0;
    ctx->mem_size = alloc_size;
    ctx->n_objects = 0;
    ctx->quarks = NULL;//_quark_new(256, 2*1024);;
    return ctx;
}
void ggml_free(struct ggml_context*  ctx){
    free(ctx);
}


/*! \brief 
 */
struct qnn_cgraph * qnn_graph_new(struct ggml_context * ctx){
    struct qnn_cgraph * gf = NULL;
    return gf;
}




/*! \brief декодирование OID 

[RFC 6256] "Using SDNVs" 3.2.  Decoding Algorithm:

o  (Initial Step) Set the result to 0.  Set an index to the first
      byte of the encoded SDNV.

o  (Recursion Step) Shift the result left 7 bits.  Add the low-order
      7 bits of the value at the index to the result.  If the high-order
      bit under the pointer is a 1, advance the index by one byte within
      the encoded SDNV and repeat the Recursion Step, otherwise return
      the current value of the result.
*/
static uint32_t sdnv_decode(uint32_t data, uint32_t* value)
{
	uint8_t msb;
	uint32_t v = 0;
	do {
		v = (v<<7)|(data&0x7F);
		msb = data&0x80;
		data>>=8; 
	} while (msb);
	*value = v; 
	return data;
}
/*! \brief кодирование идентификатора OID */
static uint8_t* sdnv_encode(uint8_t *pdu, uint32_t value){
	do {
		*pdu++ = value&0x7F | ((value>>7)?0x80:0);
	} while ((value>>=7)!=0);
	return pdu;
}

const char *name_suffix[] = {
    [_SUFFIX_TRANSPOSED]= "transposed",
    [_SUFFIX_PERMUTED]  = "permuted",
    [_SUFFIX_RESHAPED]  = "reshaped",
    [_SUFFIX_VIEW]      = "view",
    [_SUFFIX_CONT]      = "cont",
    [_SUFFIX_NONE]      = ""
};
/*! \brief добавление элемента SDNV 
	\param sdnv - указатель на начало SDNV
	\param suffix - значение 1..127
	\return указатель на конец SDNV
 */
static uint8_t *  _sdnv_suffix(uint8_t* sdnv, uint8_t suffix){
	while (sdnv[0]!=0) sdnv++;
	*sdnv++ = suffix;
	return sdnv;
}
/*! \brief формирование имени тензора с использованием суффикса

    Я не хочу использовать строки для идентификации тензора. 
    Имена должны складываться в SDNV идентификатор, из которого можно получить уникальное имя по шаблону.
    Должен дополнять cname_id.index.suffix , при выводе используется таблица имен в хэш-таблице quarks, см. ggml_ctx::quarks
    suffix = "view"|"cont"|"reshaped"|"permuted"|"transposed"
 */
void ggml_format_name(struct ggml_tensor *tensor, enum _name_suffix suffix_id, const struct ggml_tensor *src){
    tensor->sdnv = src->sdnv;
    _sdnv_suffix((uint8_t*)&tensor->sdnv, suffix_id);// пропускает заполненные элементы SDNV
}

struct _type_traits type_traits[GGML_TYPE_COUNT] = {
    [GGML_TYPE_I32]     = {.blck_size = 1, .type_size = sizeof(int32_t)},
    [GGML_TYPE_I8 ]     = {.blck_size = 1, .type_size = sizeof(int8_t)},
    [GGML_TYPE_I16]     = {.blck_size = 1, .type_size = sizeof(int16_t)},
    [GGML_TYPE_F64]     = {.blck_size = 1, .type_size = sizeof(double)},
    [GGML_TYPE_F32]     = {.blck_size = 1, .type_size = sizeof(float)},
    [GGML_TYPE_F16]     = {.blck_size = 1, .type_size = sizeof(ggml_fp16_t)},
    [GGML_TYPE_BF16]    = {.blck_size = 1, .type_size = sizeof(ggml_bf16_t)},
    [GGML_TYPE_Q2_0]    = {.blck_size = QK2_0, .type_size = sizeof(block_q2_0)},
    [GGML_TYPE_Q4_0]    = {.blck_size = QK4_0, .type_size = sizeof(block_q4_0)},
//    [GGML_TYPE_Q4_1]    = {.blck_size = QK4_1, .type_size = sizeof(block_q4_1)},
//    [GGML_TYPE_Q5_0]    = {.blck_size = QK5_0, .type_size = sizeof(block_q5_0)},
//    [GGML_TYPE_Q5_1]    = {.blck_size = QK5_1, .type_size = sizeof(block_q5_1)},
    [GGML_TYPE_Q8_0]    = {.blck_size = QK8_0, .type_size = sizeof(block_q8_0)},
    [GGML_TYPE_Q4_K]    = {.blck_size = QK_K, .type_size = sizeof(block_q4_K)},
    [GGML_TYPE_Q5_K]    = {.blck_size = QK_K, .type_size = sizeof(block_q5_K)},
    [GGML_TYPE_Q6_K]    = {.blck_size = QK_K, .type_size = sizeof(block_q6_K)},
    [GGML_TYPE_Q8_K]    = {.blck_size = QK_K, .type_size = sizeof(block_q8_K)},
};


#if 0
uint8x64_t _ternary_l2n(uint8x64_t a.p, uint8x64_t a.m){
    return _mm512_popcnt_epi8(a.p) + _mm512_popcnt_epi8(a.m);
//    return __builtin_popcount(a.p) + __builtin_popcount(a.m);
}
uint8x64_t _ternary_dot(uint8x64_t a.p, uint8x64_t a.m, uint8x64_t b.p, uint8x64_t b.m){
    return _mm512_sub_epi8(
            _mm512_popcnt_epi8((a.p & b.p) | (a.m & b.m)),
            _mm512_popcnt_epi8((a.p & b.m) | (a.m & b.p)));
}
int32_t _ternary_act(uint8x64_t v, uint8x64_t th){
    c.p = _mm512_cmpge_epi8_mask(v, thp);
    c.m = _mm512_cmple_epi8_mask(v, thm);
    return c;
}
#endif
