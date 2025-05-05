/*

Сборка 
	$ gcc -DTEST_GGUF -O3 -march=native -o test qnn_gguf.c qnn_png.c xxh64.c quarks.c `pkgconf --cflags --libs glib-2.0` -lz -lpng
	
Тестирование
	$ ./test.exe ../../llama.cpp/models/Rombos-Coder-V2.5-Qwen-14b-Q8_0.gguf -v -n blk.1.attn_q.weight -o test.png


Чего надо
	Ограничить коэффициенты при квантизации на уровне 1e-5, BF8 (E5M2)
	Реализовать кодирование и декодирование BitNet.cpp 
	Описать высоко производительный CPU+NPU backend __bf16 и __fp16
	Загрузка моделей и построение графа, экспорт графа
*/
#define FINITE_ONLY // не проверять NAN
#include "qnn.h"
#include "quarks.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <unistd.h>
#include <inttypes.h>
#include <assert.h>
#include <glib.h>


#define GGUF_MAGIC 		"GGUF"
#define GGUF_VERSION 	3
#define GGUF_DEFAULT_ALIGNMENT 32
#define GGUF_MAX_DIMS 	4// максимальный размер в файле может отличаться от макс. размерности при вычислениях GGUF_MAX_DIMS>=GGML_MAX_DIMS
// private структура должна приводиться к типу tensor_t и tensor_weight_t.
struct gguf_tensor_info {
	uint64_t sdnv;  	 //!< SDNV идентификатор имени тензора (убрать в структуру ggml_tensor)
    enum ggml_type type; //!< type of data: FP32, INT32, ...
	enum ggml_type op;   //!< compute opcode: OP_NONE
    size_t ne[GGUF_MAX_DIMS];//!< размеры тензора
    //uint32_t n_dims; -- убрал 

    uint64_t offset;	//!< смещение массива данных в файле GGUF, must be a multiple of `ALIGNMENT`, см. параметр модели `general.alignment`
	void *   data;	//!< pointer to data в памяти
    size_t 	 size;	//!< size of `data` in bytes

	struct gguf_str name;// убрать в хеш таблицу .dynstr


//    uint64_t uuid[2]; // уникальный идентификатор UUIDv5
}; 
// static_assert(offsetof(struct gguf_tensor_info,ne)==offsetof(tensor_weight_t, ne));

// static_assert(sizeof(struct gguf_tensor_info)==sizeof(tensor_weight_t));
// -------------------

struct {
	uint16_t blck_size;
	uint16_t type_size;
} type_info[GGML_TYPE_COUNT] = {
	[GGML_TYPE_F64]	={1,8},
	[GGML_TYPE_F32]	={1,4},
	[GGML_TYPE_F16]	={1,2},
	[GGML_TYPE_BF16]={1,2},
	[GGML_TYPE_I32 ]={1,4},
	[GGML_TYPE_Q8_0]={QK8_0, sizeof(block_q8_0)},
	[GGML_TYPE_Q4_0]={QK4_0, sizeof(block_q4_0)},
	[GGML_TYPE_Q4_K]={QK_K,  sizeof(block_q4_K)},
	[GGML_TYPE_Q5_K]={QK_K,  sizeof(block_q5_K)},
	[GGML_TYPE_Q6_K]={QK_K,  sizeof(block_q6_K)},
	[GGML_TYPE_Q8_K]={QK_K,  sizeof(block_q8_K)},
};
const char* GGML_TYPE_NAME[GGML_TYPE_COUNT] = {
	[GGML_TYPE_F64 ]    = "F64",
	[GGML_TYPE_F32 ]    = "F32",
	[GGML_TYPE_F16 ]    = "F16",
	[GGML_TYPE_BF16]    = "BF16",
// QNN
	[GGML_TYPE_HF8 ]    = "HF8", // E4M3 (Intel conversion rules https://www.intel.com/content/www/us/en/developer/articles/technical/introduction-to-oneapi-ml-common-extensions.html)
	[GGML_TYPE_BF8 ]    = "BF8", // E5M2
	[GGML_TYPE_E4M3FN]  = "E4M3FN",// E4M3FN
	
	[GGML_TYPE_Q4_0]    = "Q4_0",
	[GGML_TYPE_Q4_1]    = "Q4_1",
	[GGML_TYPE_Q5_0]    = "Q5_0",
	[GGML_TYPE_Q5_1]    = "Q5_1",
	[GGML_TYPE_Q8_0]    = "Q8_0",
	[GGML_TYPE_Q8_1]    = "Q8_1",
	[GGML_TYPE_Q2_K]    = "Q2_K",
	[GGML_TYPE_Q3_K]    = "Q3_K",
	[GGML_TYPE_Q4_K]    = "Q4_K",
	[GGML_TYPE_Q5_K]    = "Q5_K",
	[GGML_TYPE_Q6_K]    = "Q6_K",
	[GGML_TYPE_Q8_K]    = "Q8_K",
	[GGML_TYPE_IQ2_XXS] = "IQ2_XXS",
	[GGML_TYPE_IQ2_XS ] = "IQ2_XS",
	[GGML_TYPE_IQ3_XXS] = "IQ3_XXS",
	[GGML_TYPE_IQ1_S ]  = "IQ1_S",
	[GGML_TYPE_IQ4_NL]  = "IQ4_NL",
	[GGML_TYPE_IQ3_S ]  = "IQ3_S",
	[GGML_TYPE_IQ2_S ]  = "IQ2_S",
	[GGML_TYPE_IQ4_XS]  = "IQ4_XS",
	[GGML_TYPE_I8 ]     = "I8",
	[GGML_TYPE_I16]     = "I16",
	[GGML_TYPE_I32]     = "I32",
	[GGML_TYPE_I64]     = "I64",
	[GGML_TYPE_IQ1_M]   = "IQ1_M",
	[GGML_TYPE_TQ1_0]   = "TQ1_0",
	[GGML_TYPE_TQ2_0]   = "TQ2_0",
	[GGML_TYPE_I2_S]    = "I2_S",
	[GGML_TYPE_I8_S]    = "I8_S",
	[GGML_TYPE_TL1]     = "TL1",
	[GGML_TYPE_TL2]     = "TL2",
	[GGML_TYPE_I2_S]    = "I2_S",// BitNet
	[GGML_TYPE_TQ1_0]	= "TQ1_0",
	[GGML_TYPE_TQ2_0]	= "TQ2_0",
	};
	const size_t GGUF_TYPE_SIZE[GGUF_TYPE_COUNT] = {
		[GGUF_TYPE_UINT8 ]  = sizeof(uint8_t),
		[GGUF_TYPE_INT8  ]  = sizeof(int8_t),
		[GGUF_TYPE_UINT16]  = sizeof(uint16_t),
		[GGUF_TYPE_INT16 ]  = sizeof(int16_t),
		[GGUF_TYPE_UINT32]  = sizeof(uint32_t),
		[GGUF_TYPE_INT32 ]  = sizeof(int32_t),
		[GGUF_TYPE_FLOAT32] = sizeof(float),
		[GGUF_TYPE_BOOL  ]  = sizeof(bool),
		[GGUF_TYPE_STRING]  = sizeof(struct gguf_str),
		[GGUF_TYPE_UINT64]  = sizeof(uint64_t),
		[GGUF_TYPE_INT64 ]  = sizeof(int64_t),
		[GGUF_TYPE_FLOAT64] = sizeof(double),
		[GGUF_TYPE_ARRAY ]  = 0, // undefined
	};
/* Из FP8 (E4M3) можно сделать FP4 (E2M1) см. CUDA 10.x */
/* Понижение разрядности
	FP32 (E8M23) можно разложить на два/три BF16(E8M7) или два FP16 (E5M10)
		v2BF16.s0 = FP32_to_BF16(x) // понижение разрядности
		v2BF16.s1 = FP32_to_BF16(x-BF16_to_FP32) // _gradient_ 
	BF16 (E8M7) можно представить парой v2FP8_e4m3fn (E4M3) и множитель (E8M0):
		v2FP8.s0 = BF16_to_FP8(x, e_max) // понижение разрядности
		v2FP8.s1 = BF16_to_FP8(x-FP8_to_BF16(v2FP4.s0) , e_max+4) // остаток
   Аналогично:
	FP8 (E4M3) можно представить v2FP4 и множитель (E4M0):
		v2FP4.s0 = FP8_to_FP4(x, e_max) // понижение разрядности
		v2FP4.s1 = FP8_to_FP4(x-FP4_to_FP8(v2FP4.s0) , e_max+2) // остаток

Замечание. При проектировании алгоритмов можно использовать _gradient_ на следующем проходе (timeshift)
 в составе операции FMA или MMA. Компенсация ошибки округления! 

Вместе с понижением разрядности надо рассматривать методы BLAS
	\see <https://docs.nvidia.com/cuda/cublas/>
	\see GSF <>
	\see (LINALG.md)
Для теста надо выполнять Би-диагонализацию матрицы, QR- и LUP- разложения.
 */


// размер тензора с учетом выравнивания
static size_t _tensor_info_nbytes(struct gguf_tensor_info* info){
	return (info->ne[0]*type_info[info->type].type_size/type_info[info->type].blck_size)*info->ne[1];
}
gguf_cxt_t * gguf_init_empty(void) {
	gguf_cxt_t * ctx = g_malloc0(sizeof(struct gguf_context));
    if (!ctx) {
        fprintf(stderr, "%s: failed to allocate memory for context\n", __func__);
        return NULL;
    }

    memcpy(ctx->header.magic, GGUF_MAGIC, sizeof(ctx->header.magic));
    ctx->header.version   = GGUF_VERSION;
    ctx->header.n_tensors = 0;
    ctx->header.n_kv      = 0;

    ctx->kv    = NULL;
    ctx->infos = NULL;

    ctx->alignment = GGUF_DEFAULT_ALIGNMENT;
    ctx->offset    = 0;
    ctx->size      = 0;

    ctx->data = NULL;

    return ctx;
}
static bool gguf_fread(FILE * file, void * dst, size_t size, uint64_t *offset) {
    size_t res = fread(dst, 1, size, file);
	*offset += res;
	return res == size;
}
/*! \brief выделяет шаблон имени и индекс 
	Планируется использовать в RPC протоколе для формирования SDNV идентификаторов объектов `cname_id.index`. 
	Для идентификации объектов нужен словарь из `cname` - шаблонов имен. 
	Имя объекта восстанавливается по шаблону путем подстановки индекса в шаблон. 
 */
static int  _cname_idx(char* name){
    int index = 0;
    char* s = name;
    while (*s!='\0' && !(s[0]=='.' && isdigit(s[1]))) s++;
    if (*s!='\0'){ 
		s++;
		char * cname = s;
    	while (isdigit(*s)) index = index*10+(*s++ -'0');
		*cname++ = '*'; 
		do {*cname++ = *s++; } while (*s!='\0');// до конца строки
		return index;
	} else 
		return -1;
}
/*! \brief кодирование имени тензора в SDNV, сохраняет шаблон имени в хэш таблицу 

Критерий выделения индекса - наличие символа '.' и цифры после него. В шаблоне индекс заменяется на '*'.
Возможно тоит модифицировать под два и более индексов.

	\note В реализации Quarks используется QUARK_UNDEF=0, как признак неопределенного значения. 
	Метод _lookup() возвращает 0, если имя не найдено. В структуре хэш таблицы идентификатору 0 соответствует имя "undefined" или пустая строка.
	Значение 0 не может быть использовано в качестве идентификатора тензора.

	\param ht - указатель на хэш таблицу
	\param p - указатель на строку в формате GGUF
	\return SDNV - идентификатор содержащий {cname_id, index}, где cname_id - идентификатор шаблона имени, index - индекс слоя
 */
static uint64_t _cname_to_sdnv(QTable_t * ht, const struct gguf_str * p){
    int index = 0;
	char buf[p->n+2]; // выделить на стеке или выделить в таблице dynstr
	char* cname = buf;
    const char* s = p->data;
    const char* e = p->data+p->n;
    while (s<e && !(s[0]=='.' && isdigit(s[1]))) *cname++ = *s++;
    if (s<e){ 
		*cname++ = *s++;// '.'
    	while (s<e && isdigit(*s)) index = index*10+(*s++ -'0');
		*cname++ = '*'; 
		while (s<e) {*cname++ = *s++; } ;// до конца строки
	} else 
		index = -1;
	*cname++='\0';

	uint32_t cname_id = _quark_lookup(ht, buf);
	//printf("cname %s idx=%d %d\n", buf, index, cname_id);
	if (cname_id==QUARK_UNDEF) {
		cname_id = _quark_insert(ht, buf);
		//printf("add cname %s\n", buf);
	}
	
	uint64_t sdnv=0;
	uint8_t *v = (uint8_t *)&sdnv;
	v = _sdnv_encode(v, cname_id);
	if (index>=0) 
		v = _sdnv_encode(v, index);
	return sdnv;
}
#define STN_UNDEF 0
#define Nbucket 256
/*! \brief Структура хеш таблицы для поиска тензоров в файле GGUF */
typedef struct _HTable HTable_t;
struct _HTable {
	uint32_t nbucket;  // число признаков, ограничимся 256 например
	uint32_t nchain;   // длинный список например 1024-256
	uint32_t bucket[Nbucket];// два массива подряд =nbucket+nchain
};

// не применяется
static bool gguf_fread_cname(FILE * file, uint8_t * sdnv, uint64_t *offset, QTable_t* ht) {
	int64_t len;
	bool ok;
	ok = gguf_fread(file, &len, sizeof(len), offset);
	char buf[len+2];
	ok = ok && gguf_fread(file,  buf, len, offset);
	buf[len]='\0';

	int index = _cname_idx(buf);
	uint32_t cname_id = _quark_lookup(ht, buf);
	if (cname_id==0) cname_id = _quark_insert(ht, buf);
	uint8_t *s = sdnv;
	s = _sdnv_encode(s, cname_id);
	if (index>=0) 
		s = _sdnv_encode(s, index);
	// printf("SDNV %02X%02X%02X%02X\n", sdnv[2],sdnv[1],sdnv[0]);
	return ok;
}
static bool gguf_fread_str(FILE * file, struct gguf_str * p, uint64_t *offset) {
    p->n    = 0;
    p->data = NULL;

    bool ok;

    ok = gguf_fread(file, &p->n, sizeof(p->n), offset);

    // early exit if string length is invalid, prevents from integer overflow
    if (p->n == SIZE_MAX) {
        fprintf(stderr, "%s: invalid string length (%" PRIu64 ")\n", __func__, p->n);
        return false;
    }

    p->data = g_malloc(p->n + 1);
    if (p->data==NULL) {
        fprintf(stderr, "%s: failed to allocate memory for string of length %" PRIu64 "\n", __func__, p->n);
        return false;
    }

    ok = ok && gguf_fread(file,  p->data, p->n, offset);

    return ok;
}
/*
uint32_t gguf_get_val_u32(const struct gguf_context * ctx, int key_id) {
//    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
//    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_UINT32);
	return ctx->kv[key_id].value.uint32;
}*/
const char * gguf_get_key(const struct gguf_context * ctx, int key_id) {
//    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
	return ctx->kv[key_id].key.data;
}
int gguf_find_key(const struct gguf_context * ctx, const char * key) {
    int keyfound = -1;// return -1 if key not found
	int key_len = strlen(key);
    const int n_kv = ctx->header.n_kv;//gguf_get_n_kv(ctx);

    for (int i = 0; i < n_kv; ++i) {
        if (ctx->kv[i].key.n==key_len && strncmp(key, ctx->kv[i].key.data, key_len) == 0) {
            keyfound = i;
            break;
        }
    }

    return keyfound;
}

void gguf_free(gguf_cxt_t* ctx){
	if(ctx->header.n_tensors) 
	if(ctx->header.n_kv) 

	if(ctx->kv)
		g_free(ctx->kv);
	if(ctx->infos)
		g_free(ctx->infos);
	g_free(ctx);
}

//#define MWCx_A 0xFFEA
/*
A=FFEA i=7ff4fffe ( 21), P mod 24 =23, A mod 3 =0
A=FFD7 i=7feb7ffe ( 40), P mod 24 = 7, A mod 3 =2
A=FFBD i=7fde7ffe ( 66), P mod 24 =23, A mod 3 =0
A=FFA8 i=7fd3fffe ( 87), P mod 24 =23, A mod 3 =0
A=FF9B i=7fcd7ffe (100), P mod 24 = 7, A mod 3 =2
A=FF81 i=7fc07ffe (126), P mod 24 =23, A mod 3 =0
A=FF80 i=7fbffffe (127), P mod 24 = 7, A mod 3 =2
A=FF7B i=7fbd7ffe (132), P mod 24 =23, A mod 3 =0
A=FF75 i=7fba7ffe (138), P mod 24 =23, A mod 3 =0
A=FF48 i=7fa3fffe (183), P mod 24 =23, A mod 3 =0
A=FF3F i=7f9f7ffe (192), P mod 24 =23, A mod 3 =0
A=FF3C i=7f9dfffe (195), P mod 24 =23, A mod 3 =0
A=FF2C i=7f95fffe (211), P mod 24 = 7, A mod 3 =2
A=FF09 i=7f847ffe (246), P mod 24 =23, A mod 3 =0
A=FF03 i=7f817ffe (252), P mod 24 =23, A mod 3 =0
A=FF00 i=7f7ffffe (255), P mod 24 =23, A mod 3 =0
A=FEEB i=7f757ffe (276), P mod 24 =23, A mod 3 =0
A=FEE4 i=7f71fffe (283), P mod 24 = 7, A mod 3 =2
A=FEA8 i=7f53fffe (343), P mod 24 = 7, A mod 3 =2
A=FEA5 i=7f527ffe (346), P mod 24 = 7, A mod 3 =2
A=FEA0 i=7f4ffffe (351), P mod 24 =23, A mod 3 =0
A=FE94 i=7f49fffe (363), P mod 24 =23, A mod 3 =0
A=FE8B i=7f457ffe (372), P mod 24 =23, A mod 3 =0
A=FE72 i=7f38fffe (397), P mod 24 = 7, A mod 3 =2
A=FE4E i=7f26fffe (433), P mod 24 = 7, A mod 3 =2
A=FE30 i=7f17fffe (463), P mod 24 = 7, A mod 3 =2
A=FE22 i=7f10fffe (477), P mod 24 =23, A mod 3 =0
A=FE15 i=7f0a7ffe (490), P mod 24 = 7, A mod 3 =2
A=FE04 i=7f01fffe (507), P mod 24 =23, A mod 3 =0

*/
//#define MWCx_A 0xFE04
//#define MWCx_A 0xFE22 18
//#define MWCx_A 0xFF3C // 17 90
#define MWCx_A 0xFF75 // 13 66 66
//#define MWCx_A 0xFF81
//#define MWCx_A 0xFFA8
//#define MWCx_A 0xFE15 18
// #define MWCx_A 0xFEA5 19

uint32_t mwc32_hash (uint8_t *data, int length, const uint32_t A){
	//uint32_t P = (A<<16)-1;
	uint32_t h = 0xFFFF;
	for (int i=0 ; i < length ; i++){
		// h =  h*31u + data[i];// Спасибо Qwen'у за подсказку -- не использую
		h =  h + data[i];
		h = (h&0xFFFF)*A + (h>>16);
	}
    return h;
}

#define MWC_A0 0xfffeb81bULL
static uint64_t mwc64_hash(uint64_t seed, const uint8_t *data, size_t size)
{
    uint64_t s = seed;
    for (size_t i = 0; i<size; i++){
        uint64_t x = s + data[i];
        s = MWC_A0*(uint32_t)(x) + (x>>32);
    }
    return s;
}
static
void _htable_init(HTable_t *htable, uint32_t nbucket) 
{
	htable->nbucket = nbucket;
	htable->nchain = 0;
	uint32_t i;
	for (i=0; i<nbucket; i++){
		htable->bucket[i]=STN_UNDEF;
	}
}
static
HTable_t * _htable_new(uint32_t nchain) 
{
    HTable_t *htable = (HTable_t *)malloc(sizeof(HTable_t) + nchain * sizeof(uint32_t));
    _htable_init(htable, Nbucket) ;
    return htable;
}


static
int32_t _htable_lookup_tensor_info(HTable_t *htable, const struct gguf_str *name, struct gguf_tensor_info * infos)
{
//	uint32_t key = fnv_hash(name->data, name->n);
	uint32_t key = mwc32_hash(name->data, name->n, MWCx_A);
//	uint32_t key = mwc64_hash(0, name->data, name->n);
	uint32_t y = htable->bucket[key % (htable->nbucket)];
    const uint32_t *chain = htable->bucket + htable->nbucket;
	while (y<htable->nchain && y!=STN_UNDEF) {
		if (name->n==infos[y].name.n && strncmp(infos[y].name.data, name->data, name->n)==0) 
			return y;//infos[y].value;
		y = chain[y];
	}
	return -1;
}

static int hyst[Nbucket]={0};
static
uint32_t _htable_insert_tensor_info(HTable_t *htable, const struct gguf_str *name)
{
//	uint32_t key = fnv_hash(name->data, name->n);
	uint32_t key = mwc32_hash(name->data, name->n, MWCx_A);
//	uint32_t key = mwc64_hash(0, name->data, name->n);
	hyst[key%Nbucket]++;
	uint32_t *chain = htable->bucket + htable->nbucket;
	uint32_t y = htable->nchain++;
	uint32_t* head = &htable->bucket[key % htable->nbucket];
	chain[y] = *head;
	*head = y;
	return y;
}
/*! \brief 
	Составляет таблицу информации по файлу

 */
gguf_cxt_t * gguf_init_from_file(const char * fname, uint32_t/* struct gguf_init_params */params) {
    FILE * file = fopen(fname, "rb");
    if (file==NULL) {
        fprintf(stderr, "%s: failed to open '%s': '%s'\n", __func__, fname, strerror(errno));
        return NULL;
    }

    // offset from start of file
    uint64_t	offset = 0;
    int res;

    gguf_cxt_t * ctx = g_malloc0(sizeof(struct gguf_context));
    if (!ctx) {
        fprintf(stderr, "%s: failed to allocate memory for context\n", __func__);
        fclose(file);
        return NULL;
    }

	res = gguf_fread(file, &ctx->header, sizeof(ctx->header), &offset);
    // read the header
    {
        ctx->kv    = NULL;// key value pairs
        ctx->infos = NULL;
        ctx->data  = NULL;
		
        if (ctx->header.version == 1 ) {
            fprintf(stderr, "%s: GGUFv1 is no longer supported. please use a more up-to-date version\n", __func__);
            fclose(file);
            gguf_free(ctx);
            return NULL;
        }
        // sanity-checks to prevent from integer/buffer overflows

        res = res && (ctx->header.n_tensors < (SIZE_MAX/2)/sizeof(struct gguf_tensor_info));
//        res  = res &&  (ctx->header.n_tensors < (SIZE_MAX/2)/ggml_tensor_overhead());
        res = res && (ctx->header.n_kv      < (SIZE_MAX/2)/sizeof(struct gguf_kv));

        if (!res) {
            fprintf(stderr, "%s: failed to read header\n", __func__);
            fclose(file);
            gguf_free(ctx);
            return NULL;
        }
    }

    // read the key-value pairs
    {
        const uint64_t n_kv = ctx->header.n_kv;

        ctx->kv = g_malloc0(n_kv* sizeof(struct gguf_kv));

        for (uint64_t i = 0; i < n_kv; ++i) {
            struct gguf_kv * kv = &ctx->kv[i];

//            fprintf(stderr, "%s: reading kv %d\n", __func__, i);

            gguf_fread_str(file, &kv->key,                 &offset);
			uint32_t type = 0;
            gguf_fread (file, &type, sizeof(type), &offset);
			kv->type = type;

//            fprintf(stderr, "%s: %3d| %-.29s|\n", __func__, i, kv->key.data);

            switch (kv->type) {
			case GGUF_TYPE_UINT8:   
			case GGUF_TYPE_INT8:    
			case GGUF_TYPE_UINT16:  
			case GGUF_TYPE_INT16:   
			case GGUF_TYPE_UINT32:  
			case GGUF_TYPE_INT32:   
			case GGUF_TYPE_FLOAT32: 
			case GGUF_TYPE_UINT64:  
			case GGUF_TYPE_INT64:   
			case GGUF_TYPE_FLOAT64: 
			case GGUF_TYPE_BOOL:    
				res = res &&  gguf_fread(file, &kv->value.int32, gguf_type_size(kv->type),   &offset); break;
			case GGUF_TYPE_STRING:  
				res = res &&  gguf_fread_str(file, &kv->value.str, &offset); break;
			case GGUF_TYPE_ARRAY:
				{
					res = res &&  gguf_fread(file, &kv->value.arr.type, sizeof(kv->value.arr.type), &offset);
					res = res &&  gguf_fread(file, &kv->value.arr.n,    sizeof(kv->value.arr.n),    &offset);

					switch (kv->value.arr.type) {
					case GGUF_TYPE_UINT8:
					case GGUF_TYPE_INT8:
					case GGUF_TYPE_UINT16:
					case GGUF_TYPE_INT16:
					case GGUF_TYPE_UINT32:
					case GGUF_TYPE_INT32:
					case GGUF_TYPE_FLOAT32:
					case GGUF_TYPE_UINT64:
					case GGUF_TYPE_INT64:
					case GGUF_TYPE_FLOAT64:
					case GGUF_TYPE_BOOL:
						{

							kv->value.arr.data = g_malloc(kv->value.arr.n* gguf_type_size(kv->value.arr.type));
							if (!kv->value.arr.data) {
								fprintf(stderr, "%s: failed to allocate memory for array\n", __func__);
								fclose(file);
								gguf_free(ctx);
								return NULL;
							}

							res = res && gguf_fread(file, kv->value.arr.data, kv->value.arr.n * gguf_type_size(kv->value.arr.type), &offset);
						} break;
					case GGUF_TYPE_STRING:
						{
							kv->value.arr.data = g_malloc(kv->value.arr.n* sizeof(struct gguf_str));
							if (!kv->value.arr.data) {
								fprintf(stderr, "%s: failed to allocate memory for array\n", __func__);
								fclose(file);
								gguf_free(ctx);
								return NULL;
							}

							for (uint64_t j = 0; j < kv->value.arr.n; ++j) {
								res = res && gguf_fread_str(file, &((struct gguf_str *) kv->value.arr.data)[j], &offset);
							}
						} break;
					case GGUF_TYPE_ARRAY:
					default:
						{
							fprintf(stderr, "%s: invalid array type %d\n", __func__, kv->value.arr.type);
							res = false;
						} break;
					}
				} break;
			default:
				{
					fprintf(stderr, "%s: invalid type %d\n", __func__, kv->type);
					res = 0;
				} break;
            }

            if (!res) {
                break;
            }
        }

        if (!res) {
            fprintf(stderr, "%s: failed to read key-value pairs\n", __func__);
            fclose(file);
            gguf_free(ctx);
            return NULL;
        }
    }
	fprintf(stdout, "info offset: %u kB\n", (offset/1024));
    // read the tensor infos
    if (ctx->header.n_tensors > 0) {
		HTable_t* ht = _htable_new(ctx->header.n_tensors);
		ctx->htable = ht;
        ctx->infos = g_malloc0(ctx->header.n_tensors * sizeof(struct gguf_tensor_info));
        if (!ctx->infos) {
            fprintf(stderr, "%s: failed to allocate memory for tensor infos\n", __func__);
            fclose(file);
            gguf_free(ctx);
            return NULL;
        }

        for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
            struct gguf_tensor_info * info = &ctx->infos[i];

            for (int j = 0; j < GGUF_MAX_DIMS; ++j) info->ne[j] = 1;
// сразу в струтуру quarks записать
            res = res && gguf_fread_str(file, &info->name,                       &offset);
            //res = res && gguf_fread_cname(file, (uint8_t)&info->sdnv,              &offset);
			uint32_t n_dims = 0;
            res = res && gguf_fread (file, &n_dims, sizeof(n_dims),  &offset);

            res = res && (n_dims <= GGUF_MAX_DIMS);
			uint32_t j;
            for (j = 0; j < n_dims; ++j) {
				uint64_t ne = 0;
                res = res && gguf_fread(file, &ne, sizeof(ne), &offset);
				info->ne[j] = ne;
            }
			//for (; j < GGUF_MAX_DIMS; ++j) 	info->ne[j] = 1;
			uint32_t type = 0; // enum может кодироваться в меньшее число байт
            res = res && gguf_fread (file, &type,   sizeof(type),    &offset);
			info->type = type;
            res = res && gguf_fread (file, &info->offset, sizeof(info->offset),  &offset);

            // make sure there is no duplicated tensor names
			int32_t id = _htable_lookup_tensor_info(ht, &info->name, ctx->infos);
			if (id>=0) {res = false; 
				fprintf(stderr, "%s: tensor already exists #%d\n", __func__, i);
			}
			_htable_insert_tensor_info(ht, &info->name);

			if (0) for (uint64_t j = 0; j < i && res; ++j) {
                if (info->name.n==ctx->infos[j].name.n && strncmp(info->name.data, ctx->infos[j].name.data, info->name.n) == 0) {
                    fprintf(stderr, "%s: duplicated tensor name %s\n", __func__, info->name.data);
                    res = false;
                }
            }

            if (!res) {
                fprintf(stderr, "%s: failed to read tensor info #%d\n", __func__, i);
                fclose(file);
                gguf_free(ctx);
                return NULL;
            }
        }
		// тестирование хеш функции - построить гистограмму
		printf("=== HTABLE HYST ===\n");

		int hh[16] = {0};
		for(int k=0;k<Nbucket;k++) 
		if (ht->bucket[k]!=STN_UNDEF) {
			// uint8_t y = ht->bucket[k];
			// printf("%3d:%2d '%*s'\n", k, hyst[k], ctx->infos[y].name.n, ctx->infos[y].name.data);
			hh[hyst[k]%16] ++;
		} else hh[0] ++;
		printf("=== HTABLE H HYST ===\n");
		for(int k=0;k<16;k++) {
			if (hh[k]!=0) 
				printf("%3d:%2d\n", k, hh[k]);
		}

		//free(ht);
		//_Exit(1);
    }
fprintf(stdout, "data offset: %ld kB\n", (uint32_t)(ctx->offset/1024));

    ctx->alignment = GGUF_DEFAULT_ALIGNMENT;

    int alignment_idx = gguf_find_key(ctx, "general.alignment");
    if (alignment_idx != -1) {
        ctx->alignment = gguf_get_val_u32(ctx, alignment_idx);
    }

    // we require the data section to be aligned, so take into account any padding
    {
        const size_t offset_pad = offset % ctx->alignment;
        if (offset_pad != 0) {
            offset += ctx->alignment - offset_pad;
            fseeko(file, offset, SEEK_SET);
        }
    }
    // store the current file offset - this is where the data section starts
    ctx->offset = offset;
    // compute the total size of the data section, taking into account the alignment
    // load the tensor data only if requested
//	fprintf(stdout, "data offset: %d kB\n", ctx->offset/1024);
    fclose(file);

    return ctx;
}
/*! \brief построить таблицу имен */
QTable_t* gguf_quarks(gguf_cxt_t* ctx) {
	QTable_t* qt = _quark_new(256, 512);
	for(int i=0; i< ctx->header.n_tensors; i++){
		struct gguf_tensor_info * info = &ctx->infos[i];
		uint64_t sdnv = _cname_to_sdnv(qt, &info->name);
		info->sdnv = sdnv;
		//printf ("SDNV %04llx\n", sdnv);
	}
	_quark_to_csv(qt);
	return qt;
}
/*! \brief вывод в терминал структуры файла */
void gguf_print_header(gguf_cxt_t* ctx){
	for(uint64_t i=0; i<ctx->header.n_kv; ++i){
		struct gguf_kv *kv = &ctx->kv[i];
        fprintf(stdout, "%3d| %-36.*s|", i, kv->key.n, kv->key.data);
		switch(kv->type){
		case GGUF_TYPE_BOOL:
			fprintf(stdout, "%s\n", kv->value.bool_?"true":"false"); break;
		case GGUF_TYPE_UINT8:
		case GGUF_TYPE_UINT16:
		case GGUF_TYPE_UINT32:
		case GGUF_TYPE_UINT64:
			fprintf(stdout, "%llu\n", kv->value.uint64); break;
		case GGUF_TYPE_INT8:
		case GGUF_TYPE_INT16:
		case GGUF_TYPE_INT32:
		case GGUF_TYPE_INT64:
			fprintf(stdout, "%lld\n", kv->value.int64); break;
		case GGUF_TYPE_STRING:
			fprintf(stdout, "\"%-.*s\"\n", kv->value.str.n, kv->value.str.data); break;
		case GGUF_TYPE_FLOAT32:
			fprintf(stdout, "%f\n", kv->value.float32); break;
		case GGUF_TYPE_FLOAT64:
			fprintf(stdout, "%g\n", kv->value.float64); break;
		case GGUF_TYPE_ARRAY: {
			fprintf(stdout, "%d[",kv->value.arr.n);
			switch (kv->value.arr.type){
			case GGUF_TYPE_BOOL:
				for (int k=0, offs=0;k< kv->value.arr.n && k<10; k++ )
					fprintf(stdout, " %s", ((uint8_t *) kv->value.arr.data)[k]?"1":"0");
				break;
			case GGUF_TYPE_UINT8:
				for (int k=0, offs=0;k< kv->value.arr.n && k<10; k++ )
					fprintf(stdout, " %u", ((uint8_t *) kv->value.arr.data)[k]);
				break;
			case GGUF_TYPE_INT8:
				for (int k=0, offs=0;k< kv->value.arr.n && k<10; k++ )
					fprintf(stdout, " %d", ((int8_t *) kv->value.arr.data)[k]);
				break;
			case GGUF_TYPE_UINT16:
				for (int k=0, offs=0;k< kv->value.arr.n && k<10; k++ )
					fprintf(stdout, " %u", ((uint16_t *) kv->value.arr.data)[k]);
				break;
			case GGUF_TYPE_INT16:
				for (int k=0, offs=0;k< kv->value.arr.n && k<10; k++ )
					fprintf(stdout, " %d", ((int16_t *) kv->value.arr.data)[k]);
				break;
			case GGUF_TYPE_UINT32:
				for (int k=0, offs=0;k< kv->value.arr.n && k<10; k++ )
					fprintf(stdout, " %u", ((uint32_t *) kv->value.arr.data)[k]);
				break;
			case GGUF_TYPE_INT32:// этот вариант
				for (int k=0, offs=0;k< kv->value.arr.n && k<10; k++ )
					fprintf(stdout, " %d", ((int32_t *) kv->value.arr.data)[k]);
				break;
			case GGUF_TYPE_FLOAT32:// этот вариант
				for (int k=0, offs=0;k< kv->value.arr.n && k<10; k++ )
					fprintf(stdout, " %f", ((float *) kv->value.arr.data)[k]);
				break;
			case GGUF_TYPE_UINT64:
				for (int k=0, offs=0;k< kv->value.arr.n && k<10; k++ )
					fprintf(stdout, " %lld", ((uint64_t *) kv->value.arr.data)[k]);
				break;
			case GGUF_TYPE_STRING:
				for (int k=0, offs=0;k< kv->value.arr.n && k<10; k++ ){
					struct gguf_str *str =&((struct gguf_str *) kv->value.arr.data)[k];
					fprintf(stdout, " \"%-.*s\"", str->n, str->data);
				}
				break;
			default: 
				fprintf(stdout, "..."); break;
			}
			fprintf(stdout, "]\n"); break;
		} break;
		default: 
			fprintf(stdout, "\n"); break;
		}
	}
}
#undef  GGML_ASSERT
#define GGML_ASSERT(x) if (!(x)) _Exit(0);
enum llama_token_type {
    LLAMA_TOKEN_TYPE_UNDEFINED    = 0,
    LLAMA_TOKEN_TYPE_NORMAL       = 1,
    LLAMA_TOKEN_TYPE_UNKNOWN      = 2,
    LLAMA_TOKEN_TYPE_CONTROL      = 3,
    LLAMA_TOKEN_TYPE_USER_DEFINED = 4,
    LLAMA_TOKEN_TYPE_UNUSED       = 5,
    LLAMA_TOKEN_TYPE_BYTE         = 6,
};
void gguf_vocab_token_types(gguf_cxt_t* ctx){
	int key = gguf_find_key(ctx, "tokenizer.ggml.token_type");
	int key_tokens = gguf_find_key(ctx, "tokenizer.ggml.tokens");
	GGML_ASSERT(key>=0);
	struct gguf_kv *kv = &ctx->kv[key];
	struct gguf_kv *ts = &ctx->kv[key_tokens];

	uint32_t hist[8]={0};
	uint32_t n_tokens = kv->value.arr.n;
//	printf("type=%d\n", kv->value.arr.type);
	GGML_ASSERT(kv->value.arr.type==GGUF_TYPE_INT32);
	const uint32_t * ttypes = kv->value.arr.data;
	for(uint32_t i=0;i<n_tokens; i++){
		unsigned tt = ttypes[i];
		if (tt<8)
			hist[tt]++;
		if (tt==LLAMA_TOKEN_TYPE_CONTROL) {
			struct gguf_str *str =&((struct gguf_str *) ts->value.arr.data)[i];
			printf(" `%-.*s`", str->n, str->data);
		}
	}
	printf("=== T.TYPE HYST [%d]===\n", n_tokens);
	for (int i=0; i<8; i++) printf("%d: %d\n", i, hist[i]);
}
static int gguf_debug(gguf_cxt_t* ctx){
	gguf_print_header(ctx);
	fprintf(stdout, "2. model info:\n");
	fprintf(stdout, "| %-30.30s| %-6s| %s  | \n", "Name", "quants", "dims");
	fprintf(stdout, "|:---     |:--- |:--- |\n");
	char buf[32];
	for(uint64_t i=0; i<ctx->header.n_tensors; ++i){
		struct gguf_tensor_info * info = &ctx->infos[i];
		const char* type_name = GGML_TYPE_NAME[info->type];
		if (type_name==NULL) {
			sprintf(buf, "%d", info->type);
			type_name = buf;
		}
		fprintf(stdout, "| %-30.*s| %-6s| 0x%010llx ", info->name.n,info->name.data, (type_name!=NULL? type_name: "??"), info->offset);
		char ch='[';
		int n_dims = info->ne[3]==1? info->ne[2]==1? info->ne[1]==1? 1: 2: 3: 4;
		for (uint32_t j = 0; j < n_dims; ++j, ch=',') {
			fprintf(stdout, "%c%d", ch, info->ne[j]);
		}
		fprintf(stdout, "]\n" ); 
	}
	// Статистика по типу квантизации и по 
	fprintf(stdout, "num tensors: %ld\n", ctx->header.n_tensors);
	fprintf(stdout, "data offset: %ld kB\n", (uint32_t)(ctx->offset/1024));
}


// Загрузить таблицу хэшей
int gguf_hash_load(gguf_cxt_t * ctx_gguf, char* data, size_t size)
{
	char* s = data;
	char* e = data + size;
	char* name = NULL;
	while(s<e && s[0]!='\0'){
		if(strncmp(s, "xxh64", 5)==0){
			s+=6;
			while (isspace(*s)) s++;
			uint64_t hash = strtoull(s, &s, 16);
			while (s[0]!=':' && s[0]!='\0') s++;
			if(s[0]==':') {
				s++;
				name = s;
				while (isalnum(*s) || *s=='.'|| *s=='_') s++;
				int len = s - name;
				// проверить SDNV и загрузить хэш
				struct gguf_str str = {len, name};
				int y = _htable_lookup_tensor_info(ctx_gguf->htable, &str, ctx_gguf->infos);
				if (y>=0) {
					// ctx_gguf->infos[y].hash[0] = hash;// gguf_tensor_info
					uint64_t sdnv = ctx_gguf->infos[y].sdnv;
					printf("xxh64: %016llx #%04x :`%.*s`\n", hash, sdnv, len, name);	
				} else 
					fprintf(stderr, "xxh64: `%.*s` not found\n", len, name);
			}
		} else if (strncmp(s, "sha256", 6)==0){
		} else {
			fprintf(stderr, "\n");
			_Exit(1);
		}
		// до конца строки
		while(s[0]!='\0' && s[0]!='\n') s++;
		if (s[0]=='\n') s++;
	}
	return 0;
}


// ----------------------
typedef struct _MainOptions MainOptions;
struct _MainOptions {
	char*  input_file;
	char* output_file;
	char* group;
	char* device;
	char* name;// имя параметра для вывода в файл

	int   overwrite;
	int   verbose;
	int   version;
};
static MainOptions options= {
    .output_file = NULL,

};
static GOptionEntry entries[] =
{
  { "input",  	'i', 0, G_OPTION_ARG_FILENAME, &options.input_file,  "input filename", 	"*.gguf"},
  { "output", 	'o', 0, G_OPTION_ARG_FILENAME, &options.output_file, "output filename", "*.gguf" },
  { "name", 	'n', 0, G_OPTION_ARG_STRING,   &options.name, "name", "blk.*.attn_k.weight" },

  { "overwrite",'O', 0, G_OPTION_ARG_NONE, &options.overwrite, "overwtite output", NULL },
  { "verbose", 	'v', 0, G_OPTION_ARG_NONE, &options.verbose, "Be verbose", NULL },
  { "version", 	'V', 0, G_OPTION_ARG_NONE, &options.version, "program info", NULL },
  { NULL }
};
extern int write_png(char *file_name, uint8_t* image, int width, int height);


static inline void get_scale_min_k4(int j, const uint8_t * restrict q, uint8_t * restrict d, uint8_t * restrict m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

float dequantize_row_q4_K(const block_q4_K * restrict x, float * restrict y, int64_t k) 
{
//    assert(k % QK_K == 0);
    const int nb = k / QK_K;

	float y_max = 0;
    for (int i = 0; i < nb; i++) {
        const uint8_t * q = x[i].qs;

        const float d   = GGML_FP16_TO_FP32(x[i].d);//GGML_FP16_TO_FP32
        const float min =-GGML_FP16_TO_FP32(x[i].dmin);//GGML_FP16_TO_FP32

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
			if (d1>y_max) y_max = d1;
			if (d2>y_max) y_max = d2;
            for (int l = 0; l < 32; ++l) *y++ = d1*(q[l] & 0xF) + m1;
            for (int l = 0; l < 32; ++l) *y++ = d2*(q[l]  >> 4) + m2;
            q += 32; is += 2;
        }
    }
	return y_max;
}

void  dequantize_row_q5_K(const block_q5_K * restrict x, float * restrict y, int64_t k) {
//    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t * ql = x[i].qs;
        const uint8_t * qh = x[i].qh;

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

void dequantize_row_q4_0(const block_q4_0 * restrict x, float * restrict y, int64_t k) {
    static const int qk = QK4_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk/2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >>   4) - 8;

            y[i*qk + j + 0   ] = x0*d;
            y[i*qk + j + qk/2] = x1*d;
        }
    }
}
float dequantize_row_q8_0(const block_q8_0 * restrict x, float * restrict y, int64_t k) {
    static const unsigned int qk = QK8_0;

//    assert(k % qk == 0);
	uint32_t zero_blk=0;
    const unsigned int nb = k / qk;

	float d_max = -INFINITY;
	float d_min =  INFINITY;
//	printf("d(0)=%f %f %f...\n", x[0].d, x[1].d, x[2].d);
    for (unsigned i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
		if(d>d_max) d_max = d;
		if(d<d_min) d_min = d;
		if(d==0.0f) zero_blk++;
        for (unsigned j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
//	printf("d(%d/%d)=%f %f %f...\n", zero_blk,nb, GGML_FP16_TO_FP32(x[0].d), GGML_FP16_TO_FP32(x[1].d), GGML_FP16_TO_FP32(x[2].d));
	printf("d(%d/%d)=%f %f ...\n", zero_blk,nb, d_min, d_max);
	return d_max;
}


void quantize_row_q8_0_ref(const float * restrict x, block_q8_0 * restrict y, int64_t k) {
    assert(k % QK8_0 == 0);
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < QK8_0; j++) {
            const float v = x[i*QK8_0 + j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(d);

        for (int j = 0; j < QK8_0; ++j) {
            const float x0 = x[i*QK8_0 + j]*id;

            y[i].qs[j] = roundf(x0);
        }
    }
}
void dequantize_row_q8_0_ref(const block_q8_0 * restrict x, float * restrict y, int64_t k) {
    static const int qk = QK8_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
}

static inline int32_t nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);

    float val = fval + 0xC00000;//12582912.f;// 
    int32_t i;
	__builtin_memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

/* Теория 
1. квантизация выполняется по блокам 32 значения или по плитке 16x16 (K тип). Квантизация может быть двумерная.
2. в приоритете формат кодирования u8 i8 E4M3FN (FP8) или E5M2 (BF8) существуют варианты формата E4M3FNUZ и E5M2FNUZ,  
которые используют NaN и не используют INF, FN - финитную арифметику c насыщением вместо INF
Для хранения экспонент применяется беззнаковый формат E8M0, который можно реализовать в языке C методом ldexp(1.0, ex)
Типы имеют нормализованное представление
Кодирование может выполняться по таблице.
* Формат Q8_0 - кодирование i8[32] и d E8M7 (BF16), E5M10 (FP16) или (E8M0)
* Формат Q8_1 - кодирование u8[32] и d, dmin - смещение типы данных E8M7 (BF16) и E5M10 (FP16)
* Формат f8_1 - кодирование f8[32] и e(E8M0) f8=E4M3FN ; ExpBias = 7
* Формат f8_2 - кодирование v2f8[32] и e (E8M0) f8=E4M3FN ; ExpBias = 7 - замена для f16 и bf16
* Формат bf8  - применяется для градиентов (не использует INF), использует насыщение (clamp)

* <https://onnx.ai/onnx/technical/float8.html> 

Эффективно операции u8 i8 fp8 bf8 предполагают использование матричного умножения WMMA и векторных операций с упакованными типами DOT, конвертацию CVT.
в системе команд RDNA4 и CDNA3. У Intel присутствуют в системе команд PVC (Xe-HPC). 
*/
/* Метод квантизации по блоку 32 элемента i8[32] и масштабный множитель (E8M0) */
void    quantize_row_q8_K(const float * restrict x, block_q8_K * restrict y, int64_t k) {
    assert(k % Q8K_K == 0);
    const int nb = k / Q8K_K;
	int32_t imax = -__INT32_MAX__;
	int32_t imin = __INT32_MAX__;
	int32_t xmax = -__INT32_MAX__;
	int32_t xmin = __INT32_MAX__;
	
	int32_t imsk = 0;
	int l;
    for (int i = 0; i < nb; i++) {// по блокам

        int emax = __FLT_MIN_EXP__;
        int emin = __FLT_MAX_EXP__;
		int ex;
		unsigned esum=0;
        for (int j = 0; j < Q8K_K; ++j) {
			float e4m7 = frexpf(x[j],&ex);
			esum+=ex;
			if (ex > emax) emax=ex;
			if (ex < emin) emin=ex;
        }
//		printf("{%d,%d}",emax, emin);
		int Q = 7;
        for (int j = 0; j < Q8K_K; ++j) {
/*			float 	 e8m7 = to_BF16(x[j],&ex);// оно и так BF16 (E8M7)
			uint16_t e4m7 = to_FP12(e8m7, emax);// 12 bit
			uint8_t  e4m3_h = to_FP8_e4m3fn(e4m7, &ex);
			uint8_t  e4m3_l = to_FP8_e4m3fn(e4m7 - to_FP12(e4m3_h, ex));
*/			
			int v = nearest_int(ldexpf(x[j], Q-emax)-1.f/(1<<Q));
            y[i].qs[j] = v;//MIN((int)((1<<(Q-1))-1), v);// - применить насыщение
			if (v>imax) imax = v;
			if (v<imin) imin = v;
			imsk |= v;
			float e4m7 = frexpf(ldexpf(x[j], Q-emax),&ex);
			if (ex>xmax) xmax = ex;
			if (ex<xmin) xmin = ex;
		}
        y[i].d = ldexpf(1.f, emax-Q);//1/iscale;
		y[i].ex= emax-Q;

		//printf("%d ",emax);
        x += Q8K_K;
    }
	printf ("imax = %x, imin=%x imsk=%x xmax=%d xmin=%d\n", (unsigned)imax, (unsigned)imin, imsk, xmax, xmin);
}
// Варианты квантизации для семплера CL_SNORT_INT8 и CL_UNORM_INT8
int8_t convert_f32_to_i8_sat_rte(float x){
#if defined(FINITE_ONLY)
	return roundf(fmaxf(-127.0f, fminf(127.0f, x)));
#else
	return isnan(x)? 0: roundf(fmaxf(-127.0f, fminf(127.0f, x)));
#endif
}
uint8_t convert_f32_to_u8_sat_rte(float x){
#if defined(FINITE_ONLY)
	return roundf(fmaxf(0.f, fminf(255.f, x)));
#else
	return isnan(x)? 0: roundf(fmaxf(0.f, fminf(255.f, x)));
#endif
}
#define Q8R_K 32//2 4 8 16 32 -- число строк в плитке
static 
void  quantize_tile_q8_T(const float * restrict x, int8_t * qs, ggml_fp16_t * d, unsigned lda) {
    for (int i = 0; i < Q8R_K; ++i){
		float vmax = -INFINITY;
		for (int j = 0; j < Q8K_K; ++j)
			vmax = fmaxf(vmax, fabsf(x[i*lda+j]));
		
		_Float16 di = vmax/127.f;//GGML_FP16_TO_FP32(d[i]);
		d[i] = (ggml_fp16_t)(di);
		for (int j = 0; j < Q8K_K; ++j)
			*qs++ = convert_f32_to_i8_sat_rte(x[i*lda+j]/di);
	}
}
static 
void  quantize_tile_q8_1(const float * restrict x, uint8_t * qs, ggml_fp16_t * d, unsigned lda) {
    for (int i = 0; i < Q8R_K; ++i){
		float vmax = -INFINITY;
		float vmin = +INFINITY;
		for (int j = 0; j < Q8K_K; ++j){
			vmax = fmaxf(vmax, x[i*lda+j]);
			vmin = fminf(vmin, x[i*lda+j]);
		}
		_Float16 dmax = (vmax-vmin)/255.f;//GGML_FP16_TO_FP32(d[i]);
		_Float16 dmin = vmin;//GGML_FP16_TO_FP32(d[i]);
		d[2*i+0] = (ggml_fp16_t)(dmax);
		d[2*i+1] = (ggml_fp16_t)(dmin);
		for (int j = 0; j < Q8K_K; ++j)
			*qs++ = convert_f32_to_u8_sat_rte((x[i*lda+j]-vmin)/(vmax-vmin)*255.f);
	}
}
static 
void  dequantize_tile_q8_T(float *  y, const int8_t * qs, ggml_fp16_t * d, unsigned lda) 
{
    for (int i = 0; i < Q8R_K; ++i)
        for (int j = 0; j < Q8K_K; ++j)
            y[i*lda+j] = (d[i]) *qs[i*Q8K_K+j];
}
static 
void  dequantize_tile_q8_1(float *  y, const uint8_t * qs, ggml_fp16_t * d, unsigned lda) 
{
    for (int i = 0; i < Q8R_K; ++i)
        for (int j = 0; j < Q8K_K; ++j)
            y[i*lda+j] = d[2*i] *qs[i*Q8K_K+j]+d[2*i+1];
}
static
void convert_matrix_f32_to_q8_1(float*a, uint8_t * qs, unsigned height, unsigned width, unsigned lda)
{
	ggml_fp16_t* d = (ggml_fp16_t*)(qs + lda*height);
	for (int i = 0; i < height/Q8R_K; ++i){
		for (int j = 0; j < width/Q8K_K; ++j){
			quantize_tile_q8_1(a+i*Q8R_K*lda + Q8K_K*j, qs, d, lda);
			qs+= Q8R_K*Q8K_K; d+= 2*Q8R_K;
		}
	}
}
static
void convert_matrix_q8_1_to_f32(float*a, uint8_t * qs, unsigned height, unsigned width, unsigned lda)
{
	ggml_fp16_t* d = (ggml_fp16_t*)(qs + lda*height);
	for (int i = 0; i < height/Q8R_K; ++i)
		for (int j = 0; j < width/Q8K_K; ++j){
			dequantize_tile_q8_1(a+i*Q8R_K*lda + Q8K_K*j, qs, d, lda);
			qs+= Q8R_K*Q8K_K; d+= 2*Q8R_K;
		}
}
static
void convert_matrix_f32_to_q8_T(float*a, int8_t * qs, unsigned height, unsigned width, unsigned lda)
{
	ggml_fp16_t* d = (ggml_fp16_t*)(qs + lda*height);
	for (int i = 0; i < height/Q8R_K; ++i){
		for (int j = 0; j < width/Q8K_K; ++j){
			quantize_tile_q8_T(a+i*Q8R_K*lda + Q8K_K*j, qs, d, lda);
			qs+= Q8R_K*Q8K_K; d+= Q8R_K;
		}
	}
}
static
void convert_matrix_q8_T_to_f32(float*a, int8_t * qs, unsigned height, unsigned width, unsigned lda)
{
	ggml_fp16_t* d = (ggml_fp16_t*)(qs + lda*height);
	for (int i = 0; i < height/Q8R_K; ++i)
		for (int j = 0; j < width/Q8K_K; ++j){
			dequantize_tile_q8_T(a+i*Q8R_K*lda + Q8K_K*j, qs, d, lda);
			qs+= Q8R_K*Q8K_K; d+= Q8R_K;
		}
}
void print_tile(float*y, unsigned ib, unsigned jb, unsigned lda){
	y+= ib*Q8R_K*lda+jb*Q8K_K;
    for (int i = 0; i < Q8R_K; ++i){
        for (int j = 0; j < Q8K_K; ++j)
            printf ("%5.2f", y[i*lda+j]);// = d[i] *qs[i*Q8K_K+j];
		printf("\n");
	}
}

void print_tile_i8(int8_t*ds, unsigned ib, unsigned jb, unsigned lda){
	ds+= ib*Q8R_K*Q8K_K+jb*Q8K_K;
    for (int i = 0; i < Q8R_K; ++i){
        for (int j = 0; j < Q8K_K; ++j){
            printf ("%4d", ds[0]);
			ds++;
		}
		printf("\n");
	}
}

void  dequantize_row_q8_K(const block_q8_K * restrict x, float * restrict y, int64_t k) {
    assert(k % Q8K_K == 0);
    const int nb = k / Q8K_K;
    for (int i = 0; i < nb; i++)
        for (int j = 0; j < Q8K_K; ++j)
            *y++ = x[i].d *x[i].qs[j];
}
/*! 
	\todo оптимизацию под AVX512_BF16 и простой вариант под AVX512F
 */
static inline
void dequantize_row_bf16(ggml_bf16_t* src, float* dst, size_t k){
	for(int i=0; i<k; ++i)
		dst[i] = GGML_BF16_TO_FP32(src[i]);
}
static inline
void dequantize_row_f16(ggml_fp16_t* src, float* dst, size_t k){
	for(int i=0; i<k; ++i)
		dst[i] = GGML_FP16_TO_FP32(src[i]);
}
static inline
uint8_t convert_f32_to_f8_e4m3fn(float x, int emax){
	const int f32_bias = 127;
	const int  f8_bias = 7;// может быть 8
	union { uint32_t u; float f; } v;
	v.f=x;
	uint8_t s = (v.u>>24)&0x80;
	int e = (v.u>>23)&0xFF;
	uint32_t m= (v.u & 0x7FFFFF);
	if (e==0xFF){// inf or NaN
		return s|0x7F;
// e = 0xf
// m = (m==0)?0:(m>>20)|4; -- qNaN
	}
	e-= emax;
	if (e > (f32_bias+f8_bias) && m>0x700000){// overflow
		//printf("OVF %d\n", e);
		return s|0x7E;//MAX - saturate
// e = 0xf
// m = 0 - inf
	}
	if (e < (f32_bias-f8_bias-3)){// underflow
		return s|0;//Zero
// e = 0, m = 0
	}
	if (e <=(f32_bias-f8_bias)){// subnormal
		uint32_t 
		m1 = m | 0x800000u;
		m1 = m1 >>((f32_bias - f8_bias)+1 -e);
		m1|=(((m &0xFFFFF)+0xFFFFF)>>20);
		m = m1;
		e = 0; 
	} else
		e -= (f32_bias-f8_bias);
	/* round RNE */
	uint32_t fixup = (m>>20)&1;
	m = (m + 0x7ffffu+fixup)>>20;
	return s|(e<<3)|m;
}
static inline
uint8_t convert_f32_to_f8_e4m3(float x, int emax){
	const int f32_bias = 127;
	const int  f8_bias = 7;// может быть 8
	union { uint32_t u; float f; } v;
	v.f=x;
	uint8_t s = (v.u>>24)&0x80;
	int e = (v.u>>23)&0xFF;
	uint32_t m= (v.u & 0x7FFFFF);
	if (e==0xFF){// inf or NaN
		m = (m==0)?0:(m>>20)|4;// -- qNaN
		e = 0xf;
		return s|(e<<3)|m;
	}
	e-= emax;
	if (e > (f32_bias+f8_bias) && m>0x700000){// overflow
		m = 0x0;// - inf
		e = 0xf;
		return s|(e<<3)|m;
	}
	if (e < (f32_bias-f8_bias-3)){// underflow
		e = 0, m = 0;
		return s|(e<<3)|m;//Zero
	}
	if (e <=(f32_bias-f8_bias)){// subnormal
		uint32_t 
		m1 = m | 0x800000u;
		m1 = m1 >>((f32_bias - f8_bias)+1 -e);
		m1|=(((m &0xFFFFF)+0xFFFFF)>>20);
		m = m1;
		e = 0; 
	} else
		e -= (f32_bias-f8_bias);
	/* round RNE */
	uint32_t fixup = (m>>20)&1;
	m = (m + 0x7ffffu+fixup)>>20;
	return s|(e<<3)|m;
}
/*! 
	При конвертации выполняется смещение экспоненты
 */
static float convert_f8_e4m3fn_to_f32(uint32_t x, int emax){
	const int f32_bias = 127;
	const int  f8_bias = 7;

	union { uint32_t u; float f; } v;
	v.u =   (x&0x80)<<24;// sign
	if ((x&0x7F)==0x7F) {// qNaN
		v.u|=0x7FC00000;
		return v.f;
	} 
	if ((x&0x78)==0) {// subnormal
		int m = (x&7);
		if (m==0) return v.f;// zero
		int ex = (__builtin_clz(m)-28);
		x = m<<ex;
		v.u|= (f32_bias-f8_bias-ex)<<23;
		v.u+=(x<<20);
		return v.f;
	}
	v.u|= (((x&0x78)>>3) + (f32_bias-f8_bias)+emax)<<23;
	v.u|=   (x&0x07)<<20;
	return v.f;
}
static
void quantize_row_f8_e4m3fn(const float* v, uint8_t * q, size_t k, int emax){
	for (int i=0; i<k; ++i)
		q[i] = convert_f32_to_f8_e4m3fn(v[i], emax);
}
static
void dequantize_row_f8_e4m3fn(float* v, uint8_t * q, size_t k, int emax){
	for (int i=0; i<k; ++i)
		v[i] = convert_f8_e4m3fn_to_f32(q[i], emax);
}
#if defined(_WIN32)
    // use FILE * so we don't have to re-open the file to mmap
#include <windows.h>

uint8_t* blk_load(const char* path,  struct gguf_str * name, uint64_t offset, size_t size ){
	FILE* file = fopen(path, "rb");
	HANDLE fp_win32;
	fp_win32 = (HANDLE) _get_osfhandle(_fileno(file));
	
	if (file==NULL) {
		printf(" fopen failed (%d) %s 0x%llx\n", errno, strerror(errno), offset);
	}
//	int res = _fseeki64(file, (int64_t)offset, SEEK_SET);

	LARGE_INTEGER li;
	li.QuadPart = offset;
	BOOL ret = SetFilePointerEx(fp_win32, li, NULL, SEEK_SET);

	if (!ret) {
		printf(" seek failed (%d) %s\n", errno, strerror(errno));
	}
	uint8_t *blk = g_malloc(size);
	uint64_t offs = 0;
	do{
		size_t chunk = (size-offs)<64*1024*1024? size-offs: 64*1024*1024;
		DWORD chunk_read = 0;
		BOOL result = ReadFile(fp_win32, blk + offs, chunk, &chunk_read, NULL);
		offs+= chunk_read;
		//fread (blk+offs,1,chank, file);
	} while(size>offs);
	fclose(file);
	// расчитать хеш
	uint64_t hash = 0;
	hash = xxh64(hash, blk, size);
	printf("xxh64\t%llx\t%*s\n", hash, name->n, name->data);
	return blk;
}
#endif
#define Ftype float
/*! Вычисление скалярного произведения от двух колонок матрицы
	\param m - число строк
	\param n - число столбцов, m x n размер матрицы
	\param j - номер колонки
	\param k - номер колонки
 */
static Ftype dot_col(const Ftype *a, int m, int n, int j, int k)
{
	Ftype s=0;
	for(int i=0; i<m; i++) s = fmaf(a[i*n + j], a[i*n + k], s);
	return s;
}
static Ftype dot_row(const Ftype *A, unsigned m, unsigned n, unsigned j, unsigned k)
{
	const Ftype *a = A + n*j;
	const Ftype *b = A + n*k;
	Ftype s=0;
	for(unsigned i=0; i<m; i++) s = fmaf(a[i], b[i], s);
	return s;
}
static
void scal_col(Ftype d, Ftype* a, unsigned M, unsigned N, unsigned lda, int k){
	for (int j=0; j<M; ++j)
		a[j*lda+k] *= d;
}
static
void scal_row(Ftype d, Ftype* a, unsigned M, unsigned N, unsigned lda, int k){
	for (int j=0; j<N; ++j)
		a[k*lda+j] *= d;
}
// TODO сделать балансировку блочную по плиткам Tile 16x16
/*! \brief Балансировка матрицы D^{-1}AD
	\param A матрица MxN
	\param D диагональные элементы матрицы D 
	\param M число строк, 
	\param N число столбцов
	\param lda число столбцов исходной матрицы, по которой делается выборка блоков

	\note Балансировка работает только со степенями и не вносит ошибку квантизации
	Балансировка позволяет понижать размерность

	\todo заменить цыкл while (ldexpf()..) p++ на функцию 
		frexpf(, &ex_c) ; frexpf(, &ex_r) ; 
		p = (ex_c+ex_r+1)>>1; - есть такая операция в GPU

	\todo Синтезировать двумерную операцию квантизации из балансировки
 */
void balance(Ftype* A, short *D, unsigned M, unsigned N, unsigned lda){
	const float Q = 0.99;// степень балансировки, показатель качества
	Ftype c,r,s,f;
	int p;
	for (unsigned i=0; i<N; i++) D[i]=0;
	int converged=0;
	while( converged==0 ){
		converged = 1;
		for (unsigned i=0; i<N; i++){
			r = c = - fabsf(A[i*lda+i]); 
			for (unsigned j=0; j<N; j++){// 1-norm по строке и колонке, выделить функцию nrm
				//if (i!=j)
				{
					c += fabsf(A[j*lda+i]);
					r += fabsf(A[i*lda+j]);
				}
			}
			if (c==0 || r==0) continue;
			s = c+r;
			p = 0;
			if (c<r)
				while (ldexpf(c,p+1) < ldexpf(r,-p)) p++;
			else
				while (ldexpf(c,p-1) > ldexpf(r,-p)) p--;

			if ((ldexpf(c,p) + ldexpf(r,-p))<Q*s){
				converged = 0;
				if (p!=0){
					f = ldexpf(1.0f, p);
					D[i] += p;// можно сохранить только степень `p` E8M0 формат или как BF16 (E8M7)
					// printf("f=%f\n", f);
					scal_row(1/f, A, M,N, lda, i);
					scal_col(  f, A, M,N, lda, i);
				}
			}
		}
	}
}

/*! \brief Расбалансировка матрицы. Восстановление исходной 
	\todo для прямоугольных матриц
 */
void unbalance(Ftype* A, short *D, unsigned M, unsigned N, unsigned lda){
	for (unsigned i=0; i<N; i++)
	{
		Ftype f = ldexpf(1.0f, D[i]);
		scal_col(1/f, A, M,N, lda, i);
		scal_row(  f, A, M,N, lda, i);
	}
}
/*! Копирование матрицы */
void m_copy(float* data1_f32, float* data_f32, unsigned height, unsigned width, unsigned lda){
	unsigned i,j, N = lda;
	for(i=0; i<height; i++)
	for(j=0; j< width; j++){
		data1_f32[i*N+j] = data_f32[i*N+j];
	}
	
}
/*! Сравнение матриц */
float m_cmp(float* data1_f32, float* data_f32, unsigned height, unsigned width, unsigned lda){
	unsigned i,j, N = lda;
	float s = 0;
	for(i=0; i<height; i++)
	for(j=0; j< width; j++){
		s += data1_f32[i*N+j] - data_f32[i*N+j];
	}
	return s;
}
/*! Вычисление средне-квадратичной ошибки RMSE */
float vec_rmse(const float * src0, const float * src1, size_t k){
	double rmse=0;
	for (int i=0; i< k; ++i){
		float diff = (src0[i]-src1[i]);
		rmse += diff*diff;
	}
	return sqrt(rmse/k);
}
/*! Вычисление максимальной ошибки*/
float vec_maxe(const float * src0, const float * src1, size_t k){
	float maxe=0;
	float eps=1e-15;
	for (int i=0; i< k; ++i){
		if (!isnan(src0[i])&&!isnan(src1[i])){
			float diff = (src0[i]-src1[i]);
			maxe = fmaxf(fabsf(diff), maxe);
		}
	}
	return maxe;
}
static inline int isfinite_f16(uint16_t x){
	return (x&0x7C00)!=0x7C00;
}
static inline int isnormal_f16(uint16_t x){
	return (x&0x7C00)!=0x7C00 && (x&0x7C00)!=0;
}
static inline int issubnormal_f16(uint16_t x){
	return (x&0x7C00)==0;
}
static inline int frexp_f16(_Float16 x, int*ex){
	const int f16_bias = 15;
	union {uint16_t u; _Float16 f; } v;
	v.f = x;
	if ((v.u&0x7FFF)==0){
		*ex = 0;
		return 0;
	} else
	if ((v.u&0x7C00)==0){// subnormal
		int m = (v.u&0x3FF);
		*ex = (22-__builtin_clz(m)) - f16_bias+1;
	} else
		*ex = ((v.u&0x7C00)>>10) - f16_bias+1;
	return x;// число в интервале [1/2;1)

}
static int tile_metrics_f16( _Float16 * blk, unsigned M, unsigned N, size_t lda){
	int emax= -127;
	int emin= +128;
	float vmin = __FLT16_MAX__;
	float vmax = -0.0001;
	for (int i=0; i<M; ++i)
	for (int j=0; j<N; ++j)
	{
		float v = GGML_FP16_TO_FP32(blk[i*lda+j]);
		int ex;
		frexp_f16(blk[i*lda+j], &ex);
//		printf("Tmax %5.3g %d %5.3g\n", v, ex, ldexpf(v, 8-ex));
//		frexpf(v, &ex);
		if (ex<emin) emin=ex;
		if (ex>emax) emax=ex;
		if (v>vmax) vmax = v;
		if (v<vmin) vmin = v;
	}
	float v = (ldexpf(vmax, 8-emax));
	float u = (ldexpf(vmin, 8-emax));
//	printf("Tmax %5.1f %5.1f emax=%d\n", v, u, emax);
	return emax-emin;
}

/* методика - перебрать все значения */
static int test_f8_e4m3fn(){
	for(unsigned i=0; i<0x100; i++){
/*		if (isnan_f8_e3m3fn(i)){
			printf("%02X NaN\n", i);
		} */
		float v = convert_f8_e4m3fn_to_f32(i, 0);
		uint8_t j = convert_f32_to_f8_e4m3fn(v, 0);
		if (j!=i) 
			printf("%02X %02X  %g %s\n", i, j, v, i==j?"ok":"fail");
	}
	return 0;
}
#ifdef TEST_GGUF
int main (int argc, char *argv[]){
	GError* error=NULL;
	GOptionContext *opt_context;
    opt_context = g_option_context_new ("- command line interface");
    g_option_context_add_main_entries (opt_context, entries, NULL/*GETTEXT_PACKAGE*/);
    if (!g_option_context_parse (opt_context, &argc, &argv, &error))
    {
        printf ("option parsing failed: %s\n", error->message);
        _Exit(1);
    }
    g_option_context_free (opt_context);

	test_f8_e4m3fn();


	if (argc<2) return 0;
	char *path = argv[1];
	gguf_cxt_t *ctx = gguf_init_from_file(path, 0);

	gguf_vocab_token_types(ctx); // статистика токенов
	if (options.verbose) gguf_debug(ctx);
	// найти матрицу и показать
	if (options.name!=NULL && options.output_file!=NULL && g_str_has_suffix(options.output_file, ".png")) {
		struct gguf_tensor_info * info = NULL;
		int i;
		int name_len = strlen(options.name);
		for(i=0; i<ctx->header.n_tensors; ++i){// поиск мнформацаии о тензоре
			info = &ctx->infos[i];
			if (name_len == info->name.n && strncmp(info->name.data, options.name, info->name.n)==0) 
				break;
		}
		
		if (i==ctx->header.n_tensors) {
			printf("Tensor '%s' not found\n", options.name);
		} else {// тензор найден
			size_t height = info->ne[0], width = info->ne[1];
			const char* type_name = GGML_TYPE_NAME[info->type];
			size_t size = _tensor_info_nbytes(info);
			printf("Tensor '%s' %s %zd x %zd offs=0x%llx size=%d kB\n", options.name, type_name, width, height, info->offset, size/1024);
			float* data_f32 = g_malloc(sizeof(float)*width*height );
			//for (uint64_t i = 0; i<height; i++)
			// выполнить деквантизацию.. во внутренний формат f32
			if (info->type==GGML_TYPE_F32){
				info->size = (sizeof(float)*width*height);
				float*  blk = (float*)blk_load(path, &info->name, info->offset+ctx->offset, info->size);
				// анализ может это TF32?
				uint32_t mask= 0;
				int emax= __FLT16_MIN_EXP__;
				int emin= __FLT16_MAX_EXP__;
				double err=0;
				double rmse=0;
				for (int j=0; j<height;++j)
				for (int i=0; i<width;++i){
					float v = blk[j*width+i];
					mask |= *(uint32_t*)&v;
					ggml_bf16_t bf = GGML_FP32_TO_BF16(v);
					float x = v - GGML_BF16_TO_FP32(bf);
					rmse += x*x;
					int ex;
					(void)frexpf(v, &ex);
					if (ex>emax) {emax = ex; }
					if (ex<emin) {emin = ex; }
					err += fabsf(v-(_Float16)(v));
				}
				printf ("mask =0x%08X %s qerr=%g\n", mask, (mask&0xFFFF)==0?"BF16 as F32":"F32", err);
				printf ("rmse =%4.3g BF16 min exp = %d max exp = %d \n", sqrt(rmse/(width*height)), emin, emax);
				block_q8_K* qblk = g_malloc(sizeof(block_q8_K)*((width*height)/Q8K_K));
				quantize_row_q8_K(blk, qblk, (width*height)&~(QK_K-1));
				dequantize_row_q8_K(qblk, data_f32, (width*height)&~(QK_K-1));
				rmse = vec_rmse(data_f32, blk, (width*height)&~(QK_K-1));
				printf ("rmse =%4.3g Q8_1\n", rmse);
				g_free(qblk);
				block_q8_0* qblk0 = g_malloc(sizeof(block_q8_0)*((width*height)/QK8_0));
				quantize_row_q8_0_ref(blk, qblk0, (width*height)&~(QK_K-1));
				dequantize_row_q8_0_ref(qblk0, data_f32, (width*height)&~(QK_K-1));
				rmse = vec_rmse(data_f32, blk, (width*height)&~(QK_K-1));
				printf ("rmse =%4.3g Q8_0\n", rmse);
				_Exit(0);
			} else
			if (info->type==GGML_TYPE_F16){
				info->size = (sizeof(_Float16)*width*height);
				_Float16*  blk = (_Float16*)blk_load(path, &info->name, info->offset+ctx->offset, info->size);
				uint32_t mask= 0;
				int normal = 1;
				int finite = 1;
				int emax= -128;
				int emin= +127;
				float vmax=0, vmin=0;
				float rmse=0;
				float maxe=0;
				for (int j=0; j<height;++j)
				for (int i=0; i<width;++i){
					uint16_t v = *(uint16_t*)(blk+j*width+i);
					normal &= isnormal_f16(v);//нормальные числа
					finite &= isfinite_f16(v);//конечные числа
					int ex;
					float x = GGML_FP16_TO_FP32(blk[j*width+i]);
					if (!isfinite(x)) printf("!FN");
					float z = frexpf(x, &ex);
					if (z==0) continue;
//					frexp_f16(blk[j*width+i], &ex);
					if (ex>emax) {emax = ex;}
					if (ex<emin) {emin = ex;}
					if ( x>vmax) {vmax = x;}
					if ( x<vmin) {vmin = x;}
					ggml_bf16_t bf = GGML_FP32_TO_BF16(x);
					mask |= bf.bits;
					x -= GGML_BF16_TO_FP32(bf);
					rmse+=x*x;
					x = fabsf(x);
					if (x>maxe) maxe = x;
				}
				printf ("mask =0x%08X %s %s\n", mask, (mask&0xF)==0?"BF16 as F16":"F16", normal?"normal":finite?"finite":"");
				printf ("min exp = %d, max exp = %d, range (%f , %f) BF16 rmse=%g maxerr=%g\n", 
					emin, emax, vmin, vmax, sqrt(rmse/(width*height)), maxe);
				float* blkf32 = g_malloc(sizeof(float)*(width*height));
				dequantize_row_f16((void*)blk, blkf32, width*height);
				if (emax-emin<16 || 1){// можно представить в формате E4M3
					//printf ("suggest F8 E4M3FN\n");
					size_t lda=width;
					short *d = g_malloc(sizeof(short) *(width>height?width:height));
					if (0)
					for (int i=0;i<height; i+=16)
					for (int j=0;j<width; j+=16){
						balance(blkf32+i*lda+j, d, 16, 16, lda);
						//unbalance(blkf32+i*lda+j, d, 16, 16, lda);
						if (0){
							for (int k=0; k<16;++k) printf("%2d", (signed)d[k]);
							//if (j%64==63) 
							printf("\n");
						}
					}

					uint8_t* blkf8 = g_malloc(sizeof(uint8_t)*(width*height));
					quantize_row_f8_e4m3fn(blkf32, blkf8, width*height, emax);
					dequantize_row_f8_e4m3fn(data_f32, blkf8, width*height, emax);
					float rmse = vec_rmse(data_f32, blkf32, (width*height));
					float maxe = vec_maxe(data_f32, blkf32, (width*height));
					printf ("E4M3FN\trmse =%4.3g,  max err =%4.3g\n", rmse, maxe);
					g_free(blkf8);

					int8_t* blkq8 = g_malloc(sizeof(int8_t)*(width*height)+sizeof(ggml_fp16_t)*(width*height/Q8K_K));
					convert_matrix_f32_to_q8_T(  blkf32, blkq8, height, width, width);
					convert_matrix_q8_T_to_f32(data_f32, blkq8, height, width, width);
					rmse = vec_rmse(data_f32, blkf32, (width*height));
					maxe = vec_maxe(data_f32, blkf32, (width*height));
					printf ("Q8_T\trmse =%4.3g,  max err =%4.3g\n", rmse, maxe);
					g_free(blkq8);

					uint8_t* blkq8_1 = g_malloc(sizeof(uint8_t)*(width*height)+sizeof(ggml_fp16_t)*2*(width*height/Q8K_K));
					convert_matrix_f32_to_q8_1(  blkf32, blkq8_1, height, width, width);
					convert_matrix_q8_1_to_f32(data_f32, blkq8_1, height, width, width);
					rmse = vec_rmse(data_f32, blkf32, (width*height));
					maxe = vec_maxe(data_f32, blkf32, (width*height));
					printf ("Q8_1\trmse =%4.3g,  max err =%4.3g\n", rmse, maxe);
/*					print_tile(blkf32,  0, 0, width);
					printf ("Q8_1\n");
					print_tile(data_f32,0, 0, width);
 */
					g_free(blkq8_1);

				}
				int dmax=0;
				size_t lda = width; int M=32, N=32;
				for (int jb=0;jb<height/M; jb++)
				for (int ib=0;ib<width /N; ib++){
					int d = tile_metrics_f16(blk+ jb*width*M+ib*N, M, N, lda);
					if (d>dmax) dmax=d;
				}
				printf ("tiles %dx%d max d exp=%d\n", M,N, dmax);
				if (dmax<=2){// подходит квант Q8_0 и Q6_K

				}
				_Exit(0);
			} else
			if (info->type==GGML_TYPE_BF16){
				info->size = (sizeof(uint16_t)*width*height);
				ggml_bf16_t*  blk = (ggml_bf16_t*)blk_load(path, &info->name, info->offset+ctx->offset, info->size);
				uint32_t mask= 0;
				double err=0;
				double max_err=0;
				for (int j=0; j<height;++j)
				for (int i=0; i<width;++i){
					ggml_bf16_t x;
					x.bits = *(uint16_t*)(blk+j*width+i);
					mask |= x.bits;
					float v = GGML_BF16_TO_FP32(x);
					float e = v==0?0:fabsf((v-(_Float16)v)/v);
					err += e;
					if (e>max_err) max_err = e;// абсолютная максимальная ошибка
				}
				printf ("mask =0x%04X %s avg_err=%4.2g max_err=%4.2g\n", mask, (mask&0x3)==0?"BF16 as F16":"F32", err/(width*height), max_err);
				float* blkf32 = g_malloc(sizeof(float)*(width*height));
				// преобразовать из bf16 в f32
				dequantize_row_bf16(blk, blkf32, width*height);
				// выполнить квантизацию блока, сформировать отчет
				block_q8_K* qblk = g_malloc(sizeof(block_q8_K)*((width*height)/Q8K_K));
				quantize_row_q8_K(blkf32, qblk, (width*height)&~(QK_K-1));
				dequantize_row_q8_K(qblk, data_f32, (width*height)&~(QK_K-1));
				float rmse = vec_rmse(data_f32, blkf32, (width*height)&~(QK_K-1));
				printf ("rmse =%4.3g Q8_1\n", rmse);
				g_free(qblk);
				//_Exit(0);
			} else
			if (info->type==GGML_TYPE_Q8_0){
				info->size = (sizeof(block_q8_0)*width*height)/QK8_0;
				uint8_t*  blk = blk_load(path, &info->name, info->offset+ctx->offset, info->size);
				float d_max = dequantize_row_q8_0((const block_q8_0 *)blk, data_f32, width*height);
				//for(int i = 0; i<width*height; i++) if(data_f32[i]>d_max) d_max = data_f32[i];
				printf(" - max %e\n", (double)d_max);
			} else
			if (info->type==GGML_TYPE_Q4_0){
				info->size = (sizeof(block_q4_0)*width*height)/QK4_0;
				uint8_t*  blk = blk_load(path, &info->name, info->offset+ctx->offset, info->size);
				dequantize_row_q4_0((const block_q4_0 *)blk, data_f32, width*height);
			} else
			if (info->type==GGML_TYPE_Q4_K){
				info->size = (sizeof(block_q4_K)*width*height)/QK_K;
				uint8_t*  blk = blk_load(path, &info->name, info->offset+ctx->offset, info->size);
				float d_max = dequantize_row_q4_K((const block_q4_K *)blk, data_f32, width*height);
				printf(" - max %e\n", (double)d_max);
				uint32_t mask= 0;
				double err=0;
				double max_err=0;
				for (int j=0; j<height;++j)
				for (int i=0; i<width;++i){
					float v = data_f32[j*width+i];
					mask |= *(uint32_t*)&v;
					//uint32_t u = (*(uint32_t*)&v)&~0x0ul;
					float e = v==0?0:fabsf((v-(_Float16)v)/v);
					err += e;
					if (e>max_err) max_err = e;
				}
				printf(" - msk =0x%08X %s avg_err=%4.2g max_err=%4.2g\n", mask, (mask&0x3F)==0?"BF16 as F16":"F32", err/(width*height), max_err);
			} else {
				printf(" unsupported format `%s` \n",	GGML_TYPE_NAME[info->type]);
				_Exit(0);
			}
			// анализ матрицы
			// 1. Классификация матрицы и как ее считать Dense, Sparse
			// 
			// 3. Балансировка.
			size_t lda = width;
			short *d = g_malloc(sizeof(short) *(width>height?width:height));
			float* data1_f32 = g_malloc(sizeof(float)*width*height );
			m_copy(data1_f32, data_f32, height, width, lda);
			for (int i=0;i<height; i+=16)
			for (int j=0;j<width; j+=16){
				balance(data_f32+i*lda+j, d, 16, 16, lda);
				for (int i=0; i<16;++i) printf("%2d", (signed)d[i]);
				if (j%64==63) printf("\n");
			}

			balance(data_f32, d, height, width, lda);
			unbalance(data_f32, d, height, width, lda);
			//m_copy(data1_f32, data_f32, height, width, lda);
			for (int i=0; i<width;++i){
				printf("%2d", (signed)d[i]);
				
			}
			
			if (m_cmp(data1_f32, data_f32, height, width, lda)==0) printf("..ok\n"); 
			// 4. Сортировка
			
			
			//if (0) write_png(options.output_file, (uint8_t*)img, width, height);
			if (data_f32) g_free(data_f32);
		}
	}
	gguf_free (ctx);
	return 0;
}
#endif