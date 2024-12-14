#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <unistd.h>
#include <inttypes.h>
#include <glib.h>
#define GGML_MAX_DIMS 	4
#define GGUF_MAGIC 		"GGUF"
#define GGUF_VERSION 	3
#define GGUF_DEFAULT_ALIGNMENT 32
//typedef struct _gguf_ctx gguf_cxt_t;
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
	off_t offset; // data segment 

	size_t size;
	unsigned alignment;
};
struct gguf_str {
    uint64_t n;  // GGUFv2
    char * data;
};

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
enum ggml_type: uint32_t {
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
        // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
        // GGML_TYPE_Q4_0_4_8 = 32,
        // GGML_TYPE_Q4_0_8_8 = 33,
        GGML_TYPE_TQ1_0   = 34,
        GGML_TYPE_TQ2_0   = 35,
        // GGML_TYPE_IQ4_NL_4_4 = 36,
        // GGML_TYPE_IQ4_NL_4_8 = 37,
        // GGML_TYPE_IQ4_NL_8_8 = 38,
        GGML_TYPE_COUNT   = 39,
};
static const size_t GGUF_TYPE_SIZE[GGUF_TYPE_COUNT] = {
    [GGUF_TYPE_UINT8]   = sizeof(uint8_t),
    [GGUF_TYPE_INT8]    = sizeof(int8_t),
    [GGUF_TYPE_UINT16]  = sizeof(uint16_t),
    [GGUF_TYPE_INT16]   = sizeof(int16_t),
    [GGUF_TYPE_UINT32]  = sizeof(uint32_t),
    [GGUF_TYPE_INT32]   = sizeof(int32_t),
    [GGUF_TYPE_FLOAT32] = sizeof(float),
    [GGUF_TYPE_BOOL]    = sizeof(bool),
    [GGUF_TYPE_STRING]  = sizeof(struct gguf_str),
    [GGUF_TYPE_UINT64]  = sizeof(uint64_t),
    [GGUF_TYPE_INT64]   = sizeof(int64_t),
    [GGUF_TYPE_FLOAT64] = sizeof(double),
    [GGUF_TYPE_ARRAY]   = 0, // undefined
};
static size_t gguf_type_size(enum gguf_type type) {
    //GGML_ASSERT(0 <= type && type < GGUF_TYPE_COUNT);
    return GGUF_TYPE_SIZE[type];
}
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

struct gguf_tensor_info {
    struct gguf_str name;

    uint32_t n_dims;
    uint64_t ne[GGML_MAX_DIMS];

    enum ggml_type type;

    uint64_t offset; // offset from start of `data`, must be a multiple of `ALIGNMENT`

    // for writing API
    const void * data;
    size_t size;
};
typedef struct gguf_context gguf_cxt_t;
struct gguf_context {
    struct gguf_header header;

    struct gguf_kv          * kv;
    struct gguf_tensor_info * infos;

    size_t alignment;
    size_t offset;    // offset of `data` from beginning of file
    size_t size;      // size of `data` in bytes

    //uint8_t * padding;
    void * data;
};


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
static bool gguf_fread(FILE * file, void * dst, size_t size, off_t *offset) {
    size_t res = fread(dst, 1, size, file);
	offset += res;
	return res == size;
}
static bool gguf_fread_str(FILE * file, struct gguf_str * p, off_t * offset) {
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
uint32_t gguf_get_val_u32(const struct gguf_context * ctx, int key_id) {
//    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
//    GGML_ASSERT(ctx->kv[key_id].type == GGUF_TYPE_UINT32);
    return ctx->kv[key_id].value.uint32;
}
const char * gguf_get_key(const struct gguf_context * ctx, int key_id) {
//    GGML_ASSERT(key_id >= 0 && key_id < gguf_get_n_kv(ctx));
    return ctx->kv[key_id].key.data;
}

int gguf_find_key(const struct gguf_context * ctx, const char * key) {
    int keyfound = -1;// return -1 if key not found

    const int n_kv = ctx->header.n_kv;//gguf_get_n_kv(ctx);

    for (int i = 0; i < n_kv; ++i) {
        if (strcmp(key, gguf_get_key(ctx, i)) == 0) {
            keyfound = i;
            break;
        }
    }

    return keyfound;
}

static void gguf_free(gguf_cxt_t* ctx){
	if(ctx->header.n_tensors) 
	if(ctx->header.n_kv) 

	if(ctx->kv)
		g_free(ctx->kv);
	if(ctx->infos)
		g_free(ctx->infos);
	g_free(ctx);
}
gguf_cxt_t * gguf_init_from_file(const char * fname, uint32_t/* struct gguf_init_params */params) {
    FILE * file = fopen(fname, "rb");
    if (file==NULL) {
        //fprintf(stderr, "%s: failed to open '%s': '%s'\n", __func__, fname, strerror(errno));
        return NULL;
    }

    // offset from start of file
    off_t offset = 0;
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
        res = res &&  (ctx->header.n_kv      < (SIZE_MAX/2)/sizeof(struct gguf_kv));

        if (!res) {
            fprintf(stderr, "%s: failed to read header\n", __func__);
            fclose(file);
            gguf_free(ctx);
            return NULL;
        }
    }

    // read the kv pairs
    {
        const uint64_t n_kv = ctx->header.n_kv;

        ctx->kv = g_malloc0(n_kv* sizeof(struct gguf_kv));

        for (uint64_t i = 0; i < n_kv; ++i) {
            struct gguf_kv * kv = &ctx->kv[i];

//            fprintf(stderr, "%s: reading kv %d\n", __func__, i);

            gguf_fread_str(file, &kv->key,                    &offset);
            gguf_fread (file, &kv->type, sizeof(kv->type), &offset);

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
//return  ctx;
    // read the tensor infos
    if (ctx->header.n_tensors > 0) {
        ctx->infos = g_malloc0(ctx->header.n_tensors * sizeof(struct gguf_tensor_info));
        if (!ctx->infos) {
            fprintf(stderr, "%s: failed to allocate memory for tensor infos\n", __func__);
            fclose(file);
            gguf_free(ctx);
            return NULL;
        }

        for (uint64_t i = 0; i < ctx->header.n_tensors; ++i) {
            struct gguf_tensor_info * info = &ctx->infos[i];

            for (int j = 0; j < GGML_MAX_DIMS; ++j) info->ne[j] = 1;

            res = res && gguf_fread_str(file, &info->name,                       &offset);
            res = res && gguf_fread (file, &info->n_dims, sizeof(info->n_dims),  &offset);

            res = res && (info->n_dims <= GGML_MAX_DIMS);

            for (uint32_t j = 0; j < info->n_dims; ++j) {
                res = res && gguf_fread(file, &info->ne[j], sizeof(info->ne[j]), &offset);
            }

            res = res && gguf_fread (file, &info->type,   sizeof(info->type),    &offset);
            res = res && gguf_fread (file, &info->offset, sizeof(info->offset),  &offset);

//            res = res && gguf_tensor_info_sanitize(info);

            // make sure there is no duplicated tensor names
            for (uint64_t j = 0; j < i && res; ++j) {
				//GQuark id = g_quark_try_string (info->name.data);
                if (strcmp(info->name.data, ctx->infos[j].name.data) == 0) {
                    fprintf(stderr, "%s: duplicated tensor name %s\n", __func__, info->name.data);
                    res = false;
                }
            }

            if (!res) {
                fprintf(stderr, "%s: failed to read tensor info\n", __func__);
                fclose(file);
                gguf_free(ctx);
                return NULL;
            }
        }
    }

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
            fseek(file, offset, SEEK_SET);
        }
    }
    // store the current file offset - this is where the data section starts
    ctx->offset = offset;
    // compute the total size of the data section, taking into account the alignment
    // load the tensor data only if requested
    fclose(file);

    return ctx;
}
int gguf_debug(gguf_cxt_t* ctx){
	for(uint64_t i=0; i<ctx->header.n_kv; ++i){
		struct gguf_kv *kv = &ctx->kv[i];
        fprintf(stdout, "%3d| %-.*s|", i, kv->key.n, kv->key.data);
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
	fprintf(stdout, "model info:\n");
	for(uint64_t i=0; i<ctx->header.n_tensors; ++i){
		struct gguf_tensor_info * info = &ctx->infos[i];
		fprintf(stdout, "%-.*s ", info->name.n,info->name.data );
		char ch='[';
		for (uint32_t j = 0; j < info->n_dims; ++j, ch=',') {
			fprintf(stdout, "%c%d", ch, info->ne[j]);
		}
		fprintf(stdout, "]\n"); 
	}
		
}
int main (int argc, char *argv[]){
	if (argc<2) return 0;
	char *path = argv[1];
	gguf_cxt_t *ctx = gguf_init_from_file(path, 0);
	gguf_debug(ctx);
	gguf_free (ctx);
	return 0;
}