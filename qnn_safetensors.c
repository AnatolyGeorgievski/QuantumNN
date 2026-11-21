// qnn_safetensors - загрузка и анализ структуры
/*!
Сборка и тестирование 

```bash
gcc -O3 -mf16c -D_FILE_OFFSET_BITS=64 -o qnn-st qnn_safetensors.c json.c `pkgconf --cflags --libs glib-2.0` -lm
./qnn-st -i models/InternViT/model.safetensors --list
./qnn-st -i models/BitCPM4/model.safetensors --list -n model.layers.9.mlp.gate_proj.weight
# HF shards via index (headers-only stubs ok for --list):
./qnn-st -i /tmp/qwen35-hdr/model.safetensors.index.json --list
# FP8 pair from our encoder:
./qnn-st -i /tmp/qwen38-bf16-hdr/gate_ours.safetensors --list \
  -n model.language_model.layers.0.mlp.gate_proj.weight
```
Модель может быть разбита на части (HF: model.safetensors-00001-of-00011.safetensors + index.json;
FP8 Qwen: layers-N.safetensors + index). Рецепт сравнения с hf8-block:
QuantumNN/tools/quantize/REPRO-fp8.md.

Пример работы, вывод в формате JSON (PyTorch)
```json
{
 "__metadata__":{
  "format":"pt"
 },
 . . .
 "model.layers.2.self_attn.q_proj.weight":{
  "dtype":"BF16",
  "shape":[1536,1536],
  "data_offsets":[843061248,847779840]
 },
 . . .
}
```
Содержит смещения тензоров в файле safetensors, занимает объем порядка 64-128kB 

Для экспорта метаданных и сохранения модели используются:
* config.json -- параметры модели
* tokenizer.json -- настройки лингвистического процессора
* model.safetensors -- может быть разбит на части
* tokenizer_config.json -- опционально для языковых моделей
* vocab.json -- опционально, внутри tokenizer.json уже содержится словарь
* preprocessor_config.json -- опционально содержит настройки CLIP

 */
#include <stdint.h>
#include <stdio.h>
#include <ctype.h>
#include "json.h"
#include "qnn.h"

#include <stdbool.h>
#include <glib.h>
#define GGUF_MAX_DIMS 	4// максимальный размер в файле может отличаться от макс. размерности при вычислениях GGUF_MAX_DIMS>=GGML_MAX_DIMS
// структура должна приводиться к типу tensor_weight_t. \see qnn.h
// typedef struct _weight_tensor tensor_weight_t;

// шаблон имени файла для вычисления числа фрагментов
/* HF transformers */
#define FILENAME_TPL "model.safetensors-%05d-of-%05d.safetensors"
#define FILENAME_TPL_OLD "model_%d_of_%d.safetensors"
#define ALIGN(x, a) (((x) + ((a) - 1)) & ~((a) - 1))
/*! \brief загрузить модель из файла safetensors
    \param filename - имя файла в формате safetensors
    \param header - указатель на заголовок модели, возвращает описание модели в формате JSON 
    \param len - размер описания модели */
int qnn_load_safetensors(const char* filename, char** header, size_t *len)
{
    uint64_t length=0;
    uint8_t* hdr = *header;
    FILE* fp = fopen(filename, "rb");
    if (fp==NULL) return -1;
    fread(&length, 1, 8, fp);
    hdr = realloc(hdr, ALIGN(length+1,8));
    fread(hdr, 1, length, fp);
    fclose(fp);
    hdr[length] = 0;
    *header = hdr;
    *len = length;
    return 0;
}

typedef struct _MainOptions MainOptions;
struct _MainOptions {
    char * input_file;  //!< загрузить один или серию файлов модели
    char * output_file; //!< Сохранить результат в выбранном файле
    char * config_file;
    char * name;        //!< Анализировать выбранный тензор

    char * kernel;
	int device_idx;

    int count;
    int list;           //!< Вывести список тензоров
    int verbose  ;
    int version  ;
    int overwrite;
};
static MainOptions options = {
    .input_file  = "model.safetensors",
    .output_file = NULL,
    .config_file = "config.json",
    .name = "",
	.device_idx  = 0,
    .verbose  =false,
    .overwrite=false,
    .version  =false,
};
static GOptionEntry entries[] =
{
  { "device", 'd', 0, G_OPTION_ARG_INT, &options.device_idx, "select OpenCL device", "1..2" },

  { "input",    'i', 0, G_OPTION_ARG_FILENAME, &options.input_file,  "input  file name", "model.safetensors" },
  { "output",   'o', 0, G_OPTION_ARG_FILENAME, &options.output_file, "output file name", "*.gguf" },
  { "config",   'c', 0, G_OPTION_ARG_FILENAME, &options.config_file, "config file name", "config.json" },
  { "name", 	'n', 0, G_OPTION_ARG_STRING,   &options.name, "name", "blk.*.attn_k.weight" },
  { "list",      0 , 0, G_OPTION_ARG_NONE, &options.list,       "list tensors", NULL },
  { "overwrite",'O', 0, G_OPTION_ARG_NONE, &options.overwrite,   "overwtite output", NULL },
  { "verbose",  'v', 0, G_OPTION_ARG_NONE, &options.verbose, "Be verbose", NULL },
  { "version",  'V', 0, G_OPTION_ARG_NONE, &options.version, "program info", NULL },
  { NULL }
};
/*! \brief выделяет шаблон имени и индекс 
	Планируется использовать в RPC протоколе для формирования SDNV идентификаторов объектов `cname_id.index`. 
	Для идентификации объектов нужен словарь из `cname` - шаблонов имен. 
	Имя объекта восстанавливается по шаблону путем подстановки индекса в шаблон. 
 */
static int  _cname_idx(const char* name, char* cname){
    int index = 0;
    const char* s = name;
    while (*s!='\0' && !(s[0]=='.' && isdigit(s[1]))) s++;
    if (*s!='\0'){ 
        __builtin_memcpy(cname, name, s-name+1); cname+=s-name+1;
		s++;
    	while (isdigit(*s)) index = index*10+(*s++ -'0');
		*cname++ = '*'; 
		do {*cname++ = *s++; } while (*s!='\0');// до конца строки
        *cname = '\0';
		return index;
	} else 
		return -1;
}
// см как сделана функция для Windows \see qnn_gguf.c: blk_load()
static void* blk_load(FILE*fp, uint64_t offset, size_t size){
    int res = fseeko(fp, (off_t)offset, SEEK_SET);
    if (res) return NULL;
    void* data = malloc(size);
    if (data==NULL) return NULL;
    fread(data, 1, size, fp);
    return data;
}
#include <math.h>
#include <string.h>
#include <libgen.h>


/*! \brief safetensors F8_E4M3 (e4m3fn): 1|4|3, bias 7, без Inf */
static float e4m3fn_to_fp32(uint8_t x){
    int s = (x >> 7) & 1;
    int e = (x >> 3) & 0xF;
    int m = x & 7;
    float v;
    if (e == 0)
        v = (m == 0) ? 0.f : ldexpf((float)m / 8.f, 1 - 7);
    else if (e == 0xF)
        v = 0.f; /* fn: NaN/нет Inf — для разбора ставим 0 */
    else
        v = ldexpf(1.f + (float)m / 8.f, e - 7);
    return s ? -v : v;
}

static JsonNode* json_find_tagged(GSList* list, const char* name){
    GQuark id = g_quark_from_string(name);
    for (; list; list = list->next) {
        JsonNode* n = list->data;
        if (n->tag_id == id) return n;
    }
    return NULL;
}

/*! \brief data_offsets как байты (uint64) */
static void json_data_span(GSList* data_offs, uint64_t* off, uint64_t* end){
    *off = 0; *end = 0;
    if (!data_offs) return;
    JsonNode* a = data_offs->data;
    *off = (a->type==JSON_UINT || a->type==JSON_INT) ? a->value.u : 0;
    if (data_offs->next) {
        JsonNode* b = data_offs->next->data;
        *end = (b->type==JSON_UINT || b->type==JSON_INT) ? b->value.u : 0;
    }
}

static char* path_join_dir(const char* base_path, const char* name){
    char dirbuf[4096];
    strncpy(dirbuf, base_path, sizeof(dirbuf)-1);
    dirbuf[sizeof(dirbuf)-1] = 0;
    char* dir = dirname(dirbuf);
    size_t n = strlen(dir) + 1 + strlen(name) + 1;
    char* out = malloc(n);
    if (!out) return NULL;
    snprintf(out, n, "%s/%s", dir, name);
    return out;
}

/*! \brief -i file.safetensors | model.safetensors.index.json → список путей шардов */
static GSList* resolve_inputs(const char* input_file){
    GSList* files = NULL;
    size_t L = strlen(input_file);
    const char* suf = ".index.json";
    if (L > strlen(suf) && strcmp(input_file + L - strlen(suf), suf) == 0) {
        gchar* contents = NULL;
        gsize len = 0;
        GError* err = NULL;
        if (!g_file_get_contents(input_file, &contents, &len, &err)) {
            fprintf(stderr, "index read fail: %s\n", err ? err->message : "?");
            return NULL;
        }
        JsonNode* json = json_value(contents, NULL, &err);
        g_free(contents);
        if (!json || json->type != JSON_OBJECT) {
            fprintf(stderr, "index: bad json\n");
            return NULL;
        }
        GQuark id_wm = g_quark_from_string("weight_map");
        GHashTable* seen = g_hash_table_new(g_str_hash, g_str_equal);
        for (GSList* l = json->value.list; l; l = l->next) {
            JsonNode* n = l->data;
            if (n->tag_id != id_wm || n->type != JSON_OBJECT) continue;
            for (GSList* w = n->value.list; w; w = w->next) {
                JsonNode* t = w->data;
                if (t->type != JSON_STRING || !t->value.s) continue;
                const char* shard = t->value.s;
                if (g_hash_table_contains(seen, (gpointer)shard)) continue;
                g_hash_table_add(seen, (gpointer)shard);
                files = g_slist_append(files, path_join_dir(input_file, shard));
            }
        }
        g_hash_table_destroy(seen);
        if (!files) fprintf(stderr, "index: empty weight_map\n");
        return files;
    }
    return g_slist_append(NULL, g_strdup(input_file));
}

int main(int argc, char *argv[]){
    GOptionContext *opt_context;
    opt_context = g_option_context_new ("- command line interface");
    g_option_context_add_main_entries (opt_context, entries, NULL/*GETTEXT_PACKAGE*/);
    if (!g_option_context_parse (opt_context, &argc, &argv, NULL))
    {
        //printf ("option parsing failed: %s\n", error->message);
        _Exit(1);
    }
    g_option_context_free (opt_context);
	if (options.version) {
		printf ("QNN v0.01 (c) Anatoly Georgievski\n");
	}
    if (options.input_file) {
        printf ("input file: %s\n", options.input_file);
        GSList* inputs = resolve_inputs(options.input_file);
        if (!inputs) _Exit(1);
        char cname[72];
        int group = -1;
        for (GSList* fi = inputs; fi; fi = fi->next) {
            const char* path = fi->data;
            printf ("shard: %s\n", path);
            char* header = NULL;
            size_t header_len = 0;
            if (qnn_load_safetensors(path, &header, &header_len)) {
                fprintf(stderr, "fail load %s\n", path);
                _Exit(1);
            }
            GError * error=NULL;
            JsonNode* json = json_value(header, NULL, &error);
            if (json==NULL) _Exit(1);
            if (options.verbose) {
                GString *str = g_string_sized_new(header_len);
                json_to_string(json, str, 0);
                fprintf(stdout, "%s\n -- length=%zu", str->str, (size_t)str->len);
                g_string_free(str, TRUE);
            }
            if (options.list) {
                GSList* list = json->value.list;
                GQuark id_offs  = g_quark_from_string ("data_offsets");
                GQuark id_dtype = g_quark_from_string ("dtype");
                GQuark id_shape = g_quark_from_string ("shape");
                while (list){
                    JsonNode* node = list->data;
                    if (node->type == JSON_OBJECT) {
                        const char* dtype = json_get_string(node->value.list, id_dtype, NULL);
                        if (dtype!=NULL) {
                            GSList* shape     = json_get_array (node->value.list, id_shape);
                            size_t ne[GGUF_MAX_DIMS]= {0};
                            int i;
                            size_t len=1;
                            for (i=0; shape!=NULL && i<GGUF_MAX_DIMS; i++) {
                                JsonNode* n = shape->data;
                                ne[i] = n->value.u;
                                shape = shape->next;
                                len*= ne[i];
                            }
                            int n_dim = i;
                            GSList* data_offs = json_get_array (node->value.list, id_offs);
                            uint64_t offs = 0, end = 0;
                            json_data_span(data_offs, &offs, &end);
                            uint64_t nbytes = end > offs ? end - offs : 0;
                            const char* name  = g_quark_to_string(node->tag_id);
                            int idx = _cname_idx(name, cname);
                            if (idx!=group) {
                                fprintf(stdout, "layers[%d]:\n", idx);
                                group = idx;
                            }
                            fprintf(stdout, " '%s' (%s): \tlen=%zu k\n", idx<0?name:cname, dtype, len/1024);
                            if (options.name && options.name[0] && strcmp(name, options.name)==0) {
                                FILE * fp = fopen(path, "rb");
                                if (!fp) { fprintf(stderr, "open fail %s\n", path); break; }
                                if (strcmp(dtype, "F8_E4M3") == 0 || strcmp(dtype, "F8_E4M3FN") == 0) {
                                    /* блочный FP8: W[N,K] + W_scale_inv[N/B,K/B], B=128 (или из shape) */
                                    char scale_name[512];
                                    snprintf(scale_name, sizeof(scale_name), "%s_scale_inv", name);
                                    JsonNode* sn = json_find_tagged(json->value.list, scale_name);
                                    if (!sn || sn->type != JSON_OBJECT) {
                                        fprintf(stderr, "no %s\n", scale_name);
                                        fclose(fp); break;
                                    }
                                    GSList* ssh = json_get_array(sn->value.list, id_shape);
                                    GSList* sof = json_get_array(sn->value.list, id_offs);
                                    size_t sn0 = 0, sn1 = 0;
                                    if (ssh) { JsonNode* a=ssh->data; sn0=a->value.u; if (ssh->next){JsonNode*b=ssh->next->data; sn1=b->value.u;} }
                                    uint64_t soff=0, send=0; json_data_span(sof, &soff, &send);
                                    size_t B0 = sn0 ? ne[0]/sn0 : 128;
                                    size_t B1 = (n_dim>1 && sn1) ? ne[1]/sn1 : 128;
                                    if (!B0) B0=128; if (!B1) B1=128;
                                    size_t sn_elem = sn0 * (sn1 ? sn1 : 1);
                                    ggml_bf16_t* sdata = blk_load(fp, soff + header_len + 8, send - soff);
                                    uint8_t* wdata = blk_load(fp, offs + header_len + 8, nbytes);
                                    if (!sdata || !wdata) { fprintf(stderr, "blk_load fail\n"); fclose(fp); break; }
                                    printf("F8_E4M3 block=%zux%zu scale_inv=%zux%zu (%s)\n", B0, B1, sn0, sn1, scale_name);
                                    float smin =  __FLT_MAX__, smax = -__FLT_MAX__;
                                    for (size_t i=0; i<sn_elem; i++) {
                                        float sv = GGML_BF16_TO_FP32(sdata[i]);
                                        if (sv < smin) smin = sv;
                                        if (sv > smax) smax = sv;
                                    }
                                    printf("scale_inv min=%g max=%g  (dequant: fp8 * scale_inv)\n", smin, smax);
                                    /* превью: верхний левый блок B0 x min(B1,16) */
                                    size_t show_r = B0 < 8 ? B0 : 8;
                                    size_t show_c = B1 < 16 ? B1 : 16;
                                    float vmin =  __FLT_MAX__, vmax = -__FLT_MAX__;
                                    double err_ref = 0; /* amax vs 448*scale */
                                    for (size_t bi=0; bi<sn0; bi++) {
                                        for (size_t bj=0; bj<(sn1?sn1:1); bj++) {
                                            float si = GGML_BF16_TO_FP32(sdata[bi*(sn1?sn1:1)+bj]);
                                            float amax = 0;
                                            size_t i0=bi*B0, j0=bj*B1;
                                            size_t i1=i0+B0; if (i1>ne[0]) i1=ne[0];
                                            size_t j1=j0+B1; if (n_dim>1 && j1>ne[1]) j1=ne[1];
                                            for (size_t i=i0; i<i1; i++) {
                                                for (size_t j=j0; j<(n_dim>1?j1:1); j++) {
                                                    size_t idx = n_dim>1 ? i*ne[1]+j : i;
                                                    float v = e4m3fn_to_fp32(wdata[idx]) * si;
                                                    if (fabsf(v) > amax) amax = fabsf(v);
                                                    if (v < vmin) vmin = v;
                                                    if (v > vmax) vmax = v;
                                                }
                                            }
                                            float expect = si * 448.f;
                                            if (expect > 0) err_ref += fabsf(amax - expect) / expect;
                                        }
                                    }
                                    printf("dequant range [%g .. %g]\n", vmin, vmax);
                                    printf("block headroom vs 448*scale_inv: mean underfill=%g (1=full saturate)\n",
                                           1.0 - err_ref / (double)(sn0*(sn1?sn1:1)));
                                    printf("preview dequant [0:%zu, 0:%zu]:\n", show_r, show_c);
                                    float s00 = GGML_BF16_TO_FP32(sdata[0]);
                                    for (size_t i=0; i<show_r; i++) {
                                        for (size_t j=0; j<show_c; j++) {
                                            size_t idx = n_dim>1 ? i*ne[1]+j : i;
                                            fprintf(stdout, "%9.4g ", e4m3fn_to_fp32(wdata[idx]) * s00);
                                        }
                                        fprintf(stdout, "\n");
                                    }
                                    /* своя квантизация обратно: s' = amax/448, сравнить с scale_inv */
                                    {
                                        float amax = 0;
                                        size_t i1 = B0 < ne[0] ? B0 : ne[0];
                                        size_t j1 = (n_dim>1 && B1 < ne[1]) ? B1 : (n_dim>1?ne[1]:1);
                                        for (size_t i=0; i<i1; i++)
                                            for (size_t j=0; j<j1; j++) {
                                                size_t idx = n_dim>1 ? i*ne[1]+j : i;
                                                float v = fabsf(e4m3fn_to_fp32(wdata[idx]) * s00);
                                                if (v > amax) amax = v;
                                            }
                                        float s_own = amax / 448.f;
                                        printf("block[0,0]: scale_inv=%g  amax_dequant=%g  amax_dequant/448=%g\n",
                                               s00, amax, s_own);
                                        printf("  (own scale from already-quant data underestimates; 448*scale_inv=%g)\n",
                                               448.f * s00);
                                    }
                                    free(sdata); free(wdata);
                                } else if (strcmp(dtype, "BF16") == 0) {
                                    size_t nelem = len;
                                    ggml_bf16_t* data_f16 = blk_load(fp, offs+header_len+8, nbytes);
                                    if (!data_f16) { fclose(fp); break; }
                                    uint32_t mask = 0;
                                    float v_min =  __FLT_MAX__;
                                    float v_max = -__FLT_MAX__;
                                    float scale = 0;
                                    int e_max = -256, e_min = 256;
                                    size_t nblk = (nelem + 31) >> 5;
                                    ggml_bf16_t *scales = malloc(nblk*sizeof(ggml_bf16_t));
                                    int8_t* exp_d = malloc(nblk*sizeof(uint8_t));
                                    int8_t* exp_m = malloc(nblk*sizeof(uint8_t));
                                    for (size_t i=0; i<nelem; i++) {
                                        float v = GGML_BF16_TO_FP32(data_f16[i]);
                                        int e = 0;
                                        (void)frexpf(v, &e);
                                        if ((i&31)== 0) {
                                            scale = 0;
                                            e_max = -256, e_min = 256;
                                        }
                                        if (e<e_min) e_min=e;
                                        if (e>e_max) e_max=e;
                                        if (fabsf(v)>scale) scale = fabsf(v);
                                        if ((i&31)==31) {
                                            scales[i>>5] = GGML_FP32_TO_BF16(scale);
                                            exp_d[i>>5] = e_max - e_min;
                                            exp_m[i>>5] = e_max;
                                        }
                                    }
                                    for (size_t i=0; i<nelem; i++) {
                                        if ((i&31)==0)
                                            scale = 1.f/GGML_BF16_TO_FP32(scales[i>>5]);
                                        float v = GGML_BF16_TO_FP32(data_f16[i])*scale;
                                        if (i<512) {
                                            fprintf(stdout, "%7.3g", v);
                                            if ((i&15)==15) fprintf(stdout, "\n");
                                        }
                                        mask |= data_f16[i].bits;
                                        if (v>v_max) v_max = v;
                                        if (v<v_min) v_min = v;
                                    }
                                    printf("mask = %04x max=%g min=%g [%zu x %zu]\n", mask, v_max, v_min, ne[0], ne[1]);
                                    size_t m = n_dim>1?32:1;
                                    for (size_t i=0; i<m; i++) {
                                        for (size_t j=0; j<8 ; j++)
                                            fprintf(stdout, "%-8.2g", GGML_BF16_TO_FP32(scales[i*(ne[0]>>5)+j]));
                                        fprintf(stdout, "\n");
                                    }
                                    free(scales); free(exp_d); free(exp_m);
                                    free(data_f16);
                                } else {
                                    fprintf(stderr, "-n: dtype %s not supported yet\n", dtype);
                                }
                                fclose(fp);
                            }
                        } else if (dtype == NULL) {
                            /* __metadata__ etc. */
                        } else {
                            fprintf(stderr, "'%s': unknown type %s\n", g_quark_to_string(node->tag_id), dtype);
                        }
                    }
                    list = list->next;
                }
            }
            free(header);
        }
        g_slist_free_full(inputs, free);
    }

	return 0;
}