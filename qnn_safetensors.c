// qnn_safetensors - загрузка и анализ структуры
/*!
Сборка и тестирование 

```bash
gcc -O3 -o qnn qnn_safetensors.c json.c `pkgconf --cflags --libs glib-2.0`
./qnn.exe -i models/InternViT/model.safetensors --list
./qnn.exe -i models/BitCPM4/model.safetensors --list -n model.layers.9.mlp.gate_proj.weight
```
Модель может быть разбита на части

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
#define FILENAME_TPL "model_%d_of_%d.safetensors"
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
    int res = fseeko64(fp, offset, SEEK_SET);
    if (res) return NULL;
    void* data = malloc(size);
    if (data==NULL) return NULL;
    fread(data, 1, size, fp);
    return data;
}
#include <math.h>

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
        char* header = NULL;
        size_t header_len = 0;
        if(qnn_load_safetensors(options.input_file, &header, &header_len)) _Exit(1);

        // fprintf(stdout, "%s\n", header);
        GError * error=NULL;
        JsonNode* json = json_value(header, NULL, &error);
        if (json==NULL) _Exit(1);
        if (options.verbose) {
            GString *str = g_string_sized_new(header_len);
            json_to_string(json, str, 0);
            fprintf(stdout, "%s\n -- length=%d", str->str, str->len);
            g_string_free(str, TRUE);
        }
        char cname[72];
        if (options.list) {// вывести список всех тензоров
            GSList* list = json->value.list;
            GQuark id_offs  = g_quark_from_string ("data_offsets");
            GQuark id_dtype = g_quark_from_string ("dtype");
            GQuark id_shape = g_quark_from_string ("shape");
            // сформировать пространство имен для кэширования 
            int group = -1;
            while (list){
                JsonNode* node = list->data;
                if (node->type == JSON_OBJECT) {
                    const char* dtype = json_get_string(node->value.list, id_dtype, NULL);
                    if (dtype!=NULL) {//  && strcmp(dtype,"BF16")==0
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
                        uint64_t offs = json_array_get_uint(data_offs, 0, 0);
                        size_t   size = json_array_get_uint(data_offs, 1, 0) - offs;
                        const char* name  = g_quark_to_string(node->tag_id);
                        int idx = _cname_idx(name, cname);
                        if (idx!=group) {
                            fprintf(stdout, "layers[%d]:\n", idx);    
                            group = idx;
                        }
                        fprintf(stdout, " '%s' (%s): \tlen=%zu k\n", idx<0?name:cname, dtype, len/1024);
                        if (options.name && strcmp(name, options.name)==0) {// анализ для тензора
                        // U8:
                        // BF16: 
                            FILE * fp = fopen(options.input_file, "rb");
                            ggml_bf16_t* data_f16 = blk_load(fp, offs+header_len+8, size*sizeof(ggml_bf16_t));
                            uint32_t mask = 0;
                            float v_min =  __FLT_MAX__;
                            float v_max = -__FLT_MAX__;
                            float scale = 0;
                            int e_max = -256, e_min = 256;
                            ggml_bf16_t *scales = malloc(((size+31)>>5)*sizeof(ggml_bf16_t));
                            int8_t* exp_d = malloc(((size+31)>>5)*sizeof(uint8_t));
                            int8_t* exp_m = malloc(((size+31)>>5)*sizeof(uint8_t));
                            for (size_t i=0; i<size; i++) {// максимум нормировки
                                float v = GGML_BF16_TO_FP32(data_f16[i]);
                                int e;
                                //v = frexpf(v, &e);
                                //if (v==1.f) {v/=2, e++; }
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
                            for (size_t i=0; i<size; i++) {// это вариант для выявления BitNET 
                                if ((i&31)==0) {
                                    scale = 1.f/GGML_BF16_TO_FP32(scales[i>>5]);
                                    //if (i<512) fprintf(stdout, "%3d:%2d:", exp_m[i>>5],exp_d[i>>5]);
                                }
                                float v = GGML_BF16_TO_FP32(data_f16[i])*scale;
                                //v = ldexp(v, -exp_m[i>>5]);
                                if (i<512) {
                                    fprintf(stdout, "%7.3g", v);
                                    if ((i&15)==15) fprintf(stdout, "\n");
                                }
                                //if (isnormal(v)) 
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
                            free(scales);
                            free(data_f16);
                            fclose(fp);
                            break;
                        }

                    } else {
                        fprintf(stderr, "'%s': unknown type %s\n", g_quark_to_string(node->tag_id), dtype);
                    }
                }
                list = list->next;
            } 
        }
    }
    
	return 0;
}