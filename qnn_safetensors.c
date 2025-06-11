// qnn_safetensors - загрузка и анализ структуры
/*!
Сборка и тестирование 

```bash
gcc -O3 -o qnn qnn_safetensors.c json.c `pkgconf --cflags --libs glib-2.0`
./qqn -i models/model.safetensors
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
#include "json.h"
#include "qnn.h"

#include <stdbool.h>
#include <glib.h>

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
    char * input_file;
    char * output_file;
    char * config_file;
    char * kernel;
	int device_idx;
	int count;
    int verbose  ;
    int version  ;
    int overwrite;
};
static MainOptions options = {
    .input_file  = "model.safetensors",
    .output_file = NULL,
    .config_file = "config.json",
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

  { "overwrite",'O', 0, G_OPTION_ARG_NONE, &options.overwrite,   "overwtite output", NULL },
  { "verbose",  'v', 0, G_OPTION_ARG_NONE, &options.verbose, "Be verbose", NULL },
  { "version",  'V', 0, G_OPTION_ARG_NONE, &options.version, "program info", NULL },
  { NULL }
};

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
        size_t len = 0;
        if(qnn_load_safetensors(options.input_file, &header, &len)) _Exit(1);

        // fprintf(stdout, "%s\n", header);
        JsonNode* json = json_value(header, NULL, NULL);
        if (json==NULL) _Exit(1);
        GString *str = g_string_sized_new(len);
        json_to_string(json, str, 0);

        fprintf(stdout, "%s\n -- length=%d", str->str, str->len);
        g_string_free(str, TRUE);
    }
    
	return 0;
}