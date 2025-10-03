/*! \brief 
Из этого делаем два инструмента поиск и сравнение, анализ сцен:
1. Поиск максимума схожести кадра и фрагмента видео - возвращает задержку в микросекундах от начала фрагмента
2. Поиск границы смены сцены - возвращает список фрагментов, начало-длительность относительно первого кадра.
3. Выделение опорных кадров при анализе фрагментов: статических, в начале и в конце фрагмента - возвращает опорный кадр. 
4. Сравнение двух фрагментов (сшивание, сравнивается последний кадр первого фрагмента и первый кадр второго фрагмента).

```sh
pacman -S mingw64/mingw-w64-x86_64-opencv
```

```sh
export LD_LIBRARY_PATH=/home/ag/llama.cpp/build/bin
make LLAMA=../../llama.cpp -j 8 -f vlm.mak
ldd ./vlm
CUDA_VISIBLE_DEVICES=0 ./vlm.exe -m models/mmproj-InternVL3-14B-Instruct-Q8_0.gguf images/ch-110-4500/dvr_thumbnail_17492*.jpg
```
*/

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include "qnn_embed.h"


#include <clip.h>
//#define STB_IMAGE_IMPLEMENTATION -- в одном из мест должна быть указана 
#include "stb_image.h"

#define N_EMBED_LEN 2
struct _qnn_embedding_ctx {
    struct clip_ctx * ctx_v;//!< контекст визуальной части CLIP Vision Encoder
    int32_t n_threads;      //!< максимальное число потоков на CPU 
    int32_t n_mmproj_embd;  //!< размерность модели
    int32_t n_output_tokens;//!< Число токенов на выходе n_mmproj_embd*n_output_tokens = число элементов в эмбединг векторе
    int32_t n_embed_len;    //!< число векторов в наборе
    int32_t t_encode_us;    //!< время кодирования в микросекундах
    int32_t t_process_us;   //!< время обработки   в микросекундах
    int32_t pos;    //!< позиция записи 

    float * embd_vector[N_EMBED_LEN];
    float * embd_mean;
//    int n_mmproj_embd = llama_model_n_embd(model);
//    int n_pos_per_embd = mtmd_decode_use_mrope(ctx) ? 4 : 1;
//    int n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
};
int32_t qnn_embed_time(qnn_embed_t* mctx){
    return mctx->t_encode_us;
}
int32_t qnn_embed_process_time(qnn_embed_t* mctx){
    return mctx->t_process_us;
}
float* qnn_embed_get(qnn_embed_t* mctx, int idx){
    return (idx< mctx->n_embed_len)? mctx->embd_vector[idx]: nullptr;
}
int32_t qnn_embed_n_mmproj (qnn_embed_t* mctx){
    return mctx->n_mmproj_embd;
}
int32_t qnn_embed_n_tokens (qnn_embed_t* mctx){
    return mctx->n_output_tokens;
}
int qnn_embed_len (qnn_embed_t* mctx){
    return mctx->n_output_tokens*mctx->n_mmproj_embd;
}
// semantic similarity [0,1]
float qnn_embed_ssim  (qnn_embed_t* mctx, int offs){
    unsigned n = mctx->n_output_tokens*mctx->n_mmproj_embd;
    int i = 0;
    float* u = mctx->embd_vector[i];
    float* v = mctx->embd_vector[(i+offs)%N_EMBED_LEN];
    if (u==NULL || v==NULL) return 0.f;

//    qnn_embed_sub (u, m, )

    return qnn_embed_similarity_cos(u, v, n);
}
float qnn_embed_dist  (qnn_embed_t* mctx, int offs){
    unsigned n = mctx->n_output_tokens*mctx->n_mmproj_embd;
    int i = 0;
    float* u = mctx->embd_vector[i];
    float* v = mctx->embd_vector[(i+offs)%N_EMBED_LEN];
    if (u==NULL || v==NULL) return 0.f;
    return qnn_embed_distance(u, v, n);
}

qnn_embed_t* qnn_embed_init(const char * fname, int flags)
{
    struct clip_context_params ctx_params = 
        {.use_gpu = true, .verbosity = GGML_LOG_LEVEL_INFO};
    struct clip_init_result res = clip_init(fname, ctx_params);
    if (res.ctx_v==NULL) return NULL;
    if (!clip_has_vision_encoder(res.ctx_v)) {
        clip_free(res.ctx_v);
        return NULL;
    }

    qnn_embed_t * mctx = (qnn_embed_t *)malloc(sizeof(qnn_embed_t));
    mctx->ctx_v = res.ctx_v;
    mctx->n_threads = 511;
    mctx->n_mmproj_embd = clip_n_mmproj_embd(mctx->ctx_v);
    mctx->n_output_tokens=0;
    mctx->n_embed_len = N_EMBED_LEN;
    mctx->pos = 0;
    for (int i=0; i< N_EMBED_LEN; i++)
        mctx->embd_vector[i] = NULL;
    mctx->embd_mean = (float*)malloc(sizeof(float)*mctx->n_mmproj_embd);
    __builtin_bzero(mctx->embd_mean, sizeof(float)*mctx->n_mmproj_embd);
    return mctx;
}
void qnn_embed_free(qnn_embed_t* mctx){
    for (int i=0; i<N_EMBED_LEN; i++){
        if (mctx->embd_vector[i]!=NULL) 
            free(mctx->embd_vector[i]);
    }
    clip_free(mctx->ctx_v);
    free(mctx);
}
/*! \brief вытолкать наружу результат предыдущей операции `_rotate`
 */
static int qnn_embed_pop(qnn_embed_t* mctx) {
    float* prev = mctx->embd_vector[0];
    for (int i = 1; i<N_EMBED_LEN;i++ ){
        mctx->embd_vector[i-1] = mctx->embd_vector[i];
    }
    mctx->embd_vector[N_EMBED_LEN-1] = prev;
    mctx->pos --;

    return 0;
}
int qnn_embed_rotate(qnn_embed_t* mctx) {
    struct clip_ctx * ctx_clip = mctx->ctx_v;
    mctx->pos ++;
    float* prev = mctx->embd_vector[N_EMBED_LEN-1];
    for (int i = N_EMBED_LEN; --i>0; ){
        mctx->embd_vector[i] = mctx->embd_vector[i-1];
    }
    mctx->embd_vector[0] = prev;
    return 0;
}
static bool qnn_embed_image_batch(qnn_embed_t* mctx, struct clip_image_u8 *img){
    struct clip_ctx * ctx_clip = mctx->ctx_v;
    struct clip_image_f32_batch *   img_batch = clip_image_f32_batch_init(); 
    int64_t t0_1 = ggml_time_us();
    clip_image_preprocess(ctx_clip, img, img_batch);
    int64_t t1_1 = ggml_time_us();
    mctx->t_process_us = t1_1 - t0_1;
    bool ok = false;
    struct clip_image_f32 * img_f32 = clip_image_f32_get_img(img_batch, 0);
    mctx->n_output_tokens = clip_n_output_tokens(ctx_clip, img_f32);
    int idx = 0;
    //int idx = qnn_embed_rotate(mctx);


    if (mctx->embd_vector[0]==NULL){
        int size = clip_embd_nbytes(ctx_clip);
        mctx->embd_vector[0] = (float*)malloc(size);
    }

    float * v = mctx->embd_vector[idx];
    int64_t t0 = ggml_time_us();
    ok = clip_image_batch_encode(ctx_clip, mctx->n_threads, img_batch, v);
    int64_t t1 = ggml_time_us();
    mctx->t_encode_us = t1 - t0;

    float* m = mctx->embd_mean;
    unsigned n = mctx->n_output_tokens;
    unsigned k = mctx->n_mmproj_embd;
    qnn_embed_mean(m, v,    mctx->n_output_tokens, k);
    qnn_embed_sub (v, v, m, mctx->n_output_tokens, k);

    clip_image_f32_batch_free(img_batch);
    clip_image_u8_free(img);
    return ok;
}
/*! \brief Загрузка изображения из памяти в формате RGB nc=3 */
bool qnn_embed_image (qnn_embed_t* mctx, const uint8_t* rgb_pixels, int nx, int ny){
    struct clip_image_u8 * img = clip_image_u8_init ();
    clip_build_img_from_pixels(rgb_pixels, nx, ny, img);
    return qnn_embed_image_batch(mctx, img);
}
/*! \brief Загрузка изображения из файла в формате JPG или PNG */
bool qnn_embed_image_from_file (qnn_embed_t* mctx, const char* fname){
    int nx, ny, nc; 
    unsigned char * rgb_pixels = stbi_load(fname, &nx, &ny, &nc, 3);
    bool ok = qnn_embed_image (mctx, rgb_pixels, nx, ny);
    stbi_image_free(rgb_pixels);
    return ok;
}
bool qnn_embed_image_from_bytes(qnn_embed_t* mctx, const uint8_t* data, size_t data_len){
    int nx, ny, nc; 
    unsigned char * rgb_pixels = stbi_load_from_memory(data, data_len, &nx, &ny, &nc, 3);    
    bool ok = qnn_embed_image (mctx, rgb_pixels, nx, ny);
    stbi_image_free(rgb_pixels);
    return ok;
}

//#include "common.h"
#include "llama.h"
static void _log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    FILE* fp = (FILE*)user_data;
    if (fp) fprintf(fp, "%s\n", text);
}
/*! \brief 
    \see examples/embedding/embeddings.cpp
 */
void qnn_init()
{
    ggml_time_init();
    llama_log_set(_log_callback, stdout); // user_data
    // common_init();
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
}
void qnn_fini(llama_context * lctx)
{
//    llama_perf_context_print(lctx);
    // clean up
//    llama_batch_free(batch);
    llama_backend_free();
}
/*! 

**TEST_EMBED**

*/
#ifdef TEST_EMBED
#include <glib.h>
#include <locale.h>

#include "arg.h"
#include "llama.h"

typedef struct _MainOptions MainOptions;
struct _MainOptions {
    char * input_file;	//!< входной файл
    char *  log_file;	//!< входной файл журнала JSON Lines
    char * model;		//!< LLM модель в формате GGUF
    int verbose;		//!< выводить информацию о прогрессе
	int version;		//!< выводить информацию о версии программы
	int overwrite;		//!< перезаписывать выходной файл
};
MainOptions options = {NULL};
static GOptionEntry entries[] =
{
  { "input",  	'i', 0, G_OPTION_ARG_FILENAME, &options.input_file,  "input  `filename`", "*.gguf"},
  { "model", 	'm', 0, G_OPTION_ARG_FILENAME, &options.model, 		 "model  `filename`", "*.gguf" },
  { "log",  	'l', 0, G_OPTION_ARG_FILENAME, &options.log_file,    "log    `filename`", "*.jsonl"},
  { "overwrite",'O', 0, G_OPTION_ARG_NONE,	 &options.overwrite, "overwrite output", NULL },
  { "verbose", 	'v', 0, G_OPTION_ARG_NONE,	 &options.verbose, 	 "Be verbose", 	 NULL },
  { "version", 	'V', 0, G_OPTION_ARG_NONE,	 &options.version, 	 "program info", NULL },
  { NULL }

};
int main(int argc, char** argv)
{
	setlocale(LC_ALL, "ru_RU.UTF-8");
	setlocale(LC_NUMERIC, "C");

    GError* error=NULL;
	GOptionContext *opt_context;
    opt_context = g_option_context_new ("- command line interface");
    g_option_context_add_main_entries (opt_context, entries, NULL/*GETTEXT_PACKAGE*/);
    if (!g_option_context_parse (opt_context, &argc, &argv, &error))
    {
        fprintf (stderr, "option parsing failed: %s\n", error->message);
        return 1;
    }
    g_option_context_free (opt_context);
	if (options.version){
		fprintf (stdout, "VLM - Visual Embedding Test\n");
		return 0;
	}

    ggml_time_init();
    common_params params;

    common_init();
    params.embedding = true;
#if 0 // может убрать вовсе
	if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, NULL)) 
	{
		return 1;
	}
#endif

// utilize the full context
    if (params.n_batch < params.n_ctx) {
        fprintf(stdout, "%s: setting batch size to %d\n", __func__, params.n_ctx);
        params.n_batch = params.n_ctx;
    }

    // For non-causal models, batch size must be equal to ubatch size
    params.n_ubatch = params.n_batch;

    llama_backend_init();
    llama_numa_init(params.numa);

    if (options.log_file!=NULL 
    &&  g_file_test (options.log_file, G_FILE_TEST_IS_REGULAR)
    &&  g_str_has_suffix(options.log_file, ".jsonl")) {
        // разобрать по строчкам
    }

    const char* embed_model_name = options.model;//argv[1];
    qnn_embed_t * mctx = qnn_embed_init(embed_model_name, 0);
    if (mctx==NULL) return -2;
    int n_mmproj = qnn_embed_n_mmproj(mctx); // this should be equal to the embedding dimension of the text model

    for (int i = 1; i<argc; i++) {
        char* img_file = argv[i];
        if (!g_file_test(img_file, G_FILE_TEST_IS_REGULAR)){
            continue;
        }
        if (g_str_has_suffix(img_file, ".jpg")) {
            qnn_embed_rotate(mctx);
            bool ok = qnn_embed_image_from_file(mctx, img_file);
            int n_tok = qnn_embed_n_tokens(mctx);
            int  t  = qnn_embed_time(mctx);
            float ss= qnn_embed_ssim(mctx, 1);
            float ds= qnn_embed_dist(mctx, 1); // сделать luminance маску размером n_tok!
            fprintf(stdout, "file %s: %s, %1.4f, ds = %1.4f, time %1.1f ms/%d tokens\n", img_file, ok?"ok":"fail", ss, ds, (double)t/1000, n_tok);
        } else 
        if (g_str_has_suffix(img_file, ".ts")
        ||  g_str_has_suffix(img_file, ".mp4")){
            // файл содержит метку времени, надо найти привязку
            qnn_embed_mpeg2ts(mctx, img_file);
        }

    }
    qnn_embed_free(mctx);
    return 0;
}
#endif
