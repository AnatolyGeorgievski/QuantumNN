/*! Надо реализовать С-интерфейс для запуска моделей 
    API:
    * qnn_llama_init
    * qnn_llama_prompt  (ctx, seq_id, role, message) -- 
    * qnn_llama_classify(ctx, seq_id, role, message, image) -- 
для задач embedding и re-ranking нет фазы генерации, только декодирование. 
 */
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include <common.h>
#include <llama.h>
#include <mtmd.h>
#include <mtmd-helper.h>
#include <glib.h>
#include "json.h"
#include "qnn_embed.h"

#if 0
int _text(struct llama_context * lctx)
{
    struct llama_batch text_batch = llama_batch_init(n_batch, 0, 1);
    // заполнить batch
    for (; i < n_tokens && text_batch.n_tokens < n_batch; i++) {
    }
    int ret = llama_decode(lctx, text_batch);

    llama_batch_free(text_batch);
    return ret;
}
void _embed(float * embd){
    // encode image
    // decode image \see mtmd_helper_decode_image_chunk()
    llama_batch embd_batch;
    for (int i_batch =0; I< n_img_batch; i_batch++){// разбить на задания
        int pos_offset = i_batch*n_batch;
        int n_tokens_batch = std::min(n_batch, n_tokens - pos_offset);
        llama_batch batch_embd_view = batch_embd.get_view(pos_offset, n_tokens_batch);
        ret = llama_decode(lctx, batch_embd_view);
    }
}
#endif

/*! \brief преобразовать текст в вектор токенов 

    Преобразование использует словарь из контекста
    \param lctx контекст llama
    \param text строка с текстом
    \return вектор токенов, следует освободить с использованием free()
 */
int32_t qnn_llama_tokenize(const struct llama_context * lctx, llama_token * tokens, int32_t rsize, const char* text, size_t text_len)
{
    const llama_model * model = llama_get_model(lctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int add_special = 0;
    int parse_special = 1;
    // это возможно не самый эффективный способ выделения памяти, но он используется в качестве примера в llama.cpp
    // в некоторых случаях число токенов известно или ограничено
//    llama_token * result = (llama_token *)malloc(rsize*sizeof(llama_token));
    return llama_tokenize(vocab, text, text_len, tokens, rsize, add_special, parse_special);
/*
    if (n_tokens == ~__INT32_MAX__)
    if (n_tokens<0) {// resize
        result = (llama_token *)realloc(result, (-n_tokens)*sizeof(llama_token));
        n_tokens = llama_tokenize(vocab, text, text_len, result, rsize, add_special, parse_special);
    }
    return result; */
}
/*! \brief преобразовать токен в текст 

    Преобразование использует словарь из контекста
    \param lctx контекст llama
    \param piece буфер для результата, должен быть достаточного размера для одного токена
    \param token токен
    \param special флаг, указывающий, что токен является специальным
    \return количество символов в возвращаемом фрагменте текста. 
    Если результат не умещается в выделенной памяти, то возвращается отрицательное значение, 
    равное размеру необходимой памяти.

    \note завершение разбора последовательности токенов может быть достигнуто, 
    если возвращается специальный токен LLAMA_TOKEN_EOS.
 */
int32_t _token_to_piece(const struct llama_context * lctx, char* piece, size_t plen, llama_token token, bool special)
{
    const llama_model * model = llama_get_model(lctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    return llama_token_to_piece(vocab, token, piece, plen, 0, special);
}
/*! \brief инициализация */
#if 0
int _init(struct llama_context * lctx)
{
    const llama_model * model = llama_get_model(lctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int32_t n_tokens = llama_tokenize(vocab, text, text_len, tokens, rsize, 0, 1);
    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    const int n_embed = llama_model_n_embd(model);
    float * embed = g_malloc0(sizeof(float) * n_embed);
    llama_decode(lctx, batch);
}
#endif
/*
struct _GSList {
    void* data;
    struct _GSList * next;
};*/

#define MAX_CHANNELS 1 // число каналов в batch задании
struct mtmd_ctx {
    struct llama_model       * model;
    struct llama_context     * lctx;
    const struct llama_vocab * vocab;
    struct llama_batch         batch;
    int                      n_batch;
    int                      n_threads; // число потоков для генерации
    std::vector<mtmd_bitmap *> bitmaps;

    mtmd_context* ctx_vision;
    llama_pos n_past;
};
/*! \brief инициализация контекста визуальной модели
    \param mctx контекст llama, контекст визуальной модели
    \param options параметры модели
    \return контекст llama
 */
struct mtmd_ctx * qnn_llama_init(struct mtmd_ctx * mctx, struct mtmd_params* options)
{
    if (mctx==nullptr) {
        mctx = new mtmd_ctx();
    }
printf("start common_init\n");

    ggml_time_init();

	struct common_params params;
	params.sampling.temp = 0.2; // lower temp by default for better quality
	params.sampling.seed = 42;
#if 0 // может убрать вовсе
	if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, show_additional_info)) 
	{
		return 1;
	}
#endif
    if (options->embedding)  // text embedding or re-ranking
        params.embedding = true; // может быть специальная модель тип embedding

// common_params_parser_init()
	params.model.path  = options->model;
// llama_supports_rpc() -- определяет является ли модель RPC
//	params.model.hf_repo = ;
//	params.model.hf_file = ;
	// добавить длину контекста batch ubatch, и реализовать ротацию KV кэша. - для бесконечной генерации
	// https://github.com/ggml-org/llama.cpp/blob/4078c77f9891831f29ffc7c315c8ec6695ba5ce7/examples/main/main.cpp#L552-L585
	// --no-context-shift                      disables context shift on infinite text generation (default: disabled)
	// --cache-reuse N                         min chunk size to attempt reusing from the cache via KV shifting
	params.ctx_shift = true;
	params.swa_full = true; // see https://github.com/ggml-org/llama.cpp/pull/13194 -- этот параметр - заглушка, чтобы работал сдвиг
//	params.n_sequences  = 2;

	params.n_parallel = MIN(MAX_CHANNELS, options->n_channels);
	if (params.n_parallel == 1)
        params.kv_unified = true;
	params.n_cache_reuse = 256;
//	params.n_keep = 512;
	params.n_ctx = options->n_ctx;// 16384; 32768
    params.n_batch = 512;
	params.n_ubatch = params.n_batch;

	params.flash_attn_type = (enum llama_flash_attn_type)options->flash_attn;

    /*
        LLAMA_FLASH_ATTN_TYPE_AUTO     =-1,
        LLAMA_FLASH_ATTN_TYPE_DISABLED = 0,
        LLAMA_FLASH_ATTN_TYPE_ENABLED  = 1,
    */
//	params.defrag_thold = 0;
	params.n_gpu_layers = 999;
	params.no_op_offload = true;
	params.no_kv_offload = false;
	params.mmproj_use_gpu = true;
	params.mmproj.path = options->mmproj;
//	params.mmproj.hf_repo = "https://huggingface.co/..";
//	params.mmproj.hf_file = "model.safetensors";

	params.system_prompt = options->system_prompt;
	if (options->prompt) 
		params.prompt	= options->prompt;
	params.verbosity = 1;
	params.verbose_prompt = options->verbose;

// распределение слоев модели по 
	params.split_mode = LLAMA_SPLIT_MODE_LAYER;

    if (0) {
		params.tensor_split[0] = 1.0f;
		params.tensor_split[1] = 1.0f;
		for (size_t i = 1; i < llama_max_devices(); ++i) {// llama_max_devices - const
			params.tensor_split[i] = 0.0f;
		}
	}

    common_init();
    common_init_result llama_init  = common_init_from_params(params);

    mctx->model = llama_init.model.get();
    mctx->lctx  = llama_init.context.get();
    mctx->vocab = llama_model_get_vocab(mctx->model);
    mctx->n_threads = params.cpuparams.n_threads;
    mctx->batch = llama_batch_init(params.n_batch, 0, 1);// TODO может изменить?
    mctx->n_batch = params.n_batch;

    if (!mctx->model || !mctx->lctx) {
        exit(1);
    }
    //init_vision_context(params);
    const char * clip_path = params.mmproj.path.c_str();
    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = params.mmproj_use_gpu;
    mparams.image_marker = "<__image__>";
    mparams.print_timings = true;
    mparams.n_threads = params.cpuparams.n_threads;
    mparams.verbosity = params.verbosity > 1 ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_ERROR;

    mctx->ctx_vision = mtmd_init_from_file(clip_path, mctx->model, mparams);
    if (mctx->ctx_vision==NULL) {
        fprintf(stderr, "Failed to load vision model from %s\n", clip_path);
        exit(1);
    }
    return mctx;
}
/*! \brief освободить визуальный контекст, включая контекст медиа-проектора и контекст LLM
    \param mctx контекст визуальной модели
 */
void qnn_llama_free(struct mtmd_ctx* mctx)
{
    mtmd_free(mctx->ctx_vision);
    llama_model_free(mctx->model);
    llama_free(mctx->lctx);
    free(mctx);
}
extern "C" {// декларация функций {вынести}
	mtmd_bitmap* _stb_load(const char * filename);
	mtmd_bitmap* _stb_load_from_memory(const unsigned char * bytes, size_t bytes_length);
};
#if 0
/*! \brief загрузить изображение из буфера
    \param ctx контекст
    \param data данные изображения
    \return 0 - успешно, 1 - ошибка загрузки изображения
 */
int load_image_from_buf(struct mtmd_ctx* ctx, uint8_t * data, size_t len){
    mtmd_bitmap *bmp = _stb_load_from_memory(data, len);//mtmd_helper_bitmap_init_from_buf(data, len);
    if (bmp == nullptr) {// если убрать работать не будет
        fprintf(stderr, "Failed to load image from buffer\n");
        return 1;
    }
    ctx->bitmaps.push_back(bmp);
    return 0;
}
#endif
/*! \brief загрузить изображение из файла
    \param fname имя файла изображения
    \return указатель на изображение или NULL при ошибке
 */
static mtmd_bitmap * load_image_from_file(const char* fname){
    mtmd_bitmap *bmp = _stb_load(fname);//mtmd_helper_bitmap_init_from_buf(data, len);
    if (bmp == nullptr) {// если убрать работать не будет
        fprintf(stderr, "Failed to load image from file\n");
    }
    return bmp;
}
/*! \brief загрузить изображение из буфера
    \param data данные изображения
    \param len размер данных в байтах
    \return указатель на изображение или NULL при ошибке
 */
static mtmd_bitmap * load_image_from_data(uint8_t * data, size_t len){
    mtmd_bitmap *bmp = _stb_load_from_memory(data, len);//mtmd_helper_bitmap_init_from_buf(data, len);
    if (bmp == nullptr) {// если убрать работать не будет
        fprintf(stderr, "Failed to load image from file\n");
    }
    return bmp;
}

void _log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    FILE* fp = (FILE*)user_data;
    fprintf(fp, "%s\n", text);
}
/*! */
static int batch_decode(llama_context * lctx, llama_batch & batch, float * output)
{
    const llama_model * model = llama_get_model(lctx);
    const int n_embd = llama_model_n_embd(model);
    if (llama_decode(lctx, batch) < 0) { // run model
        printf("%s : failed to process\n", __func__);
    }
    enum llama_pooling_type pooling_type = llama_pooling_type(lctx);
    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {// TODO: rename "logits" to "output"
            continue;
        }
        const float* embd = nullptr;
        llama_seq_id embd_pos = 0;
        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            // try to get token embeddings
            embd = llama_get_embeddings_ith(lctx, i);
            embd_pos = i;
            GGML_ASSERT(embd != NULL && "failed to get token embeddings");
        } else {
            // try to get sequence embeddings - supported only when pooling_type is not NONE
            embd = llama_get_embeddings_seq(lctx, batch.seq_id[i][0]);
            embd_pos = batch.seq_id[i][0];
            GGML_ASSERT(embd != NULL && "failed to get sequence embeddings");
        }
        float * out = output + embd_pos * n_embd;
        // normalization
        float norm = 1.0f;
        if (1) {// euclidean norm
            double sum = 0;
            for (int i = 0; i < n_embd; i++)
                sum += embd[i] * embd[i];
            norm = 1.0/sqrt(sum);
        } else {
            for (int i = 0; i < n_embd; i++)
                norm = fmaxf(fabsf(embd[i]), norm);
        }
        for (int i = 0; i < n_embd; i++)
            out [i] = embd[i] * norm;
    }
    return 0;
}
static int embedding_prompt(struct mtmd_ctx* mctx, const char* prompt, float* output) {
    const llama_vocab* vocab = mctx->vocab;
    llama_token eos = llama_vocab_eos(vocab);// end of sequence
    int32_t tlen = strlen(prompt);
    int rsize = mctx->n_batch;
    llama_token* tokens = (llama_token*)malloc(rsize*sizeof(llama_token));
    int32_t n_tokens = llama_tokenize(vocab, prompt, tlen, tokens, rsize, true, true);
    if (n_tokens<0) {// resize n_tokens> n_batch
        rsize = 1-n_tokens;
        tokens = (llama_token*)realloc(tokens, rsize*sizeof(llama_token));// resize
        n_tokens = llama_tokenize(vocab, prompt, tlen, tokens, rsize-1, true, true);
    }
    tokens[n_tokens++] = eos;

    llama_seq_id seq_id=0;
    if (1/* is_embedding */)
        llama_memory_clear(llama_get_memory(mctx->lctx), true);
    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    // разделить на пакеты `batch` каждый пакет относится к одному или более  seq_id
    // TODO а что если число токенов больше n_batch?
    for (size_t i = 0; i < n_tokens; i++) {
        common_batch_add(batch, tokens[i], i, { seq_id }, true);
    }
    int res = batch_decode(mctx->lctx, batch, output);
    common_batch_clear(batch); // очищать перед следующим пакетом
    llama_batch_free(batch);
    return 0;
}
// Этот вариант для re-ranking модели
static int reranking_prompt(struct mtmd_ctx* mctx, GSList* prompts, float* output) {
    const llama_vocab* vocab = mctx->vocab;
    // if (llama_pooling_type(ctx->lctx) != LLAMA_POOLING_TYPE_RANK) return -1;
    llama_token eos = llama_vocab_eos(vocab);// end of sequence
    llama_token sep = llama_vocab_sep(vocab);// separator
    GString* str = g_string_new("");
    GSList* list = prompts;
    int offs = 0;
    int rsize = mctx->n_batch;
    llama_token* tokens = nullptr;// ???
    while (list){// составить строку запроса с разделителями
        const char* text = (const char*)list->data;
        int32_t tlen = strlen(text);
        int32_t n_tokens = llama_tokenize(vocab, text, tlen, tokens+offs, rsize-offs, true, true);
        if (n_tokens<0) {// resize
            tokens = (llama_token*)realloc(tokens, 1-n_tokens);// resize
            n_tokens = llama_tokenize(vocab, text, tlen, tokens+offs, rsize-offs, true, true);
        }
        offs += n_tokens;
        tokens[offs++] = list->next?sep:eos;
        list = list->next;
    }
    if (1/* is_embedding */)
        llama_memory_clear(llama_get_memory(mctx->lctx), true);
    int res = batch_decode(mctx->lctx, mctx->batch, output);
    return res;
}
static int eval_message(struct mtmd_ctx* ctx, llama_seq_id seq_id, const char* prompt, bool add_bos = false) 
{
	mtmd_context* mctx = ctx->ctx_vision;//->get();
	mtmd_input_text text;
    text.text          = prompt;
    text.add_special   = add_bos;
    text.parse_special = true;

	mtmd::input_chunks chunks(mtmd_input_chunks_init());
    // auto bitmaps_c_ptr = ctx.bitmaps.c_ptr();
    int32_t res = mtmd_tokenize(mctx,
                        chunks.ptr.get(), // output chunks
                        &text, // text
                        (const mtmd_bitmap**)ctx->bitmaps.data(),
                        ctx->bitmaps.size());
    if (res != 0) {
        fprintf(stderr, "Unable to tokenize prompt, res = %d\n", res);
        return 1;
    }

	//ctx.bitmaps.entries.clear();
	//ctx.bitmaps.clear();
	// see server.cpp => utils.hpp => server_tokens::process_chunk
	// mtmd_helper_eval_chunk_single (mctx, lctx, )
    llama_pos new_n_past;
    if (mtmd_helper_eval_chunks(mctx,// mctx
                ctx->lctx, 			// lctx
                chunks.ptr.get(), 	// chunks
                ctx->n_past, 		// n_past
                seq_id, // seq_id
                ctx->n_batch, // n_batch
                true, // logits last
                &new_n_past)) {
					fprintf(stderr, "Unable to eval prompt\n");
        return 1;
    }
//	ctx.bitmaps.clear();// удалить изображения
    ctx->n_past = new_n_past;
    return 0;
}
/*! \brief задать параметры контекста - сообщение от имени окружения 
    \param mctx контекст визуальной модели llama
    \param seq_id идентификатор последовательности в многопоточном режиме
    \param role имя роли: user, system, model, assistant, context
    \param prompt текст запроса
    \return код ошибки или 0
 */
int qnn_llama_prompt(struct mtmd_ctx* mctx, int32_t seq_id, const char* role, const char * prompt)
{
    GString * str = g_string_sized_new(256);
//  if (is_first_msg)  g_string_append(str, "<bos>");
    g_string_append_printf(str, 
        "<start_of_turn>%s %s<end_of_turn>\n"
        "<start_of_turn>%s " // перейти к генерации
        , role, prompt,"model");

    char * message = g_string_free(str, false);
    int res = eval_message(mctx, seq_id, message, false);
    g_free(message);
    // mctx->prompt = prompt;
    return res;
}
/*! \brief классификация изображения 
    \param mctx контекст визуальной модели llama
    \param seq_id идентификатор последовательности в многопоточном режиме
    \param role имя роли: user, system, model, assistant, context
    \param prompt текст запроса, может содержать поля для заполнения
    \param img_fname имя файла с изображением
    \return ключевое слово из списка классов

    `"${channel}" [${UTC}] классифицируй <__image__>`
 */
int qnn_llama_classify(struct mtmd_ctx* mctx, int32_t seq_id, const char* role, const char * prompt, 
        const char * img_fname)
{
    GString * str = g_string_sized_new(256);
//  if (is_first_msg)  g_string_append(str, "<bos>");
    g_string_append_printf(str, "<start_of_turn>%s ", role);
    g_string_append(str, prompt);
//				offs += " <__image__>"; -- уже добавлен в промпт
    g_string_append(str, "<end_of_turn>\n");
    g_string_append(str, "<start_of_turn>model");

    if (img_fname) { // можно вынести, сделать загрузку из памяти
        mtmd_bitmap * bmp = load_image_from_file(img_fname);
        if (bmp) mctx->bitmaps.push_back(bmp);
    }
    char* message = g_string_free(str, false);
    if (eval_message(mctx, seq_id, message, false)) {
        return 1;
    }
    g_free(message);
    // mctx->prompt = prompt;
    return 0;
}
/*! \brief 
    TODO генерация параллельно несколько сообщений
 */
int qnn_generate_response(struct mtmd_ctx* mctx, llama_seq_id seq_id){
    return 0;
}


#if defined(TEST_LLAMA)
/* Разбор конфигурационных файлов и `JSON` документов 
    выполняется с использованием кварков `GQuark`, для этого заранее готовим пространство имен
 */
#define _JCONFIG_NAMESPACE \
    TAG(manifest),TAG(mmproj),TAG(model),TAG(embed),TAG(prompt),TAG(system_prompt),TAG(n_ctx), \
    TAG(rpc),TAG(tools)
#define TAG(name) #name // stringify
static const char* names[] = {// список имен идентификаторов
	_JCONFIG_NAMESPACE,
	NULL
};
#undef TAG
#define TAG(name) JCONFIG_TAG_##name
enum {// перечисление идентификаторов имен
	_JCONFIG_NAMESPACE,
	_JCONFIG_TAG_COUNT
};

static GQuark tags[_JCONFIG_TAG_COUNT];
/*! \brief конструктор таблицы идентификаторов */
static void __attribute__((constructor)) _tag_init()
{// статический конструктор, формирует таблицу идентификаторов
	for(int i=0; i<_JCONFIG_TAG_COUNT; i++){
		tags[i] = g_quark_from_static_string(names[i]);
	}
}
#undef  TAG
#define TAG(name) tags[JCONFIG_TAG_##name]

typedef struct _MainOptions MainOptions;
struct _MainOptions {
    char * log_file;	//!< файл журнала JSON Lines
    char * input_file;	//!< входной файл
    char * output_file;	//!< выходной файл
    char * config_file;	//!< файл конфигурации в формате JSON
    struct mtmd_params llama; //!< параметры модели см. <qnn_embed.h>
    char * manifest;	//!< manifest файла, содержит список хешей от тензоров модели
    char * embed;		//!< Image Embedding модель в формате GGUF
    char * pooling_type; //!< тип вычисления вектора 
    char * normalization; //!< тип нормализации: L2, L1, inf, none

    int all;            //!< обработать все каналы
    int verbose;		//!< выводить информацию о прогрессе
	int version;		//!< выводить информацию о версии программы
	int overwrite;		//!< перезаписывать выходной файл
};
MainOptions options = {
    .llama = {
        .system_prompt = (char*)"You are a helpful assistant.",
        .prompt = (char*)"",
        .n_ctx = 8192,//16384, -- уменьшить до параметра модели
    },
};
static GOptionEntry entries[] = {
  { "input",  	'i', 0, G_OPTION_ARG_FILENAME, &options. input_file, "input  `filename`", "*.json"},
  { "output",  	'o', 0, G_OPTION_ARG_FILENAME, &options.output_file, "output `filename`", "*.json"},
  { "config",  	'C', 0, G_OPTION_ARG_FILENAME, &options.config_file, "config `filename`", "*.json"},
  { "log",  	'L', 0, G_OPTION_ARG_FILENAME, &options.   log_file, "log    `filename`", "*.jsonl"},
  { "manifest", 'M', 0, G_OPTION_ARG_FILENAME, &options.manifest,    "manifest `filename`", "*.json"},
// модели нейросетей языковых и визуальных
  { "model", 	'm', 0, G_OPTION_ARG_FILENAME, &options.llama.model, "multimodal LLM   `filename`", "models/*.gguf" },
  { "mmproj", 	'P', 0, G_OPTION_ARG_FILENAME, &options.llama.mmproj,"media projector  `filename`", "models/mmproj-*.gguf" },
  { "prompt", 	'p', 0, G_OPTION_ARG_STRING, &options.llama.prompt, "user prompt", NULL },
  { "system", 	's', 0, G_OPTION_ARG_STRING, &options.llama.system_prompt,"system prompt", NULL },
  { "ctx",      'c', 0, G_OPTION_ARG_INT,    &options.llama.n_ctx,   "context size", "0-default"},
//  { "batch",    'b', 0, G_OPTION_ARG_INT,    &options.llama.n_batch, "batch size", "0-default"},
// параметры приложения (Embeddings)
  { "embed", 	'e', 0, G_OPTION_ARG_FILENAME, &options.embed, 		 "embedding model  `filename`", "models/mmproj-*.gguf" },
  { "pooling", 	 0 , 0, G_OPTION_ARG_STRING, &options.pooling_type, "pooling type", "{last|mean}" },
  { "norm", 	'n', 0, G_OPTION_ARG_STRING, &options.normalization,"normalization", "{L2|L1|inf|none}" },
// общие параметры
  { "all",      'a', 0, G_OPTION_ARG_NONE,	 &options.all,       "process all channels", NULL },
  { "overwrite",'O', 0, G_OPTION_ARG_NONE,	 &options.overwrite, "overwrite output", NULL },
  { "verbose", 	'v', 0, G_OPTION_ARG_NONE,	 &options.verbose, 	 "Be verbose", 	 NULL },
  { "version", 	'V', 0, G_OPTION_ARG_NONE,	 &options.version, 	 "program info", NULL },
  { NULL }
};

#include <locale.h>
int main(int argc, char** argv)
{
    qnn_init();
#if 1
	setlocale(LC_ALL, "en_US.UTF-8");
	setlocale(LC_NUMERIC, "C");
#else
	setlocale(LC_ALL, "");
#endif

	GOptionContext *opt_context;
    opt_context = g_option_context_new ("- command line interface");
    g_option_context_add_main_entries (opt_context, entries, NULL/*GETTEXT_PACKAGE*/);
    if (!g_option_context_parse (opt_context, &argc, &argv, NULL)){
        //printf ("option parsing failed: %s\n", error->message);
        _Exit(1);
    }
    g_option_context_free (opt_context);
	GError* error = NULL;
	if (options.version){
		fprintf (stdout, "Embedding LLaMa test\n");
		return 0;
	}

	FILE* log_file 	 = stdout;// файл журнала в формате JSON Lines
	if (options.log_file!=NULL){// файл журнала
		log_file = fopen(options.log_file, "a");
	}
	char* config 	 = NULL;// глобальная конфигурация, текстовый файл
	JsonNode* root	 = NULL;// глобальная конфигурация чтобы удалить
	char* tools_str  = NULL;// описание инструментов
	GSList* rpc	 	 = NULL;// список серверов RPC
	GSList* tools	 = NULL;// набор инструментов
	GSList* channels = NULL;// список каналов и их описание
	if (options.config_file!=NULL){// загрузка конфигурации
		size_t max_length = 0;
		if(g_file_get_contents(options.config_file, &config, &max_length, NULL)){
			printf("config file: %s\n", options.config_file);
			root = json_value(config, NULL, NULL);
			if (root->type==JSON_OBJECT){
                options.manifest 	        = (char*)json_get_string(root->value.list, TAG(manifest), options.manifest);
				options.llama.mmproj 		= (char*)json_get_string(root->value.list, TAG(mmproj), options.llama.mmproj);
				options.llama.model  		= (char*)json_get_string(root->value.list, TAG(model),  options.llama.model );
				options.embed  		        = (char*)json_get_string(root->value.list, TAG(embed),  options.embed );
				options.llama.system_prompt = (char*)json_get_string(root->value.list, TAG(system_prompt),  options.llama.system_prompt );
				options.llama.prompt 		= (char*)json_get_string(root->value.list, TAG(prompt), options.llama.prompt );
                options.llama.n_ctx         = json_get_uint(root->value.list, TAG(n_ctx),  options.llama.n_ctx);
				rpc   	= json_get_array(root->value.list, TAG(rpc));
				tools   = json_get_array(root->value.list, TAG(tools));
				printf("model : %s\n", options.llama.model );
				if (options.llama.mmproj) 
					printf("mmproj: %s\n", options.llama.mmproj);
				if (options.embed) 
					printf("embed : %s\n", options.embed);
				printf("prompt: %s\n", options.llama.prompt);
			}
		}
	}
    struct mtmd_ctx *mctx = NULL; // мультимодальный контекст
    //TODO: qnn_rag_t *rctx = NULL; // контекст для embedding и re-ranking, поиск по семантической схожести
    // если определена языковая модель для function calling и определен визуальный энкодер
    if (options.llama.model && options.llama.mmproj) 
        mctx = qnn_llama_init(NULL, &options.llama);

    // params.embedding = true;
    // if (params.n_batch < params.n_ctx) params.n_batch = params.n_ctx;
//    params.n_ubatch = params.n_batch;
//    struct llama_model_params = llama_model_default_params();
    struct llama_model_params mparams = {
        /*.devices                     =*/ NULL,
        /*.tensor_buft_overrides       =*/ NULL,
        /*.n_gpu_layers                =*/ 999,
        /*.split_mode                  =*/ LLAMA_SPLIT_MODE_LAYER,
        /*.main_gpu                    =*/ 0,
        /*.tensor_split                =*/ NULL,
        /*.progress_callback           =*/ NULL,
        /*.progress_callback_user_data =*/ NULL,
        /*.kv_overrides                =*/ NULL,
        /*.vocab_only                  =*/ false,
        /*.use_mmap                    =*/ true,
        /*.use_mlock                   =*/ false,
        /*.check_tensors               =*/ false,
    };
// \see common_context_params_to_llama \see common_init_from_params
    if (mctx){
        llama_model_free(mctx->model);
        llama_free(mctx->lctx);
    }
    // qnn_fini();
    return 0;
}
#endif