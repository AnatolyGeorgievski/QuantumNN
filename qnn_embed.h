#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

    // параметры модели MLLM
    struct mtmd_params {
        char* model;    //!< путь к модели в формате gguf
        char* mmproj;   //!< media projector путь к модели в формате gguf
        char* system_prompt;   //!< системный prompt
        char* prompt;   //!< пользовательский prompt
        int n_ctx;      //!< размер контекста
        int n_threads;  //!< число потоков для генерации
        int n_channels; //!< число каналов в параллель 
        int flash_attn; //!< использовать макро определение операции FA: -1 (default), 0, 1
        int embedding;  //!< if text embedding or re-ranking модель
        int verbose;    //!< verbose prompt
    };

// инициализация библиотек, параметром является callback для записи журнала
     void qnn_init();
// функционал сервера LLM 
struct mtmd_ctx * qnn_llama_init  (struct mtmd_ctx* mctx, struct mtmd_params* options);
     void qnn_llama_free    (struct mtmd_ctx* mctx);
      int qnn_llama_prompt  (struct mtmd_ctx* mctx, int32_t seq_id, const char* role, const char * prompt);
      int qnn_llama_classify(struct mtmd_ctx* mctx, int32_t seq_id, const char* role, const char * prompt,
                const char * img_fname);
// операции над Visual Embedding векторами
    float qnn_embed_structural_sim(const float* u, const float* v, float* l, float *c, unsigned n);
    float qnn_embed_similarity_cos(const float* u, const float* v, unsigned n);

    float qnn_embed_distance(const float* u, const float* v, unsigned n);
    float qnn_embed_dot     (const float* u, const float* v, unsigned n);
    float qnn_embed_sad     (const float* u, const float* v, unsigned n);
    float qnn_embed_scale   (      float* r, const float* v, unsigned n, float scale);
    float qnn_embed_softmax (      float* r, const float* v, unsigned k, float tau);
    float qnn_embed_l2normalize(   float* r, const float* v, unsigned n);
    float qnn_embed_rms_norm(      float* r, const float* v, unsigned n);
     void qnn_embed_time_mix(      float* r, const float* v, unsigned n, float mu);
    float qnn_embed_layer_norm(    float* r, const float* v, unsigned n);
     void qnn_embed_sum     (      float* r, const float* v, unsigned n, unsigned k);
     void qnn_embed_sub     (      float* r, const float *a, const float *b, unsigned n, unsigned k);
     void qnn_embed_mean    (      float* r, const float* v, unsigned n, unsigned k);

     //  float qnn_embed_batch_norm(    float* r, const float* v, unsigned n, unsigned k);

    typedef struct _qnn_embedding_ctx qnn_embed_t;
    typedef struct _qnn_scene qnn_scene_t;
    struct _qnn_scene {
         int64_t pos_frame;     //!< время в микросекундах;
        uint64_t argmax;        //!< метка времени для опорного кадра
         int32_t duration;      //!< время в микросекундах от начала интервала
        float    ss_min;        //!< признак схожести с предыдущим кадром - граница
        float    ss_max;        //!< признак схожести с предыдущим кадром - значение опорного кадра
        const char* type;       //!< тип кадра, классификация
    };
    void qnn_scene_free(struct _GSList* scenes);
    
    qnn_embed_t* qnn_embed_init(const char * fname, int flags);
    bool qnn_embed_text (qnn_embed_t* mctx, const char* text, size_t mlen);
    bool qnn_embed_image(qnn_embed_t* mctx, const uint8_t* rgb_pixels, int nx, int ny);
    bool qnn_embed_image_from_bytes(qnn_embed_t* mctx, const uint8_t* data, size_t data_len);
uint64_t qnn_embed_mpeg2ts   (qnn_embed_t * emctx, const char* fname, uint64_t unix_usec, struct _GSList** scene);
uint64_t qnn_embed_mpeg2ts_at(qnn_embed_t * emctx, const char* fname, uint64_t unix_usec);
    void qnn_embed_free (qnn_embed_t* mctx);
  float* qnn_embed_get  (qnn_embed_t* mctx, int offs);
  float  qnn_embed_ssim (qnn_embed_t* mctx, int offs);
 int32_t qnn_embed_time (qnn_embed_t* mctx);
 int32_t qnn_embed_process_time(qnn_embed_t* mctx);
 int32_t qnn_embed_n_tokens (qnn_embed_t* mctx);
 int32_t qnn_embed_n_mmproj (qnn_embed_t* mctx);
     int qnn_embed_rotate   (qnn_embed_t* mctx);
#ifdef __cplusplus
};
#endif
