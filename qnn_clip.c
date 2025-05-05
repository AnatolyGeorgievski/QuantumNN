
#include <stdint.h>
#include "qnn.h"
#include "qnn_clip.h"
/*! \brief формирование контекста визуальной части модели
    \param params - параметры модели
    \return указатель на контекст
 */ 
clip_ctx_t * clip_init(struct clip_params* params)
{
    clip_ctx_t * clip = malloc(sizeof(struct _clip_ctx));
    clip->params = *params;
    return clip;
};
/*! \see [Тензорный_ассемблер](QNN_MODEL.md#Тензорный_ассемблер)
    \see [Переносимое представление графа тензорных операций](QNN_MODEL.md#Переносимое_представление_графа_тензорных_операций)
    \see [Кодирование CBOR](QNN_MODEL.md#Кодирование_CBOR)
 */
#define _MODEL(weight)      model->weight
#define _LAYER(il,weight)   model->layers[il].weight

struct _clip_model {// base class
    unsigned type       :8;    //!< type of model CLIP, SigLIP...
    unsigned n_layer    :8;    //!< number of layers
    unsigned n_tensor   :16;   //!< number of tensors per layer
    tensor_t* weights[0];//!< table of tensor weights
};
// SigLIP: self-attention with feed forward network and layer normalization. SigLIP use 16-bit weights in BF16. 
struct _siglip_layer {
    tensor_t* ln_1_w; // layer normalization weights
    tensor_t* ln_1_b;
    tensor_t* q_w;    // self-attention Q K V weights
    tensor_t* q_b;
    tensor_t* k_w;
    tensor_t* k_b;
    tensor_t* v_w;
    tensor_t* v_b;
    tensor_t* o_w;    // output weights
    tensor_t* o_b;
    tensor_t* ln_2_w; // layer normalization weights
    tensor_t* ln_2_b;
    tensor_t* ffn_i_w;// feed forward network input and output weights
    tensor_t* ffn_i_b;
    tensor_t* ffn_o_w;
    tensor_t* ffn_o_b;
};
struct _siglip_model {// :: _clip_model
    unsigned type       :8;    //!< type of model CLIP, SigLIP...
    unsigned n_layer    :8;    //!< number of layers
    unsigned n_tensor   :16;   //!< number of tensors per layer

    tensor_t* patch_embeddings;
    tensor_t* patch_bias;
    tensor_t* position_embeddings;
    tensor_t* post_ln_w;
    tensor_t* post_ln_b;
    tensor_t* mm_soft_emb_norm_w;
    tensor_t* mm_input_proj_w;
    struct _siglip_layer layers[0];
};
enum clip_tensor {
    CLIP_TENSOR_PATCH_EMB,
    CLIP_TENSOR_PATCH_BIAS,
    CLIP_TENSOR_POSITION_EMB,
    CLIP_TENSOR_PRE_LN_W,
    CLIP_TENSOR_PRE_LN_B,
    CLIP_TENSOR_POST_LN_W,
    CLIP_TENSOR_POST_LN_B,
    CLIP_TENSOR_PROJECTION,
    CLIP_TENSOR_MM_SOFT_EMB_NORM_W,
    CLIP_TENSOR_MM_INPUT_PROJ_W,

    CLIP_TENSOR_LN_1_W,
    CLIP_TENSOR_LN_1_B,
    CLIP_TENSOR_LN_2_W,
    CLIP_TENSOR_LN_2_B,
    CLIP_TENSOR_Q_W,
    CLIP_TENSOR_Q_B,
    CLIP_TENSOR_K_W,
    CLIP_TENSOR_K_B,
    CLIP_TENSOR_V_W,
    CLIP_TENSOR_V_B,
    CLIP_TENSOR_O_W,
    CLIP_TENSOR_O_B,
    CLIP_TENSOR_FFN_I_W,
    CLIP_TENSOR_FFN_I_B,
    CLIP_TENSOR_FFN_O_W,
    CLIP_TENSOR_FFN_O_B,
    CLIP_TENSOR_COUNT
};
//!< структура для сопоставления шаблона имени тензора и идентификатора
struct clip_model_name {
    enum clip_tensor id;
    const char *cname;
};
static
struct clip_model_name siglip_names[] = {
{ CLIP_TENSOR_PATCH_EMB,        "v.patch_embeddings.weight" },
{ CLIP_TENSOR_PATCH_BIAS,       "v.patch_embeddings.bias" },
{ CLIP_TENSOR_POSITION_EMB,     "v.position_embeddings" },
{ CLIP_TENSOR_PRE_LN_W,         "v.pre_ln.weight" },
{ CLIP_TENSOR_PRE_LN_B,         "v.pre_ln.bias" },
{ CLIP_TENSOR_POST_LN_W,        "v.post_ln.weight" },
{ CLIP_TENSOR_PROJECTION,       "v.projection" },
{ CLIP_TENSOR_MM_SOFT_EMB_NORM_W,"v.mm_soft_emb_norm.weight" },
{ CLIP_TENSOR_MM_INPUT_PROJ_W,  "v.mm_input_proj.weight" },
// layers 
{ CLIP_TENSOR_LN_1_W,           "v.blk.*.ln1.weight" },
{ CLIP_TENSOR_LN_1_B,           "v.blk.*.ln1.bias" },
{ CLIP_TENSOR_LN_2_W,           "v.blk.*.ln2.weight" },
{ CLIP_TENSOR_LN_2_B,           "v.blk.*.ln2.bias" },
{ CLIP_TENSOR_Q_W,              "v.blk.*.attn_q.weight" },
{ CLIP_TENSOR_Q_B,              "v.blk.*.attn_q.bias" },
{ CLIP_TENSOR_K_W,              "v.blk.*.attn_k.weight" },
{ CLIP_TENSOR_K_B,              "v.blk.*.attn_k.bias" },
{ CLIP_TENSOR_V_W,              "v.blk.*.attn_v.weight" },
{ CLIP_TENSOR_V_B,              "v.blk.*.attn_v.bias" },
{ CLIP_TENSOR_O_W,              "v.blk.*.attn_out.weight" },
{ CLIP_TENSOR_O_B,              "v.blk.*.attn_out.bias" },
{ CLIP_TENSOR_FFN_I_W,          "v.blk.*.ffn_down.weight" },
{ CLIP_TENSOR_FFN_I_B,          "v.blk.*.ffn_down.bias" },
{ CLIP_TENSOR_FFN_O_W,          "v.blk.*.ffn_up.weight" },
{ CLIP_TENSOR_FFN_O_B,          "v.blk.*.ffn_up.bias" },
};

/*! \brief создать хэш таблицу имен для поиска тензоров
    \param params - параметры модели
    \return указатель на контекст
 */ 
QTable_t* clip_vision_quarks(){
    QTable_t* quarks = _quark_new(256, 1024);
    for (int i=0; i<sizeof(siglip_names)/sizeof(struct clip_model_name); i++){
        _quark_insert(quarks, siglip_names[i].cname);
    }
    return quarks;
}
static inline
int clip_sdnv_validate(uint64_t sdnv, unsigned n_layer){
    return (sdnv>>8) < n_layer && (sdnv&0xFF) < CLIP_TENSOR_COUNT;
}
/*! \brief создать таблицу весов для визуальной части модели в переносимом виде 
    \param ctx_clip - контекст визуальной части модели
    \param quarks - указатель на таблицу имен для поиска тензоров в таблице
    \param infos - указатель на таблицу весов модели
    \param n_tensors - количество тензоров весов в модели
    \return SUCCESS (0) or ERROR (-1)

    \note Концепция. Как загружать таблицы, без разбора форматов.
    1. Я сначала формирую идентификаторы тензоров в виде sdnv,
    2. Загружаю таблицу весов в виде массива tensor_weight_t, где каждый тензор имеет идентификатор sdnv,
    3. Распределяем тензоры по слоям, используя структуру SDNV
 */
int clip_vision_tensors(clip_ctx_t* ctx_clip, tensor_weight_t *infos, int n_tensors)
{
    struct clip_vision_model * model = &ctx_clip->vision_model;
// распределить тензоры по слоям, таблица имен quarks и qnn_clip_model содержат упорядоченные имена
// \see    QTable_t* quarks = clip_vision_quarks();
#define CLIP_TENSORS_PER_LAYER (CLIP_TENSOR_COUNT-CLIP_TENSOR_LN_1_W)// число имен тензоров в слое
    uint32_t n_layer = ctx_clip->params.n_layer;
    size_t count = CLIP_TENSOR_COUNT + (n_layer-1)*CLIP_TENSORS_PER_LAYER;
    tensor_weight_t* weights = malloc(count*sizeof(void*));
    __builtin_bzero(weights, count*sizeof(void*));
    // model->weights = weights;
#define _WEIGHT(sdnv)   weights[(sdnv>>8)*CLIP_TENSORS_PER_LAYER + (sdnv&0xFF)]
    for (int it = 0; it < n_tensors; it++) { // по всем тензорам в таблице
        if (clip_sdnv_validate(infos[it].sdnv, n_layer))
            _WEIGHT(infos[it].sdnv) = infos[it];
    }
#undef _WEIGHT
/* цикл заменяет действия по заполнению таблицы весов приведенный ниже
 */
#if 0
    model->patch_embeddings = _get_tensor(infos, quarks, CLIP_TENSOR_PATCH_EMB);
    model->patch_bias   = _get_tensor(infos, quarks, CLIP_TENSOR_PATCH_BIAS);
    model->position_embeddings = _get_tensor(infos, quarks, CLIP_TENSOR_POSITION_EMB);
    model->post_ln_w    = _get_tensor(infos, quarks, CLIP_TENSOR_POST_LN_W);
    model->post_ln_b    = _get_tensor(infos, quarks, CLIP_TENSOR_POST_LN_B);
    model->projection   = _get_tensor(infos, quarks, CLIP_TENSOR_PROJECTION);
    model->mm_soft_emb_norm_w = _get_tensor(infos, quarks, CLIP_TENSOR_MM_SOFT_EMB_NORM_W);
    model->mm_input_proj_w = _get_tensor(infos, quarks, CLIP_TENSOR_MM_INPUT_PROJ_W);

    for (int il = 0; il < model->n_layer; il++) {
        struct clip_layer * layer = &model->layers[il];
        layer->ln_1_w = _get_tensor(infos, quarks, tn(il, CLIP_TENSOR_LN_1_W));
        layer->ln_1_b = _get_tensor(infos, quarks, tn(il, CLIP_TENSOR_LN_1_B));
        layer->q_w    = _get_tensor(infos, quarks, tn(il, CLIP_TENSOR_Q_W));
        layer->q_b    = _get_tensor(infos, quarks, tn(il, CLIP_TENSOR_Q_B));
        layer->k_w    = _get_tensor(infos, quarks, tn(il, CLIP_TENSOR_K_W));
        layer->k_b    = _get_tensor(infos, quarks, tn(il, CLIP_TENSOR_K_B));
        layer->v_w    = _get_tensor(infos, quarks, tn(il, CLIP_TENSOR_V_W));
        layer->v_b    = _get_tensor(infos, quarks, tn(il, CLIP_TENSOR_V_B));
        layer->o_w    = _get_tensor(infos, quarks, tn(il, CLIP_TENSOR_O_W));
        layer->o_b    = _get_tensor(infos, quarks, tn(il, CLIP_TENSOR_O_B));
        layer->ln_2_w = _get_tensor(infos, quarks, tn(il, CLIP_TENSOR_LN_2_W));
        layer->ln_2_b = _get_tensor(infos, quarks, tn(il, CLIP_TENSOR_LN_2_B));
        layer->ff_i_w = _get_tensor(infos, quarks, tn(il, CLIP_TENSOR_FFN_I_W));
        layer->ff_i_b = _get_tensor(infos, quarks, tn(il, CLIP_TENSOR_FFN_I_B));
        layer->ff_o_w = _get_tensor(infos, quarks, tn(il, CLIP_TENSOR_FFN_O_W));
        layer->ff_o_b = _get_tensor(infos, quarks, tn(il, CLIP_TENSOR_FFN_O_B));
    }
#endif
    return 0;
}


/*! \brief формирование графа вычислений для классификации изображений 
    \param ctx - контекст визуальной части модели
    \param img_batch - количество изображений в серии, для которых формируется граф
    \return указатель на граф, в форме таблицы

    \note Граф вычислений SigLIP составляется для одного изображения. 

    Модель Gemma3 состоит из двух частей: 
    - SigLIP - визуальная часть модели, которая преобразует изображение в векторное представление
    - Gemma3 - текстовая часть модели, которая преобразует текст в векторное представление

    Контекст визуальной модели содержит ссылки на веса модели ctx.model и параметры модели ctx.params.
    Основные параметры модели: image_size, patch_size, hidden_size, n_head, n_layer, 
    - влияют на размерность тензорных операций.
 */
struct qnn_cgraph * clip_image_build_graph_siglip(clip_ctx_t * ctx, int img_batch) 
{
    const struct clip_params* params = &ctx->params;
    const struct clip_vision_model * model  = &ctx->vision_model;
    const int image_size = params->image_size;
    uint32_t image_size_width  = image_size;
    uint32_t image_size_height = image_size;

    const int patch_size           = params->patch_size;
    const int num_patches          = (image_size_width / patch_size) * (image_size_height / patch_size);
    const int hidden_size          = params->hidden_size;
    const int n_head               = params->n_head;
    const int d_head               = hidden_size / n_head;
    const int n_layer              = params->n_layer;
    const float eps                = params->eps;

    GGML_ASSERT(imgs->size == 1); // batch_size == 1
#if 0
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx->buf_compute_meta.size(),
        /*.mem_buffer =*/ ctx->buf_compute_meta.data(),
        /*.no_alloc   =*/ true,
    };
#endif
    struct ggml_context * ctx0 = ggml_init(NULL, 0);// memory pool and quarks
    ctx0->quarks = _quark_new(256, 2*1024);

    struct qnn_cgraph  * gf = qnn_graph_new(ctx0);
    // input raw TODO GGML_TYPE_BF16, _F16 и Q8
    struct ggml_tensor * inp_raw = ggml_tensor_new(ctx0, GGML_TYPE_F32, (size_t[4]){image_size_width, image_size_height, 3, 1});
    ggml_set_name (ctx0, inp_raw, "inp_raw");
    inp_raw->flags|=GGML_TENSOR_FLAG_INPUT;
return gf;
    struct ggml_tensor * inp = ggml_conv_2d(ctx0, _MODEL(patch_embeddings_0), inp_raw, patch_size, patch_size, 0, 0, 1, 1);
    inp = ggml_reshape_2d(ctx0, inp, num_patches, hidden_size);
    inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
    inp = ggml_add(ctx0, inp, _MODEL(patch_bias));

    // position embeddings
    struct ggml_tensor * embeddings = ggml_add(ctx0, inp, _MODEL(position_embeddings));

    // loop over layers
    for (int il = 0; il < n_layer; il++) {
        struct ggml_tensor * cur = embeddings; // embeddings = residual, cur = hidden_states

        // ggml_layernorm(il, ln_1_w, ln_1_b)
        {
            cur = ggml_norm(ctx0, cur, eps);
            cur = ggml_mul(ctx0, cur, _LAYER(il,ln_1_w));
            cur = ggml_add(ctx0, cur, _LAYER(il,ln_1_b));
        }

        // self-attention
        {

            struct ggml_tensor * Q = // ggml_mmad_3d(ctx0, _LAYER(il,q_w), _LAYER(il,q_b), 0,2,1);
                ggml_add(ctx0, ggml_mul_mat(ctx0, _LAYER(il,q_w), cur), _LAYER(il,q_b));
// TODO выделить отдельную операцию пересатновки permute( d_head, n_head, num_patches) с перестановкой
            Q = ggml_reshape_3d(ctx0, Q, d_head, n_head, num_patches);
            Q = ggml_cont(ctx0, ggml_permute(ctx0, Q, 0, 2, 1, 3));

            struct ggml_tensor * K = // ggml_mmad_3d(ctx0, _LAYER(il,k_w), _LAYER(il,k_b), 0,2,1);
                ggml_add(ctx0, ggml_mul_mat(ctx0, _LAYER(il,k_w), cur), _LAYER(il,k_b));

            K = ggml_reshape_3d(ctx0, K, d_head, n_head, num_patches);
            K = ggml_cont(ctx0, ggml_permute(ctx0, K, 0, 2, 1, 3));

            struct ggml_tensor * V = // ggml_mmad_3d(ctx0, _LAYER(il,v_w), _LAYER(il,v_b), 1,2,0);
                ggml_add(ctx0, ggml_mul_mat(ctx0, _LAYER(il,v_w), cur), _LAYER(il,v_b));

            V = ggml_reshape_3d(ctx0, V, d_head, n_head, num_patches);
            V = ggml_cont(ctx0, ggml_permute(ctx0, V, 1, 2, 0, 3));

            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            KQ = ggml_soft_max_ext(ctx0, KQ, NULL, 1.f / sqrtf((float)d_head), 0.f);
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V, KQ);

            KQV = ggml_reshape_3d(ctx0, KQV, d_head, num_patches, n_head);
            KQV = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            cur = ggml_cont_2d(ctx0, KQV, hidden_size, num_patches);
        }

        // attention output
        cur = ggml_add(ctx0, ggml_mul_mat(ctx0, _LAYER(il,o_w), cur), _LAYER(il,o_b));

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, embeddings);

        embeddings = cur; // embeddings = residual, cur = hidden_states

        // layernorm(il, ln_2_w, ln_2_b)
        {
            cur = ggml_norm(ctx0, cur, eps);
            cur = ggml_mul(ctx0, cur, _LAYER(il,ln_2_w));
            cur = ggml_add(ctx0, cur, _LAYER(il,ln_2_b));
        }

        cur = ggml_mul_mat(ctx0, _LAYER(il,ff_i_w), cur);
        cur = ggml_add(ctx0, cur, _LAYER(il,ff_i_b));

        // siglip uses gelu
        cur = ggml_gelu(ctx0, cur);

        cur = ggml_mul_mat(ctx0, _LAYER(il,ff_o_w), cur);
        cur = ggml_add(ctx0, cur, _LAYER(il,ff_o_b));

        // residual 2
        cur = ggml_add(ctx0, embeddings, cur);

        embeddings = cur;
    }

    // post-layernorm
    if (_MODEL(post_ln_w)) {
        embeddings = ggml_norm(ctx0, embeddings, eps);
        ggml_set_name(ctx0, embeddings, "post_ln");

        embeddings = ggml_add(ctx0, ggml_mul(ctx0, embeddings, _MODEL(post_ln_w)), _MODEL(post_ln_b));
    }

    if (params->proj_type == PROJECTOR_TYPE_GEMMA3) {
        const int batch_size = 1;
        const int mm_tokens_per_image = 256; // default value for gemma3
        const int tokens_per_side = sqrt(mm_tokens_per_image);
        const int patches_per_image = sqrt(num_patches);
        const int kernel_size = patches_per_image / tokens_per_side;

        embeddings = ggml_cont(ctx0, ggml_transpose(ctx0, embeddings));
        embeddings = ggml_reshape_4d(ctx0, embeddings, patches_per_image, patches_per_image, hidden_size, batch_size);

        // doing a pool2d to reduce the number of output tokens to 256
        embeddings = ggml_pool_2d(ctx0, embeddings, GGML_OP_POOL_AVG, kernel_size, kernel_size, kernel_size, kernel_size, 0, 0);
        embeddings = ggml_reshape_3d(ctx0, embeddings, embeddings->ne[0] * embeddings->ne[0], hidden_size, batch_size);
        embeddings = ggml_cont(ctx0, ggml_transpose(ctx0, embeddings));

        // apply norm before projection
        embeddings = ggml_rms_norm(ctx0, embeddings, eps);
        embeddings = ggml_mul(ctx0, embeddings, _MODEL(mm_soft_emb_norm_w));

        // apply projection
        embeddings = ggml_mul_mat(ctx0,
            ggml_cont(ctx0, ggml_transpose(ctx0, _MODEL(mm_input_proj_w))),
            embeddings);
    }

    // build the graph
    //ggml_build_forward_expand(gf, embeddings);

    ggml_free(ctx0);

    return gf;
}