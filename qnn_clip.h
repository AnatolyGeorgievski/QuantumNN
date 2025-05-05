#ifndef _QNN_CLIP
#define _QNN_CLIP

#include <glib.h>
#include <stdint.h>
#include <stdbool.h>
#include "qnn.h"

enum projector_type {
    PROJECTOR_TYPE_MLP,
    PROJECTOR_TYPE_MLP_NORM,
    PROJECTOR_TYPE_LDP,
    PROJECTOR_TYPE_LDPV2,
    PROJECTOR_TYPE_RESAMPLER,
    PROJECTOR_TYPE_GLM_EDGE,
    PROJECTOR_TYPE_MERGER,
    PROJECTOR_TYPE_GEMMA3,
    PROJECTOR_TYPE_UNKNOWN,
};

typedef struct clip_layer clip_layer;
struct clip_layer {
    // attention
    tensor_t * k_w;
    tensor_t * k_b;
    tensor_t * q_w;
    tensor_t * q_b;
    tensor_t * v_w;
    tensor_t * v_b;

    tensor_t * o_w;
    tensor_t * o_b;

    // layernorm 1
    tensor_t * ln_1_w;
    tensor_t * ln_1_b;

    // ff
    tensor_t * ff_i_w;
    tensor_t * ff_i_b;

    tensor_t * ff_o_w;
    tensor_t * ff_o_b;

    // layernorm 2
    tensor_t * ln_2_w;
    tensor_t * ln_2_b;
};

typedef struct _clip_ctx clip_ctx_t;
struct _clip_ctx {
/*
    bool has_text_encoder    = false;
    bool has_vision_encoder  = false;
    bool has_llava_projector = false;
    bool has_minicpmv_projector = false;
    bool has_glm_projector = false;
    bool has_qwen2vl_merger = false;
    int      minicpmv_version = 2;
    bool use_gelu = false; // gemma3
*/
    QTable_t * quarks;
    uint32_t n_leafs;

    struct clip_params {
        enum projector_type proj_type;
        int32_t image_size;
        int32_t patch_size;
        int32_t hidden_size;
        int32_t n_intermediate;
        int32_t projection_dim;
        int32_t n_head;
        int32_t n_layer;
    
        float eps;
    } params;
    struct clip_vision_model {
        tensor_t * weights; // остальные убрать?
        // embeddings
        tensor_t * class_embedding;
        tensor_t * patch_embeddings_0;
        tensor_t * patch_embeddings_1;  // second Conv2D kernel when we decouple Conv3D along temproal dimension (Qwen2VL)
        tensor_t * patch_bias;
        tensor_t * position_embeddings;
    
        tensor_t * pre_ln_w;
        tensor_t * pre_ln_b;
    
        struct clip_layer * layers;
    
        tensor_t * post_ln_w;
        tensor_t * post_ln_b;
    
        tensor_t * projection;
        // gemma3
        tensor_t * mm_input_proj_w;
        tensor_t * mm_soft_emb_norm_w;
    } vision_model;

    float image_mean[3];
    float image_std [3];

    struct gguf_context * ctx_gguf;
    struct ggml_context * ctx_data;
};
extern clip_ctx_t * clip_init(struct clip_params* params);
//extern int clip_vision_model(clip_ctx_t* ctx, QTable_t* quarks, tensor_weight_t *infos, int n_tensors);
extern int clip_vision_tensors(clip_ctx_t* ctx_clip, tensor_weight_t *infos, int n_tensors);
extern struct qnn_cgraph * clip_image_build_graph_siglip(clip_ctx_t * ctx, int img_batch);
#endif //QNN_CLIP