/*! \file qnn_llm.c 
    \brief Описание языковых моделей 
    Очень выборочно, чтобы не прегружать приложение, только те модели, которые мы используем

    cname - шблон имени тензора, заполнятся номером блока из SDNV идентификатора. Идентификатор SDNV может содержать идентификатор шаблона, два индекса и идентификатор суффикса. 
    Подстановка индексов выполняется в порядке возникновения символов `.*` в шаблоне имени. 
    Суффикс может принимать значения: `transposed`, `permuted`, `reshaped`, `copy`, `view`, `cont`.

 */
#include <stdint.h>
#include <stdbool.h>
#include "qnn.h"
// #include "qnn_arch.h"
enum _llm_arch {
    LLM_ARCH_LLAMA4,
    LLM_ARCH_FALCON,
    LLM_ARCH_GPT2,
    LLM_ARCH_GRANITE,
    LLM_ARCH_GRANITE_MOE,
    LLM_ARCH_GROK,

    LLM_ARCH_PHI3,
    LLM_ARCH_PHI4,
    LLM_ARCH_QWEN2,
    LLM_ARCH_QWEN2MOE,
    LLM_ARCH_QWEN2VL,
    LLM_ARCH_QWEN3,
    LLM_ARCH_QWEN3MOE,
    LLM_ARCH_MINICPM3,
    LLM_ARCH_GEMMA3,
    LLM_ARCH_STARCODER2,
    LLM_ARCH_OPENELM,
    LLM_ARCH_DEEPSEEK,
    LLM_ARCH_DEEPSEEK2,
    LLM_ARCH_CHATGLM,
    LLM_ARCH_GLM4,
    LLM_ARCH_BITNET,
    LLM_ARCH_NEMOTRON,
    LLM_ARCH_RWKV7,
    LLM_ARCH_ARWKV7,
    LLM_ARCH_UNKNOWN,
};


enum _llm_tensor {
    LLM_TENSOR_TOKEN_EMBD,
    LLM_TENSOR_TOKEN_EMBD_NORM,
    LLM_TENSOR_TOKEN_TYPES,
    LLM_TENSOR_POS_EMBD,
    LLM_TENSOR_OUTPUT,
    LLM_TENSOR_OUTPUT_NORM,
    LLM_TENSOR_ROPE_FREQS,
    LLM_TENSOR_ROPE_FACTORS_LONG,
    LLM_TENSOR_ROPE_FACTORS_SHORT,
    LLM_TENSOR_ATTN_Q,
    LLM_TENSOR_ATTN_K,
    LLM_TENSOR_ATTN_V,
    LLM_TENSOR_ATTN_QKV,
    LLM_TENSOR_ATTN_OUT,
    LLM_TENSOR_ATTN_NORM,
    LLM_TENSOR_ATTN_NORM_2,
    LLM_TENSOR_ATTN_OUT_NORM,
    LLM_TENSOR_ATTN_POST_NORM,
    LLM_TENSOR_ATTN_ROT_EMBD,
    LLM_TENSOR_FFN_GATE_INP,
    LLM_TENSOR_FFN_GATE_INP_SHEXP,
    LLM_TENSOR_FFN_NORM,
    LLM_TENSOR_FFN_POST_NORM,
    LLM_TENSOR_FFN_GATE,
    LLM_TENSOR_FFN_DOWN,
    LLM_TENSOR_FFN_UP,
    LLM_TENSOR_FFN_ACT,
    LLM_TENSOR_FFN_DOWN_EXP,  // split experts for backward compatibility
    LLM_TENSOR_FFN_GATE_EXP,
    LLM_TENSOR_FFN_UP_EXP,
    LLM_TENSOR_FFN_NORM_EXPS,
    LLM_TENSOR_FFN_DOWN_EXPS, // merged experts
    LLM_TENSOR_FFN_GATE_EXPS,
    LLM_TENSOR_FFN_UP_EXPS,
    LLM_TENSOR_FFN_DOWN_SHEXP,
    LLM_TENSOR_FFN_GATE_SHEXP,
    LLM_TENSOR_FFN_UP_SHEXP,
    LLM_TENSOR_FFN_EXP_PROBS_B,
    LLM_TENSOR_ATTN_Q_NORM,
    LLM_TENSOR_ATTN_K_NORM,
    LLM_TENSOR_LAYER_OUT_NORM,
    LLM_TENSOR_POST_ATTN_NORM,
    LLM_TENSOR_POST_MLP_NORM,
    LLM_TENSOR_SSM_IN,
    LLM_TENSOR_SSM_CONV1D,
    LLM_TENSOR_SSM_X,
    LLM_TENSOR_SSM_DT,
    LLM_TENSOR_SSM_A,
    LLM_TENSOR_SSM_D,
    LLM_TENSOR_SSM_OUT,
    LLM_TENSOR_TIME_MIX_W0,
    LLM_TENSOR_TIME_MIX_W1,
    LLM_TENSOR_TIME_MIX_W2,
    LLM_TENSOR_TIME_MIX_A0,
    LLM_TENSOR_TIME_MIX_A1,
    LLM_TENSOR_TIME_MIX_A2,
    LLM_TENSOR_TIME_MIX_V0,
    LLM_TENSOR_TIME_MIX_V1,
    LLM_TENSOR_TIME_MIX_V2,
    LLM_TENSOR_TIME_MIX_G1,
    LLM_TENSOR_TIME_MIX_G2,
    LLM_TENSOR_TIME_MIX_K_K,
    LLM_TENSOR_TIME_MIX_K_A,
    LLM_TENSOR_TIME_MIX_R_K,
    LLM_TENSOR_TIME_MIX_LERP_X,
    LLM_TENSOR_TIME_MIX_LERP_W,
    LLM_TENSOR_TIME_MIX_LERP_K,
    LLM_TENSOR_TIME_MIX_LERP_V,
    LLM_TENSOR_TIME_MIX_LERP_R,
    LLM_TENSOR_TIME_MIX_LERP_G,
    LLM_TENSOR_TIME_MIX_LERP_FUSED,
    LLM_TENSOR_TIME_MIX_FIRST,
    LLM_TENSOR_TIME_MIX_DECAY,
    LLM_TENSOR_TIME_MIX_DECAY_W1,
    LLM_TENSOR_TIME_MIX_DECAY_W2,
    LLM_TENSOR_TIME_MIX_KEY,
    LLM_TENSOR_TIME_MIX_VALUE,
    LLM_TENSOR_TIME_MIX_RECEPTANCE,
    LLM_TENSOR_TIME_MIX_GATE,
    LLM_TENSOR_TIME_MIX_LN,
    LLM_TENSOR_TIME_MIX_OUTPUT,
    LLM_TENSOR_CHANNEL_MIX_LERP_K,
    LLM_TENSOR_CHANNEL_MIX_LERP_R,
    LLM_TENSOR_CHANNEL_MIX_KEY,
    LLM_TENSOR_CHANNEL_MIX_RECEPTANCE,
    LLM_TENSOR_CHANNEL_MIX_VALUE,
    LLM_TENSOR_ATTN_Q_A,
    LLM_TENSOR_ATTN_Q_B,
    LLM_TENSOR_ATTN_KV_A_MQA,
    LLM_TENSOR_ATTN_KV_B,
    LLM_TENSOR_ATTN_K_B,
    LLM_TENSOR_ATTN_V_B,
    LLM_TENSOR_ATTN_Q_A_NORM,
    LLM_TENSOR_ATTN_KV_A_NORM,
    LLM_TENSOR_ATTN_SUB_NORM,
    LLM_TENSOR_FFN_SUB_NORM,
    LLM_TENSOR_DEC_ATTN_NORM,
    LLM_TENSOR_DEC_ATTN_Q,
    LLM_TENSOR_DEC_ATTN_K,
    LLM_TENSOR_DEC_ATTN_V,
    LLM_TENSOR_DEC_ATTN_OUT,
    LLM_TENSOR_DEC_ATTN_REL_B,
    LLM_TENSOR_DEC_CROSS_ATTN_NORM,
    LLM_TENSOR_DEC_CROSS_ATTN_Q,
    LLM_TENSOR_DEC_CROSS_ATTN_K,
    LLM_TENSOR_DEC_CROSS_ATTN_V,
    LLM_TENSOR_DEC_CROSS_ATTN_OUT,
    LLM_TENSOR_DEC_CROSS_ATTN_REL_B,
    LLM_TENSOR_DEC_FFN_NORM,
    LLM_TENSOR_DEC_FFN_GATE,
    LLM_TENSOR_DEC_FFN_DOWN,
    LLM_TENSOR_DEC_FFN_UP,
    LLM_TENSOR_DEC_OUTPUT_NORM,
    LLM_TENSOR_ENC_ATTN_NORM,
    LLM_TENSOR_ENC_ATTN_Q,
    LLM_TENSOR_ENC_ATTN_K,
    LLM_TENSOR_ENC_ATTN_V,
    LLM_TENSOR_ENC_ATTN_OUT,
    LLM_TENSOR_ENC_ATTN_REL_B,
    LLM_TENSOR_ENC_FFN_NORM,
    LLM_TENSOR_ENC_FFN_GATE,
    LLM_TENSOR_ENC_FFN_DOWN,
    LLM_TENSOR_ENC_FFN_UP,
    LLM_TENSOR_ENC_OUTPUT_NORM,
    LLM_TENSOR_CLS,
    LLM_TENSOR_CLS_OUT,
    LLM_TENSOR_CONV1D,
    LLM_TENSOR_CONVNEXT_DW,
    LLM_TENSOR_CONVNEXT_NORM,
    LLM_TENSOR_CONVNEXT_PW1,
    LLM_TENSOR_CONVNEXT_PW2,
    LLM_TENSOR_CONVNEXT_GAMMA,
    LLM_TENSOR_POS_NET_CONV1,
    LLM_TENSOR_POS_NET_CONV2,
    LLM_TENSOR_POS_NET_NORM,
    LLM_TENSOR_POS_NET_NORM1,
    LLM_TENSOR_POS_NET_NORM2,
    LLM_TENSOR_POS_NET_ATTN_NORM,
    LLM_TENSOR_POS_NET_ATTN_Q,
    LLM_TENSOR_POS_NET_ATTN_K,
    LLM_TENSOR_POS_NET_ATTN_V,
    LLM_TENSOR_POS_NET_ATTN_OUT,
};

enum llm_tensor_layer {
    LLM_TENSOR_LAYER_INPUT,
    LLM_TENSOR_LAYER_REPEATING,
    LLM_TENSOR_LAYER_OUTPUT,
};
struct _llm_tensor_names {
    enum _llm_tensor param_id; 
    const char* cname;
};
const struct _llm_tensor_names *LLM_TENSOR_NAMES[] = {
    [LLM_ARCH_QWEN2] = (const struct _llm_tensor_names[]){
        { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
        { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
        { LLM_TENSOR_OUTPUT,          "output" },
        { LLM_TENSOR_ATTN_NORM,       "blk.*.attn_norm" },
        { LLM_TENSOR_ATTN_Q,          "blk.*.attn_q" },
        { LLM_TENSOR_ATTN_K,          "blk.*.attn_k" },
        { LLM_TENSOR_ATTN_V,          "blk.*.attn_v" },
        { LLM_TENSOR_ATTN_OUT,        "blk.*.attn_output" },
        { LLM_TENSOR_FFN_NORM,        "blk.*.ffn_norm" },
        { LLM_TENSOR_FFN_GATE,        "blk.*.ffn_gate" },
        { LLM_TENSOR_FFN_DOWN,        "blk.*.ffn_down" },
        { LLM_TENSOR_FFN_UP,          "blk.*.ffn_up" },
    },
    [LLM_ARCH_QWEN2VL] = (const struct _llm_tensor_names[]){
        { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
        { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
        { LLM_TENSOR_OUTPUT,          "output" },
        { LLM_TENSOR_ATTN_NORM,       "blk.*.attn_norm" },
        { LLM_TENSOR_ATTN_Q,          "blk.*.attn_q" },
        { LLM_TENSOR_ATTN_K,          "blk.*.attn_k" },
        { LLM_TENSOR_ATTN_V,          "blk.*.attn_v" },
        { LLM_TENSOR_ATTN_OUT,        "blk.*.attn_output" },
        { LLM_TENSOR_FFN_NORM,        "blk.*.ffn_norm" },
        { LLM_TENSOR_FFN_GATE,        "blk.*.ffn_gate" },
        { LLM_TENSOR_FFN_DOWN,        "blk.*.ffn_down" },
        { LLM_TENSOR_FFN_UP,          "blk.*.ffn_up" },
    },
    [LLM_ARCH_QWEN2MOE] = (const struct _llm_tensor_names[]){
        { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
        { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
        { LLM_TENSOR_OUTPUT,             "output" },
        { LLM_TENSOR_ATTN_NORM,          "blk.*.attn_norm" },
        { LLM_TENSOR_ATTN_Q,             "blk.*.attn_q" },
        { LLM_TENSOR_ATTN_K,             "blk.*.attn_k" },
        { LLM_TENSOR_ATTN_V,             "blk.*.attn_v" },
        { LLM_TENSOR_ATTN_OUT,           "blk.*.attn_output" },
        { LLM_TENSOR_FFN_NORM,           "blk.*.ffn_norm" },
        { LLM_TENSOR_FFN_GATE_INP,       "blk.*.ffn_gate_inp" },
        { LLM_TENSOR_FFN_GATE_EXPS,      "blk.*.ffn_gate_exps" },
        { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.*.ffn_down_exps" },
        { LLM_TENSOR_FFN_UP_EXPS,        "blk.*.ffn_up_exps" },
        { LLM_TENSOR_FFN_GATE_INP_SHEXP, "blk.*.ffn_gate_inp_shexp" },
        { LLM_TENSOR_FFN_GATE_SHEXP,     "blk.*.ffn_gate_shexp" },
        { LLM_TENSOR_FFN_DOWN_SHEXP,     "blk.*.ffn_down_shexp" },
        { LLM_TENSOR_FFN_UP_SHEXP,       "blk.*.ffn_up_shexp" },
    },
    [LLM_ARCH_QWEN3] = (const struct _llm_tensor_names[]){
        { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
        { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
        { LLM_TENSOR_OUTPUT,          "output" },
        { LLM_TENSOR_ATTN_NORM,       "blk.*.attn_norm" },
        { LLM_TENSOR_ATTN_Q,          "blk.*.attn_q" },
        { LLM_TENSOR_ATTN_Q_NORM,     "blk.*.attn_q_norm" },
        { LLM_TENSOR_ATTN_K,          "blk.*.attn_k" },
        { LLM_TENSOR_ATTN_K_NORM,     "blk.*.attn_k_norm" },
        { LLM_TENSOR_ATTN_V,          "blk.*.attn_v" },
        { LLM_TENSOR_ATTN_OUT,        "blk.*.attn_output" },
        { LLM_TENSOR_FFN_NORM,        "blk.*.ffn_norm" },
        { LLM_TENSOR_FFN_GATE,        "blk.*.ffn_gate" },
        { LLM_TENSOR_FFN_DOWN,        "blk.*.ffn_down" },
        { LLM_TENSOR_FFN_UP,          "blk.*.ffn_up" },
    },
    [LLM_ARCH_QWEN3MOE] = (const struct _llm_tensor_names[]){
        { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
        { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
        { LLM_TENSOR_OUTPUT,             "output" },
        { LLM_TENSOR_ATTN_NORM,          "blk.*.attn_norm" },
        { LLM_TENSOR_ATTN_Q,             "blk.*.attn_q" },
        { LLM_TENSOR_ATTN_Q_NORM,        "blk.*.attn_q_norm" },
        { LLM_TENSOR_ATTN_K,             "blk.*.attn_k" },
        { LLM_TENSOR_ATTN_K_NORM,        "blk.*.attn_k_norm" },
        { LLM_TENSOR_ATTN_V,             "blk.*.attn_v" },
        { LLM_TENSOR_ATTN_OUT,           "blk.*.attn_output" },
        { LLM_TENSOR_FFN_NORM,           "blk.*.ffn_norm" },
        { LLM_TENSOR_FFN_GATE_INP,       "blk.*.ffn_gate_inp" },
        { LLM_TENSOR_FFN_GATE_EXPS,      "blk.*.ffn_gate_exps" },
        { LLM_TENSOR_FFN_DOWN_EXPS,      "blk.*.ffn_down_exps" },
        { LLM_TENSOR_FFN_UP_EXPS,        "blk.*.ffn_up_exps" },
    },
    [LLM_ARCH_GEMMA3] =  (const struct _llm_tensor_names[]){
        { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
        { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
        { LLM_TENSOR_OUTPUT,          "output" },
        { LLM_TENSOR_ATTN_NORM,       "blk.*.attn_norm" },
        { LLM_TENSOR_ATTN_Q,          "blk.*.attn_q" },
        { LLM_TENSOR_ATTN_Q_NORM,     "blk.*.attn_q_norm" },
        { LLM_TENSOR_ATTN_K,          "blk.*.attn_k" },
        { LLM_TENSOR_ATTN_K_NORM,     "blk.*.attn_k_norm" },
        { LLM_TENSOR_ATTN_V,          "blk.*.attn_v" },
        { LLM_TENSOR_ATTN_OUT,        "blk.*.attn_output" },
        { LLM_TENSOR_ATTN_POST_NORM,  "blk.*.post_attention_norm" },
        { LLM_TENSOR_FFN_NORM,        "blk.*.ffn_norm" },
        { LLM_TENSOR_FFN_GATE,        "blk.*.ffn_gate" },
        { LLM_TENSOR_FFN_DOWN,        "blk.*.ffn_down" },
        { LLM_TENSOR_FFN_UP,          "blk.*.ffn_up" },
        { LLM_TENSOR_FFN_POST_NORM,   "blk.*.post_ffw_norm" },
    },
    [LLM_ARCH_GRANITE] = (const struct _llm_tensor_names[]){
        { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
        { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
        { LLM_TENSOR_OUTPUT,          "output" },
        { LLM_TENSOR_ATTN_NORM,       "blk.*.attn_norm" },
        { LLM_TENSOR_ATTN_Q,          "blk.*.attn_q" },
        { LLM_TENSOR_ATTN_K,          "blk.*.attn_k" },
        { LLM_TENSOR_ATTN_V,          "blk.*.attn_v" },
        { LLM_TENSOR_ATTN_OUT,        "blk.*.attn_output" },
        { LLM_TENSOR_FFN_NORM,        "blk.*.ffn_norm" },
        { LLM_TENSOR_FFN_GATE,        "blk.*.ffn_gate" },
        { LLM_TENSOR_FFN_DOWN,        "blk.*.ffn_down" },
        { LLM_TENSOR_FFN_UP,          "blk.*.ffn_up" },
    },
    [LLM_ARCH_GRANITE_MOE] = (const struct _llm_tensor_names[]){
        { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
        { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
        { LLM_TENSOR_OUTPUT,          "output" },
        { LLM_TENSOR_ATTN_NORM,       "blk.*.attn_norm" },
        { LLM_TENSOR_ATTN_Q,          "blk.*.attn_q" },
        { LLM_TENSOR_ATTN_K,          "blk.*.attn_k" },
        { LLM_TENSOR_ATTN_V,          "blk.*.attn_v" },
        { LLM_TENSOR_ATTN_OUT,        "blk.*.attn_output" },
        { LLM_TENSOR_FFN_NORM,        "blk.*.ffn_norm" },
        { LLM_TENSOR_FFN_GATE_INP,    "blk.*.ffn_gate_inp" },
        { LLM_TENSOR_FFN_GATE_EXPS,   "blk.*.ffn_gate_exps" },
        { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.*.ffn_down_exps" },
        { LLM_TENSOR_FFN_UP_EXPS,     "blk.*.ffn_up_exps" },
    },
    [LLM_ARCH_BITNET] = (const struct _llm_tensor_names[]){
        { LLM_TENSOR_TOKEN_EMBD,         "token_embd" },
        { LLM_TENSOR_OUTPUT_NORM,        "output_norm" },
        { LLM_TENSOR_ATTN_Q,             "blk.*.attn_q" },
        { LLM_TENSOR_ATTN_K,             "blk.*.attn_k" },
        { LLM_TENSOR_ATTN_V,             "blk.*.attn_v" },
        { LLM_TENSOR_ATTN_OUT,           "blk.*.attn_output" },
        { LLM_TENSOR_ATTN_NORM,          "blk.*.attn_norm" },
        { LLM_TENSOR_ATTN_SUB_NORM,      "blk.*.attn_sub_norm" },
        { LLM_TENSOR_FFN_GATE,           "blk.*.ffn_gate" },
        { LLM_TENSOR_FFN_DOWN,           "blk.*.ffn_down" },
        { LLM_TENSOR_FFN_UP,             "blk.*.ffn_up" },
        { LLM_TENSOR_FFN_NORM,           "blk.*.ffn_norm" },
        { LLM_TENSOR_FFN_SUB_NORM,       "blk.*.ffn_sub_norm" },
    },
    [LLM_ARCH_GROK] = (const struct _llm_tensor_names[]){
        { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
        { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
        { LLM_TENSOR_OUTPUT,          "output" },
        { LLM_TENSOR_ROPE_FREQS,      "rope_freqs" },
        { LLM_TENSOR_ATTN_NORM,       "blk.*.attn_norm" },
        { LLM_TENSOR_ATTN_Q,          "blk.*.attn_q" },
        { LLM_TENSOR_ATTN_K,          "blk.*.attn_k" },
        { LLM_TENSOR_ATTN_V,          "blk.*.attn_v" },
        { LLM_TENSOR_ATTN_OUT,        "blk.*.attn_output" },
        { LLM_TENSOR_ATTN_ROT_EMBD,   "blk.*.attn_rot_embd" },
        { LLM_TENSOR_FFN_GATE_INP,    "blk.*.ffn_gate_inp" },
        { LLM_TENSOR_FFN_NORM,        "blk.*.ffn_norm" },
        { LLM_TENSOR_FFN_GATE_EXP,    "blk.*.ffn_gate.*" },
        { LLM_TENSOR_FFN_DOWN_EXP,    "blk.*.ffn_down.*" },
        { LLM_TENSOR_FFN_UP_EXP,      "blk.*.ffn_up.*" },
        { LLM_TENSOR_FFN_GATE_EXPS,   "blk.*.ffn_gate_exps" },
        { LLM_TENSOR_FFN_DOWN_EXPS,   "blk.*.ffn_down_exps" },
        { LLM_TENSOR_FFN_UP_EXPS,     "blk.*.ffn_up_exps" },
        { LLM_TENSOR_LAYER_OUT_NORM,  "blk.*.layer_output_norm" },
        { LLM_TENSOR_ATTN_OUT_NORM,   "blk.*.attn_output_norm" },
    },
    [LLM_ARCH_GPT2] = (const struct _llm_tensor_names[]){
        { LLM_TENSOR_TOKEN_EMBD,      "token_embd" },
        { LLM_TENSOR_POS_EMBD,        "position_embd" },
        { LLM_TENSOR_OUTPUT_NORM,     "output_norm" },
        { LLM_TENSOR_OUTPUT,          "output" },
        { LLM_TENSOR_ATTN_NORM,       "blk.*.attn_norm" },
        { LLM_TENSOR_ATTN_QKV,        "blk.*.attn_qkv" },
        { LLM_TENSOR_ATTN_OUT,        "blk.*.attn_output" },
        { LLM_TENSOR_FFN_NORM,        "blk.*.ffn_norm" },
        { LLM_TENSOR_FFN_UP,          "blk.*.ffn_up" },
        { LLM_TENSOR_FFN_DOWN,        "blk.*.ffn_down" },
    },
};
typedef struct _llm_params llm_params_t;
struct _llm_params {
    uint16_t n_layer;
    uint16_t n_tokens;       //!< размер словаря
    uint16_t n_head;
    uint16_t n_head_kv;
    uint16_t n_ff;
    uint16_t n_embd;        //!< 
    uint16_t n_embd_head;
    uint16_t n_embd_head_k;
    uint16_t n_embd_head_v;
    uint16_t n_expert;
    uint16_t n_expert_used;
};

static bool weight_buft_supported(const llm_params_t * hparams, tensor_t * w, enum ggml_op op, ggml_backend_buffer_type_t buft, ggml_backend_dev_t dev) {
    GGML_ASSERT(w != nullptr);

    if (op == GGML_OP_NONE) 
        return true;
#if 0
    ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead()*8,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    if (!ctx_ptr) {
        throw std::runtime_error(format("failed to create ggml context"));
    }
#endif
    ggml_context * ctx = ggml_init(NULL, 8);
    ggml_tensor  * op_tensor = NULL;
    const size_t BATCH = 512;
    switch (op) {
        case GGML_OP_GET_ROWS:
            {
                ggml_tensor * b = ggml_tensor_new(ctx, GGML_TYPE_I32, (const size_t[4]){BATCH,1,1,1});
                op_tensor = ggml_get_rows(ctx, w, b);
            } break;
        case GGML_OP_MUL_MAT:
            {
                ggml_tensor * b = ggml_tensor_new(ctx, GGML_TYPE_F32, (const size_t[4]){w->ne[0], BATCH, w->ne[2], w->ne[3]});
                op_tensor = ggml_mul_mat(ctx, w, b);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                int n_expert_used = hparams->n_expert_used;
                ggml_tensor * b = ggml_tensor_new(ctx, GGML_TYPE_F32, (const size_t[4]){w->ne[0], n_expert_used, BATCH,1});
                ggml_tensor * ids = ggml_tensor_new(ctx, GGML_TYPE_I32, (const size_t[4]){n_expert_used, BATCH,1,1});
                op_tensor = ggml_mul_mat_id(ctx, w, b, ids);
            } break;
        case GGML_OP_ADD:
            {
                ggml_tensor * a = ggml_tensor_new(ctx, GGML_TYPE_F32, w->ne);
                op_tensor = ggml_add(ctx, a, w);
            } break;
        case GGML_OP_MUL:
            {
                ggml_tensor * a = ggml_tensor_new(ctx, GGML_TYPE_F32, w->ne);
                op_tensor = ggml_mul(ctx, a, w);
            } break;
        case GGML_OP_ROPE:
            {
                const unsigned int n_embd_head = hparams->n_embd_head_v;
                const unsigned int n_head = hparams->n_head;
                ggml_tensor * a = ggml_tensor_new(ctx, GGML_TYPE_F32, (const size_t[4]){n_embd_head, n_head, BATCH, 1});
                ggml_tensor * b = ggml_tensor_new(ctx, GGML_TYPE_I32, (const size_t[4]){BATCH,1,1,1});
                op_tensor = ggml_rope_ext(ctx, a, b, w,
                    0, 0, 0, 0, 0,
                    0, 0, 0, 0
                );

            } break;
        case GGML_OP_IM2COL:
            {
                const int n_embd = hparams->n_embd;
                ggml_tensor * b = ggml_tensor_new(ctx, GGML_TYPE_F32, (const size_t[4]){n_embd, w->ne[1], 1, 1});
                op_tensor = ggml_im2col(ctx, w, b, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F16);
            } break;
        default:
            GGML_ABORT("%s: missing test for op %s for tensor %s", __func__, ggml_op_name(op), w->name);
    }

    // create a temporary dummy buffer for the weight so that supports_op can check the buffer type
    GGML_ASSERT(w->buffer == NULL);
    w->buffer = ggml_backend_buft_alloc_buffer(buft, 0);
    bool op_supported = ggml_backend_dev_supports_op(dev, op_tensor);
    ggml_tensor_free(ctx, op_tensor);
    ggml_backend_buffer_free(w->buffer);
    w->buffer = NULL;

    return op_supported;
}

// see _LAYER(il, cname) _MODEL(cname)
#define _MODEL(cname)       model->cname
#define _LAYER(il,  name)   model->layers[il].name
#define _WEIGHT(il, offs)   model->weights[il*model->n_tensor + offs]
// базовый класс, который позволяет находить тензоры весов модели по индексу model->weights[cname_idx + il*n_tensor]
struct _llm_model {
    enum _llm_arch arch; // идентификатор модели OID = {LLM_ARCH_GEMMA3}
    const int n_layer;       //!< number of layers
    const int n_tensor;      //!< number of tensors per layer
    tensor_weight_t weights[0];
    tensor_weight_t tok_embd;
    tensor_weight_t output_norm;
    tensor_weight_t output_norm_b;
    tensor_weight_t output;
};
// У каждой модели есть свой набор тензоров в структуре слоя
struct _gpt2_layer {
    tensor_weight_t attn_norm;
    tensor_weight_t attn_norm_b;
    tensor_weight_t wqkv;
    tensor_weight_t bqkv;
    tensor_weight_t wo;
    tensor_weight_t bo;
    tensor_weight_t ffn_norm;
    tensor_weight_t ffn_norm_b;
    tensor_weight_t ffn_down;
    tensor_weight_t ffn_down_b;
    tensor_weight_t ffn_up;
    tensor_weight_t ffn_up_b;
};
struct _exaone_layer {
    tensor_weight_t attn_norm;
    tensor_weight_t wq;
    tensor_weight_t wk;
    tensor_weight_t wv;
    tensor_weight_t wo;
    tensor_weight_t ffn_norm;
    tensor_weight_t ffn_gate;
    tensor_weight_t ffn_down;
    tensor_weight_t ffn_up;
    tensor_weight_t rope_freqs;
};
struct _gemma_layer {
    tensor_weight_t attn_norm;
    tensor_weight_t wq;
    tensor_weight_t wk;
    tensor_weight_t wv;
    tensor_weight_t wo;
    tensor_weight_t ffn_norm;
    tensor_weight_t ffn_gate;
    tensor_weight_t ffn_down;
    tensor_weight_t ffn_up;
};
struct _gemma2_layer {
    tensor_weight_t attn_norm;
    tensor_weight_t wq;
    tensor_weight_t wk;
    tensor_weight_t wv;
    tensor_weight_t wo;
    tensor_weight_t ffn_norm;
    tensor_weight_t ffn_gate;
    tensor_weight_t ffn_down;
    tensor_weight_t ffn_up;
    tensor_weight_t ffn_post_norm;
};
struct _gemma3_layer {
    tensor_weight_t attn_norm;
    tensor_weight_t wq;
    tensor_weight_t wk;
    tensor_weight_t wv;
    tensor_weight_t wo;
    tensor_weight_t attn_post_norm;
    tensor_weight_t attn_k_norm;
    tensor_weight_t attn_q_norm;
    tensor_weight_t ffn_norm;
    tensor_weight_t ffn_gate;
    tensor_weight_t ffn_up;
    tensor_weight_t ffn_down;
    tensor_weight_t ffn_post_norm;
};
struct _starcoder2_layer {
    tensor_weight_t attn_norm;
    tensor_weight_t attn_norm_b;
    tensor_weight_t wq;
    tensor_weight_t bq;
    tensor_weight_t wk;
    tensor_weight_t bk;
    tensor_weight_t wv;
    tensor_weight_t bv;
    tensor_weight_t wo;
    tensor_weight_t bo;
    tensor_weight_t ffn_norm;
    tensor_weight_t ffn_norm_b;
    tensor_weight_t ffn_down;
    tensor_weight_t ffn_down_b;
    tensor_weight_t ffn_up;
    tensor_weight_t ffn_up_b;
};
struct _nemotron_layer {
    tensor_weight_t attn_norm;
    tensor_weight_t attn_norm_b;
    tensor_weight_t wq;
    tensor_weight_t bq;
    tensor_weight_t wk;
    tensor_weight_t bk;
    tensor_weight_t wv;
    tensor_weight_t bv;
    tensor_weight_t wo;
    tensor_weight_t bo;
    tensor_weight_t ffn_norm;
    tensor_weight_t ffn_norm_b;
    tensor_weight_t ffn_down;
    tensor_weight_t ffn_down_b;
    tensor_weight_t ffn_up;
    tensor_weight_t ffn_up_b;
};
struct _bitnet_layer {
    tensor_weight_t attn_norm;
    tensor_weight_t attn_sub_norm;
    tensor_weight_t wq;
    tensor_weight_t wq_scale;
    tensor_weight_t wk;
    tensor_weight_t wk_scale;
    tensor_weight_t wv;
    tensor_weight_t wv_scale;
    tensor_weight_t wo;
    tensor_weight_t wo_scale;
    tensor_weight_t ffn_norm;
    tensor_weight_t ffn_sub_norm;
    tensor_weight_t ffn_gate;
    tensor_weight_t ffn_gate_scale;
    tensor_weight_t ffn_down;
    tensor_weight_t ffn_down_scale;
    tensor_weight_t ffn_up;
    tensor_weight_t ffn_up_scale;
};
struct _grok_layer {
    tensor_weight_t attn_norm;
    tensor_weight_t wq;
    tensor_weight_t wk;
    tensor_weight_t wv;
    tensor_weight_t wo;
    tensor_weight_t attn_out_norm;
    tensor_weight_t ffn_norm;
    tensor_weight_t ffn_gate_inp;
    tensor_weight_t ffn_gate_exps;
    tensor_weight_t ffn_down_exps;
    tensor_weight_t ffn_up_exps;
    tensor_weight_t layer_out_norm;
};
struct _qwen2_layer {// Qwen2 Qwen2VL
    tensor_weight_t attn_norm;
    tensor_weight_t wq;
    tensor_weight_t bq;
    tensor_weight_t wk;
    tensor_weight_t bk;
    tensor_weight_t wv;
    tensor_weight_t bv;
    tensor_weight_t wo;
    tensor_weight_t ffn_norm;
    tensor_weight_t ffn_gate;
    tensor_weight_t ffn_down;
    tensor_weight_t ffn_up;
};
struct _qwen3_layer {
    tensor_weight_t attn_norm;
    tensor_weight_t wq;
    tensor_weight_t wk;
    tensor_weight_t wv;
    tensor_weight_t wo;
    tensor_weight_t attn_k_norm;
    tensor_weight_t attn_q_norm;
    tensor_weight_t ffn_norm;
    tensor_weight_t ffn_gate;
    tensor_weight_t ffn_down;
    tensor_weight_t ffn_up;
};
struct _qwen3moe_layer {
    tensor_weight_t attn_norm;
    tensor_weight_t wq;
    tensor_weight_t wk;
    tensor_weight_t wv;
    tensor_weight_t wo;
    tensor_weight_t attn_k_norm;
    tensor_weight_t attn_q_norm;
    tensor_weight_t ffn_norm;
    tensor_weight_t ffn_gate_inp;
    tensor_weight_t ffn_gate_exps;
    tensor_weight_t ffn_down_exps;
    tensor_weight_t ffn_up_exps;
};
struct _phi3_layer {
    tensor_weight_t attn_norm;
    tensor_weight_t wqkv;
    tensor_weight_t wo;
    tensor_weight_t ffn_norm;
    tensor_weight_t ffn_down;
    tensor_weight_t ffn_up;
    tensor_weight_t rope_long;
    tensor_weight_t rope_short;
};
struct _gemma3_model {
    enum _llm_arch arch; // идентификатор модели:= LLM_ARCH_GEMMA3
    const int n_layer;
    const int n_tensor; //!< число тензоров весов в слое \see struct gemma3_layer_t
    tensor_weight_t tok_embd;
    tensor_weight_t output_norm;
    tensor_weight_t output;
};
struct _llm_model* gemma3_new(int n_layer){
    struct _gemma3_model * model = (struct _gemma3_model*) malloc(sizeof(struct _llm_model) + sizeof(struct gemma3_layer_t) * n_layer);
    model->n_layer  = n_layer;
    model->n_tensor = sizeof(struct gemma3_layer_t) /sizeof(tensor_weight_t);
    return (struct _llm_model*)model;
}

static inline const uint64_t tn(enum _llm_tensor tensor_id, unsigned int suffix, unsigned int il){ // кодирование имени тензора SDNV
    uint64_t sdnv = (suffix);
    sdnv = (sdnv<<8) | (il&127);
    return (sdnv<<8) | (tensor_id & 127);
}
int _model(const llm_params_t * hparams, enum _llm_arch arch){

    const size_t n_head         = hparams->n_head;
    const size_t n_head_kv      = hparams->n_head_kv;
    const size_t n_embd_head_k  = hparams->n_embd_head_k;
    const size_t n_embd_head_v  = hparams->n_embd_head_v;
    const size_t n_embd_k_gqa   = n_embd_head_k * n_head_kv;
    const size_t n_embd_v_gqa   = n_embd_head_v * n_head_kv;
    const size_t n_ff           = hparams->n_ff;
    const size_t n_embd  = hparams->n_embd;
    const size_t n_vocab = hparams->n_tokens;
    const size_t n_layer = hparams->n_layer;
    const size_t n_expert = hparams->n_expert;
    const size_t n_expert_used = hparams->n_expert_used;

    switch(arch){
    case LLM_ARCH_GEMMA3: {
        tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), (const size_t[4]){n_embd, n_vocab,1,1}, 0);

        // output
        _MODEL(output_norm) = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), (const size_t[4]){n_embd,1,1,1}, 0);
        _MODEL(output)      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), (const size_t[4]){n_embd, n_vocab,1,1}, TENSOR_NOT_REQUIRED);

        // if output is NULL, init from the input tok embed
        if (_MODEL(output) == NULL) {
            _MODEL(output) = create_tensor_dup(tn(LLM_TENSOR_TOKEN_EMBD,   "weight"), (const size_t[4]){n_embd, n_vocab,1,1}, TENSOR_DUPLICATED);
        }

        for (int il = 0; il < n_layer; ++i) {
            _LAYER(il, attn_norm) = create_tensor(tn(LLM_TENSOR_ATTN_NORM,"weight", i), (const size_t[4]){n_embd,1,1,1}, 0);
            _LAYER(il, wq       ) = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), (const size_t[4]){n_embd, n_embd_head_k * n_head,1,1}, 0);
            _LAYER(il, wk       ) = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), (const size_t[4]){n_embd, n_embd_k_gqa,1,1}, 0);
            _LAYER(il, wv       ) = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), (const size_t[4]){n_embd, n_embd_v_gqa,1,1}, 0);
            _LAYER(il, wo       ) = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), (const size_t[4]){n_embd_head_k * n_head, n_embd}, 0);

            _LAYER(il, attn_post_norm) = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), (const size_t[4]){n_embd,1,1,1}, 0);
            _LAYER(il, attn_k_norm)    = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM,    "weight", i), (const size_t[4]){n_embd_head_k,1,1,1}, 0);
            _LAYER(il, attn_q_norm)    = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), (const size_t[4]){n_embd_head_k,1,1,1}, 0);

            _LAYER(il, ffn_norm) = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), (const size_t[4]){n_embd, 1,1,1}, 0);
            _LAYER(il, ffn_gate) = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), (const size_t[4]){n_embd, n_ff, 1,1}, 0);
            _LAYER(il, ffn_up)   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), (const size_t[4]){n_embd, n_ff, 1,1}, 0);
            _LAYER(il, ffn_down) = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), (const size_t[4]){n_ff, n_embd, 1,1}, 0);

            _LAYER(il, ffn_post_norm)  = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), (const size_t[4]){n_embd, 1,1,1}, 0);
        }
    } break;
    case LLM_ARCH_QWEN3:  {
        tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

        // output
        output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
        output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
        // if output is NULL, init from the input tok embed
        if (output == NULL) {
            output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
        }

        for (int i = 0; i < n_layer; ++i) {
            auto & layer = layers[i];

            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_gqa}, 0);
            layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_gqa}, 0);
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

            layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
        }
    } break;
    default: 
        return -1;
    }
    return 0;
}