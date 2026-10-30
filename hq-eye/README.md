# hq-eye — глаз / лесенка OpenCL ([[nil]])

Код ступени глаза, принятый в QuantumNN после прогона на phantom A770.

| путь | роль |
|------|------|
| `tools/check_im2col_op_standalone.py` | шаг1: `im2col_tile` vs CPU |
| `tools/check_patch_emb_op.py` | шаг2a: patch emb (im2col RGB + GEMM) vs CPU |
| `tools/check_pool2d_op.py` | POOL_2D avg vs CPU |
| `opencl/grain/` | `pool_2d_tile.cl`, `nil_grain.cl`, fma/warp |
| `fixtures/eye/asm/` | дампы графа mtmd A770 (сверка ассемблера) |

Не путать с Vulkan Whisper unit и с `im2col` внутри llama.cpp — здесь зерно глаза.

## Контракт

Я ([[cursor]]) или ты ([[george]] / [[nil]]):

1. На phantom с ICD Arc: `cd hq-eye && python3 tools/check_im2col_op_standalone.py -P 16 -S 224` (и `-P 14`).
2. Готово шага1: stdout `OK`, exit 0.
3. Дальше: `python3 tools/check_patch_emb_op.py` → `OK`; `python3 tools/check_pool2d_op.py` → `OK`.
4. Патч в этот каталог — после сверки с [[george]] / [[cursor]]. Я не подменяю `im2col_tile` Нила своим Whisper-путём.

Прогон 2026-10-30 phantom A770: im2col P16/P14 OK · patch_emb OK · pool2d OK.

**подпись:** Cursor (`[[cursor]]`)
