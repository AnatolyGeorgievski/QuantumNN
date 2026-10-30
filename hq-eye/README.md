# hq-eye — глаз / лесенка OpenCL ([[nil]])

Код ступени глаза в QuantumNN. Не ggml-op и не Whisper.

| путь | роль |
|------|------|
| `tools/check_im2col_op_standalone.py` | шаг1: `im2col_tile` vs CPU |
| `tools/check_patch_emb_op.py` | шаг2a: patch emb vs CPU |
| `tools/check_pool2d_op.py` | POOL_2D avg; LWS=16 (=subgroup) |
| `tools/ocl_dev.py` | выбор устройства `-d` |
| `opencl/grain/` | `pool_2d_tile.cl`, `nil_grain.cl`, fma/warp |
| `fixtures/eye/asm/` | дампы графа mtmd A770 |

## Контракт

Я ([[cursor]]) или ты ([[george]] / [[nil]]):

1. На phantom: `cd hq-eye && python3 tools/check_im2col_op_standalone.py -P 16 -S 224` (и `-P 14`).
2. Готово: stdout `OK`, exit 0.
3. `python3 tools/check_patch_emb_op.py` → `OK`; `python3 tools/check_pool2d_op.py` → `OK` (в логе `LWS=16`).
4. Патч сюда — после сверки с [[george]] / [[cursor]].

Ревью: `log/2026-10-30 ревью кода Нила.md`.

**подпись:** Cursor (`[[cursor]]`)
