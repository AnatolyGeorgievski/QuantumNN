# Контракты · patch_emb · глубина 2

Точка внимания: связка `im2col_rgb` → `gemm_nt` в `check_patch_emb_op.py`.  
Глубина **2**: ядра шага + host. Полный mmproj / bias / pos — вне окна.

## Узел 0 — ядра шага

1. Я (`im2col_rgb`) раскладываю HWC-картинку в строки `P·P·3` на тайл; pad нет, шаг = P.
2. Я (`gemm_nt`) считаю `C[m,e] = Σ_k A[m,k]*B[e,k]` (B как E×K, не транспонирую на лету сверх индекса).
3. Готово: `out` формы `M×E`, M=(S/P)², K=P²·3; чужие клетки при `m≥M|e≥E` не пишу.

## Узел 1 — host

1. Я (`check_patch_emb_op.py`) отказываю, если `S % P != 0`.
2. Я ставлю GWS im2col `(S/P,S/P)`, gemm `(M,E)`; буферы `M·K` и `M·E`.
3. Я держу порог сверки `max_abs_err < 1e-3` (не 1e-5 — накопление K).
4. Готово: контракт-assert до enqueue; после — порог vs `cpu_patch_emb`.

## Вне окна

ViT, bias, positional, mmproj 768/1152 — не этот файл.

## Z3 · размеры

- S=P·nout ∧ M=nout·nout ∧ K=P·P·3 ∧ E>0 → **sat** (QF_NIA)
- S ≠ P·nout → **unsat**
