# Контракты · im2col_tile · глубина 2

Точка внимания: `kernel im2col_tile` (тело в `tools/check_im2col_op_standalone.py`).  
Глубина **2**: ядро + host. ggml IM2COL / pad / dilation — вне окна.

## Узел 0 — ядро

1. Я (`im2col_tile`) копирую тайл `P×P` из `src` в `dst` при **без pad**, шаг = P (нарезка сетки).
2. Я выхожу, если `ox >= W/P` или `oy >= H/P`.
3. Я пишу `P*P` float начиная с `dst + (oy*nout_x+ox)*(P*P)`.
4. Готово: каждая клетка выхода — точная копия входа; чужие клетки за пределами `(nout_y,nout_x)` не пишу.

## Узел 1 — host

1. Я (`check_im2col_op_standalone.py`) отказываю запуск, если `S % P != 0` (H=W=S).
2. Я ставлю `global = (W/P, H/P)`, local = NULL (пусть ICD).
3. Я выделяю `dst` на `(H/P)*(W/P)*P*P` float.
4. Готово: до enqueue — кратность и размер; после — `max_abs_err < 1e-5` vs CPU.

## Переделка при unsat

- S не кратно P → не звать ядро.
- Нужен pad/dilation → другой контракт / другое ядро (не штопать это).

## Z3 · глубина 2

- S%P=0 ∧ nout=S/P ∧ dst_len = nout²·P² ∧ src_len=S² → **sat**
- S=224 P=15 → **unsat** (остаток)


> Z3: `(* P nout)` и `mod` — **QF_NIA**. Под QF_LIA solver врёт/игнор — не опираться.
