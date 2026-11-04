# Контракты · float_fma.cl · глубина 2

Точка внимания: `kernel float_fma`.  
Глубина **2**: ядро + host/буфер.

## Узел 0 — ядро

1. Я (`float_fma` в `opencl/grain/float_fma.cl`) — FP32 FMA/mul-add probe, не ggml.
2. Я пишу только `data[get_global_id(0)] = y` после 512 итераций.
3. Готово: одна запись на поток; чтение `data` мне не нужно (только запись по gid).

## Узел 1 — host

1. Ты даёшь буфер `data` длиной **> max get_global_id(0)** (строго ≥ global_size).
2. Ты запускаешь `-k float_fma` (или из `nil_grain.cl` тот же символ).
3. Готово: нет OOB на записи; enqueue rc==0.

## Вне окна

Эталон числа y, FP_FAST_FMAF vs soft — не глубина 2.

## Z3

- len > max_gid ∧ max_gid = global0-1 ∧ global0>0 → **sat**
- len = global0-1 при max_gid=global0-1 → **unsat** (равенство len>max_gid ложно)
