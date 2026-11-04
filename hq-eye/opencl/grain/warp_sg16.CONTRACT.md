# Контракты · warp_sg16.cl · глубина 2

Точка внимания: `kernel warp_sg16`.  
Глубина **2**: ядро + host/буфер. Adreno / Mesa — вне окна.

## Узел 0 — ядро

1. Я (`warp_sg16` в `opencl/grain/warp_sg16.cl`) — probe Intel subgroup 16. На Adreno меня не вызывают.
2. Я требую `intel_reqd_sub_group_size(16)`; `ix = get_sub_group_local_id()` ∈ [0,16).
3. Я читаю `data[i+ix]` при `i = 0,16,…,496` и пишу `data[get_global_id(0)] = s`.
4. Готово: одна запись на global_id; чтение не за пределами буфера **если** host дал длину.

## Узел 1 — host / буфер

1. Ты (запускающий: `ocl -i warp_sg16.cl` или скрипт) даёшь `data` длиной **≥ 512** float (доступ `i+ix` до 496+15=511).
2. Ты ставишь local/global так, что `get_global_id(0) < length(data)` на запись.
3. Ты не собираешь это ядро на ICD без Intel subgroups как «общий suite».
4. Готово: длина буфера ≥ 512; enqueue без CL_OUT_OF_RESOURCES по OOB в этом контракте.

## Вне окна

Смысл half2-битов в float-ячейках, сравнение с эталоном — не глубина 2.

## Переделка при unsat

- length < 512 → увеличить буфер на host, не «надеюсь».
- Нет Intel SG → не звать это ядро.

## Z3 · буфер

- len≥512 ∧ max_ix=15 ∧ max_i=496 ∧ max_i+max_ix < len → **sat**
- len=256 при тех же i → **unsat**
