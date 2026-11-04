# Контракты · pool_2d_tile.cl · глубина 2

Точка внимания: `kernel pool_2D_avg`.  
Глубина окна: **2**. Курсор: ядро + host. Дальше не иду.

## Узел 0 — ядро

1. Я (`pool_2D_avg` в `opencl/grain/pool_2d_tile.cl`) считаю avg-pool при **s==k** и **pad=0**. Иначе меня не вызывают.
2. Я пишу `dst[mad24(row,ne0,pos)] = sum/(kh*kw)` только если `tid==0`.
3. Я читаю окно `kh×kw` с базы `src + im*ne00*ne01 + row*kh*ne00 + kw*pos`.
4. Готово: одно конечное float на клетку `(im,row,pos)` в пределах сетки; чужие клетки не трогаю.

## Узел 1 — host

1. Я (`check_pool2d_op.py`) ставлю `local_size = (POOL_SG, 1, 1)` при `POOL_SG=16`. Preferred WG 64 как local не ставлю.
2. Я ставлю `global_size = (ne0 * POOL_SG, ne1, C)` — тогда `get_group_id(0)∈[0,ne0)`, `(1)∈[0,ne1)`, `(2)∈[0,C)`.
3. Я даю аргументы `ne00=ne01=S`, `ne0=ne1=S/k`, `kw=kh=k` и отказываю запуск, если `S % k != 0`.
4. Готово: до enqueue контракт-assert проходит; после — `max_abs_err < 1e-5` против CPU avg.

## Вне окна

Subgroup-reduce детали, Mesa, pad≠0 — не сейчас.

## Переделка при unsat

- local≠16 → правлю host, не ослабляю контракт.
- S не кратно k → не запускаю ядро.

## Сверка host↔контракт (факт дерева)

`check_pool2d_op.py` уже: LWS=16, GWS=(ow*16, oh, C), S%k. Ребро **sat** по коду. GPU-прогон здесь без pyopencl — числа не гонял; контракт-assert на хосте — да.

## Z3 · глубина 2

- NDRange: local0=16 ∧ global0=ne0·16 ∧ ne00≥kw·ne0 ∧ ne01≥kh·ne1 ∧ S%k=0 (через ne0·kw=ne00 при квадрате) → **sat**
- local0=64 ∧ local0=POOL_SG → **unsat**
