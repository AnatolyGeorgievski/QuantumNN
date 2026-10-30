# 2026-10-29 — OpenCL: старт на phantom (тесты)

<tags>opencl, phantom, arc770, benchmark, nil, тест, llama, inbox, прямая-речь</tags>

<topo>
name: [[cursor]]
links:
  - [[george]]
  - [[nil]]
path: QuantumNN/log/2026-10-29 OpenCL старт phantom.md
</topo>

<toc>
- [Сейчас](#сейчас)
- [Контракт: старт на phantom](#контракт-старт-на-phantom)
- [Порядок без календаря](#порядок-без-календаря)
- [Контракт: Нил](#контракт-нил)
- [Базовая линия phantom](#базовая-линия-phantom)
</toc>

<refs>
- OpenCL-Benchmark/README.md
- inbox/benchmark.cl
- LLaMa.md — GGML_OPENCL
- log/2026-10-15.md — метка opencl
- log/2026-10-29 phantom-opencl-bench.txt
- godminer/log/2026-10-29 оркестратор не редактор.md
- /coord/notes/2026-10-29-курсор-нил-opencl-phantom.md
- /home/george/godminer/log/2026-10-29 MCP и OpenCL концепт.md
- /home/george/godminer/log/2026-10-29 OpenCL лестница тестов.md
- /home/george/godminer/log/2026-10-29 OpenCL быстрый старт.md
</refs>

Срок я не обещаю. Plan-карточку не раздуваю: этот лист — рабочий старт.

## Сейчас

| факт | значение |
|------|----------|
| Хост | phantom `.185`, user `george` |
| GPU | Intel Arc A770, OpenCL 3.0, ICD ок |
| Пакеты | `intel-opencl-icd`, `ocl-icd-opencl-dev`, headers — стоят |
| Bench | `~/opencl-bench` на phantom; `./benchmark.exe 0` прошёл 2026-10-29 |

Phantom — лабораторный OpenCL-носитель (слой B/C). Я не подменяю им gpu03-Qwen.

## Контракт: старт на phantom

Я ([[cursor]]) или ты ([[george]]) на phantom:

1. Я проверяю устройство: `clinfo -l` → Device #0 Arc A770.
2. Я собираю и гоняю bench: `cd ~/opencl-bench && make && ./benchmark.exe 0`.
3. Готово: FP32/FP16 в stdout; сверяю с таблицей «Базовая линия» ниже (±5% ок).

Канон сборки: [OpenCL-Benchmark/README.md](../OpenCL-Benchmark/README.md).  
Исходники на arc2: `/home/george/QuantumNN/OpenCL-Benchmark/` → rsync в `~/opencl-bench`.

Следующий кернел после bench: я беру [inbox/benchmark.cl](../inbox/benchmark.cl) через `qnn/opencl` (`ocl`).

## Порядок без календаря

Зависимости, не дедлайны. Лестница: godminer [OpenCL лестница тестов](/home/george/godminer/log/2026-10-29%20OpenCL%20лестница%20тестов.md).

1. **Базовая линия** — я закрыл: `clinfo` + `benchmark.exe 0`.
2. **miner/opencl** — на диске `/home/george/qnn/opencl/`; зеркало `inbox/`.
3. **llama.cpp** — я гоняю `test-backend-ops test -b OpenCL` (op vs CPU); чат-модель не сейчас.
4. **Стык курса / DPAS** — после зелёных 2–3; LLaMa.md.
5. **MCP** — tools по ступеням ([концепт](/home/george/godminer/log/2026-10-29%20MCP%20и%20OpenCL%20концепт.md)).

Я не смешиваю Windows Contour / Patriot с phantom OpenCL. Личное Нила — `/geolab/nil`; площадка — phantom + этот журнал.

## Контракт: Нил

Ты ([[nil]]) вписываешься без права ломать канон:

1. Ты на phantom (доступ — слово [[george]]) гоняешь `clinfo -l` и `./benchmark.exe 0`.
2. Ты кладёшь в `/geolab/nil` или coord таблицу чисел + driver version; сверяешь с «Базовая линия» ниже.
3. Дальше — один кернел из inbox или свой минимальный. Патч в QuantumNN — только после сверки с [[george]] / [[cursor]].
4. Готово шага 1–2: числа в coord и совпадение driver `26.05.037020` (или явно отмеченное отличие).

Coord: `2026-10-29-курсор-нил-opencl-phantom.md`.

## Базовая линия phantom

Дата: 2026-10-29. Driver: `26.05.037020 (Linux)`. Device 0 Arc A770, 16287 MB VRAM, 512 CU @ 2400 MHz.

| метрика | значение |
|---------|----------|
| FP32 | 12.210 TFLOPs/s |
| FP16 | 18.754 TFLOPs/s |
| INT32 | 5.765 TIOPs/s |
| INT16 | 36.193 TIOPs/s |
| DP4A | 36.747 TIOPs/s |
| MWC | 2.397 TBOPs/s |
| TNN | 86.938 TBOPs/s |
| MMA.h | 225.756 TIOPs/s |
| BW block R/W | 1746 / 4220 GB/s |
| PCIe bidir | 11.01 GB/s (Gen4 x16) |

FP64: not supported (ожидаемо для Arc).

Сырой хвост: [log/2026-10-29 phantom-opencl-bench.txt](2026-10-29%20phantom-opencl-bench.txt).

**подпись:** Cursor (`[[cursor]]`)
