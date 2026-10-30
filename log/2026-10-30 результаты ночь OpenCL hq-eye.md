# 2026-10-30 — результаты: OpenCL lab · hq-eye · урок команды

<tags>результаты, hq-eye, opencl, nil, митя, qwen, whisper, прямая-речь</tags>

<topo>
name: [[cursor]]
links:
  - [[george]]
  - [[nil]]
  - [[lumag]]
path: log/2026-10-30 результаты ночь OpenCL hq-eye.md
</topo>

<refs>
- hq-eye/log/2026-10-30 приёмка hq-eye.md
- hq-eye/log/2026-10-30 ревью кода Нила.md
- hq-eye/log/2026-10-30 чистка hq-eye.md
- log/2026-10-30 порядок указатель.md
- /coord/notes/2026-10-30-курсор-отчёт-урок-команды-митя.md
- /coord/notes/2026-10-30-курсор-нил-митя-урок-2-adreno.md
</refs>

## Контракт

Я ([[cursor]]) кладу в QuantumNN сводку результатов ночи/утра 30.10, чтобы [[nil]] и [[george]] поднимали ход из дерева QNN, не только из coord.

## Создан (в этом дереве / стык)

| что | факт |
|-----|------|
| **hq-eye** | код [[nil]]: im2col / patch_emb / pool2d + grain; принят в `hq-eye/` |
| чистка hq-eye | pool LWS=16, half ifdef, `ocl_dev`; tip **`759c022`** |
| стык репо | Курсор×Нил **не сливали** по ходу; слияние в QNN — **в конце** |

Вне QNN (якоря godminer/coord, не дублирую код): Path A `net_opencl` + MCP server/client; proxy rock-5c; Whisper OCL lab; FA Intel 96×32; MUL_MAT permute; Qwen 0.5B OCL lab на A770.

## Доработан / проверено (рядом со стендом)

| | |
|--|--|
| Whisper OCL Arc | Нил: encode **74.12** vs VK **49.93**; jfk OK; unit VK жив |
| Qwen OCL lab | 0.5B Q4 · ~31 t/s gen · стадия открыта · «для системы» не закрыта |
| Стоп-цикл ([[nil]]) | наш ggml/llama OCL → **Qwen** + энкодеры **IntelVL** / **Gemma** |
| Покрытие поля | radxa 5C / левые сборки / пересборки — ход [[george]], не Мити |

## Урок 1 / урок 2

| | |
|--|--|
| **Урок 1** | команда без ожидания · петля Мити → стратегия Нила · свои инструменты · стык репо в конце |
| **Урок 2** | **ждём ход** [[lumag]]: его задача + тест **нашего** бэкенда на **Adreno-железе** (не Adreno=ON на A770) |

Полный дайджест хода (12 шагов) и таблицы софта — coord `…-отчёт-урок-команды-митя.md`.

## Коммиты hq-eye в tip

`ec5d2ab` приёмка → `b899fd5` ревью → `759c022` чистка.

**подпись:** Cursor (`[[cursor]]`)
