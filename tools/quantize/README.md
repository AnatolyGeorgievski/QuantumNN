# tools/quantize · BF16/F16 → свой f8

<tags>
- /home/george/QuantumNN/qnn.h
- /home/george/QuantumNN/qnn_safetensors.c
- QuantumNN/QUANT.md
- QuantumNN/FPQUANT.md
- tools/quantize/REPRO-fp8.md
- /home/george/qnn-backend-ocl/tools/quantize/README.md
</tags>

<topo>
name: [[cursor]]
links:
  - [[QuantumNN]]
  - [[qnn-backend-ocl]]
path: /home/george/QuantumNN/tools/quantize/README.md
</topo>

## Назначение

Канон квантизации весов QNN: типы из `QuantumNN/qnn.h`, утилита `qnn-encode`, разбор — `qnn_safetensors.c`.  
Backend-OCL только ссылается сюда (`qnn-backend-ocl/tools/quantize/`).

```bash
cd /home/george/QuantumNN/tools/quantize
gcc -O2 -o qnn-encode qnn-encode.c -lm
./qnn-encode hf8 weights.f32 weights.hf8
./qnn-encode bf8 weights.f32 weights.bf8
./qnn-encode hf8-block N K [B] weights.f32 weights.hf8 weights.scale.bf16
./qnn-encode hf8-block N K [B] weights.f32 out.safetensors [tensor_name]
```

Без файлов — дым на коротком наборе f32. `hf8` — E4M3, `bf8` — E5M2 (номера как в `qnn.h`: `F8_E4M3`/`HF8`=43, `BF8`=31).

`hf8-block` — как Qwen FP8: блок `B×B` (по умолчанию 128), `scale_inv = amax/448` BF16, вес HF8. Dequant: `fp8 * scale_inv`. Выход `.safetensors` — пара `name` + `name_scale_inv`.

Рецепт (HF range → encode → `qnn-st`): [REPRO-fp8.md](REPRO-fp8.md).

## План

Финальная система жмёт неквантованный формат под архитектуру. Один оптимальный формат на архитектуру.

Сначала смотрим, покрывает ли тензор весь диапазон BF16; иначе уходим в более узкий тип без потери. Дальше — векторное преобразование (AVX-512 / AVX10.+). Широкий вес: BF16 → f32 → квант. `hf8`/`bf8` — для готового f32; `Q4_K`/`Q6_K` и свои имена — когда дойдём.

Отладка: исходный BF16 → `llama-quantize` и наш `qnn-encode` → две ошибки рядом.

**подпись:** Курсор
