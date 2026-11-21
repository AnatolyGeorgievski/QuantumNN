# Повторить · блочный FP8 (Qwen3.8) и наш hf8-block

<tags>
- /home/george/QuantumNN/tools/quantize/qnn-encode.c
- /home/george/QuantumNN/tools/quantize/README.md
- /home/george/QuantumNN/qnn.h
- /home/george/QuantumNN/qnn_safetensors.c
- QuantumNN/QUANT.md
- QuantumNN/FPQUANT.md
</tags>

<topo>
name: [[cursor]]
links:
  - [[qnn-backend-ocl]]
  - [[QuantumNN]]
path: /home/george/QuantumNN/tools/quantize/REPRO-fp8.md
</topo>

Канон кода: `QuantumNN/tools/quantize/` + `QuantumNN/qnn.h`.

Дата прогона: 2026-11-21. Эталон: `gate_proj` слоя 0, вырезка `[0:256, 0:256]`.

## Что проверяем

1. Разобрать чужой FP8 (`F8_E4M3` + `weight_scale_inv`, блок 128×128).
2. Взять тот же кусок BF16 из базовой модели.
3. Сжать своим `qnn-encode hf8-block` в `.safetensors`.
4. Прочитать выход через `qnn-st -n` (QuantumNN).

База BF16: https://huggingface.co/Qwen/Qwen3.8-27B  
FP8: https://huggingface.co/Qwen/Qwen3.8-27B-FP8  

`quantization_config`: `fmt=e4m3`, `weight_block_size=[128,128]`.  
Dequant: `w ≈ e4m3(q) * scale_inv[i//128][j//128]`, где `scale_inv = amax/448`.

## Сборка

```bash
# наш квант (этот репозиторий)
cd /home/george/QuantumNN/tools/quantize
gcc -O2 -o qnn-encode qnn-encode.c -lm
./qnn-encode   # дым: короткий набор f32

# чтение safetensors (QuantumNN)
cd /home/george/QuantumNN
gcc -O3 -mf16c -D_FILE_OFFSET_BITS=64 -o /tmp/qnn-st \
  qnn_safetensors.c json.c `pkgconf --cflags --libs glib-2.0` -lm
```

## A · заголовки FP8 (без весов модели)

Шарды FP8: `layers-0..63.safetensors`, `outside.safetensors`, `mtp.safetensors` + `model.safetensors.index.json`.

```bash
python3 <<'PY'
import struct, urllib.request, pathlib, json
base = "https://huggingface.co/Qwen/Qwen3.8-27B-FP8/resolve/main"
out = pathlib.Path("/tmp/qwen38-fp8-hdr"); out.mkdir(exist_ok=True)
urllib.request.urlretrieve(f"{base}/model.safetensors.index.json", out / "model.safetensors.index.json")
idx = json.load(open(out / "model.safetensors.index.json"))
for name in sorted(set(idx["weight_map"].values())):
    url = f"{base}/{name}"
    def get(a, b):
        req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()
    ln = struct.unpack("<Q", get(0, 7))[0]
    (out / name).write_bytes(get(0, 7 + ln))
    print(name, "header", ln)
PY

/tmp/qnn-st -i /tmp/qwen38-fp8-hdr/model.safetensors.index.json --list | head
/tmp/qnn-st -i /tmp/qwen38-fp8-hdr/layers-0.safetensors --list
```

Ожидание в `layers-0`: `BF16` на мелких тензорах и `F8_E4M3` + `*_scale_inv` на крупных матрицах.

## B · вырезка BF16 256×256 с базовой модели

Тензор: `model.language_model.layers.0.mlp.gate_proj.weight`, shape `[17408, 5120]`.

```bash
python3 <<'PY'
import struct, json, urllib.request, pathlib, numpy as np
base = "https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main"
out = pathlib.Path("/tmp/qwen38-bf16-hdr"); out.mkdir(exist_ok=True)
urllib.request.urlretrieve(f"{base}/model.safetensors.index.json", out / "model.safetensors.index.json")
wm = json.load(open(out / "model.safetensors.index.json"))["weight_map"]
key = "model.language_model.layers.0.mlp.gate_proj.weight"
shard = wm[key]
url = f"{base}/{shard}"

def get(a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.read()

ln = struct.unpack("<Q", get(0, 7))[0]
hdr = get(0, 7 + ln)
meta = json.loads(hdr[8:8 + ln])
info = meta[key]
Wn = Wk = 256
K = info["shape"][1]
base_off = 8 + ln
raw = get(base_off + info["data_offsets"][0],
          base_off + info["data_offsets"][0] + Wn * K * 2 - 1)
W = np.frombuffer(raw, dtype=np.uint16).reshape(Wn, K)[:, :Wk].copy()
f = (W.astype(np.uint32) << 16).view(np.float32)
f.tofile(out / "gate_256.f32")
print("amax", float(np.max(np.abs(f))), "->", out / "gate_256.f32")
PY
```

Ожидание: `amax ≈ 0.0688`.

## C · наш блочный квант → safetensors

```bash
cd /home/george/QuantumNN/tools/quantize
NAME=model.language_model.layers.0.mlp.gate_proj.weight
./qnn-encode hf8-block 256 256 \
  /tmp/qwen38-bf16-hdr/gate_256.f32 \
  /tmp/qwen38-bf16-hdr/gate_ours.safetensors \
  "$NAME"
```

Ожидание (порядок величины, прогон 2026-11-21):

```
hf8-block N=256 K=256 B=128 scales=2x2 max_abs=0.00122 rmse=0.000253
wrote ...gate_ours.safetensors
```

Сырой выход без контейнера:

```bash
./qnn-encode hf8-block 256 256 \
  /tmp/qwen38-bf16-hdr/gate_256.f32 \
  /tmp/qwen38-bf16-hdr/gate.hf8 \
  /tmp/qwen38-bf16-hdr/gate.scale.bf16
```

## D · разбор нашего файла

```bash
NAME=model.language_model.layers.0.mlp.gate_proj.weight
/tmp/qnn-st -i /tmp/qwen38-bf16-hdr/gate_ours.safetensors --list -n "$NAME"
```

Ожидание: `F8_E4M3 block=128x128 scale_inv=2x2`, превью dequant, `scale_inv` порядка `1e-4`.

## E · эталонные ошибки vs BF16

| метод | max\|err\| | rmse |
|--|--|--|
| Qwen FP8 block128 | ≈0.00125 | ≈0.000255 |
| наш hf8 поэлементно | ≈0.00195 | ≈0.000568 |
| наш hf8-block 128 | ≈0.00122 | ≈0.000253 |

Поэлементный (хуже по RMSE):

```bash
./qnn-encode hf8 /tmp/qwen38-bf16-hdr/gate_256.f32 /tmp/qwen38-bf16-hdr/gate_elem.hf8
```

## Замечания

- Веса целиком не качать: index + header или HTTP Range нужного тензора.
- Имена шардов FP8 — `layers-N.safetensors`, не `model-00001-of-…`.
- `qnn-st -n` на F8 ищет парный `*_scale_inv` в том же файле.
- Бинарник `qnn-encode` в git не класть — только исходник `.c`.

**подпись:** Курсор
