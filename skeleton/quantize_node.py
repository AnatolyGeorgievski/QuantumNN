#!/usr/bin/env python3
"""Учебный узел слоя C: симметричная int8-квантизация с обучаемым scale.

Учитель = float-вектор. Ученик = quantize/dequantize. Loss = MSE.
Без внешних LLM. Контракт приёмки снаружи (пакет особи / baseline).
"""
from __future__ import annotations

import math


def quantize(x: list[float], scale: float) -> list[int]:
    s = max(scale, 1e-8)
    return [max(-127, min(127, int(round(v / s)))) for v in x]


def dequantize(q: list[int], scale: float) -> list[float]:
    return [v * scale for v in q]


def mse(a: list[float], b: list[float]) -> float:
    n = max(len(a), 1)
    return sum((x - y) ** 2 for x, y in zip(a, b)) / n


def best_scale(teacher: list[float]) -> float:
    """Один параметр: scale = max|x|/127 — классический PTQ; затем локальный спуск."""
    peak = max((abs(v) for v in teacher), default=1.0) or 1.0
    scale = peak / 127.0
    best_s, best_l = scale, mse(teacher, dequantize(quantize(teacher, scale), scale))
    # grid refine around peak-based scale
    for k in range(-20, 21):
        s = scale * (1.0 + 0.02 * k)
        if s <= 0:
            continue
        l = mse(teacher, dequantize(quantize(teacher, s), s))
        if l < best_l:
            best_l, best_s = l, s
    return best_s


def distill_step(teacher: list[float], student_scale: float) -> tuple[float, float, float]:
    before = mse(teacher, dequantize(quantize(teacher, student_scale), student_scale))
    new_scale = best_scale(teacher)
    after = mse(teacher, dequantize(quantize(teacher, new_scale), new_scale))
    return before, after, new_scale


def main() -> None:
    teacher = [math.sin(i / 3.0) * (1.0 + 0.1 * i) for i in range(64)]
    rough = 0.5  # заведомо грубый ученик (переквант)
    before, after, scale = distill_step(teacher, rough)
    print(f"scale={scale:.6f}")
    print(f"mse_before={before:.6e}")
    print(f"mse_after={after:.6e}")
    ok = after <= before and after < 1e-2
    print("ACCEPT" if ok else "REJECT")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
