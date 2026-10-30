#!/usr/bin/env python3
"""Общий выбор OpenCL-устройства для check_*.py."""
from __future__ import annotations


def list_devices(cl):
    return [d for p in cl.get_platforms() for d in p.get_devices()]


def pick_device(cl, index: int):
    devs = list_devices(cl)
    if not devs:
        raise SystemExit("no OpenCL devices")
    if index < 0 or index >= len(devs):
        raise SystemExit(f"-d {index} out of range (0..{len(devs) - 1})")
    return devs[index]
