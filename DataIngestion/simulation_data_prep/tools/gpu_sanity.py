#!/usr/bin/env python
"""gpu_sanity.py – quick CUDA check inside the data-prep container.

Usage (from repo root):
    docker compose run --rm --gpus all data-prep \
        python -m tools.gpu_sanity

Outputs:
  • detected GPU name + compute capability
  • matmul timing to confirm the kernel runs on GPU
"""
from __future__ import annotations

import time

import cupy as cp


def main() -> None:  # pragma: no cover
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    name = props["name"].decode()
    cc = f"{props['major']}.{props['minor']}"
    print(f"Detected GPU: {name} (compute capability {cc})")

    x = cp.random.rand(4000, 4000, dtype=cp.float32)
    cp.cuda.Device().synchronize()

    t0 = time.perf_counter()
    _ = x @ x.T
    cp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - t0

    print(f"✓ CuPy matmul on GPU took {elapsed:.3f} s")


if __name__ == "__main__":
    main()
