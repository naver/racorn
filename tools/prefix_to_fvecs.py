#!/usr/bin/env python3
# RACORN
# Copyright (c) 2026-present NAVER Corp.
# Apache-2.0
r"""Writes the first N vectors of a Big ANN benchmark file, for SIFT 100M and Text2Image 40M.

A Big ANN file (u8bin or fbin) is an 8-byte header of two uint32, the row count and the dimension,
followed by the rows. The script reads such a file from stdin. Only the first N rows need to
arrive (curl -r). It writes to stdout
  --to fvecs: float32 fvecs, each row prefixed by its dimension as an int32 (from u8bin or fbin);
  --to fbin:  fbin of float32 rows with the row count of the header set to N (with --type u8
              the rows are converted to float32).

  curl -sf -r 0-$((8 + N * D - 1)) <url>/base.1B.u8bin \
      | python3 prefix_to_fvecs.py --type u8 --n N --d D --to fvecs > sift_base.fvecs
"""
import argparse
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--type", choices=["u8", "f32"], required=True, help="element type of the input")
ap.add_argument("--n", type=int, required=True, help="rows to write")
ap.add_argument("--d", type=int, required=True, help="dimension, checked against the header")
ap.add_argument("--to", choices=["fvecs", "fbin"], required=True)
ap.add_argument("--chunk", type=int, default=1 << 20, help="rows per read")
a = ap.parse_args()
src, dst = sys.stdin.buffer, sys.stdout.buffer
rows, dims = np.frombuffer(src.read(8), dtype=np.uint32)
assert dims == a.d and rows >= a.n, (rows, dims)
dt = np.uint8 if a.type == "u8" else np.float32
if a.to == "fbin":
    dst.write(np.array([a.n, a.d], dtype=np.uint32).tobytes())
left = a.n
while left:
    k = min(left, a.chunk)
    buf = src.read(k * a.d * np.dtype(dt).itemsize)
    x = np.frombuffer(buf, dtype=dt).reshape(k, a.d).astype(np.float32)
    if a.to == "fvecs":
        out = np.empty((k, a.d + 1), dtype=np.float32)
        out[:, 0] = np.array([a.d], dtype=np.int32).view(np.float32)[0]
        out[:, 1:] = x
        dst.write(out.tobytes())
    else:
        dst.write(x.tobytes())
    left -= k
