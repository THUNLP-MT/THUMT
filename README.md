# THUMT: An Open Source Toolkit for Neural Machine Translation

This is the experimental PyTorch implementation of THUMT.

## Prerequisite

* Python 3
* TensorFlow v2 (CPU version, for data reading only)
* PyTorch

## Current Status

* Architecture: Transformer
* Gradient aggregation: Yes
* Multi-GPU training: Yes
* Distributed training: Yes
* Mixed precision training/decoding: Yes
* Model validation: No
* Multi-GPU decoding: No

## Benchmarks

| Dataset     | Size | Steps | GPUs | Batch/GPU |   Mode   |  BLEU  |
|-------------|------|-------|------|-----------|----------|--------|
| WMT14 En-De | Base | 100k  |   8  |   4096    |   FP32   | 26.95  |
| WMT14 En-De | Base |  86k  |   8  |  2*4096   |   FP32   | 27.21  |
| WMT14 En-De | Big  |  20k  |  16  |  8*4096   | Dist+FP16| 28.68  |
