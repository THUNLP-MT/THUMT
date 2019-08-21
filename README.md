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
* Distributed training: Experimental
* Mixed precision training/decoding: Experimental
* Model validation: No
* Multi-GPU decoding: No

## Benchmarks

| Dataset     | Size | Steps | GPUs | Batch/GPU |   Mode   |  BLEU  |
|-------------|------|-------|------|-----------|----------|--------|
| WMT14 En-De | Base | 100k  |   8  |   4096    |   FP32   | 26.95  |
| WMT14 En-DE | Base |  86k  |   8  |  2*4096   |   FP32   | 27.21  |
