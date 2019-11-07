# THUMT: An Open Source Toolkit for Neural Machine Translation

This is the experimental PyTorch implementation of THUMT.

## Prerequisite

* Python 3
* PyTorch
* TensorFlow-CPU 2.0 (for data reading, do NOT use GPU version)

## Features

* Multi-GPU training
* Multi-worker distributed training
* Gradient aggregation
* Mixed precision training/decoding
* TensorBoard for visualization

## Benchmarks

| Dataset     | Size | Steps | GPUs | Batch/GPU |   Mode   |  BLEU  |
|-------------|------|-------|------|-----------|----------|--------|
| WMT14 En-De | Base | 100k  |   8  |   4096    |   FP32   | 26.95  |
| WMT14 En-De | Base |  86k  |   8  |  2*4096   |   FP32   | 27.21  |
| WMT14 En-De | Big  |  20k  |  16  |  8*4096   | Dist+FP16| 28.68  |

## Usage

See the document of TensorFlow version THUMT.

## Changes

* `learning_rate_decay` renamed to `learning_rate_schedule`
* `constant_batch_size` renamed to `fixed_batch_size`
* `--distribute` changed to `--distributed`
* Add `--hparam_set` to select predefined hyper-parameters
* Some features are not implemented (e.g. validation)
* Model ensemble and multi-GPU decoding are currently not available
