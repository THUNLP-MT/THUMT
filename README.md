# THUMT: An Open Source Toolkit for Neural Machine Translation

This is the PyTorch implementation of THUMT.

## Prerequisite

* Python 3
* PyTorch
* TensorFlow-CPU 2.0 (for data reading, do NOT use GPU version)

## Features

* Multi-GPU training/decoding
* Multi-worker distributed training
* Mixed precision training/decoding
* Gradient aggregation
* TensorBoard for visualization

## Usage

See the document of TensorFlow version THUMT.

## Changes

* `learning_rate_decay` renamed to `learning_rate_schedule`.
* `constant_batch_size` renamed to `fixed_batch_size`.
* `--distribute` changed to `--distributed`.
* Add `--hparam_set` to select predefined hyper-parameters.
* Model ensemble is not available.

## Benchmarks

| Dataset   |   Model   | Size | Steps | GPUs | Batch/GPU |   Mode   |  BLEU  |
|:---------:|:---------:|:----:|:-----:|:----:|:---------:|:--------:|:------:|
|WMT14 En-De|Transformer| Base | 100k  |   4  |  2*4096   |   FP16   | 26.85  |
|WMT14 En-De|Transformer| Base | 100k  |   4  |  2*4096   |   FP32   | 26.91  |
|WMT14 En-De|Transformer| Base | 100k  |   8  |   4096    |   FP32   | 26.95  |
|WMT14 En-De|Transformer| Base |  86k  |   8  |  2*4096   |   FP32   | 27.21  |
|WMT14 En-De|Transformer| Big  | 300k  |   8  |   4096    |   FP16   | 28.71  |
|WMT14 En-De|Transformer| Big  |  20k  |  16  |  8*4096   | DistFP16 | 28.68  |
|WMT17 Zh-En|Transformer| Big  | 300k  |   8  |   4096    |   FP16   | 24.43  |
