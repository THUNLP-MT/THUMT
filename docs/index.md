# THUMT Documentation

[THUMT](https://github.com/thumt/THUMT/tree/pytorch) is an open-source toolkit for neural machine translation developed by the Tsinghua Natural Language Processing Group. This page describes the document of [THUMT-PyTorch](https://github.com/thumt/THUMT/tree/pytorch).

## Contents

* [Basics](#basics)
  * [Prerequisite](#prerequisite)
  * [Installation](#installation)
  * [Features](#features)
* [Walkthrough](#walkthrough)
* [Benchmarks](#benchmarks)

## Basics

### Prerequisites

* CUDA 10.0
* PyTorch
* TensorFlow-2.0 (CPU version)

### Installation

```bash
pip install --upgrade pip
pip install thumt
```

### Features

* Multi-GPU training & decoding
* Multi-worker distributed training
* Mixed precision training & decoding
* Model ensemble & averaging
* Gradient aggregation
* TensorBoard for visualization

## Walkthrough

We provide a step-by-step [walkthrough](walkthrough.md) with a running example: WMT 2018 Chinese-English news translation shared task.

## Benchmarks

We provide benchmarks on several datasets. See [here](benchmarks.md).
