# THUMT Documentation

[THUMT](https://github.com/thumt/THUMT/tree/pytorch) is an open-source toolkit for neural machine translation developed by the Tsinghua Natural Language Processing Group.

## Benchmarks

We benchmark THUMT on the following datasets:

* [WMT14 En-DE](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8)
* [WMT18 Zh-En](http://data.statmt.org/wmt18/translation-task/preprocessed/zh-en/)

The testsets for WMT14 En-De and WMT18 Zh-En are `newstest2014` and `newstest2017` respectively.

| Dataset   |   Model   | Size | Steps | GPUs | Batch/GPU |   Mode   |  BLEU  |
|:---------:|:---------:|:----:|:-----:|:----:|:---------:|:--------:|:------:|
|WMT14 En-De|Transformer| Base | 100k  |   4  |  2*4096   |   FP16   | 26.85  |
|WMT14 En-De|Transformer| Base | 100k  |   4  |  2*4096   |   FP32   | 26.91  |
|WMT14 En-De|Transformer| Base | 100k  |   8  |   4096    |   FP32   | 26.95  |
|WMT14 En-De|Transformer| Base |  86k  |   8  |  2*4096   |   FP32   | 27.21  |
|WMT14 En-De|Transformer| Big  | 300k  |   8  |   4096    |   FP16   | 28.71  |
|WMT14 En-De|Transformer| Big  |  20k  |  16  |  8*4096   | DistFP16 | 28.68  |
|WMT18 Zh-En|Transformer| Big  | 300k  |   8  |  2*4096   |   FP16   | 24.07  |

[return to index](index.md)
