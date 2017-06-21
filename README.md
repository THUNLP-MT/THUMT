# THUMT: An Open Source Toolkit for Neural Machine Translation
## Contents
* [Introduction](#introduction)
* [Website](#website)
* [User Manual](#user-manual)
* [Documentation](#documentation)
* [License](#license)
* [Citation](#citation)
* [Development Team](#development-Team)
* [Contact](#contact)

## Introduction

THUMT is a data-driven machine translation system developed by [the Natural Language Processing Group at Tsinghua University](http://nlp.csai.tsinghua.edu.cn/site2/index.php?lang=en).

Machine translation is a natural language processing task that aims to translate natural languages using computers automatically. Recent several years have witnessed the rapid development of end-to-end neural machine translation, which has become the new mainstream method in practical MT systems.

On top of [Theano](http://deeplearning.net/software/theano/), THUMT is an open-source toolkit for neural machine translation with the following features:

* **Attention-based translation model.** THUMT implements the standard attention-based encoder-decoder framework for NMT.
* **Minimum risk training.** Besides standard maximum likelihood estimation (MLE), THUMT also supports minimum risk training (MRT) that aims to find a set of model parameters that minimize the expected loss calculated using evaluation metrics such as BLEU on the training data.
* **Exploiting monolingual data.** THUMT provides semi-supervised training (SST) for NMT that is capable of exploiting abundant monolingual corpora to improve the learning of both source-to-target and target-to-source NMT models.
* **Visualization.** To better understand the internal workings of NMT, THUMT features a visualization tool to demonstrate the relevance between each intermediate state and its relevant contextual words.

## Website

[http://thumt.thunlp.org](http://thumt.thunlp.org)

## User Manual

This [user manual](http://thumt.thunlp.org/static/THUMT.pdf) describes how to install and use THUMT.

## Documentation

This [documentation](http://thumt.thunlp.org/static/document/index.html) provides detailed information about the functions in THUMT.

## License

The source code is dual licensed. Open source licensing is under the [BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause), which allows free use for research purposes. For commercial licensing, please email [thumt17@gmail.com](mailto:thumt17@gmail.com).

## Citation

Please cite the following paper:

> Jiacheng Zhang, Yanzhuo Ding, Shiqi Shen, Yong Cheng, Maosong Sun, Huanbo Luan, Yang Liu. 2017. [THUMT: An Open Source Toolkit for Neural Machine Translation](https://arxiv.org/abs/1706.06415). 

## Development Team

Project leaders: [Maosong Sun](http://www.thunlp.org/site2/index.php/zh/people?id=16), [Yang Liu](http://nlp.csai.tsinghua.edu.cn/~ly/), Huanbo Luan

Project members: Jiacheng Zhang, Yanzhuo Ding, [Shiqi Shen](http://nlp.csai.tsinghua.edu.cn/~ssq/), Yong Cheng

## Contact

If you have questions, suggestions and bug reports, please email [thumt17@gmail.com](mailto:thumt17@gmail.com).





