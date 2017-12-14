# THUMT: An Open Source Toolkit for Neural Machine Translation
## Contents
* [Introduction](#introduction)
* [Versions](#versions)
* [License](#license)
* [Citation](#citation)
* [Development Team](#development-team)
* [Contact](#contact)

## Introduction

THUMT is a data-driven machine translation system developed by [the Natural Language Processing Group at Tsinghua University](http://nlp.csai.tsinghua.edu.cn/site2/index.php?lang=en).

Machine translation is a natural language processing task that aims to translate natural languages using computers automatically. Recent several years have witnessed the rapid development of end-to-end neural machine translation, which has become the new mainstream method in practical MT systems.

On top of [TensorFlow](http://tensorflow.org), THUMT is an open-source toolkit for neural machine translation with the following features:

* **Attention-based translation model.** THUMT implements the standard attention-based encoder-decoder framework for NMT. It also implements the latest Transformer architecture.
* **Minimum risk training.** Besides standard maximum likelihood estimation (MLE), THUMT also supports minimum risk training (MRT) that aims to find a set of model parameters that minimize the expected loss calculated using evaluation metrics such as BLEU on the training data.

Besides, THUMT also has Theano and DyNet implementations.

## Versions
* [THUMT-TensorFlow](https://github.com/thumt/THUMT)
* [THUMT-Theano](https://github.com/thumt/THUMT/tree/theano)
* THUMT-DyNet (coming soon)
* THUMT-Caffe (coming soon)

## License

The source code is dual licensed. Open source licensing is under the [BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause), which allows free use for research purposes. For commercial licensing, please email [thumt17@gmail.com](mailto:thumt17@gmail.com).

## Citation

Please cite the following paper:

> Jiacheng Zhang, Yanzhuo Ding, Shiqi Shen, Yong Cheng, Maosong Sun, Huanbo Luan, Yang Liu. 2017. [THUMT: An Open Source Toolkit for Neural Machine Translation](https://arxiv.org/abs/1706.06415). arXiv:1706.06415.

## Development Team

Project leaders: [Maosong Sun](http://www.thunlp.org/site2/index.php/zh/people?id=16), [Yang Liu](http://nlp.csai.tsinghua.edu.cn/~ly/), Huanbo Luan

Project members: Jiacheng Zhang, Yanzhuo Ding, Shiqi Shen, Yong Cheng

## Contributors 
* [Zhixing Tan](mailto:playinf@stu.xmu.edu.cn) (Xiamen University)

## Contact

If you have questions, suggestions and bug reports, please email [thumt17@gmail.com](mailto:thumt17@gmail.com).
