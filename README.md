# THUMT: An Open Source Toolkit for Neural Machine Translation
## Contents
* [Introduction](#introduction)
* [Implementations](#implementations)
* [License](#license)
* [Citation](#citation)
* [Development Team](#development-team)
* [Contributors](#Contributors)
* [Contact](#contact)

## Introduction

Machine translation is a natural language processing task that aims to translate natural languages using computers automatically. Recent several years have witnessed the rapid development of end-to-end neural machine translation, which has become the new mainstream method in practical MT systems.

THUMT is an open-source toolkit for neural machine translation developed by [the Natural Language Processing Group at Tsinghua University](http://nlp.csai.tsinghua.edu.cn/site2/index.php?lang=en).


## Implementations
THUMT has currently two main implementations:

* [THUMT-TensorFlow](https://github.com/thumt/THUMT): a new implementation developed with [TensorFlow](https://github.com/tensorflow/tensorflow). It implements the sequence-to-sequence model (**Seq2Seq**) ([Sutskever et al., 2014](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)), the standard attention-based model (**RNNsearch**) ([Bahdanau et al., 2014](https://arxiv.org/pdf/1409.0473.pdf)), and the Transformer model (**Transformer**) ([Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)).

* [THUMT-Theano](https://github.com/thumt/THUMT/tree/theano): the original project developed with [Theano](https://github.com/Theano/Theano), which is no longer updated because MLA put an end to [Theano](https://github.com/Theano/Theano). It implements the standard attention-based model (**RNNsearch**) ([Bahdanau et al., 2014](https://arxiv.org/pdf/1409.0473.pdf)), minimum risk training (**MRT**) ([Shen et al., 2016](http://nlp.csai.tsinghua.edu.cn/~ly/papers/acl2016_mrt.pdf)) for optimizing model parameters with respect to evaluation metrics, semi-supervised training (**SST**) ([Cheng et al., 2016](http://nlp.csai.tsinghua.edu.cn/~ly/papers/acl2016_semi.pdf)) for exploiting monolingual corpora to learn bi-directional translation models, and layer-wise relevance propagation (**LRP**) ([Ding et al., 2017](http://nlp.csai.tsinghua.edu.cn/~ly/papers/acl2017_dyz.pdf)) for visualizing and anlayzing RNNsearch.


The following table summarizes the features of two implementations:

| Implementation | Model | Criterion | Optimizer | LRP |
| :------------: | :---: | :--------------: | :--------------: | :----------------: |
| Theano       |  RNNsearch | MLE, MRT, SST | SGD, AdaDelta, Adam | RNNsearch |   
| TensorFlow   |  Seq2Seq, RNNsearch, Transformer | MLE| Adam |n/a |

We recommend using [THUMT-TensorFlow](https://github.com/thumt/THUMT), which delivers better translation performance than [THUMT-Theano](https://github.com/thumt/THUMT/tree/theano). We will keep adding new features to [THUMT-TensorFlow](https://github.com/thumt/THUMT).

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
