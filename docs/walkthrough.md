# THUMT Documentation

[THUMT](https://github.com/thumt/THUMT/tree/pytorch) is an open-source toolkit for neural machine translation developed by the Tsinghua Natural Language Processing Group.

## Walkthrough

* [Data Preparation](#data-preparation)
  * [Obtaining the Datasets](#obtaining-the-datasets)
  * [Running BPE](#running-bpe)
  * [Shuffling Training Set](#shuffling-training-set)
  * [Generating Vocabularies](#generating-vocabularies)
* [Training](#training)
* [Decoding](#decoding)

We provide a step-by-step guide with a running example: WMT 2018 Chinese-English news translation shared task.

## Data Preparation

### Obtaining the Datasets

Running THUMT involves three types of datasets:

* **Training set**: a set of parallel sentences used for training NMT models.
* **Validation set**: a set of source sentences paired with single or multiple target translations used for model selection and hyper-parameter optimization.
* **Test set**: a set of source sentences paired with single or multiple target translations used for evaluating translation performance on unseen texts.

In this walkthrough, we'll use the preprocessed official [dataset](http://data.statmt.org/wmt18/translation-task/preprocessed/zh-en/). Download and unpack the files `corpus.gz`:

```bash
gzip -d corpus.gz
```

The resulting file is `corpus.tsv`. Use the following command to generate source and target files:

```bash
cut -f 1 corpus.tsv > corpus.tc.zh
cut -f 2 corpus.tsv > corpus.tc.en
```

`corpus.tc.zh` and `corpus.tc.en` serve as the training set, which contains 24,752,392 pairs of sentences. Note that the Chinese sentences are tokenized and the English sentences are tokenized and truecased. Unpack the file `dev.tgz` using the following command:

```bash
tar xvfz dev.tgz
```

After unpacking,  `newdev2017.tc.zh` and `newsdev2017.tc.en` serve as the validation set, which contains 2,002 pairs of sentences. The test set we use is `newstest2017.tc.zh` and `newstest2017.tc.en`, which consists of 2,001 pairs of sentences. Note that both the validation and test sets use single references since there is only one gold-standard English translation for each Chinese sentence.

### Running BPE

For efficiency reasons, only a fraction of the full vocabulary can be used in neural machine translation systems. The most widely used  approach for addressing the open vocabulary problem is to use the Byte Pair Encoding (BPE). We recommend using BPE for THUMT.

First, download the source code of BPE using the following command:

```bash
git clone https://github.com/rsennrich/subword-nmt.git
```

To encode the training corpora using BPE, you need to generate BPE operations first. The following command will create two files named `bpe.zh` and `bpe.en`, which contain 32k BPE operations.

```bash
python subword-nmt/learn_bpe.py -s 32000 -t < corpus.tc.zh > bpe.zh
python subword-nmt/learn_bpe.py -s 32000 -t < corpus.tc.en > bpe.en
```

Then, the `apply_bpe.py` script runs to encode the training set using the generated BPE operations.

```bash
python subword-nmt/apply_bpe.py -c bpe.zh < corpus.tc.zh > corpus.tc.32k.zh
python subword-nmt/apply_bpe.py -c bpe.en < corpus.tc.en > corpus.tc.32k.en
```

The source side of the validation set and the test set also needs to be processed using the `apply_bpe.py` script.

```bash
python subword-nmt/apply_bpe.py -c bpe.zh < newsdev2017.tc.zh > newsdev2017.tc.32k.zh
python subword-nmt/apply_bpe.py -c bpe.zh < newstest2017.tc.zh > newstest2017.tc.32k.zh
```

Kindly note that while the source side of the validation set and test set is applied with BPE operations, the target side of them are not needed to be applied. This is because when evaluating the translation outputs, we will restore them in the normal tokenization and compare them with the original ground-truth sentences.

### Shuffling Training Set

The next step is to shuffle the training set, which proves to be helpful for improving the translation quality. Simply run the following command:

```bash
shuffle_corpus.py --corpus corpus.tc.32k.zh corpus.tc.32k.en
```

The resulting files `corpus.tc.32k.zh.shuf` and `corpus.tc.32k.en.shuf` rearrange the sentence pairs randomly.

### Generating Vocabularies

We need to generate vocabulary from the shuffled training set. This can be done by running the `build_vocab.py` script:

```bash
build_vocab.py corpus.tc.32k.zh.shuf vocab.32k.zh
build_vocab.py corpus.tc.32k.en.shuf vocab.32k.en
```

The resulting files `vocab.32k.zh.txt` and `vocab.32k.en.txt` are final source and target vocabularies used for model training.

## Training

We recommend using the Transformer model that delivers the best translation performance among all the three models supported by THUMT. The command for training a Transformer model is given by

```bash
thumt-trainer \
  --input corpus.tc.32k.zh.shuf corpus.tc.32k.en.shuf \
  --vocabulary vocab.32k.zh.txt vocab.32k.en.txt \
  --model transformer \
  --validation newsdev2017.tc.32k.zh \
  --references newsdev2017.tc.en \
  --parameters=batch_size=4096,device_list=[0,1,2,3],update_cycle=2 \
  --hparam_set base
```

Note that we set the `batch_size` on each device (e.g. GPU) to 4,096 words instead of 4,096 sentences. By default, the batch size for the Transformer model is defined in terms of word number rather than sentence number in THUMT. We set `update_cycle` to 2, which means the model parameters are updated every 2 batches. This effectively simulates the setting of `batch_size=32768` and requires less GPU memory. If you still run out of GPU memory, try to use smaller `batch_size` and larger `update_cycle`. For newer GPUs like Tesla V100, you can add `--half` to enable mixed-precision training, which can improves training speed and reduces memory usage.

`device_list=[0,1,2,3]` suggests that `gpu0-3` is used to train the model. THUMT supports to train NMT models on multiple GPUs. If `gpu0-7` are available, simply set `device_list=[0,1,2,3,4,5,6,7]` for the same batch size of 32768 (but with the training speed doubled). You may use the `nvidia-smi` command to find unused GPUs.

By setting `hparams_set=base`, we will train a base Transformer model. the training process will terminate at iteration 100,000 by default. During the training, the `thumt-trainer` command creates a `train` directory to store intermediate models called `checkpoints`, which will be evaluated on the validation set periodically.

Please kindly note again that while the source side of the validation set is applied with BPE operations (`newsdev2017.tc.32k.zh`), the target side of the validation set is in the original tokenization (`newsdev2017.tc.en`).

Only a small number of checkpoints that achieves highest BLEU scores on the validation set will be saved in the `train/eval` directory. This directory will be used in decoding.

## Decoding

The command for translating the test set using the trained Transformer model is given by

```bash
thumt-translator \
  --models transformer \
  --input newstest2017.tc.32k.zh \
  --output newstest2017.trans \
  --vocabulary vocab.32k.zh.txt vocab.32k.en.txt \
  --checkpoints train/eval \
  --parameters=device_list=[0],decode_alpha=1.2
```

Please kindly note that a lot of decoding techniques are actually working on the test set here, i.e. `decode_alpha`; varying `decode_alpha` during the command for training process only leads to varied translation performances on the evaluation set.

The translation file output by the `thumt-translator` is `newstest2017.trans`, which needs to be restored to the normal tokenization using the following command:

```bash
sed -r 's/(@@ )|(@@ ?$)//g' < newstest2017.trans > newstest2017.trans.norm
```

Finally, BLEU scores can be calculated using the [`multi-bleu.perl`](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl):

```bash
multi-bleu.perl -lc newstest2017.tc.en < newstest2017.trans.norm > evalResult
```

The resulting `evalResult` stores the calculated BLEU score.

[return to index](index.md)
