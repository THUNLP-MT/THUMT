import numpy
import theano
import theano.tensor as tensor
from nmt import RNNsearch
from binmt import BiRNNsearch
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import tools
from layer import LayerFactory
from config import * 
from optimizer import adadelta, SGD
from data import *

import cPickle
import json
import argparse
import signal
import time

parser = argparse.ArgumentParser("train the translation model")
parser.add_argument('-i', '--input', required = True, help = 'path to sentences to be translated')
parser.add_argument('-o','--output', required = True, help = 'path to translations')
parser.add_argument('-m', '--model', required = True, help = 'path to NMT model')
parser.add_argument('-bs', '--beam-size', type = int, default = 10)
parser.add_argument('-unk', '--unk-replace', action = "store_true", help = 'replacing unknown words in translation')
parser.add_argument('-t', '--trans', help = 'path to translation with UNK. If set, do not translate source sentences')
parser.add_argument('-map', '--mapping', help = ' mapping file for unkreplace')
parser.add_argument('-v', '--verbose', action = "store_true", help = 'print each translation')

def unkreplace(src, trg, attention, mapping):
	attention = attention.reshape((attention.shape[0], attention.shape[1]))
	for i in range(len(trg)):
		if trg[i] != 'UNK':
			continue
		index = numpy.argmax(attention[i][:-1])
		trg[i] = src[index]
		if mapping.has_key(trg[i]):
			trg[i] = mapping[trg[i]]
	return ' '.join(trg)

if __name__ == "__main__":
	args = parser.parse_args()

	# load vocab and model
	print 'Loading vocab and model'
	values = numpy.load(args.model)
	config = values['config']
	config = json.loads(str(config))
	model = eval(config['model'])(config)
	model.build()
	values = model.load(args.model, decode = True)

	data = DataCollection(config, train = False)
	data.vocab_src = json.loads(str(values['vocab_src']))
	data.ivocab_src = json.loads(str(values['ivocab_src']))
	data.vocab_trg = json.loads(str(values['vocab_trg']))
	data.ivocab_trg = json.loads(str(values['ivocab_trg']))
	data.encode_vocab()

	try:
		mapping = json.loads(str(values['mapping']))
		mapping = {i.encode('utf-8'): mapping[i].encode('utf-8') for i in mapping}
	except:
		mapping = None

	# load source sentences
	input = open(args.input, 'r')
	output = open(args.output, 'w')
	if args.unk_replace:
		if args.mapping:
			mapping = cPickle.load(open(args.mapping, 'r'))
		if args.trans:
			trans = open(args.trans, 'r')

	line = input.readline()
	num = 0
	print 'Start translating'
	st = time.time()
	while line != '':
		line = line.strip()
		src = line.split(' ')
		src_index = data.toindex_source(src)
		if args.trans:
			# load translation result
			result = trans.readline().strip().split(' ')
			result = numpy.transpose(data.toindex_target(result))[0]
		else:
			# translate
			result = model.translate(src_index)
			result = numpy.asarray(result)
		if args.unk_replace:
			# replace unknown words
			trg_index = numpy.transpose(numpy.asarray([result]))
			attention = model.get_attention(src_index, numpy.ones(src_index.shape, dtype = numpy.float32), trg_index, numpy.ones(trg_index.shape, dtype = numpy.float32))
			trg = data.print_target(result)
			result = unkreplace(src, trg.split(' '), attention[0], mapping)
			print >> output, result
		else:
			print >> output, data.print_target(result)
		num += 1
		if num % 100 == 0:
			ed = time.time()
			print num, 'sentences translated, average', (ed - st) / num, 'seconds per sentence'
		line = input.readline()


