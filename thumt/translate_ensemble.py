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
parser.add_argument('-m', '--model', required = True, help = 'path to NMT model', nargs = '+')
parser.add_argument('-bs', '--beam-size', type = int, default = 10)
parser.add_argument('-unk', '--unk-replace', action = 'store_true', help = 'replacing unknown words in translation')
parser.add_argument('-ln', '--length-norm', action = 'store_true', help = 'use length normalization')
parser.add_argument('-t', '--trans', help = 'path to translation with UNK. If set, do not translate source sentences')
parser.add_argument('-map', '--mapping', help = 'mapping file for unkreplace')
parser.add_argument('-v', '--verbose', action = 'store_true', help = 'print each translation')

def translate_ensemble(models, config, x, beam_size=10, length_norm = True):
	num_models = len(models)
	
	result = [[]]
	loss = [0.]
	result_eos = []
	loss_eos = []
	beam = beam_size

	c = []
	state = []
	emb_y = []
	for i in range(num_models):
		tmpc, tmpstate = models[i].get_context_and_init(x)
		c.append(tmpc)
		state.append(tmpstate)
		emb_y.append(numpy.zeros((1, config['dim_emb_trg']),dtype='float32'))
	
	for l in range(x.shape[0]*3):
		ctx = []
		probs = []
		for i in range(num_models):
			tmpenergy, tmpctx = models[i].get_probs(numpy.repeat(c[i], len(result), axis=1), state[i], emb_y[i])
			ctx.append(tmpctx)
			probs.append(tools.softmax(tmpenergy))
		losses = -numpy.log(probs[0])
		for i in range(1, num_models):
			losses -= numpy.log(probs[i])
		if l < x.shape[0]/2:
			losses[:, config['index_eos_trg']] = numpy.inf
		for i in range(len(loss)):
			if length_norm:
				losses[i] += (loss[i] * l)
				losses[i] /= (l+1)
			else:
				losses[i] += loss[i]
		best_index_flatten = numpy.argpartition(losses.flatten(), beam)[:beam]
		best_index = [(index/config['num_vocab_trg'],index%config['num_vocab_trg']) for index in best_index_flatten]
		
		new_y = []
		new_result = []
		new_loss = []
		new_ctx = []
		new_state = []
		for i in range(num_models):
			new_ctx.append(numpy.zeros((beam, 2*config['dim_rec_enc']), dtype='float32'))
			new_state.append(numpy.zeros((beam, config['dim_rec_dec']), dtype='float32'))
		for i in range(beam):
			index = best_index[i]
			new_result.append(result[index[0]]+[index[1]])
			new_loss.append(losses[index[0], index[1]])
			new_y.append(index[1])
			for j in range(num_models):
				new_ctx[j][i] = ctx[j][index[0]]
				new_state[j][i] = state[j][index[0]]

		new_emby = []
		for i in range(num_models):
			new_emby.append(models[i].get_trg_embedding(numpy.asarray(new_y, dtype='int64'))[0])
			new_state[i] = models[i].get_next(new_ctx[i], new_state[i], new_emby[i])

		state = []
		emb_y = []
		result = []
		loss = []
		for i in range(num_models):
			state.append([])
			emb_y.append([])
		for i in range(beam):
			if new_result[i][-1] == config['index_eos_trg']:
				result_eos.append(new_result[i])
				loss_eos.append(new_loss[i])
				beam -=1
			else:
				result.append(new_result[i])
				loss.append(new_loss[i])
				for j in range(num_models):
					state[j].append(new_state[j][i])
					emb_y[j].append(new_emby[j][i])

		if beam <= 0:
			break
		for i in range(num_models):
			state[i] = numpy.asarray(state[i], dtype='float32')
			emb_y[i] = numpy.asarray(emb_y[i], dtype='float32')

	if len(result_eos) > 0:
		return result_eos[numpy.argmin(loss_eos)]
	elif beam_size > 100:
		logging.warning('cannot find translation in beam size %d' % beam_size)
		return []
	else:
		logging.info('cannot find translation in beam size %d, try %d' % (beam_size, beam_size*2))
		return translate_ensemble(models, config, x, beam_size=beam_size*2)


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
	ensemble = False

	if len(args.model) >= 2:
		path_models = args.model
		args.model = args.model[0]
		ensemble = True
	else:
		args.model = args.model[0]

	# load vocab and model
	print 'Loading vocab and model'
	values = numpy.load(args.model)
	config = config()
	config = update_config(config, json.loads(str(values['config'])))
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

	# load other models for ensemble
	if ensemble:
		models = [model]
		for i in range(1, len(path_models)):
			print 'Loading model #' + str(i + 1)
			next_model = eval(config['model'])(config)
			next_model.build()
			next_model.load(path_models[i])
			models.append(next_model)
	else:
		models = [model]

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
			result = translate_ensemble(models, config, src_index, length_norm = args.length_norm)
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
		print num
		if num % 100 == 0:
			ed = time.time()
			print num, 'sentences translated, average', (ed - st) / num, 'seconds per sentence'
		line = input.readline()


