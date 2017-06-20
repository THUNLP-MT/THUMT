# -*- encoding: utf-8 -*-
import numpy
import theano
import theano.tensor as tensor
from nmt import RNNsearch
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

def load_model_and_data(model_path):
	values = numpy.load(model_path)
	config = values['config']
	config = json.loads(str(config))
	model = eval(config['model'])(config)
	model.build(verbose=True)
	values = model.load(model_path, decode=True)
	data = DataCollection(config, train=False)
	data.vocab_src = json.loads(str(values['vocab_src']))
	data.ivocab_src = json.loads(str(values['ivocab_src']))
	data.vocab_trg = json.loads(str(values['vocab_trg']))
	data.ivocab_trg = json.loads(str(values['ivocab_trg']))
	data.encode_vocab()
	return model, data, config

def generate_tmp_val(model,data,src,num):
	src_index = data.toindex_source(src.split())
	result = model.translate(src_index)
	result = numpy.asarray(result)
	trg_index = numpy.transpose(numpy.asarray([result]))
	trg = data.print_target(trg_index)
	layers = model.get_layer(src_index, numpy.ones(src_index.shape, dtype=numpy.float32), trg_index, numpy.ones(trg_index.shape, dtype=numpy.float32))
	
	layers['enc_for_x'] = src_index
	layers['enc_back_x'] = src_index[::-1]
	layers['dec_y'] = trg_index
	numpy.savez('./val_' + str(num) + '.npz',**layers)
	f = open('sent_' + str(num) +'.txt','w')
	f.write(src + ' eos\n')
	f.write(trg + ' eos\n')
	f.close()
