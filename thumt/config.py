import cPickle
import json
import logging
import re

def config():
	'''
		Get the default configuration.  
	'''
	config = {}
	config['model'] = 'RNNsearch' # the NMT model. Only change to "BiRNNsearch" on semi-supervised training

	# data
	config['src_text'] = '' # source corpus path
	config['trg_text'] = '' # target corpus path
	config['src'] = '' # preprocessed source corpus path 
	config['trg'] = '' # preprocessed target corpus path
	config['src_shuf'] = '' # shuffled source corpus path
	config['trg_shuf'] = '' # shuffled target corpus path
	config['vocab_src'] = '' # source index2vocab path
	config['vocab_trg'] = '' # target index2vocab path
	config['ivocab_src'] = '' # source vocab2index path
	config['ivocab_trg'] = '' # target vocab2index path
	config['data_corpus'] = 'json' # preprocessing tool for corpus, "cPickle" or "json", "json" is recommended
	config['data_vocab'] = 'cPickle' # do not modify this
	 
	config['index_unk_src'] = 1 # index for "UNK" symbol 
	config['index_unk_trg'] = 1
	config['index_eos_src'] = 30000 # vocabulary size
	config['index_eos_trg'] = 30000
	config['num_vocab_src'] = 30001 # vocabulary size + 1
	config['num_vocab_trg'] = 30001

	# neural networks
	config['dim_rec_enc'] = 1000 # encoder hidden layer dimension
	config['dim_rec_dec'] = 1000 # decoder hidden layer dimension
	config['dim_emb_src'] = 620 # word embedding dimension
	config['dim_emb_trg'] = 620
	config['maxout'] = 2 # maxout count

	# training
	config['maxlength'] = 50 # max sentence length
	config['sort_batches'] = 20 # number of batches sorted for efficiency
	config['batchsize'] = 80 # number of sentences in one batch
	config['max_iter'] = 1000000 # stop if reached
	config['try_iter'] = 100000 # stop if no improvement

	# sampling
	config['sample_freq'] = 100 # the frequency to sample 
	config['sample_length'] = config['maxlength'] # the maximum length for sampled sentences
	config['n_samples'] = 1 # number of sentences to sample
	config['sample_times'] = 1 # number of samples for each sentence
	config['sample_sentence'] = '' # if not empty, will always sample this sentence
	
	# checkpoint & save model
	config['save'] = True # periodically save and validate the model
	config['save_freq'] = 2000 # the frequency to save the model
	config['save_path'] = 'models' # the path to save the models
	config['checkpoint_freq'] = 2000 # the frequency to save a checkpoint model
	config['checkpoint_model'] = 'checkpoint_model.npz' # the checkpoint model. 
	config['checkpoint_status'] = 'checkpoint_status.pkl' # the checkpoint training status. 

	# optimizing
	config['optimizer'] = 'adadelta' # the optimizer
	# SGD
	config['lr'] = 1.
	# Adadelta
	config['rho'] = 0.95
	config['epsilon'] = 1e-6
	config['clip'] = 1. 
	# Adam
	config['alpha_adam'] = 0.0005
	config['alphadecay_adam'] = 0.998
	config['beta1_adam'] = 0.9
	config['beta2_adam'] = 0.999
	config['eps_adam'] = 1e-8

	# MRT
	config['MRT'] = False 
	config['init_model'] = 'models/model_best.npz' # the path of initialization model
	config['sampleN'] = 100 # number of samples for each sentence
	config['LenRatio'] = 1.5 # the multiplier of maximum length of samples
	config['MRT_alpha'] = 0.005 # the hyper parameter

	# semi-supervised training
	config['semi_learning'] = False
	config['reconstruct_lambda'] = 0.1
	config['auto_lambda_1'] = 1
	config['auto_lambda_2'] = 10
	config['semi_sampleN'] = 10
	config['sample_num'] = 10

	# validaiation and test
	config['valid_dir'] = 'validation'
	config['test_dir'] = 'eval'
	config['valid_src'] = ''
	config['valid_ref'] = ''
	config['test_src'] = []
	config['test_ref'] = []
	config['beam_size'] = 10

	return config

def update_config(config, dic):
	'''
		Update the configuration
	'''
	for key in dic:
		config[key] = dic[key]
	return config

def print_config(config):
	'''
		Print the configuration to screen
	'''
	logging.info("=====config=====")
	for key in config:
		logging.info('"' + key + '": ' + str(config[key]))
	logging.info("")

def load_config(config):
	'''
		Load the configuration file, which can be json format or the format defined below 
	'''
	try:
		result = json.loads(config)
	except:
		result = parse_config(config)
	return result

def parse_config(config):
	'''
		Parse the configuration file in the key : value format, each item is listed in one line.
		"#" symbol can be used for comment

		Example:
		batchsize: 50 # comment
		save_freq: 10000 

		will be parsed into {'batchsize': 50, 'save_freq': 10000}
	'''
	result = {}
	lines = config.split('\n')
	for line in lines:
		comment_mark = line.find('#')
		if comment_mark != -1:
			line = line[0 : comment_mark]
		spliter = line.find(':')
		if spliter != -1:
			key = line[0 : spliter]
			value = line[spliter + 1 :]
			result[eval(key.strip())] = eval(value.strip())
	return result
