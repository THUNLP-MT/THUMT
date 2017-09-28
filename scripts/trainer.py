#! /usr/bin/env python

import sys
import os
import types
import datetime

root_dir = '/home/zhangjiacheng/git/THUMT_170913'
code_dir = root_dir + '/thumt'

def version():
	'''Display version.'''
	s = '----------------------------------------\n' + \
		' trainer v0.1\n' + \
		' 2017/06/19 - 2017/06/19\n' + \
		' (c) 2017 Thumt\n' + \
		'----------------------------------------\n'
	print s

def help():
	'''Display helping message.'''
	s = 'Usage: trainer [--help] ...\n' + \
		'Required arguments:\n' + \
		'  --config-file <file>            configuration file\n' + \
		'  --trn-src-file <file>           training set, source file\n' + \
		'  --trn-trg-file <file>           training set, target file\n' + \
		'  --vld-src-file <file>           validation set, source file\n' + \
		'  --vld-trg-file <file>           validation set, target file\n' + \
		'  --device {cpu, gpu0, ...}       device\n' + \
		'Optional arguments:\n' + \
		'  --training-criterion {0, 1, 2}  training criterion\n' + \
		'                                    0: maximum likelihood estimation (default)\n' + \
		'                                    1: minimum risk training\n' + \
		'                                    2: semi-supervise training\n' + \
		'  --mono-src-file <file>          monilingual training set, source file\n' + \
		'  --mono-trg-file <file>          monilingual training set, target file\n' + \
		'  --init-model-file <file>        initialization model file\n' + \
		'  --replace-unk {0, 1}            replacing unknown words\n' + \
		'                                    0: off (default)\n' + \
		'                                    1: on\n' + \
		'  --save-all-models {0, 1}        saving all intermediate models:\n' + \
		'                                    0: only save the best model (default)\n' + \
		'                                    1: save all intermediate models\n' + \
		'  --debug {0, 1}                  displaying debugging info\n' + \
		'                                    0: off (default)\n' + \
		'                                    1: on\n' + \
		'  --help                          displaying this message\n'
	print s
	sys.exit()

def convert_config_format(training_criterion, \
		                  config_file, \
		                  trn_src_file, \
						  trn_trg_file, \
						  vld_src_file, \
						  vld_trg_file, \
						  mono_src_file, \
						  mono_trg_file, \
						  init_model_file, \
						  debug, \
						  output_file):
	'''Convert configuration file format.'''
	d = {}
	if training_criterion == 0:
		d['MRT'] = False
		d['semi_learning'] = False
	elif training_criterion == 1:
		d['MRT'] = True
		d['semi_learning'] = False
	else:
		d['MRT'] = False
		d['semi_learning'] = True
		d['model'] = 'BiRNNsearch'
	d['src_text'] = trn_src_file
	d['trg_text'] = trn_trg_file
	d['valid_src'] = vld_src_file
	d['valid_ref'] = vld_trg_file
	d['init_model'] = init_model_file
	if debug == 0:
		d['verbose_level'] = 'info'
	else:
		d['verbose_level'] = 'debug'
	d['src'] = 'corpus/train.zh.json'
	d['trg'] = 'corpus/train.en.json'
	d['src_shuf'] = 'corpus/train.zh.json.shuf'
	d['trg_shuf'] = 'corpus/train.en.json.shuf'
	d['vocab_src'] = 'corpus/vocab.zh.pkl'
	d['vocab_trg'] = 'corpus/vocab.en.pkl'
	d['ivocab_src'] = 'corpus/ivocab.zh.pkl'
	d['ivocab_trg'] = 'corpus/ivocab.en.pkl'
	d['data_corpus'] = 'json'
	if mono_src_file == '':
		d['src_mono_text'] = ''
		d['src_mono'] = ''
		d['src_mono_shuf'] = ''
	else:
		d['src_mono_text'] = mono_src_file
		d['src_mono'] = 'corpus/mono.zh.json'
		d['src_mono_shuf'] = 'corpus/mono.zh.json.shuf'
	if mono_trg_file == '':
		d['trg_mono_text'] = ''
		d['trg_mono'] = ''
		d['trg_mono_shuf'] = ''
	else:
		d['trg_mono_text'] = mono_trg_file
		d['trg_mono'] = 'corpus/mono.en.json'
		d['trg_mono_shuf'] = 'corpus/mono.en.json.shuf'

	f1 = open(config_file, 'r')
	while True:
		line = f1.readline()
		if line == '':
			break
		if '[source vocabulary size]' in line:
			d['index_eos_src'] = int(line.split(']')[-1].strip())
			d['num_vocab_src'] = d['index_eos_src'] + 1
		elif '[target vocabulary size]' in line:
			d['index_eos_trg'] = int(line.split(']')[-1].strip())
			d['num_vocab_trg'] = d['index_eos_trg'] + 1
		elif '[source word embedding dimension]' in line:
			d['dim_emb_src'] = int(line.split(']')[-1].strip())
		elif '[target word embedding dimension]' in line:
			d['dim_emb_trg'] = int(line.split(']')[-1].strip())
		elif '[encoder hidden layer dimension]' in line:
			d['dim_rec_enc'] = int(line.split(']')[-1].strip())
		elif '[decoder hidden layer dimension]' in line:
			d['dim_rec_dec'] = int(line.split(']')[-1].strip())
		elif '[dropout ratio]' in line:
			d['dropout_maxout'] = float(line.split(']')[-1].strip())
		elif '[MRT sample size]' in line:
			d['sampleN'] = int(line.split(']')[-1].strip()) 
		elif '[MRT length ratio limit]' in line:
			d['LenRatio'] = float(line.split(']')[-1].strip())
		elif '[maximum sentence length]' in line:
			d['maxlength'] = int(line.split(']')[-1].strip())
		elif '[mini-batch size]' in line:
			d['batchsize'] = int(line.split(']')[-1].strip())
			if training_criterion == 1:  # batch size must be set to 1 in MRT
				d['batchsize'] = 1
		elif '[mini-batch sorting size]' in line:
			d['sort_batches'] = int(line.split(']')[-1].strip()) 
		elif '[iteration limit]' in line:
			d['max_iter'] = int(line.split(']')[-1].strip()) 
		elif '[convergence limit]' in line:
			d['try_iter'] = int(line.split(']')[-1].strip())
		elif '[optimizer]' in line:
			x = int(line.split(']')[-1].strip())
			if x == 0: 
				d['optimizer'] = 'SGD'
			elif x == 1:
				d['optimizer'] = 'adadelta'
			elif x == 2:
				d['optimizer'] = 'adam_slowstart'
			else:
				d['optimizer'] = 'adam'
		elif '[clip]' in line:
			d['clip'] = float(line.split(']')[-1].strip())
		elif '[SGD learning rate]' in line:
			d['lr'] = float(line.split(']')[-1].strip())
		elif '[AdaDelta rho]' in line:
			d['rho'] = float(line.split(']')[-1].strip())
		elif '[AdaDelta epsilon]' in line:
			d['epsilon'] = float(line.split(']')[-1].strip())
		elif '[Adam alpha]' in line:
			d['alpha_adam'] = float(line.split(']')[-1].strip()) 
		elif '[Adam alpha recover]' in line:
			d['alphadecay_adam'] = float(line.split(']')[-1].strip()) 
		elif '[Adam decay]' in line:
			d['decay_adam'] = float(line.split(']')[-1].strip()) 
		elif '[Adam beta1]' in line:
			d['beta1_adam'] = float(line.split(']')[-1].strip())
		elif '[Adam beta2]' in line:
			d['beta2_adam'] = float(line.split(']')[-1].strip()) 
		elif '[Adam eps]' in line:
			d['eps_adam'] = float(line.split(']')[-1].strip()) 
		elif '[beam size]' in line:
			d['beam_size'] = int(line.split(']')[-1].strip()) 
		elif '[model dumping iteration]' in line:
			d['save_freq'] = int(line.split(']')[-1].strip()) 
		elif '[checkpoint iteration]' in line:
			d['checkpoint_freq'] = int(line.split(']')[-1].strip()) 
	# generate configuation file in the internal format
	f2 = open(output_file, 'w')
	for key in d:
		if type(d[key]) is types.StringType:
			f2.write('"' + key + '": "' + str(d[key]) + '"\n')
		else:
			f2.write('"' + key + '": ' + str(d[key]) + '\n')

if __name__ == '__main__':
	# display version
	version()
	sys.stdout.flush()
	tb = datetime.datetime.now()
	# initialize arguments
	config_file = ''         # configuration file
	trn_src_file = ''        # training set, source file
	trn_trg_file = ''        # training set, target file
	vld_src_file = ''        # validation set, source file
	vld_trg_file = ''        # validation set, target file
	device = ''              # device
	training_criterion = 0   # training criterion
	unkreplace = 0           # replacing unknown words
	save_all = 0             # saving all intermediate models
	mono_src_file = ''       # monolingual training set, source file
	mono_trg_file = ''       # monolingual training set, target file
	init_model_file = ''     # initialization model file
	debug = 0                # debug
	# analyze command-line arguments
	i = 1
	while i < len(sys.argv):
		if sys.argv[i] == '--config-file':
			config_file = sys.argv[i + 1]
		elif sys.argv[i] == '--trn-src-file':
			trn_src_file = sys.argv[i + 1]
		elif sys.argv[i] == '--trn-trg-file':
			trn_trg_file = sys.argv[i + 1]
		elif sys.argv[i] == '--vld-src-file':
			vld_src_file = sys.argv[i + 1]
		elif sys.argv[i] == '--vld-trg-file':
			vld_trg_file = sys.argv[i + 1]
		elif sys.argv[i] == '--device':
			device = sys.argv[i + 1]
		elif sys.argv[i] == '--training-criterion':
			training_criterion = int(sys.argv[i + 1])
		elif sys.argv[i] == '--replace-unk':
			unkreplace = int(sys.argv[i + 1])
		elif sys.argv[i] == '--save-all-models':
			save_all = int(sys.argv[i + 1])
		elif sys.argv[i] == '--mono-src-file':
			mono_src_file = sys.argv[i + 1]
		elif sys.argv[i] == '--mono-trg-file':
			mono_trg_file = sys.argv[i + 1]
		elif sys.argv[i] == '--init-model-file':
			init_model_file = sys.argv[i + 1]
		elif sys.argv[i] == '--debug':
			debug = int(sys.argv[i + 1])
		else:
			print 'incorrect argument:', sys.argv[i]
			help()
		i += 2
	# check required arguments
	if config_file == '' or \
	   trn_src_file == '' or \
	   trn_trg_file == '' or \
	   vld_src_file == '' or \
	   vld_trg_file == '' or \
	   device == '':
		help()
	# convert configuration file format
	convert_config_format(training_criterion, \
			              config_file, \
			              trn_src_file, \
						  trn_trg_file, \
						  vld_src_file, \
						  vld_trg_file, \
						  mono_src_file, \
						  mono_trg_file, \
						  init_model_file, \
						  debug, \
			              '_config')
	# create directories and export environment variables
	os.system('mkdir -p corpus')
	os.system('mkdir -p models')
	os.system('mkdir -p validation')
	# pre-processing
	os.system('python ' + code_dir + \
			  '/preprocess.py -c _config')
	if unkreplace == 1:
		os.system('python ' + root_dir + '/thumt/mapping.py' + \
				  ' --src-file ' + trn_src_file + \
				  ' --trg-file ' + trn_trg_file + \
				  ' --out-file mapping.pkl')
	# training
	optional = ''
	if debug == 1:
		optional += ' --debug'
	if save_all == 1:
		optional += ' --save-all'
	if unkreplace == 1:
		optional += ' --map mapping.pkl'
	os.system('THEANO_FLAGS=floatX=float32,device=' + device + \
			  ',lib.cnmem=0.98 python ' + code_dir + \
			  '/train.py -c _config' + optional)
	# clean
	# os.system('rm _config')
	# show training time
	te = datetime.datetime.now()
	print 'The training started at ' + tb.strftime("%Y-%m-%d %H:%M:%S") + \
		  ' and ended at ' + te.strftime("%Y-%m-%d %H:%M:%S") + \
		  '. The total training time is %.2f hour(s).' % ((te - tb).seconds / 3600.0 + 24. * (te - tb).days)
