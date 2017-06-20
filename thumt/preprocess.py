import numpy
import cPickle
import argparse
from config import *
import random
import re
import json
import logging
import os

logging.basicConfig(level = logging.DEBUG,
	                format = '[%(asctime)s %(levelname)s] %(message)s',
	                datefmt = '%d %b %H:%M:%S')


parser = argparse.ArgumentParser("generate vocabulary file and pickled corpus")
parser.add_argument('-c', '--config', help = 'path to config file')
parser.add_argument('-wd', '--withdict', action = "store_true", help = 'preprocess with existing vocabulary file')
parser.add_argument('-gr', '--fromgr', action = "store_true", help = 'transfer from Groundhog vocabulary file, must use with --wd')
parser.add_argument('-np', '--nopreprocess', action = "store_true", help = 'skip preprocessing')
parser.add_argument('-ns', '--noshuffle', action = "store_true", help = 'skip shuffling')

def preprocess(num, \
		       vocab_f, \
			   ivocab_f, \
			   input_f, \
			   output_f, \
			   data_vocab = 'cPickle', \
			   data_corpus = 'json', \
			   withdict = False, \
			   fromgr = False):
	'''
		Count the most frequent words in the training corpus.
		Then, generate the vocabulary file and index the corpus. 

		:type vocab_f: string
		:param vocab_f: the path to vocabulary file

		:type ivocab_f: string
		:param ivocab_f: the path to vocabulary to index file

		:type input_f: string
		:param input_f: the path to corpus (text) file

		:type output_f: string
		:param output_f: the path to indexed corpus file

		:type withdict: bool
		:param withdict: if set to True, vocabulary will be loaded from existing file instead.
	'''
	if not withdict:
		logging.info('Building dictionary')
		input = open(input_f, 'r')
		line = input.readline()
		count = {}

		# building a dictionary
		while line != '':
			line = line.strip()
			words = re.split(' ', line)
			for word in words:
				if not count.has_key(word):
					count[word] = 1
				else:
					count[word] += 1

			line = input.readline()

		count_sort = sorted(count.items(), key=lambda x: x[1], reverse = True)
		vocab = ['<s>', 'UNK']
		ivocab = {'UNK': 1, '<s>': 0}
		vnum = 2
		while vnum < num:
			if vnum-2 >= len(count_sort):
				break
			vocab.append(count_sort[vnum - 2][0])
			ivocab[count_sort[vnum - 2][0]] = vnum
			vnum += 1

		with open(vocab_f, 'wb') as f:
			cPickle.dump(vocab, f)
		with open(ivocab_f, 'wb') as f:
			cPickle.dump(ivocab, f)

		input.close()
	else:
		logging.info('Loading dictionary')
		vocab_old = cPickle.load(open(vocab_f, 'rb'))
		ivocab_old = cPickle.load(open(ivocab_f, 'rb'))
		
		if fromgr:
			vocab = [0]*num 
			for key in ivocab_old:
				vocab[key] = ivocab_old[key]
			vocab[0] = '<s>'

			ivocab = vocab_old
			if ivocab.has_key('</s>'):
				ivocab['<s>'] = ivocab['</s>']
				del ivocab['</s>'] 

			with open(vocab_f, 'wb') as f:
				cPickle.dump(vocab, f)
			with open(ivocab_f, 'wb') as f:
				cPickle.dump(ivocab, f)
		else:
			vocab = vocab_old
			ivocab = ivocab_old

	# indexing the corpus
	logging.info('Indexing the corpus')
	input = open(input_f, 'r')
	line = input.readline()
	corpora = []
	num_words = 0
	UNKs = 0
	num_sentences = 0
	while line != '':
		corpora.append([])
		line = line.strip()
		words = line.split(' ')
		for word in words:
			if not ivocab.has_key(word):
				corpora[num_sentences].append(1)
				UNKs += 1
			else:	
				corpora[num_sentences].append(ivocab[word])
			num_words += 1
		num_sentences += 1
		line = input.readline()

	if not withdict:
		logging.info('%d unique words in %d sentences' % (len(count_sort), num_sentences))
		logging.info('Perserving %d words (including the UNK and empty symbols) and covering %2f of the words.' % (len(vocab), 100. * (num_words - UNKs) / num_words))
	if data_corpus == 'cPickle':
		with open(output_f, 'wb') as f:
			cPickle.dump(corpora, f)
	elif data_corpus == 'json':
		with open(output_f, 'wb') as f:
			json.dump(corpora, f)

def shuffle(src, trg, src_shuf, trg_shuf, data_corpus = 'json'):
	'''
		Randomly shuffling the parallel corpus

		:type src: string
		:param src: the path to indexed source corpus file

		:type trg: string
		:param trg: the path to indexed target corpus file

		:type src_shuf: string
		:param src_shuf: the path to shuffled indexed source corpus file

		:type trg_shuf: string
		:param trg_shuf: the path to shuffled indexed target corpus file
	'''

	# source
	logging.info('Loading source corpus')
	if data_corpus == 'cPickle':
		before_source = cPickle.load(open(src, 'rb'))
	elif data_corpus == 'json':
		before_source = json.load(open(src, 'rb'))
	length = len(before_source)
	logging.info('Total number of source sentences: %d' % length)

	idx = range(length)
	random.shuffle(idx)
	logging.info('Shuffling source corpus')
	after_source = []
	for i in range(len(idx)):
		if (i + 1) % 100000 == 0:
			logging.debug(str(i + 1))
		index = idx[i]
		after_source.append(before_source[index])

	logging.info('Dumping shuffled source corpus')

	if data_corpus == 'cPickle':
		with open(src_shuf, 'wb') as f:
			cPickle.dump(after_source, f)
			f.close()
	elif data_corpus == 'json':
		with open(src_shuf, 'wb') as f:
			json.dump(after_source, f)
			f.close()

	del before_source
	del after_source

	# target
	logging.info('Loading target corpus')
	if data_corpus == 'cPickle':
		before_target = cPickle.load(open(trg, 'rb'))
	elif data_corpus == 'json':
		before_target = json.load(open(trg, 'rb'))
	assert length == len(before_target)

	logging.info('Shuffling target corpus')
	after_target = []
	for i in range(len(idx)):
		if (i+1) % 100000 == 0:
			logging.debug(str(i+1))
		index = idx[i]
		after_target.append(before_target[index])

	logging.info('Dumping shuffled target corpus')

	if data_corpus == 'cPickle':
		with open(trg_shuf, 'wb') as f:
			cPickle.dump(after_target, f)
	elif data_corpus == 'json':
		with open(trg_shuf, 'wb') as f:
			json.dump(after_target, f)


def shuffle_mono(src, src_shuf, data_corpus = 'json'):
	'''
		Randomly shuffling the monolingual corpus.

		:type src: string
		:param src: the path to indexed corpus file

		:type src_shuf: string
		:param src_shuf: the path to indexed corpus file
	'''

	if data_corpus == 'cPickle':
		before_source = cPickle.load(open(src, 'rb'))
	elif data_corpus == 'json':
		before_source = json.load(open(src, 'rb'))
	length = len(before_source)

	idx = range(length)
	random.shuffle(idx)
	after_source = []
	for i in range(len(idx)):
		if (i + 1) % 100000 == 0:
			logging.debug(str(i + 1))
		index = idx[i]
		after_source.append(before_source[index])
	
	if data_corpus == 'cPickle':
		with open(src_shuf, 'wb') as f:
			cPickle.dump(after_source, f)
			f.close()
	elif data_corpus == 'json':
		with open(src_shuf, 'wb') as f:
			json.dump(after_source, f)
			f.close()

if __name__ == "__main__":
	args = parser.parse_args()

	#init config
	cfg = config()
	if args.config:
		cfg = update_config(cfg, load_config(open(args.config, 'r').read()))

	logging.info('STEP 1: Processing\n')
	if os.path.exists(cfg['src_shuf']) and os.path.exists(cfg['trg_shuf']):
		logging.info('File exists! Skipped.\n')
	else:
		if not args.nopreprocess:
			logging.info('STEP 1.1: Processing source corpus')
			preprocess(cfg['index_eos_src'], 
					   cfg['vocab_src'], 
					   cfg['ivocab_src'], 
					   cfg['src_text'], 
					   cfg['src'],
					   cfg['data_vocab'],
					   cfg['data_corpus'],
					   args.withdict,
					   args.fromgr)
			logging.info('Done!\n')
			logging.info('STEP 1.2: Processing target corpus')
			preprocess(cfg['index_eos_trg'], 
				   	   cfg['vocab_trg'], 
					   cfg['ivocab_trg'], 
					   cfg['trg_text'], 
					   cfg['trg'],
					   cfg['data_vocab'],
					   cfg['data_corpus'],
					   args.withdict,
					   args.fromgr)
			logging.info('Done!\n')

		if not args.noshuffle:
			logging.info('STEP 1.3: Shuffling')
			shuffle(cfg['src'],
				    cfg['trg'],
				    cfg['src_shuf'],
				    cfg['trg_shuf'],
				    cfg['data_corpus'])
			logging.info('Done!\n')

	if cfg['src_mono_shuf'] != '':
		if os.path.exists(cfg['src_mono_shuf']):
			logging.info('Monolingual source file exists! Skipped.\n')
		else:
			logging.info('STEP 1.4: Processing monolingual source corpus')
			preprocess(cfg['index_eos_src'], 
					   cfg['vocab_src'], 
					   cfg['ivocab_src'], 
					   cfg['src_mono_text'], 
					   cfg['src_mono'],
					   cfg['data_vocab'],
					   cfg['data_corpus'],
					   True,
					   args.fromgr)
			logging.info('Shuffling\n')
			shuffle_mono(cfg['src_mono'],
					     cfg['src_mono_shuf'],
					     cfg['data_corpus'])
			logging.info('Done!\n')
	if cfg['trg_mono_shuf'] != '':
		if os.path.exists(cfg['trg_mono_shuf']):
			logging.info('Monolingual target file exists! Skipped.\n')
		else:
			logging.info('STEP 1.5: Processing monolingual target corpus.')
			preprocess(cfg['index_eos_trg'], 
				 	   cfg['vocab_trg'], 
					   cfg['ivocab_trg'], 
					   cfg['trg_mono_text'], 
					   cfg['trg_mono'],
					   cfg['data_vocab'],
					   cfg['data_corpus'],
					   True,
					   args.fromgr)
			logging.info('Shuffling\n')
			shuffle_mono(cfg['trg_mono'],
					     cfg['trg_mono_shuf'],
					     cfg['data_corpus'])
			logging.info('Done!\n')
		
