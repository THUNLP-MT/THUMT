import numpy
import theano
import theano.tensor as tensor
import cPickle
import json
import datetime
import logging

def getbatch(lx, ly, config):
	'''
		Get a batch for training.

		:type lx: numpy array
		:param lx: 2-D numpy arrays, each row contains an indexed source sentence

		:type ly: numpy array
		:param ly: 2-D numpy arrays, each row contains an indexed target sentence

		:type config: dict
		:param config: the configuration
	'''
	assert len(lx) == len(ly)
	# get the length of longest sentence in one batch
	xlen = min(max([len(i) for i in lx]), config['maxlength']) + 1
	ylen = min(max([len(i) for i in ly]), config['maxlength']) + 1

	# filter sentences that are too long.
	# Although sentences have been filtered in data preparation, 
	# we filter again for robusty.
	tx = []
	ty = []
	for i in range(len(lx)):
		if len(lx[i]) <= config['maxlength'] and len(ly[i]) <= config['maxlength']:
			tx.append(numpy.concatenate((lx[i], [config['index_eos_src']])))
			ty.append(numpy.concatenate((ly[i], [config['index_eos_trg']])))
	assert len(tx) == len(ty)
	if len(tx) > 0:
		pass 
	else:
		xlen = 0
		ylen = 0

	# prepare the masks that indicate the length of sentences in one batch
	x = config['index_eos_src'] * numpy.ones((xlen, len(tx)), dtype = 'int64')
	y = config['index_eos_trg'] * numpy.ones((ylen, len(ty)), dtype = 'int64')
	xmask = numpy.zeros((xlen, len(tx)), dtype = 'float32')
	ymask = numpy.zeros((ylen, len(ty)), dtype = 'float32')
	for i in range(len(tx)):
		x[:len(tx[i]),i] = tx[i]
		xmask[:len(tx[i]),i] = 1.
		y[:len(ty[i]),i] = ty[i]
		ymask[:len(ty[i]),i] = 1.

	return x, xmask, y, ymask

class DataCollection(object):
	'''
		The data manager. It also reserve the training status.

		:type config: dict
		:param config: the configuration

		:type train: bool 
		:param train: Only set to true on training. If true, the vocabulary and corpus will be loaded, and the training status will be recorded.
	'''

	def __init__(self, config, train = True):
		self.config = config
		if not train:
			return
		self.load_vocab()
		self.inited = False
		self.inited_mono = False
		self.peeked_batch = None
		self.peeked_batch_mono = None
		self.batch_id = 0
		self.next_offset = 0
		self.valid_result = {}
		self.valid_time = {}
		self.num_iter = 0
		self.time = 0.
		self.updatetime = 0.
		self.train_cost = []
		self.status_variable = ['next_offset', 'num_iter', 'time', 'updatetime', 'valid_result', 'valid_time', 'train_cost']
		self.load_data()
		if self.config['semi_learning']:
			self.next_offset_source = 0
			self.next_offset_target = 0
			self.status_variable += ['next_offset_source', 'next_offset_target']
			self.load_data_mono()

	def next(self):
		'''
			Get the next batch of training corpus

			:returns: x, y are 2-D numpy arrays, each row contains an indexed source/target sentence
		'''
		# read several batches and sort them by length for training efficiency
		if not self.inited or self.batch_id == self.config['sort_batches']:
			self.batch_id = 0
			self.inited = True
			startid = self.next_offset
			endid = self.next_offset + self.config['batchsize'] * self.config['sort_batches']
			self.peeked_batch = []
			while endid >= self.num_sentences:
				cx = self.source[startid : self.num_sentences]
				cy = self.target[startid : self.num_sentences]
				self.peeked_batch += [[x, y] for x,y in zip(cx, cy)]
				endid -= self.num_sentences
				startid = 0
			self.next_offset = endid
			cx = self.source[startid : endid]
			cy = self.target[startid : endid]
			if startid < endid:
				self.peeked_batch += [[x, y] for x, y in zip(cx, cy)]
			self.peeked_batch = sorted(self.peeked_batch, key = lambda x : max(len(x[0]), len(x[1])))
		# return a batch of sentences
		x = numpy.asarray(numpy.asarray(self.peeked_batch[self.batch_id * self.config['batchsize'] : (self.batch_id + 1) * self.config['batchsize']])[:, 0])
		y = numpy.asarray(numpy.asarray(self.peeked_batch[self.batch_id * self.config['batchsize'] : (self.batch_id + 1) * self.config['batchsize']])[:, 1])
		self.batch_id += 1
		
		return x, y
	
	def next_mono(self):
		'''
			Get the next batch of monolingual training corpus. Only used in semi-supervised training.

			:returns: x, y are 2-D numpy arrays, each row contains an indexed source/target sentence
		'''
		if not self.inited_mono or self.batch_id_mono == self.config['sort_batches']:
			self.batch_id_mono = 0
			self.inited = True
			startid_src = self.next_offset_source
			startid_trg = self.next_offset_target
			endid_src = self.next_offset_source + self.config['batchsize'] * self.config['sort_batches']
			endid_trg = self.next_offset_target + self.config['batchsize'] * self.config['sort_batches']
			self.peeked_batch_mono = []
			cx = []
			cy = []
			while endid_src >= self.num_sentences_mono_source:
				cx += self.source_mono[startid : self.num_sentences_mono_source]
				endid_src -= self.num_mono_sentences_mono_source
				startid_src = 0
			self.next_offset_source = endid_src
			while endid_trg >= self.num_sentences_mono_target:
				cy += self.target_mono[startid : self.num_sentences_mono_target]
				endid_trg -= self.num_sentences_mono_target
				startid_trg = 0
			self.next_offset_target = endid_trg
			if startid_src < endid_src:
				cx += self.source_mono[startid_src : endid_src]
			if startid_trg < endid_trg:
				cy += self.target_mono[startid_trg : endid_trg]
			self.peeked_batch_mono = [[x, y] for x, y in zip(cx, cy)]
			self.peeked_batch_mono = sorted(self.peeked_batch_mono, key = lambda x : max(len(x[0]), len(x[1])))
		x = numpy.asarray(numpy.asarray(self.peeked_batch_mono[self.batch_id_mono * self.config['batchsize'] : (self.batch_id_mono + 1) * self.config['batchsize']])[:, 0])
		y = numpy.asarray(numpy.asarray(self.peeked_batch_mono[self.batch_id_mono * self.config['batchsize'] : (self.batch_id_mono + 1) * self.config['batchsize']])[:, 1])
		self.batch_id_mono += 1

		return x, y

	def load_data(self):
		'''
			Load training corpus.
		'''
		# load corpus from file
		if self.config['data_corpus'] == 'cPickle':
			self.source_old = cPickle.load(open(self.config['src_shuf'], 'rb')) 
			self.target_old = cPickle.load(open(self.config['trg_shuf'], 'rb')) 
		elif self.config['data_corpus'] == 'json':
			self.source_old = json.load(open(self.config['src_shuf'], 'rb'))
			self.target_old = json.load(open(self.config['trg_shuf'], 'rb'))
		assert len(self.target_old) == len(self.source_old)
		logging.info('total %d sentences' % len(self.source_old))
	
		# filter sentences that are too long
		self.source = []
		self.target = []
		num = 0
		while num < len(self.source_old):
			if len(self.source_old[num]) <= self.config['maxlength'] and len(self.target_old[num]) <= self.config['maxlength']:
				self.source.append(self.source_old[num])
				self.target.append(self.target_old[num])
			num += 1
			if num % 100000 == 0:
				logging.debug(str(num))
		assert len(self.target) == len(self.source)
		logging.info('Discarding long sentences. %d sentences left.' % len(self.source))
		self.num_sentences = len(self.source)
	
	def load_data_mono(self):
		'''
			Load monolingual training courpus. Only used in semi-supervised training.
		'''
		if self.config['src_mono_shuf']:
			logging.info('Loading monolingual source corpus.')
			self.source_mono = json.load(open(self.config['src_mono_shuf'], 'rb'))
			num = 0
			while num < len(self.source_mono):
				if len(self.source_mono[num]) > self.config['maxlength']:
					del self.source_mono[num]
					num -= 1
				num += 1
				if num % 100000 == 0:
					logging.debug(str(num))
			logging.info('%d monolingual source sentences' % len(self.source_mono))
			self.num_sentences_mono_source = len(self.source_mono)

		if self.config['trg_mono_shuf']:
			logging.info('Loading monolingual target corpus.')
			self.target_mono = json.load(open(self.config['trg_mono_shuf'], 'rb'))
			num = 0
			while num < len(self.target_mono):
				if len(self.target_mono[num]) > self.config['maxlength']:
					del self.target_mono[num]
					num -= 1
				num += 1
				if num % 100000 == 0:
					logging.debug(str(num))
			logging.info('%d monolingual target sentences' % len(self.target_mono))
			self.num_sentences_mono_target = len(self.target_mono)

	def load_vocab(self):
		'''
			Load the vocabulary.
		'''
		self.vocab_src = cPickle.load(open(self.config['vocab_src'], 'rb')) 
		self.vocab_trg = cPickle.load(open(self.config['vocab_trg'], 'rb')) 
		self.ivocab_src = cPickle.load(open(self.config['ivocab_src'], 'rb')) 
		self.ivocab_trg = cPickle.load(open(self.config['ivocab_trg'], 'rb'))
		return

	def index_word_target(self, index):
		'''
			Get the target word given index.

			:type index: int
			:param index: the word index

			:returns: string, the corresponding word.
		'''
		if index == self.config['index_eos_trg']:
			return '<eos>'
		return self.vocab_trg[index]
			
	def print_sentence(self, sentence, vocab, index_eos):
		'''
			get the text form of a sentence represented by an index vector.

			:type sentence: numpy array
			:param sentence: indexed sentence. size:(length, 1)

			:type vocab: list
			:param vocab: vocabulary

			:type index_eos: int
			:param index_eos: the index of the end-of-sentence symbol

			:returns: string, the text form of the sentence
		'''
		result = []
		for pos in range(sentence.shape[0]):
			word = sentence[pos]
			if len(word.shape) != 0:
				word = word[0]
			if word == index_eos:
				break
			else:
				if word < len(vocab):
					result.append(vocab[word])
				else:
					result.append('UNK')
		return ' '.join(result)

	def print_source(self, sentence):
		'''
			Print a source sentence represented by an index vector.

			:type sentence: numpy array
			:param sentence: indexed sentence. size:(length, 1)

			:returns: string, the text form of the source sentence
		'''
		return self.print_sentence(sentence, self.vocab_src, self.config['index_eos_src'])

	def print_target(self, sentence):
		'''
			Print a target sentence represented by an index vector.

			:type sentence: numpy array
			:param sentence: indexed sentence. size:(length, 1)

			:returns: string, the text form of the target sentence
		'''
		return self.print_sentence(sentence, self.vocab_trg, self.config['index_eos_trg'])

	def toindex(self, sentence, ivocab, index_unk, index_eos):
		'''
			Transform a sentence text to indexed sentence.

			:type sentence: string
			:param sentence: sentence text

			:type ivocab: dict
			:param ivocab: the vocabulary to index 

			:type indexed_unk: int
			:param indexed_unk: the index of unknown word symbol

			:type index_eos: int
			:param index_eos: the index of end-of-sentence symbol

			:returns: numpy array, the indexed sentence
		'''
		result = []
		for word in sentence:
			if ivocab.has_key(word):
				result.append(ivocab[word])
			else:
				result.append(index_unk)
		result.append(index_eos)
		return numpy.asarray(result, dtype = 'int64').reshape((len(result), 1))

	def toindex_source(self, sentence):
		'''
			Transform a source language word list to index list.

			:type sentence: string
			:param sentence: sentence text

			:returns: numpy array, the indexed source sentence
		'''
		return self.toindex(sentence, self.ivocab_src, self.config['index_unk_src'], self.config['index_eos_src'])

	def toindex_target(self, sentence):
		'''
			Transform a target language word list to index list.

			:type sentence: string
			:param sentence: sentence text

			:returns: numpy array, the indexed target sentence
		'''
		return self.toindex(sentence, self.ivocab_trg, self.config['index_unk_trg'], self.config['index_eos_trg'])

	def save_status(self, path):
		'''
			Save the training status to file.

			:type path: string
			:param path: the path to a file
		'''
		status = {}
		for st in self.status_variable:
			exec('status["' + st + '"] = self.' + st)
		with open(path, 'wb') as f:
			cPickle.dump(status, f)

	def load_status(self, path):
		'''
			Load the training status from file.

			:type path: string
			:param path: the path to a file
		'''
		try:
			status = cPickle.load(open(path, 'rb'))
			for st in self.status_variable:
				exec('self.' + st + ' = status["' + st + '"]')
		except:
			logging.info('No status file. Starting from scratch.')

	def last_improved(self, last = False):
		'''
			:type last: bool
			:param last: if True, considering getting the same result as improved. And vice versa.

			:returns: int. The number of iteration passed after the latest improvement
		'''
		recent = -1
		recent_iter = -1
		best = -1
		best_iter = -1
		for i in self.valid_result:
			if i > recent_iter:
				recent_iter = i
				recent = self.valid_result[i]
			if self.valid_result[i] > best:
				best_iter = i
				best = self.valid_result[i]
			elif self.valid_result[i] == best:
				if last:
					if i > best_iter:
						best_iter = i
				else:
					if i < best_iter:
						best_iter = i
		return recent_iter - best_iter 
	
	def format_num(self, x):
		result = str(round(x, 2))
		while result[-3] != '.':
			result += '0'
		return result

	def print_log(self):
		validations = sorted(self.valid_result.items(), key = lambda x : x[0])
		result = ' ' * 15 + 'Time' + ' ' * 5 + 'Iteration' + \
					' ' * 8 + 'Cost' + ' ' * 8 + 'BLEU\n' + \
					'-' * 58 + '\n'
		for i in validations:
			time = str(self.valid_time[i[0]])
			iteration = str(i[0])
			cost = sum(self.train_cost[(i[0] - self.config['save_freq']): i[0]]) / self.config['save_freq']
			cost = str(self.format_num(cost))
			bleu = str(self.format_num(i[1]))
			result += time + \
					  ' ' * (14 - len(iteration)) + iteration + \
					  ' ' * (12 - len(cost)) + cost + \
					  ' ' * (12 - len(bleu)) + bleu + '\n'
		return result

	def print_valid(self):
		result = sorted(self.valid_result.items(), key = lambda x : x[0])
		for i in result:
			logging.info('iter %d: %.2f' % (i[0], i[1]))
		return

	def encode_vocab(self, encoding='utf-8'):
		'''
			Change the encoding of the vocabulary.
		'''
		self.vocab_src = [i.encode(encoding) for i in self.vocab_src]
		self.vocab_trg = [i.encode(encoding) for i in self.vocab_trg]
		self.ivocab_src = {i.encode(encoding): self.ivocab_src[i] for i in self.ivocab_src}
		self.ivocab_trg = {i.encode(encoding): self.ivocab_trg[i] for i in self.ivocab_trg}
