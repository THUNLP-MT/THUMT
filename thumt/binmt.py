import numpy
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import tools
from layer import LayerFactory

import traceback
import cPickle
from nmt import RNNsearch
from nmt import model
import copy
import logging

class BiRNNsearch(model):
	'''
		The bidirectional RNNsearch model used for semi-supervised training.
	'''

	def __init__(self, config):
		self.config = config
		model.__init__(self)

	def build(self):
		'''
			Building the computational graph.
		'''
		# building forward NMT
		logging.info("Building forward NMT")
		self.fwd_nmt = RNNsearch(self.config, '')
		self.fwd_nmt.build()

		# building backward NMT
		logging.info("Building backward NMT")
		config = copy.deepcopy(self.config)
		config['index_unk_src'], config['index_unk_trg'] = config['index_unk_trg'], config['index_unk_src']
		config['index_eos_src'], config['index_eos_trg'] = config['index_eos_trg'], config['index_eos_src']
		config['num_vocab_src'], config['num_vocab_trg'] = config['num_vocab_trg'], config['num_vocab_src']
		self.bwd_nmt = RNNsearch(config, 'inv_')
		self.bwd_nmt.build()
		
		# merging parameters and objectives
		self.creater = LayerFactory()
		self.creater.params = self.fwd_nmt.creater.params + self.bwd_nmt.creater.params
		self.creater.layers = self.fwd_nmt.creater.layers + self.bwd_nmt.creater.layers
		cost0 = self.fwd_nmt.cost_per_sample
		cost1 = self.bwd_nmt.cost_per_sample
		valid = tensor.vector('valid', dtype = 'float32')
		self.inputs = self.fwd_nmt.inputs + self.bwd_nmt.inputs + [valid,] 
		self.get_addition_grads(cost0, cost1, valid)

	def get_addition_grads(self, cost0, cost1, now_s):
		'''
			Updating the total cost of bidirectional NMT.
		'''
		bs = self.config['batchsize']
		sample_n = self.config['sample_num']
		sum_cost_0 = cost0[bs : bs + bs * sample_n] * \
		             self.config['auto_lambda_1'] + \
					 cost1[bs : bs + bs * sample_n] * \
					 self.config['auto_lambda_2']
		sum_cost_1 = cost0[bs + bs * sample_n :] * \
		             self.config['auto_lambda_2'] + \
					 cost1[bs + bs * sample_n :] * \
					 self.config['auto_lambda_1']
		sum_cost = tensor.concatenate([sum_cost_0, sum_cost_1], axis = 0)
		now_re_cost_mask = now_s[bs:]
		re_cost = sum_cost * now_re_cost_mask
		final_re_cost = re_cost
		print "Caculating gradients in bidirectional NMT"
		self.cost = cost0[:bs].sum() + cost1[:bs].sum() + final_re_cost.sum()
		coff_array = []
		for i in range(bs * 2):
			coff = -1 * sum_cost[sample_n * i : sample_n * (i + 1)]
			tmp_mask = now_re_cost_mask[sample_n * i : sample_n * (i + 1)]
			coff = coff + tensor.cast(tensor.eq(tmp_mask,0), dtype = 'float32') * (numpy.float32(-1000000))
			coff = coff - coff.max()
			coff = tensor.exp(coff)
			coff = cof / coff.sum()
			coff = coff * tmp_mask
			coff_array.append(coff)
		coff = tensor.concatenate(coff_array, axis = 0)
		coff = coff * numpy.float32(self.config['reconstruct_lambda'])
		grads = tensor.grad(self.cost, self.creater.params, known_grads = {re_cost: coff})
		self.grads = grads

	def is_valid(self, sents, eos, unk):
		'''
			Validating the sentences.

			:type sents: theano variable
			:param sents: the indexed sentences

			:type eos: int
			:param eos: the index of end-of-sentence symbol
		'''
		for s in sents:
			if s == eos:
				return True
		return False

	def get_inputs_batch(self, xp, yp, xm, ym):
		'''
			Getting a batch for semi-supervised training.

			:type xp: numpy array
			:param xp: the indexed source sentences in parallel corpus

			:type yp: numpy array
			:param yp: the indexed target sentences in parallel corpus

			:type xm: numpy array
			:param xm: the indexed source sentences in monolingual corpus

			:type ym: numpy array
			:param ym: the indexed target sentences in monolingual corpus
		'''
		# preparation
		null_x = self.config['index_eos_src']
		null_y = self.config['index_eos_trg']
		unk_x = self.config['index_unk_src']
		unk_y = self.config['index_unk_trg']
		sample_n = self.config['sample_num']
		bs = self.config['batchsize']
		x = []
		y = []
		valid = []
		trans_xs = []
		trans_ys = []
 
		for i in range(bs):
			trans_x = self.fwd_nmt.translate(tools.cut_sentence(xm[:, i], null_x), 
					sample_n, return_array = True)
			trans_y = self.bwd_nmt.translate(tools.cut_sentence(ym[:, i], null_y), 
					sample_n, return_array = True)
			while len(trans_x) < sample_n:
				trans_x.append([unk_y])
			while len(trans_y) < sample_n:
				trans_y.append([unk_x])
			for xx in trans_x:
				trans_xs.append(xx)
			for yy in trans_y:
				trans_ys.append(yy)
	
		for i in range(bs):
			indx = numpy.where(xp[:, i] == null_x)[0][0]
			x.append(xp[: indx + 1, i])
			indx = numpy.where(yp[:, i] == null_y)[0][0]
			y.append(yp[: indx + 1, i])
			valid.append(1)

		for i in range(bs):
			now_x = xm[:numpy.where(xm[:, i] == null_x)[0][0] + 1, i]
			for j in range(sample_n):
				x.append(now_x)
				now_len = min(80, len(trans_xs[i * sample_n + j]))
				y.append(trans_xs[i * sample_n + j][: now_len])
				valid.append(self.is_valid(trans_xs[i * sample_n + j], null_y, unk_y))

		for i in range(bs):
			now_y = ym[: numpy.where(ym[:, i] == null_y)[0][0] + 1, i]
			for j in range(sample_n):
				y.append(now_y)
				now_len = min(80, len(trans_ys[i * sample_n + j]))
				x.append(trans_ys[i * sample_n + j][: now_len])
				valid.append(self.is_valid(trans_ys[i * sample_n + j], null_x, unk_x))

		max_x = max([len(xx) for xx in x ])
		max_y = max([len(yy) for yy in y ])
		valid = numpy.asarray(valid, dtype = 'float32')
		new_x = numpy.zeros((max_x, len(x)), dtype = 'int64')
		new_y = numpy.zeros((max_y, len(y)), dtype = 'int64')
		new_x_mask = numpy.zeros((max_x, len(x)), dtype = 'float32')
		new_y_mask = numpy.zeros((max_y, len(y)), dtype = 'float32')

		for i in range(len(x)):
			for j in range(len(x[i])):
				new_x[j][i] = x[i][j]
				new_x_mask[j][i] = 1.

		for i in range(len(y)):
			for j in range(len(y[i])):
				new_y[j][i] = y[i][j];
				new_y_mask[j][i] = 1.
		print new_x.shape, new_x_mask.shape, new_y.shape, new_y_mask.shape, valid.shape
		return new_x, new_x_mask, new_y, new_y_mask, valid

	def sample(self, x, length, n_samples = 1):
		'''
			Sampling with source-to-target network.

			:type x: numpy array
			:param x: the indexed source sentence

			:type length: int
			:param length: the length limit of samples

			:type n_samples: int
			:param n_samples: number of samples

			:returns: a numpy array, the indexed sample results
		'''
		return self.fwd_nmt.sample(x, length, n_samples)

	def translate(self, x, beam_size = 10):
		'''
			Beam search with source-to-target network.

			:type x: numpy array
			:param x: the indexed source sentence

			:type beam_size: int
			:param beam_size: beam size

			:returns: a numpy array, the indexed translation result
		'''
		return self.fwd_nmt.translate(x, beam_size)

	def sample_inv(self, x, length, n_samples = 1):
		'''
			Sampling with target-to-source network.

			:type x: numpy array
			:param x: the indexed target sentence

			:type length: int
			:param length: the length limit of samples

			:type n_samples: int
			:param n_samples: number of samples

			:returns: a numpy array, the indexed sample results
		'''
		return self.bwd_nmt.sample(x, length, n_samples)

	def translate_inv(self, x, beam_size = 10):
		'''
			Beam search with target-to-source network.

			:type x: numpy array
			:param x: the indexed target sentence

			:type beam_size: int
			:param beam_size: beam size

			:returns: a numpy array, the indexed translation result
		'''
		return self.bwd_nmt.translate(x, beam_size)
