import numpy
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
#from theano.tensor.shared_randomstreams import RandomStreams
import tools
from layer import LayerFactory

import json
import traceback
import cPickle
import logging

class model(object):
	'''
		The parent class of NMT models
	'''

	def __init__(self):
		pass

	def sample(self, x, length, n_samples = 1):
		'''
			Sample with probability.

			:type x: numpy array
			:param x: the indexed source sentence

			:type length: int
			:param length: the length limit of samples

			:type n_samples: int
			:param n_samples: number of samples

			:returns: a numpy array, the indexed sample results
		'''
		sample, probs = self.get_sample(x.reshape((x.shape[0],1)), length, n_samples)
		return numpy.asarray(sample, dtype = 'int64').transpose(), probs

	def translate(self, x, beam_size = 10, return_array = False):
		'''
			Decode with beam search.

			:type x: numpy array
			:param x: the indexed source sentence

			:type beam_size: int
			:param beam_size: beam size

			:returns: a numpy array, the indexed translation result
		'''
		# initialize variables
		result = [[]]
		loss = [0.]
		result_eos = []
		loss_eos = []
		beam = beam_size

		# get encoder states
		c, state = self.get_context_and_init(x)
		emb_y = numpy.zeros((1, self.config['dim_emb_trg']), dtype = 'float32')

		for l in range(x.shape[0]*3):
			# get word probability
			energy, ctx = self.get_probs(numpy.repeat(c, len(result), axis = 1), state, emb_y)
			probs = tools.softmax(energy)
			losses = -numpy.log(probs)

			# prevent translation to be too short.
			if l < x.shape[0] / 2:
				losses[:, self.config['index_eos_trg']] = numpy.inf
			for i in range(len(loss)):
				losses[i] += loss[i]

			# get the n-best partial translations
			best_index_flatten = numpy.argpartition(losses.flatten(), beam)[:beam]
			best_index = [(index / self.config['num_vocab_trg'], index % self.config['num_vocab_trg']) for index in best_index_flatten]

			# save the partial translations in the beam
			new_ctx = numpy.zeros((beam, 2 * self.config['dim_rec_enc']), dtype = 'float32')
			new_y = []
			new_state = numpy.zeros((beam, self.config['dim_rec_dec']), dtype = 'float32')
			new_result = []
			new_loss = []
			for i in range(beam):
				index = best_index[i]
				new_result.append(result[index[0]] + [index[1]])
				new_loss.append(losses[index[0], index[1]])
				new_ctx[i] = ctx[index[0]]
				new_y.append(index[1])
				new_state[i] = state[index[0]]

			# get the next decoder hidden state
			new_emby = self.get_trg_embedding(numpy.asarray(new_y, dtype = 'int64'))[0]
			new_state = self.get_next(new_ctx, new_state, new_emby)

			# remove finished translation from the beam
			state = []
			emb_y = []
			result = []
			loss = []
			for i in range(beam):
				if new_result[i][-1] == self.config['index_eos_trg']:
					result_eos.append(new_result[i])
					loss_eos.append(new_loss[i])
					beam -= 1
				else:
					result.append(new_result[i])
					loss.append(new_loss[i])
					state.append(new_state[i])
					emb_y.append(new_emby[i])

			if beam <= 0:
				break

			state = numpy.asarray(state, dtype = 'float32')
			emb_y = numpy.asarray(emb_y, dtype = 'float32')

		# only used in semi-supervised training
		if return_array:
			if len(result_eos) > 0:
				return result_eos
			else:
				return [result[-1][:1]]

		if len(result_eos) > 0:
			# return the best translation
			return result_eos[numpy.argmin(loss_eos)]
		elif beam_size > 100:
			# double the beam size on failure
			logging.warning('cannot find translation in beam size %d' % beam_size)
			return []
		else:
			logging.info('cannot find translation in beam size %d, try %d' % (beam_size, beam_size * 2))
			return self.translate(x, beam_size = beam_size * 2)

	def save(self, path, data = None, mapping = None):
		'''
			Save the model in npz format.

			:type path: string
			:param path: the path to a file

			:type data: DataCollection
			:param data: the data manager, will save the vocabulary into the model if set.

			:type mapping: dict
			:param mapping: the mapping file used in UNKreplace, will save it to the model if set
		'''
		values = {}
		for p in self.creater.params:
			values[p.name] = p.get_value()
		values['config'] = json.dumps(self.config)
		if data:
			values['vocab_src'] = json.dumps(data.vocab_src)
			values['ivocab_src'] = json.dumps(data.ivocab_src)
			values['vocab_trg'] = json.dumps(data.vocab_trg)
			values['ivocab_trg'] = json.dumps(data.ivocab_trg)
		if mapping:
			values['mapping'] = json.dumps(mapping)
		numpy.savez(path, **values)

	def load(self, path, decode = False):
		'''
			Load the model from npz format. It will load from the checkpoint model.
			If checkpoint model does not exist, it will initialize a new model (MLE) or load from given model (MRT or semi)

			:type path: string
			:param path: the path to a file

			:type decode: bool
			:param decode: Set to True only on decoding
		'''
		try:
			# load model parameters from file
			values = numpy.load(path)
			for p in self.creater.params:
				if p.name in values:
					if values[p.name].shape != p.get_value().shape:
						logging.warning(p.name + ' needs ' + str(p.get_value().shape) + ', given ' + str(values[p.name].shape) + ' , initializing')
					else:
						p.set_value(values[p.name])
						logging.debug(p.name + ' loaded ' + str(values[p.name].shape))
				else:
					logging.warning('No parameter ' + p.name + ', initializing')
			if decode:
				return values
		except:
			if self.config['MRT'] or self.config['semi_learning']:
				# load from initialization model
				logging.info('Initializing the model from ' + str(self.config['init_model']))
				self.load(self.config['init_model'])
			else:
				logging.info('No model file. Starting from scratch.')

class RNNsearch(model):
	'''
		The attention-based NMT model
	'''

	def __init__(self, config, name = ''):
		self.config = config
		self.name = name
		self.creater = LayerFactory()
		self.trng = RandomStreams(numpy.random.randint(int(10e6)))

	def sampling_step(self, state, prev, context):
		'''
			Build the computational graph which samples the next word.

			:type state: theano variables
			:param state: the previous hidden state

			:type prev: theano variables
			:param prev: the last generated word

			:type context: theano variables
			:param context: the context vectors.
		'''
		emb = self.emb_trg.forward(prev)
		energy, c = self.decoderGRU.decode_probs(context, state, emb)
		probs = tensor.nnet.softmax(energy)
		
		sample = self.trng.multinomial(pvals = probs, dtype = 'int64').argmax(axis = -1)

		newemb = self.emb_trg.forward(sample)
		newstate = self.decoderGRU.decode_next(c, state, newemb)

		return newstate, sample, probs

	def decode_sample(self, state_init, c, length, n_samples):
		'''
			Build the decoder graph for sampling.

			:type state_init: theano variables
			:param state_init: the initial state of decoder

			:type c: theano variables
			:param c: the context vectors

			:type length: int
			:param length: the limitation of sample length

			:type n_samples: int
			:param n_samples: the number of samples
		'''

		state = tensor.repeat(state_init, n_samples, axis = 0)
		sample = tensor.zeros((n_samples,), dtype = 'int64')
		c = tensor.repeat(c, n_samples, axis = 1)

		result, updates = theano.scan(self.sampling_step,
							outputs_info = [state, sample, None],
							non_sequences = [c],
							n_steps = length)
		
		samples = result[1]
		probs = result[2]
		y_idx = tensor.arange(samples.flatten().shape[0]) * self.config['num_vocab_trg'] + samples.flatten()
		probs = probs.flatten()[y_idx]
		probs.reshape(samples.shape)
		return samples, probs, updates

	def build(self, verbose = False):
		'''
			Build the computational graph.

			:type verbose: bool
			:param verbose: only set to True on visualization
		'''
		config = self.config 

		# create layers
		logging.info('Initializing layers')
		self.emb_src = self.creater.createLookupTable(self.name + 'emb_src',
			config['num_vocab_src'], config['dim_emb_src'], offset = True)
		self.emb_trg = self.creater.createLookupTable(self.name + 'emb_trg',
			config['num_vocab_trg'], config['dim_emb_trg'], offset = True)
		self.encoderGRU = self.creater.createGRU(self.name + 'GRU_enc',
			config['dim_emb_src'], config['dim_rec_enc'], verbose = verbose)
		self.encoderGRU_back = self.creater.createGRU(self.name + 'GRU_enc_back',
			config['dim_emb_src'], config['dim_rec_enc'], verbose = verbose)
		self.decoderGRU = self.creater.createGRU_attention(self.name + 'GRU_dec',
			config['dim_emb_trg'], 2 * config['dim_rec_enc'],
			config['dim_rec_dec'], config['num_vocab_trg'], verbose = verbose)
		self.initer = self.creater.createFeedForwardLayer(self.name + 'initer',
			config['dim_rec_enc'], config['dim_rec_dec'], offset = True)

		# create input variables
		self.x = tensor.matrix('x', dtype = 'int64') # size: (length, batchsize)
		self.xmask = tensor.matrix('x_mask', dtype = 'float32') # size: (length, batchsize)
		self.y = tensor.matrix('y', dtype = 'int64') # size: (length, batchsize)
		self.ymask = tensor.matrix('y_mask', dtype = 'float32') # size: (length, batchsize)

		if 'MRT' in config and config['MRT'] is True:
			self.MRTLoss = tensor.vector('MRTLoss')
			self.inputs = [self.x, self.xmask, self.y, self.ymask, self.MRTLoss]
		else:
			self.MRTLoss = None
			self.inputs = [self.x, self.xmask, self.y, self.ymask]


		# create computational graph for training
		logging.info('Building computational graph')
		# ----encoder-----
		emb = self.emb_src.forward(self.x.flatten()) # size: (length, batch_size, dim_emb)
		back_emb = self.emb_src.forward(self.x[::-1].flatten())
		
		self.encode_forward = self.encoderGRU.forward(emb, self.x.shape[0], batch_size = self.x.shape[1], mask = self.xmask) # size: (length, batch_size, dim)
		self.encode_backward = self.encoderGRU_back.forward(back_emb, self.x.shape[0], batch_size = self.x.shape[1], mask = self.xmask[::-1]) # size: (length, batch_size, dim)
		context_forward = self.encode_forward[0]
		context_backward = self.encode_backward[0][::-1]
		self.context = tensor.concatenate((context_forward, context_backward), axis=2) # size: (length, batch_size, 2*dim)

		# ----decoder----
		self.init_c = context_backward[0]
		self.state_init = self.initer.forward(context_backward[0])
		emb = self.emb_trg.forward(self.y.flatten()) # size: (length, batch_size, dim_emb)
		self.decode = self.decoderGRU.forward(emb, self.y.shape[0],
									self.context, state_init = self.state_init, 
									batch_size = self.y.shape[1], mask = self.ymask, 
									cmask = self.xmask) # size: (length, batch_size, dim)
		
		energy = self.decode[1]
		self.attention = self.decode[2]
		self.softmax = tensor.nnet.softmax(energy)
		# compute costs and grads
		y_idx = tensor.arange(self.y.flatten().shape[0]) * self.config['num_vocab_trg'] + self.y.flatten()
		cost = self.softmax.flatten()[y_idx]
		cost = -tensor.log(cost)
		self.cost = cost.reshape((self.y.shape[0], self.y.shape[1])) * self.ymask
		self.cost_per_sample = self.cost.sum(axis = 0)
		if 'MRT' in config and config['MRT'] is True:
			self.cost_per_sample = self.cost.sum(axis = 0)
			tmp = self.cost_per_sample
			tmp *= config['MRT_alpha']
			tmp -= tmp.min()
			tmp = tensor.exp(-tmp)
			tmp /= tmp.sum()
			tmp *= self.MRTLoss
			tmp = -tmp.sum()	
			self.cost = tmp
		else:
			self.cost = self.cost.sum()

		# build sampling graph
		self.x_sample = tensor.matrix('x_sample', dtype = 'int64')
		self.n_samples = tensor.scalar('n_samples', dtype = 'int64')
		self.length_sample = tensor.scalar('length', dtype = 'int64')
		emb_sample = self.emb_src.forward(self.x_sample.flatten()) # (length, batch_size, dim_emb)
		back_emb_sample = self.emb_src.forward(self.x_sample[::-1].flatten())
		encode_forward_sample = self.encoderGRU.forward(emb_sample, self.x_sample.shape[0], batch_size = self.x_sample.shape[1]) # (length, batch_size, dim)
		encode_backward_sample = self.encoderGRU_back.forward(back_emb_sample, self.x_sample.shape[0], batch_size = self.x_sample.shape[1]) # (length, batch_size, dim)
		context_sample = tensor.concatenate((encode_forward_sample[0], encode_backward_sample[0][::-1]), axis = 2) # (length, batch_size, 2*dim)
		state_init_sample =self.initer.forward(encode_backward_sample[0][::-1][0])
		self.state_init_sample = state_init_sample 
		self.context_sample = context_sample
		self.samples, self.probs_sample, self.updates_sample = self.decode_sample(state_init_sample, context_sample, 
													self.length_sample, self.n_samples)

		# parameter for decoding
		self.y_decode = tensor.vector('y_decode', dtype = 'int64')
		self.context_decode = tensor.tensor3('context_decode', dtype = 'float32')
		self.c_decode = tensor.matrix('c_decode', dtype = 'float32')
		self.state_decode = tensor.matrix('state_decode', dtype = 'float32')
		self.emb_decode = tensor.matrix('emb_decode', dtype = 'float32')

	def encode(self, x):
		'''
			Encode source sentence to context vector.
		'''
		if not hasattr(self, "encoder"):
			self.encoder = theano.function(inputs = [self.x,self.xmask],
											outputs = [self.context])
		x = numpy.reshape(x, (x.shape[0], 1))
		xmask = numpy.ones(x.shape, dtype = 'float32')
		return self.encoder(x, xmask)

	def get_trg_embedding(self, y):
		'''
			Get the embedding of target sentence.
		'''
		if not hasattr(self, "get_trg_embeddinger"):
			self.get_trg_embeddinger = theano.function(inputs = [self.y_decode],
											outputs = [self.emb_trg.forward(self.y_decode)])
		return self.get_trg_embeddinger(y)

	def get_init(self, c):
		'''
			Get the initial decoder hidden state with context vector.
		'''
		if not hasattr(self, "get_initer"):
			self.get_initer = theano.function(inputs = [self.context],
											outputs = [self.initer.forward(context_backward[0])])
		return self.get_initer(c)

	def get_context_and_init(self, x):
		'''
			Encode source sentence to context vectors and get the initial decoder hidden state.
		'''
		if not hasattr(self, "get_context_and_initer"):
			self.get_context_and_initer = theano.function(inputs = [self.x,self.xmask],
											outputs = [self.context, self.state_init])
		x = numpy.reshape(x, (x.shape[0], 1))
		xmask = numpy.ones(x.shape, dtype = 'float32')
		return self.get_context_and_initer(x, xmask)

	def get_probs(self, c, state, emb):
		'''
			Get the probability of the next target word.
		'''
		if not hasattr(self, "get_probser"):
			self.get_probser = theano.function(inputs = [self.context_decode, \
					                                     self.state_decode, \
														 self.emb_decode], \
											   outputs = self.decoderGRU.decode_probs(self.context_decode, \
												                                      self.state_decode, \
																					  self.emb_decode))
		return self.get_probser(c, state, emb)

	def get_next(self, c, state, emb):
		'''
			Get the next hidden state.
		'''
		if not hasattr(self, "get_nexter"):
			self.get_nexter = theano.function(inputs = [self.c_decode, \
					                                    self.state_decode, \
														self.emb_decode],
											  outputs = self.decoderGRU.decode_next(self.c_decode, \
												                                    self.state_decode, \
																					self.emb_decode))
		return self.get_nexter(c, state, emb)

	def get_cost(self, x, xmask, y, ymask):
		'''
			Get the negative log-likelihood of parallel sentences.
		'''
		if not hasattr(self, "get_coster"):
			self.get_coster = theano.function(inputs = [self.x, self.xmask, self.y, self.ymask],
											  outputs = [self.cost])
		return self.get_coster(x, xmask, y, ymask)

	def get_sample(self, x, length, n_samples):
		'''
			Get sampling results.
		'''
		if not hasattr(self, "get_sampler"):
			self.get_sampler = theano.function(inputs = [self.x_sample, self.length_sample, self.n_samples],
											   outputs = [self.samples, self.probs_sample],
											   updates = self.updates_sample)
		return self.get_sampler(x, length, n_samples)

	def get_attention(self, x, xmask, y, ymask):
		'''
			Get the attention weight of parallel sentences.
		'''
		if not hasattr(self, "get_attentioner"):
			self.get_attentioner = theano.function(inputs = [self.x, self.xmask, self.y, self.ymask],
												   outputs = [self.attention])
		return self.get_attentioner(x, xmask, y, ymask)


	def get_layer(self, x, xmask, y, ymask):
		'''
			Get the hidden states essential for visualization
		'''
		if not hasattr(self, "get_layerer"):
			self.get_layerer = theano.function(inputs = [self.x, self.xmask, self.y, self.ymask],
										       outputs = self.encode_forward + \
											             self.encode_backward + \
														 tuple(self.decode[0]) + tuple(self.decode[1:]))
			
		layers = self.get_layerer(x, xmask, y, ymask)
		enc_names = ['h', 'gate', 'reset', 'state', 'reseted', 'state_in', 'gate_in', 'reset_in']
		dec_names = ['h', 'c', 'att', 'gate_cin', 'gate_preactive', 'gate', 'reset_cin', 'reset_preactive', 'reset', 'state_cin', 'reseted', 'state_preactive', 'state']
		dec_names += ['outenergy', 'state_in', 'gate_in', 'reset_in', 'state_in_prev', 'readout', 'maxout', 'outenergy_1', 'outenergy_2']
		value_name = ['enc_for_' + name for name in enc_names]
		value_name += ['enc_back_' + name for name in enc_names]
		value_name += ['dec_' + name for name in dec_names]
		result = {}
		for i in range(len(layers)):
			if value_name[i] != '':
				result[value_name[i]] = layers[i]
		return result
