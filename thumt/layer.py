import numpy
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import cPickle
import tools

class Layer(object):
	'''
		The parent class of neural network layers.
	'''

	def __init__(self):
		pass

class FeedForwardLayer(Layer):
	'''
		A single-layer feed-forward neural network.

		:type dim_in: int
		:param dim_in: the dimension of input vectors

		:type dim_out: int
		:param dim_out: the dimension of output vectors

		:type active: function
		:param active: the activation function

		:type offset: bool
		:param offset: if true, this layer will contain bias
	'''

	def __init__(self, name, dim_in, dim_out,
					active = tensor.tanh,
					offset = False):
		'''
			Initialize the parameters
		'''

		self.name = name
		self.active = active
		self.offset = offset
		self.W = tools.init_weight((dim_in, dim_out), name + '_W')
		self.params = [self.W]
		if offset:
			self.b = tools.init_bias((dim_out), name + '_b')
			self.params += [self.b]

	def forward(self, state_in):
		'''
			Build the computational graph.

			:type state_in: theano variable
			:param state_in: the input state
		'''
		state_out = tensor.dot(state_in,self.W)
		if self.offset:
			state_out += self.b
		state_out = self.active(state_out)
		return state_out

class LookupTable(Layer):
	'''
		A lookup table layer which reserves the word embeddings.

		:type num: int
		:param num: the number of words

		:type dim_embed: int
		:param dim_embed: the dimension of the word embedding

		:type offset: bool
		:param offset: if true, this layer will contain bias
	'''

	def __init__(self, name, num, dim_embed, offset = True):
		'''
			Initialize the parameters of the layer
		'''
		self.name = name
		self.emb = tools.init_weight((num, dim_embed), name + '_emb')
		self.offset = offset
		self.params = [self.emb]
		if offset:
			self.b = tools.init_bias((dim_embed), name + '_b')
			self.params += [self.b]

	def forward(self, index):
		'''
			Build the computational graph.

			:type index: theano variable
			:param index: the input state (word indice)
		'''
		if self.offset:
			return self.emb[index] + self.b
		else:
			return self.emb[index]

class GatedRecurrentLayer(Layer):
	'''
		The gated recurrent layer used to encode the source sentences.

		:type dim_in: int
		:param dim_in: the number of input units

		:type dim: int
		:param dim: the number of hidden state units

		:type active: function
		:param active: the activation function

		:type verbose: bool
		:param verbose: only set to True on visualization
	'''

	def __init__(self, name, dim_in, dim,
					active = tensor.tanh,
					verbose = False):
		self.name = name
		self.active = active
		self.dim = dim
		self.verbose = verbose
		self.input_emb = tools.init_weight((dim_in, dim), name + '_inputemb')
		self.gate_emb = tools.init_weight((dim_in, dim), name + '_gateemb')
		self.reset_emb = tools.init_weight((dim_in, dim), name + '_resetemb')
		self.input_hidden = tools.init_weight((dim, dim), name + '_inputhidden')
		self.gate_hidden = tools.init_weight((dim, dim), name + '_gatehidden')
		self.reset_hidden = tools.init_weight((dim, dim), name + '_resethidden')
		self.params = [self.input_emb, self.gate_emb, self.reset_emb,
						self.input_hidden, self.gate_hidden, self.reset_hidden]
		self.input_emb_offset = tools.init_bias((dim), name + '_inputoffset')
		self.params += [self.input_emb_offset]
		
	def forward_step(self, state_before, state_in, gate_in, reset_in, mask = None):
		'''
			Build the one-step computational graph which computes the next hidden state.

			:type state_before: theano variable
			:param state_before: The previous hidden state

			:type state_in: theano variable
			:param state_in: the input state

			:type gate_in: theano variable
			:param gate_in: the input to update gate
			
			:type reset_in: theano variable
			:param reset_in: the input to reset gate

			:type mask: theano variable
			:param mask: indicate the length of each sequence in one batch
		'''
		gate = tensor.nnet.sigmoid(gate_in + tensor.dot(state_before, self.gate_hidden))
		reset = tensor.nnet.sigmoid(reset_in + tensor.dot(state_before, self.reset_hidden))
		reseted = reset*state_before
		state = self.active(state_in +
						tensor.dot(reset * state_before, self.input_hidden))
		h = (1 - gate) * state_before + gate * state

		if mask is not None:
			mask = mask.dimshuffle(0, 'x')
			hidden = mask * h + (1 - mask) * state_before
		else:
			hidden = h

		if self.verbose:
			return hidden, gate, reset, state, reseted
		else:
			return hidden

	def forward(self, emb_in, length, state_init = None, batch_size = 1, mask = None):
		'''
			Build the computational graph which computes the hidden states.

			:type emb_in: theano variable
			:param emb_in: the input word embeddings

			:type length: theano variable
			:param length: the length of the input

			:type batch_size: int
			:param batch_size: the batch size

			:type mask: theano variable
			:param mask: indicate the length of each sequence in one batch
		'''

		# init with zero vectors
		state_init = tensor.alloc(numpy.float32(0.), batch_size, self.dim)
		
		# calculate the input vector for inputter, updater and reseter
		state_in = (tensor.dot(emb_in, self.input_emb) + self.input_emb_offset).reshape((length, batch_size, self.dim))
		gate_in = tensor.dot(emb_in, self.gate_emb).reshape((length, batch_size, self.dim))
		reset_in = tensor.dot(emb_in, self.reset_emb).reshape((length, batch_size, self.dim))

		if mask:
			scan_inp = [state_in, gate_in, reset_in, mask]
			scan_func = lambda x, g, r, m, h : self.forward_step(h, x, g, r, m)
		else:
			scan_inp = [state_in, gate_in, reset_in]
			scan_func = lambda x, g, r, h : self.forward_step(h, x, g, r)

		if self.verbose:
			outputs_info = [state_init, None, None, None, None]
		else:
			outputs_info = [state_init]

		hiddens, updates = theano.scan(scan_func,
							sequences = scan_inp,
							outputs_info = outputs_info)
		if self.verbose:
			return hiddens[0], hiddens[1], hiddens[2], hiddens[3], hiddens[4], state_in, gate_in, reset_in
		else:
			return hiddens, state_in, gate_in, reset_in, self.input_emb, self.input_emb_offset

class GatedRecurrentLayer_attention(Layer):
	'''
		The gated recurrent layer with attention mechanism, used as decoder.
	
		:type dim_in: int
		:param dim_in: the number of input units

		:type dim_c: int
		:param dim_c: the number of context units

		:type dim: int
		:param dim: the number of hidden units

		:type dim_class: int
		:param dim_class: the number of target vocabulary

		:type active: function
		:param active: the activation function

		:type maxout: int
		:param maxout: the number of maxout parts

		:type verbose: bool
		:param verbose: only set to True on visualization
	'''

	def __init__(self, name, dim_in, dim_c, dim, dim_class, 
					active = tensor.tanh,
					maxout = 2,
					verbose = False):
		'''
			Initialize the parameters of the layer
		'''
		self.name = name
		self.active = active
		self.dim = dim
		self.maxout = maxout
		self.verbose = verbose
		self.readout_emb = tools.init_weight((dim_in, dim), name + '_readoutemb')
		self.input_emb = tools.init_weight((dim_in, dim), name + '_inputemb')
		self.gate_emb = tools.init_weight((dim_in, dim), name + '_gateemb')
		self.reset_emb = tools.init_weight((dim_in, dim), name + '_resetemb')
		self.readout_context = tools.init_weight((dim_c, dim), name + '_readoutcontext')
		self.input_context = tools.init_weight((dim_c, dim), name + '_inputcontext')
		self.gate_context = tools.init_weight((dim_c, dim), name + '_gatecontext')
		self.reset_context = tools.init_weight((dim_c, dim), name + '_resetcontext')
		self.readout_hidden = tools.init_weight((dim, dim), name + '_readouthidden')
		self.input_hidden = tools.init_weight((dim, dim), name + '_inputhidden')
		self.gate_hidden = tools.init_weight((dim, dim), name + '_gatehidden')
		self.reset_hidden = tools.init_weight((dim, dim), name + '_resethidden')
		self.att_hidden = tools.init_weight((dim, dim), name + '_atthidden')
		self.att_context = tools.init_weight((dim_c, dim), name + '_attcontext')
		self.att = tools.init_weight((dim, 1), name + '_att')
		self.params = [self.input_emb, self.gate_emb, self.reset_emb,
						self.input_context, self.gate_context, self.reset_context,
						self.input_hidden, self.gate_hidden, self.reset_hidden,
						self.readout_hidden, self.readout_context, self.readout_emb,
						self.att_hidden, self.att_context, self.att]

		self.probs_emb = tools.init_weight((dim / maxout, dim_in), name+'_probsemb')
		self.probs = tools.init_weight((dim_in, dim_class), name + '_probs')
		self.params += [self.probs_emb, self.probs]
			
		self.input_emb_offset = tools.init_bias((dim), name + '_inputoffset')
		self.readout_offset = tools.init_bias((dim), name + '_readoutoffset')
		self.probs_offset = tools.init_bias((dim_class), name + '_probsoffset')
		self.params += [self.input_emb_offset, self.readout_offset, self.probs_offset]
		
	def decode_probs(self, context, state, emb):
		'''
			Get the probability of the next word. Used in beam search and sampling.

			:type context: theano variable
			:param context: the context vectors

			:type state: theano variable
			:param state: the last hidden state

			:type emb: theano variable
			:param emb: the embedding of the last generated word
		'''
		att_c = tools.dot3d(context, self.att_context)
		att_before = tensor.dot(state, self.att_hidden) # size: (batch_size,dim)
		energy = tensor.dot(tensor.tanh(att_c + att_before.dimshuffle('x', 0, 1)), self.att).reshape((context.shape[0], context.shape[1])) # size: (length, batch_size)
		energy = tensor.exp(energy)
		normalizer = energy.sum(axis = 0)
		attention = energy / normalizer # size: (length, batch_size)
		c = (context * attention.dimshuffle(0, 1, 'x')).sum(axis = 0) # size: (batch_size, dim_c)
		
		readout = tensor.dot(emb, self.readout_emb) + \
					tensor.dot(c, self.readout_context) + \
					tensor.dot(state, self.readout_hidden)
		readout += self.readout_offset
		maxout = tools.maxout(readout, self.maxout)

		outenergy = tensor.dot(maxout, self.probs_emb)
		outenergy = tensor.dot(outenergy, self.probs)

		outenergy += self.probs_offset

		return outenergy, c

	def decode_next(self, c, state, emb):
		'''
			Get the next hidden state. Used in beam search and sampling. 

			:type c: theano variable
			:param c: the current context reading

			:type state: theano variable
			:param state: the last hidden state

			:type emb: theano variable
			:param emb: the embedding of the last generated word
		'''
		state_in = tensor.dot(emb, self.input_emb) + self.input_emb_offset
		gate_in = tensor.dot(emb, self.gate_emb)
		reset_in = tensor.dot(emb, self.reset_emb)
		gate_c = tensor.dot(c, self.gate_context)
		gate_before	= tensor.dot(state, self.gate_hidden)
		gate = tensor.nnet.sigmoid(gate_in + gate_before + gate_c)
		reset_c = tensor.dot(c, self.reset_context)
		reset_before = tensor.dot(state, self.reset_hidden)
		reset = tensor.nnet.sigmoid(reset_in + reset_before + reset_c)
		newstate = state_in + tensor.dot(c, self.input_context) + \
				tensor.dot(reset * state, self.input_hidden)

		newstate = self.active(newstate)
		newstate = (1 - gate) * state + gate * newstate
		return newstate

	def forward_step(self, state_before, state_in, gate_in, reset_in, context, att_c, mask = None, cmask = None):
		'''
			Build the one-step computational graph which calculates the next hidden state.

			:type state_before: theano variable
			:param state_before: The previous hidden state

			:type state_in: theano variable
			:param state_in: the input state

			:type gate_in: theano variable
			:param gate_in: the input to update gate
			
			:type reset_in: theano variable
			:param reset_in: the input to reset gate

			:type mask: theano variable
			:param mask: indicate the length of each sequence in one batch

			:type cmask: theano variable
			:param cmask: indicate the length of each context sequence in one batch

			:type context: theano variable
			:param context: the context vectors

			:type att_c: theano variable
			:param att_c: the attention vector from context
				
		'''
		# calculate context and attention
		att_before = tensor.dot(state_before, self.att_hidden).dimshuffle('x', 0, 1) # size: (batch_size,dim)
		energy = tensor.dot(tensor.tanh(att_c + att_before), self.att).reshape((context.shape[0], state_before.shape[0])) # size: (length, batch_size)
		energy = tensor.exp(energy)
		if cmask:
			energy *= cmask
		normalizer = energy.sum(axis = 0)
		attention = energy / normalizer # size: (length, batch_size)
		c = (context * attention.dimshuffle(0, 1, 'x')).sum(axis = 0) # size: (batch_size, dim_c)

		# calculate hidden state
		gate_cin = tensor.dot(c, self.gate_context)
		gate_in += gate_cin
		gate_preactive = tensor.dot(state_before, self.gate_hidden) + gate_in
		gate = tensor.nnet.sigmoid(gate_preactive)
		
		reset_cin = tensor.dot(c, self.reset_context)
		reset_in += reset_cin
		reset_preactive = tensor.dot(state_before, self.reset_hidden) + reset_in
		reset = tensor.nnet.sigmoid(reset_preactive)
		
		state_cin = tensor.dot(c, self.input_context)
		state_in += state_cin
		reseted = reset * state_before
		state = tensor.dot(reseted, self.input_hidden) + state_in
		preactive = state
		state = self.active(state)

		h = gate * state + (1 - gate) * state_before

		if mask is not None:
			mask = mask.dimshuffle(0, 'x')
			newstate = mask * h + (1 - mask) * state_before
		else:
			newstate = h

		if self.verbose:
			return newstate, c, attention, gate_cin, gate_preactive, gate, reset_cin, \
					reset_preactive, reset, state_cin, reseted, preactive, state
		else:
			return newstate, c, attention

	def forward(self, emb_in, length, context, state_init, batch_size = 1, mask = None, cmask = None):
		'''
			Build the computational graph which computes the hidden states.

			:type emb_in: theano variable
			:param emb_in: the input word embeddings

			:type length: theano variable
			:param length: the length of the input

			:type context: theano variable
			:param context: the context vectors

			:type state_init: theano variable
			:param state_init: the inital states 

			:type batch_size: int
			:param batch_size: the batch size

			:type mask: theano variable
			:param mask: indicate the length of each sequence in one batch

			:type cmask: theano variable
			:param cmask: indicate the length of each context sequence in one batch
		'''
		
		# calculate the input vector for inputter, updater and reseter
		att_c = tools.dot3d(context, self.att_context) # size: (length, batch_size,dim)
		state_in = (tensor.dot(emb_in, self.input_emb)+self.input_emb_offset).reshape((length, batch_size, self.dim))
		gate_in = tensor.dot(emb_in, self.gate_emb).reshape((length, batch_size, self.dim))
		reset_in = tensor.dot(emb_in, self.reset_emb).reshape((length, batch_size, self.dim))
		
		if mask:
			scan_inp = [state_in, gate_in, reset_in, mask]
			scan_func = lambda x, g, r, m, h, c, attc, cm : self.forward_step(h, x, g, r, c, attc,m, cm)
		else:
			scan_inp = [state_in, gate_in, reset_in]
			scan_func = lambda x, g, r, h, c, attc : self.forward_step(h, x, g, r, c, attc)

		if self.verbose:
			outputs_info=[state_init, None, None, None, None, None, None, None, None, None, None, None, None]
		else:
			outputs_info=[state_init, None, None]

		# calculate hidden states
		hiddens, updates = theano.scan(scan_func,
							sequences = scan_inp,
							outputs_info = outputs_info,
							non_sequences = [context, att_c, cmask],
							n_steps = length)
		c = hiddens[1]
		attentions = hiddens[2]

		# Add the initial state and discard the last hidden state
		state_before = tensor.concatenate((state_init.reshape((1, state_init.shape[0], state_init.shape[1]))
											, hiddens[0][:-1]))

		state_in_prev = tensor.dot(emb_in, self.readout_emb).reshape((length, batch_size, self.dim))
		
		# calculate the energy for each word
		readout_c = tensor.dot(c, self.readout_context)
		readout_h = tensor.dot(state_before, self.readout_hidden)
		readout_h += self.readout_offset
		state_in_prev = tools.shift_one(state_in_prev)
		readout = readout_c + readout_h + state_in_prev
		readout = readout.reshape((readout.shape[0] * readout.shape[1], readout.shape[2]))
		maxout = tools.maxout(readout, self.maxout)

		outenergy = tensor.dot(maxout, self.probs_emb)
		outenergy_1 = outenergy
		outenergy = tensor.dot(outenergy, self.probs)
		outenergy_2 = outenergy

		outenergy += self.probs_offset
		if self.verbose:
			return hiddens, outenergy, state_in, gate_in, reset_in, state_in_prev, readout, maxout, outenergy_1, outenergy_2
		else:
			return hiddens, outenergy, attentions

class LayerFactory(object):
	'''
		The factory to build and monitor all neural network layers.
	'''

	def __init__(self):
		self.layers = []
		self.params = []

	def createLookupTable(self, name, num, dim_embed, **args):
		newLayer = LookupTable(name, num, dim_embed, **args)
		self.layers += [newLayer]
		self.params += newLayer.params
		return newLayer

	def createFeedForwardLayer(self, name, dim_in, dim_out, **args):
		newLayer = FeedForwardLayer(name, dim_in, dim_out, **args)
		self.layers += [newLayer]
		self.params += newLayer.params
		return newLayer

	def createGRU(self, name, dim_in, dim, **args):
		newLayer = GatedRecurrentLayer(name, dim_in, dim, **args)
		self.layers += [newLayer]
		self.params += newLayer.params
		return newLayer

	def createGRU_context(self, name, dim_in, dim_c, dim, **args):
		newLayer = GatedRecurrentLayer_context(name, dim_in, dim_c, dim, **args)
		self.layers += [newLayer]
		self.params += newLayer.params
		return newLayer

	def createGRU_attention(self, name, dim_in, dim_c, dim, dim_class, **args):
		newLayer = GatedRecurrentLayer_attention(name, dim_in, dim_c, dim, dim_class, **args)
		self.layers += [newLayer]
		self.params += newLayer.params
		return newLayer

		
	
