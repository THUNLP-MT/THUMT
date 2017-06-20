import numpy
import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import tools

class optimizer(object):
	'''
		The parent class of SGD algorithms
	'''

	def __init__(self):
		pass

class adadelta(optimizer):

	def __init__(self, config, params):
		self.config = config
		self.params = params
		self.gc = [tools.init_zeros(p.get_value().shape) for p in params]
		self.g2 = [tools.init_zeros(p.get_value().shape) for p in params]
		self.u2 = [tools.init_zeros(p.get_value().shape) for p in params]

	def build(self, cost, inp):
		grads_noclip = tensor.grad(cost, self.params)
		grads, grad_norm = tools.clip(grads_noclip, self.config['clip'], params = self.params)
		
		gc_up = [(gc, gr) for gc, gr in zip(self.gc, grads)]
		g2_up = [(g2, self.config['rho'] * g2 + (1. - self.config['rho']) * (gr ** 2.)) 
						for g2, gr in zip(self.g2, grads)]
		#noclip = theano.function(inp, [cost]+grads_noclip)
		#noupdate_grads = theano.function(inp, [cost, grad_norm])
		update_grads = theano.function(inp, [cost, grad_norm], updates = gc_up + g2_up)

		delta = [tensor.sqrt(u2 + self.config['epsilon']) / tensor.sqrt(g2 + self.config['epsilon']) * gr
				for g2, u2, gr in zip(self.g2, self.u2, self.gc)]
		u2_up = [(u2, self.config['rho'] * u2 + (1. - self.config['rho']) * (d ** 2.))
				for u2, d in zip(self.u2, delta)]
		param_up = [(p, p - d)
					for p, d in zip(self.params, delta)]
		#update_params = theano.function([], [], updates=param_up+u2_up)
		update_params = theano.function([], [], updates = param_up + u2_up)
		return update_grads, update_params

class adam(optimizer):

	def __init__(self, config, params):
		self.config = config
		self.params = params
		self.gc = [tools.init_zeros(p.get_value().shape) for p in params]
		self.m = [tools.init_zeros(p.get_value().shape) for p in params]
		self.v = [tools.init_zeros(p.get_value().shape) for p in params]
		self.beta1t = theano.shared(numpy.float32(config['beta1_adam']))
		self.beta2t = theano.shared(numpy.float32(config['beta2_adam']))

	def build(self, cost, inp):
		grads_noclip = tensor.grad(cost, self.params)
		grads, grad_norm = tools.clip(grads_noclip, self.config['clip'], params=self.params)

		update_ab = [(self.beta1t, self.beta1t * self.config['beta1_adam']),
					(self.beta2t, self.beta2t * self.config['beta2_adam'])]
		update_gc = [(gc, gr) for gc, gr in zip(self.gc, grads)]
		update_gc = [(gc, gr) for gc, gr in zip(self.gc, grads)]
		m_up = [(m, self.config['beta1_adam'] * m + (1. - self.config['beta1_adam']) * gr) for m, gr in zip(self.m, grads)]
		v_up = [(v, self.config['beta2_adam'] * v + (1. - self.config['beta2_adam']) * (gr ** 2)) for v, gr in zip(self.v, grads)]
		update_grads = theano.function(inp, [cost, grad_norm], updates=update_gc + m_up + v_up)
		param_up = [(p, p - self.config['alpha_adam'] * (m / (1. - self.beta1t)) / (tensor.sqrt(v / (1. - self.beta2t)) + self.config['eps_adam'])) for p, m, v in zip(self.params, self.m, self.v)]
		update_params = theano.function([],[], updates = update_ab + param_up)
		return update_grads, update_params

class adam_slowstart(optimizer):
	'''
		Adam with lowered learning rate at the beginning
	'''

	def __init__(self, config, params):
		self.config = config
		self.params = params
		self.gc = [tools.init_zeros(p.get_value().shape) for p in params]
		self.m = [tools.init_zeros(p.get_value().shape) for p in params]
		self.v = [tools.init_zeros(p.get_value().shape) for p in params]
		self.beta1t = theano.shared(numpy.float32(config['beta1_adam']))
		self.beta2t = theano.shared(numpy.float32(config['beta2_adam']))
		self.alphadecayt = theano.shared(numpy.float32(config['alphadecay_adam']))

	def build(self, cost, inp):
		grads_noclip = tensor.grad(cost, self.params)
		grads, grad_norm = tools.clip(grads_noclip, self.config['clip'], params=self.params)

		update_ab = [(self.beta1t, self.beta1t * self.config['beta1_adam']),
					(self.beta2t, self.beta2t * self.config['beta2_adam']),
					(self.alphadecayt, self.alphadecayt * self.config['alphadecay_adam'])]
		update_gc = [(gc, gr) for gc, gr in zip(self.gc, grads)]
		update_gc = [(gc, gr) for gc, gr in zip(self.gc, grads)]
		m_up = [(m, self.config['beta1_adam'] * m + (1. - self.config['beta1_adam']) * gr) for m, gr in zip(self.m, grads)]
		v_up = [(v, self.config['beta2_adam'] * v + (1. - self.config['beta2_adam']) * (gr ** 2)) for v, gr in zip(self.v, grads)]
		update_grads = theano.function(inp, [cost, grad_norm], updates=update_gc + m_up + v_up)
		self.alphat = (1.0 - self.alphadecayt) * self.config['alpha_adam']
		param_up = [(p, p - self.alphat * (m / (1. - self.beta1t)) / (tensor.sqrt(v / (1. - self.beta2t)) + self.config['eps_adam'])) for p, m, v in zip(self.params, self.m, self.v)]
		update_params = theano.function([],[], updates = update_ab + param_up)
		return update_grads, update_params

class SGD(optimizer):
	'''
		SGD with fixed learning rate
	'''

	def __init__(self, config, params):
		self.config = config
		self.params = params
		self.gc = [tools.init_zeros(p.get_value().shape) for p in params]

	def build(self, cost, inp):
		grads_noclip = tensor.grad(cost, self.params)
		grads, grad_norm = tools.clip(grads_noclip, self.config['clip'], square = False, params = self.params)

		gc_up = [(gc, gr) for gc, gr in zip(self.gc, grads)]
		update_grads = theano.function(inp, [cost, tensor.sqrt(grad_norm)], updates = gc_up)

		lr = numpy.float32(self.config['lr'])
		delta = [-lr * gr for gr in self.gc]
		params_up = [(p, p - lr * gr)
					for p, gr in zip(self.params, self.gc)]
		update_params = theano.function([], [], updates = params_up)
		return update_grads, update_params



