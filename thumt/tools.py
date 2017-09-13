import numpy
import theano
import theano.tensor as tensor
import random
import time
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import os
import math

def init_weight(size, name, scale = 0.01, shared = True):
	'''
		Randomly initialize weight parameter in neural networks.

		:type size: tuple or list
		:param size: the size of the parameter

		:type name: string
		:param name: the name of the parameter

		:type scale: float
		:param scale: the scale of the parameter
	'''
	W = scale * numpy.random.randn(*size).astype('float32')
	if shared:
		return theano.shared(W, name = name)
	else:
		return W

def init_bias(size, name, scale = 0.01, shared = True):
	'''
		Initialize bias paramater in neural networks.

		:type size: tuple or list
		:param size: the size of the parameter

		:type name: string
		:param name: the name of the parameter

		:type scale: float
		:param scale: the scale of the parameter
	'''
	b = scale * numpy.ones(size, dtype = 'float32')
	if shared:
		return theano.shared(b, name = name)
	else:
		return b

def init_zeros(size, shared = True):
	'''
		Initialize a zero matrix.

		:type size: tuple or list
		:param size: the size of the matirx
	'''
	t = numpy.zeros(size, dtype = 'float32')
	if shared:
		return theano.shared(t)
	else:
		return t

def dropout(input, drop_rate):
	'''
		The dropout operation

		:type drop_rate: float
		:param drop_rate: the probability of dropout
	'''
	trng = RandomStreams()
	output = input * trng.binomial(input.shape, n = 1, p = 1 - drop_rate, dtype = input.dtype)
	return output

def maxout(input, max_num = 2):
	'''
		Build the computational graph of maxout operation.

		:type input: theano variable
		:param input: the input variable

		:type max_num: int
		:param max_num: the number of maxout parts
	'''
	return input.reshape((input.shape[0], input.shape[1] / max_num, max_num)).max(2)

def padzero(input):
	'''
		Build the computational graph that pads zeros to the left of the input.

		:type input: theano variable
		:param input: the input variable
	'''
	zero = tensor.zeros((1, input.shape[1], input.shape[2]))
	return tensor.concatenate([zero, input])

def softmax3d(input):
	'''
		Build the computational graph of the softmax operation.

		:type input: theano variable
		:param input: the input variable
	'''
	input_reshape = input.reshape((input.shape[0] * input.shape[1], input.shape[2]))
	return tensor.nnet.softmax(input_reshape).reshape((input.shape[0], input.shape[1], input.shape[2]))

def dot3d(input, weight):
	'''
		Build the computational graph of 3-d matrix multiply operation.

		:type input: theano variable
		:param input: the input variable

		:type weight: theano variable
		:param weight: the weight parameter
	'''
	return tensor.dot(input.reshape((input.shape[0] * input.shape[1], input.shape[2])), weight).reshape((input.shape[0], input.shape[1], weight.shape[1]))

def clip(grads, threshold, square = True, params = None):
	'''
		Build the computational graph that clips the gradient if the norm of the gradient exceeds the threshold. 

		:type grads: theano variable
		:param grads: the gradient to be clipped

		:type threshold: float
		:param threshold: the threshold of the norm of the gradient

		:returns: theano variable. The clipped gradient.
	'''
	grads_norm2 = sum(tensor.sum(g ** 2) for g in grads)
	if square:
		grads_norm2 = tensor.sqrt(grads_norm2)
	grads_clip = [tensor.switch(tensor.ge(grads_norm2, threshold),
	  				g / grads_norm2 * threshold, g) 
					for g in grads]
	#deal with nan
	grads_clip = [tensor.switch(tensor.isnan(grads_norm2), 0.01 * p, g) for p, g in zip(params, grads_clip)]
	return grads_clip, grads_norm2

def duplicate(input, times):
	'''
		Broadcast a 2-D tensor given times on axis 1.

		:type input: theano variable
		:param input: the input variable
	'''
	return tensor.alloc(input, times, input.shape[1])

def cut_sentence(sentence, index_eos):
	'''
		Cut the sentence after the end-of-sentence symbol.

		:type sentence: numpy array
		:param sentence: the indexed sentence

		:type index_eos: int
		:param index_eos: the index of end-of-sentence symbol
	'''
	result = []
	for pos in range(sentence.shape[0]):
		word = sentence[pos]
		if word == index_eos:
			result.append(word)
			break
		else:
			result.append(word)
	return numpy.asarray(result, dtype = 'int64')

def softmax(energy, axis = 1):
	'''
		The softmax operation.

		:type energy: theano variable
		:param energy: the energy value for each class

	'''
	exp = numpy.exp(energy - numpy.max(energy, axis = 1).reshape((energy.shape[0], 1)))
	normalizer = numpy.sum(exp, axis = axis)
	return exp / normalizer.reshape(normalizer.shape[0], 1)

def print_time(time):
	'''
		:type time: float
		:param time: the number of seconds

		:returns: string, the text format of time
	'''
	if time < 60:
		return '%.3f sec' % time
	elif time < 3600:
		return '%.3f min' % (time / 60)
	else:
		return '%.3f hr' % (time / 3600)

def shift_one(input):
	'''
		Add a zero vector to the left side of input and remove the rightmost vector.

		:type input: theano variable
		:param input: the input variable
	'''
	result = tensor.zeros_like(input)
	result = tensor.set_subtensor(result[1:], input[:-1])
	return result

def merge_dict(d1, d2):
	'''
		Merge two dicts. The count of each item is the maximum count in two dicts.
	'''
	result = d1
	for key in d2:
		value = d2[key]
		if result.has_key(key):
			result[key] = max(result[key], value)
		else:
			result[key] = value
	return result

def sentence2dict(sentence, n):
	'''
		Count the number of n-grams in a sentence.

		:type sentence: string
		:param sentence: sentence text

		:type n: int 
		:param n: maximum length of counted n-grams
	'''
	words = sentence.split(' ')
	result = {}
	for n in range(1, n + 1):
		for pos in range(len(words) - n + 1):
			gram = ' '.join(words[pos : pos + n])
			if result.has_key(gram):
				result[gram] += 1
			else:
				result[gram] = 1
	return result

def bleu(hypo_c, refs_c, n):
	'''
		Calculate BLEU score given translation and references.

		:type hypo_c: string
		:param hypo_c: the translations

		:type refs_c: list
		:param refs_c: the list of references

		:type n: int
		:param n: maximum length of counted n-grams
	'''
	correctgram_count = [0] * n
	ngram_count = [0] * n
	hypo_sen = hypo_c.split('\n')
	refs_sen = [refs_c[i].split('\n') for i in range(len(refs_c))]
	hypo_length = 0
	ref_length = 0

	for num in range(len(hypo_sen)):
		hypo = hypo_sen[num]
		h_length = len(hypo.split(' '))
		hypo_length += h_length

		refs = [refs_sen[i][num] for i in range(len(refs_c))]
		ref_lengths = sorted([len(refs[i].split(' ')) for i in range(len(refs))])
		ref_distances = [abs(r - h_length) for r in ref_lengths]

		ref_length += ref_lengths[numpy.argmin(ref_distances)]
		refs_dict = {}
		for i in range(len(refs)):
			ref = refs[i]
			ref_dict = sentence2dict(ref, n)
			refs_dict = merge_dict(refs_dict, ref_dict)

		hypo_dict = sentence2dict(hypo, n)

		for key in hypo_dict:
			value = hypo_dict[key]
			length = len(key.split(' '))
			ngram_count[length - 1] += value
			if refs_dict.has_key(key):
				correctgram_count[length - 1] += min(value, refs_dict[key])

	result = 0.
	bleu_n = [0.] * n
	if correctgram_count[0] == 0:
		return 0.
	for i in range(n):
		if correctgram_count[i] == 0:
			correctgram_count[i] += 1
			ngram_count[i] += 1
		bleu_n[i] = correctgram_count[i] * 1. / ngram_count[i]
		result += math.log(bleu_n[i]) / n
	bp = 1
	if hypo_length < ref_length:
		bp = math.exp(1 - ref_length * 1.0 / hypo_length)
	return bp * math.exp(result)

def bleu_file(hypo, refs):
	'''
		Calculate the BLEU score given translation files and reference files.

		:type hypo: string
		:param hypo: the path to translation file

		:type refs: list
		:param refs: the list of path to reference files
	'''
	hypo = open(hypo, 'r').read()
	refs = [open(ref, 'r').read() for ref in refs]
	return bleu(hypo, refs, 4)

def get_ref_files(ref):
	'''
		Get the list of reference files by prefix.
		Suppose nist02.en0, nist02.en1, nist02.en2, nist02.en3 are references and nist02.en does not exist,
		then get_ref_files("nist02.en") = ["nist02.en0", "nist02.en1", "nist02.en2", "nist02.en3"]

		:type ref: string
		:param ref: the prefix of reference files
	'''
	if os.path.exists(ref):
		return [ref]
	else:
		ref_num = 0
		result = []
		while os.path.exists(ref + str(ref_num)):
			result.append(ref + str(ref_num))
			ref_num += 1
		return result
