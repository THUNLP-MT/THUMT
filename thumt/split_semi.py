import sys
import numpy
import json
from nmt import RNNsearch
from binmt import BiRNNsearch
from config import * 
from data import *
 

if __name__ == "__main__":

	#load bidirectional model
	values = numpy.load(sys.argv[1])
	config = values['config']
	config = json.loads(str(config))
	model = eval(config['model'])(config)
	model.build()
	values = model.load(sys.argv[1], decode = True)

	data = DataCollection(config, train = False)
	data.vocab_src = json.loads(str(values['vocab_src']))
	data.ivocab_src = json.loads(str(values['ivocab_src']))
	data.vocab_trg = json.loads(str(values['vocab_trg']))
	data.ivocab_trg = json.loads(str(values['ivocab_trg']))
	data.encode_vocab()
	
	havemap = False
	try:
		mapping = json.loads(str(values['mapping']))
		mapping = {i.encode('utf-8'): mapping[i].encode('utf-8') for i in mapping}
		havemap = True
	except:
		mapping = None

	# extract source-to-target model
	src2trg = {}
	for key, value in values.iteritems():
		if not "vocab" in key and not "config" in key and not "mapping" in key and not "inv_" in key:
			src2trg[key] = values[key]

	src2trg['config'] = values['config']
	src2trg['vocab_src'] = values['vocab_src']
	src2trg['ivocab_src'] = values['ivocab_src']
	src2trg['vocab_trg'] = values['vocab_trg']
	src2trg['ivocab_trg'] = values['ivocab_trg']
	
	if havemap:
		src2trg['mapping'] = values['mapping']

	numpy.savez(sys.argv[2], **src2trg)

	# extract target-to-source model
	trg2src = {}
	for key, value in values.iteritems():
		if 'inv_' in key:
			trg2src[key[4:]] = values[key]

	trg2src['config'] = json.loads(str(values['config']))
	trg2src['config']['index_unk_src'] = config['index_unk_trg']
	trg2src['config']['index_unk_trg'] = config['index_unk_src']
	trg2src['config']['index_eos_src'] = config['index_eos_trg']
	trg2src['config']['index_eos_trg'] = config['index_eos_src']
	trg2src['config']['num_vocab_src'] = config['num_vocab_trg']
	trg2src['config']['num_vocab_trg'] = config['num_vocab_src']
	trg2src['config'] = json.dumps(trg2src['config'])

	trg2src['vocab_src'] = values['vocab_trg']
	trg2src['ivocab_src'] = values['ivocab_trg']
	trg2src['vocab_trg'] = values['vocab_src']
	trg2src['ivocab_trg'] = values['ivocab_src']

	if havemap:
		trg2src['mapping'] = {}
		mapping = json.loads(str(values['mapping']))
		for key in mapping:
			trg2src['mapping'][mapping[key]] = key
		trg2src['mapping'] = json.dumps(trg2src['mapping'])

	numpy.savez(sys.argv[3], **trg2src)
