#! /usr/bin/env python

import sys
import os
import argparse
import cPickle
import string

# path for binary file of fast align
aligner = '~/fast_align/build/fast_align'

def version():
	'''Display version.'''
	s = '----------------------------------------\n' + \
	    ' test v0.1\n' + \
		' 2017/04/26 - 2017/04/26\n' + \
		' (c) 2017 Thumt\n' + \
		'----------------------------------------\n'
	print s

def help():
	'''Display helping message.'''
	s = 'Usage: prepare_mapping [--help] ...\n' + \
	    'Required arguments:\n' + \
		'  --src-file <file>      train set, source file\n' + \
		'  --trg-file <file>      train set, target file\n' + \
		'  --out-file <file>      output mapping file\n' + \
		'  --help                      displaying this message\n'
	print s
	sys.exit()

if __name__ == '__main__':
	#display version
	version()
	# initialize arguments
	src_file = ''  # train set, source file
	trg_file = ''  # train set, target file
	out_file = ''  # output mapping file
	i = 1
	while i < len(sys.argv):
		if sys.argv[i] == '--src-file':
			src_file = sys.argv[i + 1]
		elif sys.argv[i] == '--trg-file':
			trg_file = sys.argv[i + 1]
		elif sys.argv[i] == '--out-file':
			out_file = sys.argv[i + 1]
		else:
			help()
		i += 2
	# check required arguments
	if src_file == '' or \
	   trg_file == '' or \
	   out_file == '' :
		help()

	# generate corpus for fast_align
	src = open(src_file , 'r')
	trg = open(trg_file , 'r')
	f1 = open('for_fast_align.txt', 'w')
	srcline = src.readline()
	trgline = trg.readline()
	while srcline != '':
		f1.write(srcline.strip() + ' ||| ' + trgline.strip() + '\n')
		srcline = src.readline()
		trgline = trg.readline()
	f1.close()	
	
	# fast align
	cmd = aligner + ' -i for_fast_align.txt -d -o -v -p probs.txt > align.txt'
	os.system(cmd)
	
	# generate mapping file
	mapping = {}
	p = open('probs.txt', 'r')
	pline = p.readline().decode('utf-8')
	while pline != '':
		src, trg, logp = pline.strip().split('\t')
		logp = string.atof(logp)
		if mapping.has_key(src):
			if mapping[src][1] < logp:
				mapping[src] = [trg, logp]
		else:
			mapping[src] = [trg, logp]
		pline = p.readline()
	mapping = {key: mapping[key][0] for key in mapping}
	cPickle.dump(mapping, open(out_file, 'w'))	

	# clean
	os.remove('for_fast_align.txt')
	os.remove('probs.txt')
	os.remove('align.txt')

