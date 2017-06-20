#! /usr/bin/env python

import sys
import os

root_dir = '/data/disk1/private/ly/THUMT'
code_dir = root_dir + '/thumt'

def version():
	'''Display version.'''
	s = '--------------------------------------------\n' + \
	    '  visualize v0.1\n' + \
		'  2017/06/19 - 2017/06/19\n' + \
		'  (c) 2017 THUMT\n' + \
		'--------------------------------------------'
	print s

def help():
	'''Display helping message.'''
	s = 'Usage: visualize [--help] ...\n' + \
	    'Required arguments:\n' + \
		'  --model-file <file>         translation model file\n' + \
		'  --input-file <file>         input source file\n' + \
		'  --output-dir <dir>          output directory\n' + \
		'  --device {cpu, gpu0, ...}   device\n' + \
		'Optional arguments:\n' + \
		'  --help                      displaying this message'
	print s
	sys.exit()

if __name__ == '__main__':
	# display version
	version()
	sys.stdout.flush()
	# initialize arguments
	model_file = ''   # model file
	input_file = ''   # input file
	output_dir = ''   # output directory
	device = ''       # device
	# analyze command-line arguments
	i = 1
	while i < len(sys.argv):
		if sys.argv[i] == '--model-file':
			model_file = sys.argv[i + 1]
		elif sys.argv[i] == '--input-file':
			input_file = sys.argv[i + 1]
		elif sys.argv[i] == '--output-dir':
			output_dir = sys.argv[i + 1]
		elif sys.argv[i] == '--device':
			device = sys.argv[i + 1]
		else:
			print 'ERROR: incorrect argument:', sys.argv[i]
			help
		i += 2
	# check required arguments
	if model_file == '' or \
	   input_file == '' or \
	   output_dir == '' or \
	   device == '':
	   	print 'ERROR: required arguments missing!'
		help()
	# run lrp.py
	os.system('THEANO_FLAGS=floatX=float32,device=' + device + \
	          ',lib.cnmem=0.95 python ' + code_dir + \
			  '/lrp.py -m ' + model_file + \
			  ' -i ' + input_file + \
			  ' -o ' + output_dir)
