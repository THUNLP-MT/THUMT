import json
import sys
import cPickle

'''
	This script is used to filter sentences with unknown words.
	the input and output file should be pickled or json format.

	usage:
		python filter_semi.py [input] [output]
'''

index_unk = 1

js = True
try:
	input = json.load(open(sys.argv[1], 'r'))
except:
	js = False
	input = cPickle.load(open(sys.argv[1], 'r'))

result = []
count = 0
for i in range(len(input)):
	if i % 100000 == 0:
		print i, count
	if index_unk in input[i]:
		continue
	result.append(input[i])
	count += 1
print 'remaining:', count

if js:
	output = json.dump(result, open(sys.argv[2], 'w'))
else:
	output = cPickle.dump(result, open(sys.argv[2], 'w'))
