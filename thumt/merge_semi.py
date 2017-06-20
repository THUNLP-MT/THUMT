import numpy
import sys
print sys.argv

'''
	This script is used to merge two RNNsearch models 
	in different directions to BiRNNsearch (eg. zh-en and en-zh)
	for semi-supervised learning 

	usage:
		python merge_semi.py [src-trg model] [trg-src model] [bidirectional model]
'''

vals = numpy.load(sys.argv[1])
rev_vals = numpy.load(sys.argv[2])

print sys.argv
new_vals = {}

for key,value in vals.iteritems():

   new_vals[key] = value;

for key,value in rev_vals.iteritems():
   new_vals["inv_" + key] = value;

numpy.savez(sys.argv[3], **new_vals)






