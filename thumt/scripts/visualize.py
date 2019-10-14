import os
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy
import re
from matplotlib import font_manager


def _open(filename, mode="r", encoding="utf-8"):
    if sys.version_info.major == 2:
        return open(filename, mode=mode)
    elif sys.version_info.major == 3:
        return open(filename, mode=mode, encoding=encoding)
    else:
        raise RuntimeError("Unknown Python version for running!")


def parse_numpy(string):
    string = string.replace('[', ' ').replace(']', ' ').replace(',', ' ')
    string = re.sub(' +', ' ', string)
    result = numpy.fromstring(string, sep=' ')
    return result

# set font
fontP = font_manager.FontProperties()
fontP.set_family('SimHei')
fontP.set_size(14)

# parse from text
result = _open(sys.argv[1], 'r').read()
src = re.findall('src: (.*?)\n', result)[0]
src = src.decode('utf-8')
trg = re.findall('trg: (.*?)\n', result)[0]
rlv = re.findall('result: ([\s\S]*)', result)[0]
rlv = parse_numpy(rlv)
src_words = src.split(' ')
src_words.append('<eos>')
trg_words = trg.split(' ')
trg_words.append('<eos>')

len_t = len(trg_words)
len_s = len(src_words)
rlv = rlv[:len_t*len_s]
rlv = numpy.reshape(rlv, [len_t, len_s])

# set the scale
maximum = numpy.max(numpy.abs(rlv))
plt.matshow(rlv, cmap="RdBu_r", vmin=-maximum, vmax=maximum)

fontname = "Times"
plt.colorbar()
plt.xticks(range(len_s), src_words, fontsize=14, family=fontname,
           rotation='vertical')
plt.yticks(range(len_t), trg_words, fontsize=14, family=fontname)

matplotlib.rcParams['font.family'] = "Times"
plt.show()

