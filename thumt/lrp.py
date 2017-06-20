import sys
import numpy
import time
import os
import logging
import tools
import theano
import random
import theano.tensor as TT
from theano import scan
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import random
import argparse
import json
from get_layer import *


parser = argparse.ArgumentParser("visualizing the NMT model")
parser.add_argument('-m', '--model', required = True, help = 'path to NMT model')
parser.add_argument('-i', '--input-file', required = True, help = 'path to input file')
parser.add_argument('-o', '--output-dir', required = True, help = 'path to output directory')

def init_weight(size, name, shared=True):
    '''
        Initialize weight matrice
    '''
    W = numpy.zeros(size, dtype= 'float32')
    if shared: 
        return theano.shared(W,name = name)
    else:
        return W
        
def init_idx(size, name, shared = True):
    '''
        Initialize words indexes in sentence
    '''
    W = numpy.zeros(size, dtype = 'int64')
    if shared:
        return theano.shared(W, name = name)
    else:
        return W
     
class DecoderVal(object):
    '''
        The class which stores the intermediate variables produced by decoder in NMT mdoel
        
        :type src_word_num: int
        :param src_word_num: the length of source sentence
        
        :type trg_word_num: int
        :param trg_word_num: the length of target sentence
        
        :type config: dict
        :param config: the configuration file
                            
    '''
    def __init__(self, src_word_num, trg_word_num, config):
        self.name = "dec_"
        name = "dec_"
        self.params = []
        dim = config['dim_rec_dec']
        self.dim = dim
        dim_c = config['dim_rec_enc'] * 2
        self.dim_c = dim_c
        self.dim_in = config['dim_emb_trg']
        word_num = trg_word_num
        self.word_num = word_num
        self.src_word_num = src_word_num
        dim_o = config['dim_emb_trg']
        src_vocab_size = config['num_vocab_src']
        trg_vocab_size = config['num_vocab_trg']
        
        self.y_idx = init_idx((word_num),name + 'y')
        self.h = init_weight((word_num, dim),name + 'h')
        self.c = init_weight((word_num, dim_c), name + 'c')
        self.dec_att = init_weight((word_num, src_word_num), name + 'att')
        self.gate_cin = init_weight((word_num, dim), name + 'gate_cin')
        self.gate_preactive = init_weight((word_num, dim), name + 'gate_preactive')
        self.gate = init_weight((word_num, dim), name + 'gate')
        self.reset_cin = init_weight((word_num, dim), name + 'reset_cin')
        self.reset_preactive = init_weight((word_num, dim), name + 'reset_preactive')
        self.reset = init_weight((word_num, dim), name + 'reset')
        self.state_cin = init_weight((word_num, dim), name + 'state_cin')
        self.reseted = init_weight((word_num, dim), name + 'reseted')
        self.state_preactive = init_weight((word_num, dim), name + 'state_preactive')
        self.state = init_weight((word_num, dim), name + 'state')
        self.state_in = init_weight((word_num, dim), name + 'state_in')
        self.gate_in = init_weight((word_num, dim), name + 'gate_in')
        self.reset_in = init_weight((word_num, dim), name + 'reset_in')
        self.state_in_prev = init_weight((word_num, dim), name + 'state_in_prev')
        self.readout = init_weight((word_num, dim), name + 'readout')
        self.maxout = init_weight((word_num, dim / 2), name + 'maxout')
        self.outenergy_1 = init_weight((word_num, dim_o), name + 'outenergy_1')
        self.outenergy_2 = init_weight((word_num, trg_vocab_size), name + 'outenergy_2')
        self.outenergy   = init_weight((word_num, trg_vocab_size), name + 'outenergy')
        
        self.params = [self.h,self.c, self.gate_cin, self.gate_preactive, self.gate,
                            self.reset_cin, self.reset_preactive, self.reset, self.state_cin,
                            self.reseted, self.state_preactive, self.state, self.state_in,
                            self.gate_in, self.reset_in, self.state_in_prev]
        self.params2 = [self.readout, self.maxout, self.outenergy, self.outenergy_1, self.outenergy_2] 
        self.y = None
        
    def readData(self, filename):
        '''
           Load intermediate variables produced by decoder in NMT model
           
           :type filename: string
           :param filename: the file which stores intermediate variables
        '''
        values = numpy.load(filename)
        for p in self.params:
            if p.name in values:
                p.set_value(values[p.name].reshape([values[p.name].shape[0], values[p.name].shape[2]]))
        for p in self.params2:
            if p.name in values: 
                p.set_value(values[p.name])
        self.dec_att.set_value(values['dec_att'].reshape([values['dec_att'].shape[0], values['dec_att'].shape[1]]))
        self.y_idx.set_value(values['dec_y'].reshape([values['dec_y'].shape[0]]))
        self.h_before = values['dec_h'][:-1].reshape([values['dec_h'].shape[0] - 1, values['dec_h'].shape[2]])
        self.h_before = numpy.concatenate((numpy.zeros((self.dim), dtype= 'float32').reshape([1, self.dim]), self.h_before))
        self.h_before = theano.shared(self.h_before, name = 'dec_h_before')
        
class EncoderVal(object):
    '''
        The class which stores the intermediate variables produced by forward encoder in NMT mdoel
        
        :type word_num: int
        :param word_num: the length of source sentence               
        
        :type config: dict
        :param config: the configuration file                       
    '''
    
    def __init__(self, word_num, config):
        self.name = "enc_for_"
        name = "enc_for_"
        self.params = []
        dim = config['dim_rec_enc']
        self.dim = dim
        
        self.x_idx = init_idx((word_num), name + 'x')
        self.h = init_weight((word_num, dim), name +'h')
        self.gate = init_weight((word_num, dim), name + 'gate')
        self.reset = init_weight((word_num, dim), name + 'reset')
        self.state = init_weight((word_num, dim), name + 'state')
        self.reseted = init_weight((word_num, dim), name + 'reseted')
        self.state_in = init_weight((word_num, dim), name + 'state_in')
        self.reset_in = init_weight((word_num, dim), name + 'reset_in')
        self.gate_in = init_weight((word_num, dim), name + 'gate_in')
        
        self.params = [self.h, self.gate, self.reset, self.state, self.reseted,
                        self.state_in, self.reset_in, self.gate_in]
        self.x = None
        
    def readData(self, filename):
        '''
            Load intermediate variables produced by forward encoder in NMT model
           
            :type filename: string
            :param filename: the file which stores intermediate variables
        '''
        values = numpy.load(filename)
        for p in self.params:
            if p.name in values:
                p.set_value(values[p.name].reshape([values[p.name].shape[0], values[p.name].shape[2]]))
                
        self.x_idx.set_value(values['enc_for_x'].reshape([values['enc_for_x'].shape[0]]))
        self.h_before = values['enc_for_h'][:-1].reshape([values['enc_for_h'].shape[0] - 1, values['enc_for_h'].shape[2]])        
        self.h_before = numpy.concatenate((numpy.zeros((self.dim), dtype= 'float32').reshape([1, self.dim]), self.h_before))
        self.h_before = theano.shared(self.h_before, name = 'enc_for_h_before')

class BackEncoderVal(object):
    '''
        The class which stores the intermediate variables produced by backward encoder in NMT mdoel
        
        :type word_num: int
        :param word_num: the length of source sentence               
        
        :type config: dict
        :param config: the configuration file                           
    '''
    def __init__(self, word_num, config):
        self.name = "enc_back_"
        name = "enc_back_"
        self.params = []
        dim = config['dim_rec_enc']
        self.dim = dim
       
        self.x_idx = init_idx((word_num), name + 'x')
        self.h = init_weight((word_num, dim), name +'h')
        self.gate = init_weight((word_num, dim), name + 'gate')
        self.reset = init_weight((word_num, dim), name + 'reset')
        self.state = init_weight((word_num, dim), name + 'state')
        self.reseted = init_weight((word_num, dim), name + 'reseted')
        self.state_in = init_weight((word_num, dim), name + 'state_in')
        self.reset_in = init_weight((word_num, dim), name + 'reset_in')
        self.gate_in = init_weight((word_num, dim), name + 'gate_in')
        self.params = [self.h, self.gate, self.reset, self.state, self.reseted,
                        self.state_in, self.reset_in, self.gate_in]
        self.x = None
        
    def readData(self, filename):
        '''
            Load intermediate variables produced by backward encoder in NMT model
           
            :type filename: string
            :param filename: the file which stores intermediate variables
        '''
        values = numpy.load(filename)
        for p in self.params:
            if p.name in values:
                p.set_value(values[p.name].reshape([values[p.name].shape[0],values[p.name].shape[2]]))
        
        self.x_idx.set_value(values['enc_for_x'].reshape([values['enc_for_x'].shape[0]])[::-1])
        self.h_before = values['enc_back_h'][:-1].reshape([values['enc_back_h'].shape[0] - 1, values['enc_back_h'].shape[2]])
        self.h_before = numpy.concatenate((numpy.zeros((self.dim), dtype = 'float32').reshape([1, self.dim]), self.h_before))
        self.h_before = theano.shared(self.h_before, name = 'enc_back_h_before')

class Model(object):
    '''
        The class which calculates the relevance
        
        :type src_word_num: int 
        :param src_word_num: the length of source sentence
        
        :type trg_word_num: int
        :param trg_word_num: the length of target sentence
        
        :type config: dict
        :param config: the configuration file
    '''
    def __init__(self, src_word_num, trg_word_num, config):
        self.encoder_val = EncoderVal(src_word_num, config)
        self.back_encoder_val = BackEncoderVal(src_word_num, config)
        self.decoder_val = DecoderVal(src_word_num,trg_word_num, config)
        self.name = 'GRU_enc'
        self.src_word_num = src_word_num
        self.trg_word_num = trg_word_num
        dim_in = config['dim_emb_src']
        self.dim_in = dim_in
        dim = config['dim_rec_enc']
        self.dim = dim
        dim_c = config['dim_rec_enc'] * 2
        self.dim_c = dim_c
        dim_class = config['num_vocab_trg']
        src_vocab_size = config['num_vocab_src']
        trg_vocab_size = config['num_vocab_trg']
        self.weight = 0.5
        
        self.src_emb = tools.init_weight((src_vocab_size, dim_in), 'emb_src_emb')
        self.src_emb_offset = tools.init_bias((dim_in), 'emb_src_b')
        self.trg_emb = tools.init_weight((trg_vocab_size, dim_in), 'emb_trg_emb')
        self.trg_emb_offset = tools.init_bias((dim_in), 'emb_trg_b')
        self.input_emb = tools.init_weight((dim_in, dim), self.name + '_inputemb')
        self.gate_emb = tools.init_weight((dim_in, dim), self.name + '_gateemb')
        self.reset_emb = tools.init_weight((dim_in, dim), self.name + '_resetemb')
        self.input_hidden = tools.init_weight((dim, dim), self.name + '_inputhidden')
        self.gate_hidden = tools.init_weight((dim, dim), self.name + '_gatehidden')
        self.reset_hidden = tools.init_weight((dim, dim), self.name + '_resethidden')
        self.params = [self.input_emb, self.gate_emb, self.reset_emb,
                            self.input_hidden, self.gate_hidden, self.reset_hidden]
        self.input_emb_offset = tools.init_bias((dim), self.name + '_inputoffset')
        self.params += [self.input_emb_offset, self.src_emb, self.src_emb_offset]  
        
        
        self.name = 'GRU_enc_back'
        self.input_emb_back = tools.init_weight((dim_in, dim), self.name +'_inputemb')
        self.gate_emb_back = tools.init_weight((dim_in, dim), self.name +'_gateemb')
        self.reset_emb_back = tools.init_weight((dim_in, dim), self.name +'_resetemb')
        self.input_hidden_back = tools.init_weight((dim, dim), self.name +'_inputhidden')
        self.gate_hidden_back = tools.init_weight((dim, dim), self.name +'_gatehidden')
        self.reset_hidden_back = tools.init_weight((dim, dim), self.name +'_resethidden')
        self.params += [self.input_emb_back, self.gate_emb_back, self.reset_emb_back,
                            self.input_hidden_back, self.gate_hidden_back, self.reset_hidden_back]
        self.input_emb_offset_back = tools.init_bias((dim), self.name +'_inputoffset')
        self.params += [self.input_emb_offset_back]
        
        
        self.name = 'GRU_dec'
        self.dec_readout_emb = tools.init_weight((dim_in, dim), self.name +'_readoutemb')
        self.dec_input_emb = tools.init_weight((dim_in, dim), self.name +'_inputemb')
        self.dec_gate_emb = tools.init_weight((dim_in, dim), self.name +'_gateemb')
        self.dec_reset_emb = tools.init_weight((dim_in, dim), self.name +'_resetemb')
        self.dec_readout_context = tools.init_weight((dim_c, dim), self.name +'_readoutcontext')
        self.dec_input_context = tools.init_weight((dim_c, dim), self.name +'_inputcontext')
        self.dec_gate_context = tools.init_weight((dim_c, dim), self.name +'_gatecontext')
        self.dec_reset_context = tools.init_weight((dim_c, dim), self.name +'_resetcontext')
        self.dec_readout_hidden = tools.init_weight((dim, dim),self.name +'_readouthidden')
        self.dec_input_hidden = tools.init_weight((dim, dim), self.name +'_inputhidden')
        self.dec_gate_hidden = tools.init_weight((dim, dim), self.name +'_gatehidden')
        self.dec_reset_hidden = tools.init_weight((dim, dim), self.name +'_resethidden')
        self.dec_input_emb_offset = tools.init_bias((dim), self.name +'_inputoffset')
        self.dec_probs_emb = tools.init_weight((dim / 2, dim_in), self.name +'_probsemb')
        self.dec_probs = tools.init_weight((dim_in, dim_class), self.name + '_probs')
        
        self. params  += [self.dec_readout_emb, self.dec_input_emb, self.dec_gate_emb, 
                            self.dec_reset_emb, self.dec_readout_context, self.dec_input_context, 
                            self.dec_gate_context, self.dec_reset_context, self.dec_readout_hidden,
                            self.dec_input_hidden, self.dec_gate_hidden, self.dec_reset_hidden,
                            self.dec_input_emb_offset, self.dec_probs_emb, self.dec_probs]
        
        self.ep = numpy.float32(0.4)
        self.ep = theano.shared(self.ep, name = 'ep')
        self.idx = TT.iscalar()
        self.R_c_x = TT.tensor3()
        
    def readData(self, param_filename, val_filename):
        '''
            Load the parameters of NMT models and intermediate variables
            
            :type param_filename: string
            :param param_filename: the file which stores the parameters of NMT models
            
            :type val_filename: string
            :param val_filename: the file which stores intermediate variables
        '''
        values = numpy.load(param_filename)
        for p in self.params:
            if p.name in values: 
                p.set_value(values[p.name])
        
        self.encoder_val.readData(val_filename)
        self.back_encoder_val.readData(val_filename)
        self.encoder_val.x = self.src_emb[self.encoder_val.x_idx]
        self.back_encoder_val.x = self.src_emb[self.back_encoder_val.x_idx]
        self.decoder_val.readData(val_filename)
        self.decoder_val.y = self.trg_emb[self.decoder_val.y_idx]
        self.decoder_val.y_before = tools.shift_one(self.decoder_val.y)
      
    def cal_encoder(self):
        '''
            Calculate the relevance in forward encoder
            
            :returns: 4-D numpy array, the relevance between forward encoder hidden states and x
        '''
        f = theano.function([], outputs = self.cal_encoder_step(self.encoder_val))
        R = numpy.zeros((self.src_word_num, self.src_word_num, self.dim,self.dim_in), dtype = 'float32')
        out = f()
        
        o0 = out[0]
        o1 = out[1]
        for i in range(self.src_word_num):
            for j in range(i + 1):
                if i == j:
                    R[i][j] = o0[i]
                else:
                    if i == 0:
                        tmp = numpy.zeros([self.dim, self.dim_in])   
                    else:
                        tmp = R[i - 1][j]
                    R[i][j] = o1[i].dot(tmp)
         
        return R
        
    def cal_back_encoder(self):
        '''
            Calculate the relevance in backward encoder
            
            :returns: 4-D numpy array, the relevance between backward encoder hidden states and x
        '''
        f = theano.function([], outputs = self.cal_encoder_step(self.back_encoder_val))
        R = numpy.zeros((self.src_word_num, self.src_word_num, self.dim, self.dim_in), dtype = 'float32')
        out = f()

        o0 = out[0]
        o1 = out[1]
        for i in range(self.src_word_num):
            for j in range(i + 1):
                if i == j:
                    R[i][j] = o0[i]
                else:
                    if i == 0:
                        tmp = numpy.zeros([self.dim, self.dim_in])
                        
                    else:
                        tmp = R[i - 1][j]
                    R[i][j] = o1[i].dot(tmp)
        for i in range(self.src_word_num):
            R[i] = R[i][::-1]
        return R[::-1]
    
    def cal_decoder(self, R_x):
        '''
            Calculate the relevance in decoder
            
            :type R_x: theano sharedVariable
            :param R_x: the relevance bewteen inputs and the hidden states of encoder.
        
            :returns: R_c_x,R_h_x,R_o_x,R_h_y,R_o_y are numpy arrays, they are relevance 
                      between context and x, relevance bewteen decoder hidden states and x, relevance 
                      bewteen readout and x, relevance bewteen decoder hidden states and y, relevance 
                      bewteen readout and y
        '''
        f = theano.function([self.idx, self.R_c_x], outputs = self.cal_decoder_step(self.decoder_val))
        R_c_x = []
        dec_att = self.decoder_val.dec_att.get_value()
        out = []
        
        #print 'Calculate weight ratio in decoder...'
        for i in range(self.trg_word_num):
            c = numpy.zeros([self.src_word_num, self.dim_c, self.dim_in])
            c = numpy.array(c, dtype = 'float32')
            for j in range(self.src_word_num):
                for k in range(self.src_word_num):
                    c[k] += dec_att[i][j] * R_x[j][k]
            R_c_x.append(c)
            out.append(f(i, c))
        #print 'Done'
        #print 'Calculating relevance bewteen context and x...'
        R_c_x = numpy.array(R_c_x, dtype = 'float32')
        #print 'Done'
        #print 'Calculating relevance bewteen decoder hidden states and x...'
        R_h_x = numpy.zeros((self.trg_word_num, self.src_word_num, self.dim, self.dim_in))
        for i in range(self.trg_word_num):
            for j in range(self.src_word_num): 
                if i == 0:
                    tmp = numpy.zeros([self.dim, self.dim_in])
                else:
                    tmp = R_h_x[i - 1][j]
                R_h_x[i][j] = out[i][1][j] + out[i][0].dot(tmp)
        #print 'Done'
        #print 'Calculating relevance bewteen readout and x...'
        R_o_x = numpy.zeros((self.trg_word_num, self.src_word_num, 1, self.dim_in))
        for i in range(self.trg_word_num):
            for j in range(self.src_word_num):
                if i == 0:
                    tmp = numpy.zeros([self.dim, self.dim_in])
                else:
                    tmp = R_h_x[i - 1][j]
                R_o_x[i][j] = out[i][4][j] + out[i][3].dot(tmp)
        #print 'Done'        
        #print 'Calculating relevance bewteen decoder hidden states and y...'      
        R_h_y = numpy.zeros((self.trg_word_num, self.trg_word_num, self.dim, self.dim_in))
        for i in range(self.trg_word_num):
            for j in range(i):
                if i == 0:
                    continue
                if i-1 == j:
                    R_h_y[i][j] = out[i - 1][2]
                else:
                    if i -1 == 0:
                        tmp = numpy.zeros([self.dim, self.dim_in])
                    else:
                        tmp = R_h_y[i - 1][j]
                    R_h_y[i][j] = numpy.dot(out[i-1][0], tmp)
        #print 'Done'            
        #print 'Calculating relevance bewteen readout and y...'
        R_o_y = numpy.zeros((self.trg_word_num, self.trg_word_num, 1, self.dim_in))
        for i in range(self.trg_word_num):
            for j in range(i):
                if i == j:
                    R_o_y[i][j] = out[i][5]
                else:
                    if i == 0 :
                        tmp = numpy.zeros([self.dim, self.dim_in])
                    else:
                        tmp = R_h_y[i][j]
                    R_o_y[i][j] = numpy.dot(out[i][3], tmp)
        #print 'Done'
        return R_c_x, R_h_x, R_o_x, R_h_y, R_o_y

    def cal_decoder_step(self, decoder_val):
        '''
            Calculate the weight ratios in decoder
            
            :type decoder_val: class
            :param decoder_val: the class which stores the intermediate variables in decoder
            
            :returns: R_h_h, R_h_x, R_h_y, R_outenergy_2_h, R_outenergy_2_x, R_outenergy_2_y_before are theano variables, weight ratios in decoder.
        '''
        y = decoder_val.y[self.idx].dimshuffle(0, 'x')
        R_state_in_y = (y * self.dec_input_emb + self.dec_input_emb_offset[self.idx]) / ( decoder_val.state_in[self.idx] + self.ep * TT.sgn(decoder_val.state_in[self.idx])).dimshuffle('x', 0)
        R_state_in_y = R_state_in_y.dimshuffle(1, 0)
        R_reset_in_y = y * self.dec_reset_emb / (decoder_val.reset_in[self.idx] + self.ep * TT.sgn(decoder_val.reset_in[self.idx])).dimshuffle('x', 0)
        R_reset_in_y = R_reset_in_y.dimshuffle(1, 0)
        R_gate_in_y = y * self.dec_gate_emb / (decoder_val.gate_in[self.idx] + self.ep * TT.sgn(decoder_val.gate_in[self.idx])).dimshuffle('x', 0)
        R_gate_in_y = R_gate_in_y.dimshuffle(1, 0)
        c = decoder_val.c[self.idx].dimshuffle(0, 'x')
        R_gate_cin  = c * self.dec_gate_context / (decoder_val.gate_cin[self.idx] + self.ep * TT.sgn(decoder_val.gate_cin[self.idx])).dimshuffle('x', 0)
        R_gate_cin = R_gate_cin.dimshuffle(1, 0)
        R_reset_cin = c * self.dec_reset_context / (decoder_val.reset_cin[self.idx] + self.ep * TT.sgn(decoder_val.reset_cin[self.idx])).dimshuffle('x', 0)
        R_reset_cin = R_reset_cin.dimshuffle(1, 0)
        R_state_cin = c * self.dec_input_context / (decoder_val.state_cin[self.idx] + self.ep * TT.sgn(decoder_val.state_cin[self.idx])).dimshuffle('x', 0)
        R_state_cin = R_state_cin.dimshuffle(1, 0)
        R_gate_cin_x = TT.dot(R_gate_cin, self.R_c_x).dimshuffle(1, 0, 2)
        R_reset_cin_x = TT.dot(R_reset_cin, self.R_c_x)
        R_reset_cin_x = R_reset_cin_x.dimshuffle(1, 0, 2)
        R_state_cin_x = TT.dot(R_state_cin, self.R_c_x)
        R_state_cin_x = R_state_cin_x.dimshuffle(1, 0, 2)
        h_before = decoder_val.h_before[self.idx].dimshuffle(0, 'x')
        R_gate_h = h_before * self.dec_gate_hidden / (decoder_val.gate[self.idx] + self.ep * TT.sgn(decoder_val.gate[self.idx])).dimshuffle('x', 0)
        R_gate_h = R_gate_h.dimshuffle(1, 0)
        R_reset_h = h_before * self.dec_reset_hidden / (decoder_val.reset[self.idx] + self.ep * TT.sgn(decoder_val.reset[self.idx])).dimshuffle('x', 0)
        R_reset_h = R_reset_h.dimshuffle(1, 0)
        R_gate_y = R_gate_in_y * (decoder_val.gate_in[self.idx] / (decoder_val.gate[self.idx] + self.ep * TT.sgn(decoder_val.gate[self.idx]))).dimshuffle(0, 'x')
        R_reset_y = R_reset_in_y * (decoder_val.reset_in[self.idx] / (decoder_val.reset[self.idx] + self.ep * TT.sgn(decoder_val.reset[self.idx]))).dimshuffle(0, 'x')
        R_gate = (decoder_val.gate_cin[self.idx] / (decoder_val.gate[self.idx] + self.ep * TT.sgn(decoder_val.gate[self.idx]))).dimshuffle('x', 0, 'x')
        R_gate_x = R_gate * R_gate_cin_x
        R_reset = (decoder_val.reset_cin[self.idx] / (decoder_val.reset[self.idx] + self.ep * TT.sgn(decoder_val.reset[self.idx]))).dimshuffle('x', 0, 'x')
        R_reset_x = R_reset * R_reset_cin_x
        R_reseted_h = R_reset_h * self.weight + TT.eye(self.dim, self.dim) * self.weight
        R_reseted_y = R_reset_y * self.weight
        R_reseted_x = R_reset_x * self.weight
        R_state_x = R_state_cin_x * (decoder_val.state_cin[self.idx] / (decoder_val.state[self.idx] + self.ep * TT.sgn(decoder_val.state[self.idx]))).dimshuffle('x', 0, 'x')
        R_state_y = R_state_in_y * (decoder_val.state_in[self.idx] / (decoder_val.state[self.idx] + self.ep * TT.sgn(decoder_val.state[self.idx]))).dimshuffle(0, 'x')        
        reseted = decoder_val.reseted[self.idx].dimshuffle(0, 'x')
        R_state_reseted = reseted * self.dec_input_hidden[self.idx] / (decoder_val.state[self.idx] + self.ep * TT.sgn(decoder_val.state[self.idx])).dimshuffle(0, 'x')
        R_state_reseted = R_state_reseted.dimshuffle(1, 0)
        R_state_h = TT.dot(R_state_reseted, R_reseted_h)
        R_state_x += TT.dot(R_state_reseted, R_reseted_x).dimshuffle(1, 0, 2)
        R_state_y = TT.dot(R_state_reseted, R_reseted_y)
        R_h = (decoder_val.gate[self.idx] * decoder_val.state[self.idx] / (decoder_val.h[self.idx] + self.ep * TT.sgn(decoder_val.h[self.idx]))).dimshuffle(0, 'x') * self.weight
        R_h_h = R_gate_h * R_h + R_state_h * R_h
        R_h2 = ((1 - decoder_val.gate[self.idx]) * decoder_val.h_before[self.idx] / (decoder_val.h[self.idx] + self.ep * TT.sgn(decoder_val.h[self.idx]))).dimshuffle(0, 'x')
        R_h_h += TT.identity_like(R_h_h) * R_h2
        R_h_y = R_gate_y * R_h + R_state_y * R_h
        R_h = (decoder_val.gate[self.idx] * decoder_val.state[self.idx] / (decoder_val.h[self.idx] + self.ep * TT.sgn(decoder_val.h[self.idx]))).dimshuffle('x', 0, 'x') * self.weight
        R_h_x = R_gate_x * R_h + R_state_x * R_h
        
        R_readout_c = c * self.dec_readout_context / (decoder_val.readout[self.idx] + self.ep * TT.sgn(decoder_val.readout[self.idx])).dimshuffle('x', 0)
        R_readout_c = R_readout_c.dimshuffle(1, 0)
        R_readout_x = TT.dot(R_readout_c, self.R_c_x).dimshuffle(1, 0, 2)
        R_readout_h = h_before * self.dec_readout_hidden / (decoder_val.readout[self.idx] + self.ep * TT.sgn(decoder_val.readout[self.idx])).dimshuffle('x', 0)
        R_readout_h = R_readout_h.dimshuffle(1, 0)
        y_before = decoder_val.y_before[self.idx].dimshuffle(0, 'x')
        R_readout_y_before = y_before * self.dec_readout_emb / (decoder_val.readout[self.idx] + self.ep * TT.sgn(decoder_val.readout[self.idx])).dimshuffle('x', 0)
        R_readout_y_before = R_readout_y_before.dimshuffle(1, 0)
        dim1 = decoder_val.maxout[self.idx].shape[0]
        maxout = decoder_val.maxout[self.idx].reshape([dim1 / 2, 2])
        maxout = TT.argmax(maxout, axis = 1)
        maxout = maxout.reshape([dim1 / 2])
        L = TT.arange(dim1 / 2)
        maxout = maxout + L * 2  + L * dim1
        R_maxout = TT.zeros((self.dim * self.dim / 2))
        R_maxout = TT.set_subtensor(R_maxout[maxout.flatten()], 1.0)
        R_maxout = R_maxout.reshape([self.dim / 2, self.dim])
        R_maxout_y_before = TT.dot(R_maxout, R_readout_y_before)
        R_maxout_h = TT.dot(R_maxout, R_readout_h)
        R_maxout_x = TT.dot(R_maxout, R_readout_x).dimshuffle(1, 0, 2)
        maxout = decoder_val.maxout[self.idx].dimshuffle(0, 'x')
        R_outenergy1_maxout = maxout * self.dec_probs_emb  / (decoder_val.outenergy_1[self.idx] + self.ep * TT.sgn(decoder_val.outenergy_1[self.idx])).dimshuffle('x', 0)
        R_outenergy1_maxout = R_outenergy1_maxout.dimshuffle(1, 0)
        R_outenergy1_y_before = TT.dot(R_outenergy1_maxout, R_maxout_y_before)
        R_outenergy1_h = TT.dot(R_outenergy1_maxout, R_maxout_h)
        R_outenergy1_x = TT.dot(R_outenergy1_maxout, R_maxout_x).dimshuffle(1, 0, 2)
        probs = self.dec_probs.dimshuffle(1, 0)[decoder_val.y_idx[self.idx]].dimshuffle(0, 'x')
        outenergy_1 = decoder_val.outenergy_1[self.idx].dimshuffle(0, 'x')
        idx = decoder_val.y_idx[self.idx]
        outenergy_2 = (decoder_val.outenergy_2[self.idx][idx])
        R_outenergy_2 = outenergy_1 * probs / (outenergy_2 + self.ep * outenergy_2)
        R_outenergy_2 = R_outenergy_2.dimshuffle(1, 0)
        R_outenergy_2_y_before = TT.dot(R_outenergy_2, R_outenergy1_y_before)
        R_outenergy_2_h = TT.dot(R_outenergy_2, R_outenergy1_h)
        R_outenergy_2_x = TT.dot(R_outenergy_2, R_outenergy1_x).dimshuffle(1, 0, 2)
        return R_h_h, R_h_x, R_h_y, R_outenergy_2_h, R_outenergy_2_x, R_outenergy_2_y_before
       
    def cal_encoder_step(self, encoder_val):
        '''
            Calculate the weight ratios in encoder
            
            :type decoder_val: class
            :param decoder_val: the class which stores the intermediate variables in encoder
            
            :returns: R_h_x, R_h_h are theano variables, weight ratios in encoder
        '''
        encoder_val.x = encoder_val.x.dimshuffle(0, 1, 'x')
        R_state_in_x = (encoder_val.x * self.input_emb + self.input_emb_offset ) / (self.ep * TT.sgn(encoder_val.state_in) + encoder_val.state_in).dimshuffle(0, 'x', 1)
        R_state_in_x = R_state_in_x.dimshuffle(0, 2, 1)
        R_reset_in_x = encoder_val.x * self.reset_emb / (encoder_val.reset_in + self.ep * TT.sgn(encoder_val.reset_in)).dimshuffle(0, 'x', 1)
        R_reset_in_x = R_reset_in_x.dimshuffle(0,2,1)
        R_gate_in_x  = encoder_val.x * self.gate_emb / (encoder_val.gate_in + self.ep * TT.sgn(encoder_val.gate_in)).dimshuffle(0, 'x', 1)
        R_gate_in_x = R_gate_in_x.dimshuffle(0, 2, 1)
        h_before = encoder_val.h_before.dimshuffle(0, 1, 'x')
        R_gate_h = h_before * self.gate_hidden / (encoder_val.gate + self.ep * TT.sgn(encoder_val.gate)).dimshuffle(0, 'x', 1)
        R_gate_x = R_gate_in_x * (encoder_val.gate_in / (encoder_val.gate + self.ep * TT.sgn(encoder_val.gate))).dimshuffle(0, 1, 'x')
        R_reset_h = h_before * self.reset_hidden / (encoder_val.reset + self.ep * TT.sgn(encoder_val.reset)).dimshuffle(0, 'x', 1)
        R_reset_x = R_reset_in_x * (encoder_val.reset_in / (encoder_val.reset + self.ep * TT.sgn(encoder_val.reset))).dimshuffle(0, 1, 'x')
        R_reseted_h = R_reset_h * self.weight + TT.eye(self.dim, self.dim) * self.weight
        R_reseted_x = R_reset_x * self.weight
        encoder_val.reseted = encoder_val.reseted.dimshuffle(0, 1, 'x')
        R_state_reseted = encoder_val.reseted * self.input_hidden / (encoder_val.state + self.ep * TT.sgn(encoder_val.state)).dimshuffle(0, 'x', 1)
        R_state_reseted = R_state_reseted.dimshuffle(0, 2, 1)
        R_state_h = TT.batched_dot(R_state_reseted, R_reseted_h)
        R_state_x = TT.batched_dot(R_state_reseted, R_reseted_x)
        R_state_x += R_state_in_x * ((encoder_val.state_in / (encoder_val.state + self.ep * TT.sgn(encoder_val.state))).dimshuffle(0, 1, 'x'))
        R_h = (encoder_val.gate * encoder_val.state / (encoder_val.h + self.ep * TT.sgn(encoder_val.h))).dimshuffle(0, 1, 'x') * self.weight
        R_h_h = R_state_h * R_h +R_gate_h * R_h 
        R_h2 = ((1 - encoder_val.gate) * encoder_val.h_before / (encoder_val.h + self.ep * TT.sgn(encoder_val.h))).dimshuffle(0, 1, 'x')
        R_h_h =  TT.identity_like(R_h_h[0]) * R_h2
        R_h_x = R_gate_x * R_h + R_state_x * R_h 
        return R_h_x, R_h_h

def normalize(R, f):
    '''
        Normalize the relevance vector and write to result into output file
        
        :type R: 4-D numpy array
        :param R: the relevance vector to be normalized
        
        :type f: file
        :param f: output file
    '''
    dim = R.shape
    res = numpy.zeros([dim[0], dim[1]])
    for i in range(dim[0]):
        for j in range(dim[1]):
            res[i][j] = abs(R[i][j].sum())
    max = -100000
    for i in range(dim[0]):
        for j in range(dim[1]):
            if res[i][j] > max:
                max = res[i][j]
    for i in range(dim[0]):
        for j in range(dim[1]):
            res[i][j] = res[i][j] / max
            f.write(str(res[i][j]) + ' ')
        f.write('\n')
    
def save(att, f):
    '''
        write attentions into output file
        
        :type att: 2-D numpy array
        :param att: attention information
        
        :type f: file
        :param f: output file
    '''
    dim = att.shape
    for i in range(dim[0]):
        for j in range(dim[1]):
            f.write(str(att[i][j]) + ' ')
        f.write('\n')    



if __name__ == "__main__":
	tb = time.time()
	# load translation model
	dirs = os.getcwd()
	args = parser.parse_args()
	sys.stdout.write('\nLoading the translation model ... ')
	sys.stdout.flush()
	nmt_model, data, config= load_model_and_data(args.model)
	sys.stdout.write('done!\n')
	filename = args.input_file
	f_input = open(filename)
	os.system("mkdir " + dirs +"/" + args.output_dir)
	num = 0
	for line in f_input:
		_tb = time.time()
		print '\n========== sentence ' + str(num + 1) + ' =========='
		src_sentence = line.strip()
		# generate temporary values
		sys.stdout.write('Generating temporary values ... ')
		sys.stdout.flush()
		generate_tmp_val(nmt_model,data,src_sentence,num)
		sys.stdout.write('done!\n')
		filename = dirs + "/sent_" + str(num) + '.txt'
		f_sentence = open(filename)
		f_w = open(os.path.join(args.output_dir,str(num) + '.txt'),'w')
		src = f_sentence.readline().strip() 
		trg = f_sentence.readline().strip()
		f_w.write(src + '\n')
		f_w.write(trg + '\n')
		model = Model(len(src.split()), len(trg.split()), config)
		model.readData(args.model, dirs + "/val_" + str(num) + '.npz')
		sys.stdout.write('Calculating encoder relevance ... ')
		sys.stdout.flush()
		R_enc_forward = model.cal_encoder()
		R_enc_back = model.cal_back_encoder()
		R_enc = numpy.concatenate((R_enc_forward, R_enc_back), axis = 2)
		sys.stdout.write('done!\n')
		sys.stdout.write('Calculating decoder relevance ... ')
		sys.stdout.flush()
		R_c_x, R_h_x, R_o_x, R_h_y, R_o_y = model.cal_decoder(R_enc)
		sys.stdout.write('done!\n')
		sys.stdout.flush()
		normalize(R_enc_forward, f_w)
		normalize(R_enc_back, f_w)
		normalize(R_enc, f_w)
		normalize(R_c_x, f_w)
		normalize(R_h_x, f_w)
		normalize(R_h_y,f_w)
		save(model.decoder_val.dec_att.get_value(), f_w)
		normalize(R_o_x, f_w)
		normalize(R_o_y, f_w)
		f_w.close()
		os.system("rm " + dirs + "/val_"+ str(num) +".npz")
		os.system("rm " + dirs + "/sent_"+ str(num) +".txt")
		num += 1
		_te = time.time()
		print '%d source words, %d target words, %.2f seconds' % \
			  (len(src.split()), len(trg.split()), _te - _tb)
	sys.stdout.write('\n')
	os.system("zip -r " + args.output_dir + ".zip " + args.output_dir)
	os.system("rm -rf " + args.output_dir)
	te = time.time()
	print '\nTotal time: %.2f seconds' % (te - tb)
	print 'Average time: %.2f seconds' % ((te - tb) / float(num))
