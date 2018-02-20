#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
"""
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from pylab import *
import pickle
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.structure import TanhLayer # or other types


def normalise(m,m_min=[],m_max=[]):
    '''
    normalise every row of the input matrix m between [-1,1]
    
    Parameters
    ------
    m: input matrix to be normalised
    m_min: minimum of each column of m 
    m_max: maximum of each column of m
    '''
    if m_min == []:
        m_min = m.min(0)
    if m_max == []:
        m_max = m.max(0)
    m = 2*(m-m_min)/(m_max-m_min)-1
    return m, m_min, m_max

'''training dataset with random phase screens (2018.01.17)'''
mx = np.abs(trnslos_all).max()
ntrnslos_all = trnslos_all/mx
n_frame = 20
norm_inp = ntrnslos_all[:,:n_frame*72]
trnds = SupervisedDataSet(72*n_frame,1)
#norm_inp, trnslos_min, trnslos_max = normalise(trnslos)
trnds.setField('input',norm_inp)
trn_tar = np.empty([6000,1])
trn_tar[:,0] = np.arange(5,11).repeat(1000)/15
trnds.setField('target',trn_tar)

'''learning process'''
net = buildNetwork(72*n_frame, 1000, 1, hiddenclass = TanhLayer)
lr = 0.001
momentum = 0
lrdecay = 1
wdecay = 0
t = BackpropTrainer(net, trnds, learningrate=lr, lrdecay=lrdecay,
                 momentum=momentum, verbose=True, batchlearning=False,
                 weightdecay=wdecay)

trnerr, vderr, trnData, vdData =  t.trainUntilConvergence(maxEpochs=30,validationProportion=0.1)