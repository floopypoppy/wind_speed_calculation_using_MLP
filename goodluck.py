#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
"""
from __future__ import division
import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.structure import TanhLayer # or other types
from astropy.io import fits
from matplotlib import pyplot as plt
import pandas as pd
from random import shuffle


'''import training slope data'''
#hdu = fits.open("testslos_20180118.fits")
#testslos = hdu[0].data
#
#def stack(vec,stride,totalen,n):
#    '''
#    My explanation of this function
#    
#    Parameters
#    ------
#    vec : The vector of things
#    stride : lots of strides
#    totalen : total length of something
#    
#    Outputs:
#        something else
#    '''
#    for i in range(totalen-stride+1):
#        yield vec[i*n:(i+stride)*n]

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
    

'''basic parameters'''
#n_frame = 5
#total_frame = 200
#####nb_hidnod = int(np.sqrt(72*n_frame))

'''non-random training and validation samples (no testing)'''
#trn_size = total_frame-n_frame+1  ## number of training sequences for each v
#trnds = SupervisedDataSet(72*n_frame,1) # training dataset
#inp = []
#for i in range(5,11):
#    vec = allslopes[i-5]
#    inp.extend([j for j in stack(vec,n_frame,total_frame,72)])
#inp = np.array(inp)
#tar = np.empty([6*trn_size,1])
#tar[:,0] = np.arange(5,11).repeat(trn_size)
#trnds.setField('input',inp)
#trnds.setField('target',tar)

'''testing dataset'''
#h = fits.open('trnslos_20180118.fits')
#trnslos = h[0].data
#h.close()
#h = fits.open('testspd_20171210.fits')
#testspd = h[0].data
#h.close()
#
#test_size = total_frame-n_frame+1
#testds = SupervisedDataSet(72*n_frame,1)
#test_inp = []
#for i in range(5):
#    vec = testslos[i]
#    test_inp.extend([j for j in stack(vec,n_frame,total_frame,72)])
#test_inp = np.array(test_inp)
#test_tar = np.empty([5*test_size,1])
#test_tar[:,0] = testspd.repeat(test_size)
#testds.setField('input',test_inp)
#testds.setField('target',test_tar)

'''
random training and testing samples
all data are randomly split into training, validation and testing set
'''
#ratio_trnset = 0.8
#trn_size = int(total_frame*ratio_trnset)
#test_size = total_frame-n_frame+1-trn_size
#trnds = SupervisedDataSet(72*n_frame,1) # training dataset
#testds = SupervisedDataSet(72*n_frame,1)
#trn_inp = []        
#test_inp = []
#for i in range(6):
#    dummy = []
#    vec = allslopes[i]
#    dummy.extend([j for j in stack(vec,n_frame,total_frame,72)])
#    shuffle(dummy)
#    trn_inp.extend(dummy[:trn_size])
#    test_inp.extend(dummy[trn_size:])
#trn_inp = np.array(trn_inp)
#test_inp = np.array(test_inp)
###inp = inp/2.0
#trnds.setField('input',trn_inp)
#testds.setField('input',test_inp)
##
#trn_tar = np.empty([6*trn_size,1])
#trn_tar[:,0] = np.arange(5,11).repeat(trn_size)
#test_tar = np.empty([6*test_size,1])
#test_tar[:,0] = np.arange(5,11).repeat(test_size)
##tar = tar/20
#trnds.setField('target',trn_tar)
#testds.setField('target',test_tar)

'''training dataset with random phase screens (2018.01.17)'''
n_frame = 50
trnds = SupervisedDataSet(72*n_frame,1)
#trnds.setField('input',allslos[:,:360])
norm_inp, trnslos_min, trnslos_max = normalise(trnslos)
trnds.setField('input',norm_inp)
trn_tar = np.empty([6000,1])
trn_tar[:,0] = np.arange(5,11).repeat(1000)
#trn_tar[:,0] = np.array([-0.5,0,0.5]).repeat(1000)
#trn_tar = np.log10(trn_tar)
trnds.setField('target',trn_tar/15)
#trnds.setField('target',trn_tar)


'''learning process'''
#rand_ind = list(2*np.random.random(5)+1) #random number between 1 and 3
#rand_ind.extend([2,3])
net = buildNetwork(72*n_frame, 1000, 1, hiddenclass = TanhLayer)
lr = 0.001
momentum = 0
lrdecay = 1
wdecay = 0
t = BackpropTrainer(net, trnds, learningrate=lr, lrdecay=lrdecay,
                 momentum=momentum, verbose=True, batchlearning=False,
                 weightdecay=wdecay)

trnerr, vderr = t.trainUntilConvergence(maxEpochs=30,validationProportion=0.1)

'''mini-batch learning'''
#inp_batch, permutation = trnds.randomBatches('input',10)
#tar_batch = trnds.batches('target',10,permutation)
#mini_batch = zip(inp_batch, tar_batch)
#maxepoch = 100
#sub_ds = SupervisedDataSet(72*n_frame,1)
#t = BackpropTrainer(net, None, learningrate = 0.0001, momentum = 0.1, verbose = True, batchlearning = True)
#for epoch in xrange(maxepoch):
#    for sub_inp, sub_tar in mini_batch:
#        sub_ds.setField('input',sub_inp)
#        sub_ds.setField('target',sub_tar)
#        t.ds = sub_ds
#        t.train()
        
'''net training and testing output'''
out_err = np.empty((6000,1))
for i in range(6000):
    out_err[i] = out_trn[i]-trnds['target'][i]
plt.scatter(trn_tar/15,out_err,s=0.05)
plt.title('normalised network output deviation VS normalised expected output')
plt.xlabel('normalised expected output')
plt.ylabel('normalised network output deviation')
#plt.axis([0.3, 0.7,0.3, 0.7])
#plt.gca().set_aspect('equal', adjustable='box')
    
plt.scatter(np.arange(6000),out_err,s=0.1)
plt.xlabel('training sequence')
plt.ylabel('output-target')

out_trn = np.empty((6000,1))
for i in range(6000):
    out_trn[i] = net.activate(trnds['input'][i])
plt.hist(out_trn*15)

testslos,_,_ = normalise(testslos,trnslos_min,trnslos_max)
out_tst = np.empty((2000,1))
for i in range(2000):
    out_tst[i] = net.activate(testslos[i])*20
plt.hist(out_tst)
    
#out_trn = np.empty((6*trn_size,1))
#for i in range(6*trn_size):
#    out_trn[i] = net.activate(trnds['input'][i])
#trnmse = t.testOnData(trnds)*2
#out_test = []
#for j in range(5*test_size):
#    out_test.extend([net.activate(testds['input'][j])])
#testmse = t.testOnData(testds)*2
#
#plt.scatter(range(6*trn_size),out_trn)
#plt.xlabel('training sequence')
#plt.ylabel('training output')
#plt.text(10,10.1,'lr=%.5f'%lr,fontsize=12)
#plt.text(10,9.8,'momentum=%.5f'%momentum,fontsize=12)
#plt.text(10,9.5,'lrdecay=%.5f'%lrdecay,fontsize=12)
#plt.text(10,9.2,'wdecay=%.6f'%wdecay,fontsize=12)
#plt.text(10,8.9,'#epoch=%i'%t.epoch,fontsize=12)
#plt.text(10,8.6,'trnmse=%.4f'%trnmse,fontsize=12)
#plt.text(10,8.3,'testmse=%.4f'%testmse,fontsize=12)
#plt.savefig('./ParaTuning/lr_2/trn_lr=%.5f.eps'%lr,dfi=600)
#plt.close()
#
#plt.scatter(range(5*test_size),out_test)
#plt.xlabel('testing sequence')
#plt.ylabel('testing output')
#plt.text(10,10.1,'lr=%.5f'%lr,fontsize=12)
#plt.text(10,9.8,'momentum=%.5f'%momentum,fontsize=12)
#plt.text(10,9.5,'lrdecay=%.5f'%lrdecay,fontsize=12)
#plt.text(10,9.2,'wdecay=%.6f'%wdecay,fontsize=12)
#plt.text(10,8.9,'#epoch=%i'%t.epoch,fontsize=12)
#plt.text(10,8.6,'trnmse=%.4f'%trnmse,fontsize=12)
#plt.text(10,8.3,'testmse=%.4f'%testmse,fontsize=12)
#plt.savefig('./ParaTuning/lr_2/test_lr=%.5f.eps'%lr,dfi=600)
#plt.close()
#
#'''save outputs and params'''        
##        lrdict = {'lr':[],'trnerr':[],'vderr':[],'#epoch':[],'trnmse':[],'testmse':[]}
#lrdict['lr'].extend([lr])
#lrdict['trnerr'].append(trnerr)
#lrdict['vderr'].append(vderr)
#lrdict['#epoch'].extend([t.epoch])
#lrdict['trnmse'].extend([trnmse])  
#lrdict['testmse'].extend([testmse])
#writer = pd.ExcelWriter('./ParaTuning/lr_2/lrdict_2.xlsx')
#df1 = pd.DataFrame(lrdict)
#df1.to_excel(writer,'Sheet1')
#writer.save()
#
##    moparams = net.params
#lrparams = np.vstack((lrparams,net.params))
#fits.writeto('./ParaTuning/lr_2/lrparams_2.fits',lrparams,overwrite=True)