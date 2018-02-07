#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
auxiliaries of trial.py to evaluate network

"""
from pylab import *
import numpy as np
from matplotlib import pyplot as plt

'''training data augmentation'''
trnslos = np.empty((6000,72*n_frame))
for i in range(5,11):
    h = fits.open('trnslos_20180202_%i.fits'%i)
    trnslos[(i-5)*1000:(i-4)*1000] = h[0].data
    
'''semilog training error vs epoch'''
total_epoch = 50
trnerr_2 = np.array(trnerr)*2 

'''calculate actual output'''
out_trn = np.empty((6000,1))
for i in range(6000):
    out_trn[i] = net.activate(trnds['input'][i])
#plt.hist(out_trn*15)
#plt.scatter(np.arange(6000),out_trn,s=0.1)
fits.writeto('outtrn_40_1000.fits',out_trn)

'''error bar of normalised training error vs expected ouput'''
nor_tar = np.arange(5,11)/15.0
e_mean = np.empty(6)
e_std = np.empty(6)
for i in range(6):
    data = out_trn[i*1000:(i+1)*1000]
    temp = data-nor_tar[i]
    e_mean[i] = np.mean(temp)
    e_std[i] = np.std(temp)

'''error bar of normalised training square error vs expected ouput'''
se_mean = np.empty(6)
se_std = np.empty(6)
for i in range(6):
    data = out_trn[i*1000:(i+1)*1000]
    temp = (data-nor_tar[i])**2
    se_mean[i] = np.mean(temp)
    se_std[i] = np.std(temp)

'''error bar of normalised output vs expected output'''
x_mean = np.empty(6)
x_std = np.empty(6)
for i in range(6):
    data = out_trn[i*1000:(i+1)*1000]
    x_mean[i] = np.mean(data)
    x_std[i] = np.std(data)

'''subplots'''
subplots_adjust(hspace=0.4,wspace=0.4)
plt.figure(1)
plt.subplot(221)
plt.loglog(trnerr_2[:total_epoch+1])
plt.xlabel('epoch')
plt.ylabel('training MSE')
plt.tick_params(labelsize='small')
plt.subplot(222)
plt.errorbar(nor_tar,se_mean,yerr=se_std,fmt='.',color='grey',elinewidth=1)
plt.xlabel('normalised expected output (divided by 15)')
plt.ylabel('normalised square error')
plt.tick_params(labelsize='small')
plt.subplot(223)
plt.errorbar(nor_tar,x_mean,yerr=x_std,fmt='.',color='grey',elinewidth=1)
plt.axis([0.3, 0.7,0.3, 0.7])
plt.gca().set_aspect('equal', adjustable='box') # square axis
plt.xlabel('normalised expected output')
plt.ylabel('normalised network output') 
plt.tick_params(labelsize='small')
plt.subplot(224)
plt.errorbar(nor_tar,e_mean,yerr=e_std,fmt='.',color='grey',elinewidth=1)
plt.xlabel('normalised expected output')
plt.ylabel('normalised error')
plt.tick_params(labelsize='small')
plt.show()

'''subplots without trnerr vs epoch'''
subplots_adjust(hspace=0.4,wspace=0.4)
plt.figure(1)
plt.subplot(222)
plt.errorbar(nor_tar,se_mean,yerr=se_std,fmt='.',color='grey',elinewidth=1)
plt.xlabel('normalised expected output (divided by 15)')
plt.ylabel('normalised square error')
plt.tick_params(labelsize='small')
plt.subplot(223)
plt.errorbar(nor_tar,x_mean,yerr=x_std,fmt='.',color='grey',elinewidth=1)
plt.axis([0.3, 0.7,0.3, 0.7])
plt.gca().set_aspect('equal', adjustable='box') # square axis
plt.xlabel('normalised expected output')
plt.ylabel('normalised network output') 
plt.tick_params(labelsize='small')
plt.subplot(224)
plt.errorbar(nor_tar,e_mean,yerr=e_std,fmt='.',color='grey',elinewidth=1)
plt.xlabel('normalised expected output')
plt.ylabel('normalised error')
plt.tick_params(labelsize='small')
plt.show()