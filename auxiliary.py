#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
auxiliaries of trial.py to evaluate network

"""

'''training data augmentation'''
trnslos = np.empty((6000,72*n_frame))
for i in range(5,11):
    h = fits.open('trnslos_20180202_%i.fits'%i)
    trnslos[(i-5)*1000:(i-4)*1000] = h[0].data
    
'''open training slopes'''
h = fits.open('trnslos_20180202.fits')
trnslos_all = h[0].data
trnslos = trnslos_all[:,:72*n_frame]

'''return actual output and target'''
trnout, trntarg = t.outPut(trnData)
vdout, vdtarg = t.outPut(vdData)
allout = t.outPut(trnds,return_targets=False)
trnres = np.array([trnout,trntarg])
vdres = np.array([vdout,vdtarg])
fits.writeto('trnres_20_40_0.001_200(1).fits',trnres)
fits.writeto('vdres_20_40_0.001_200(1).fits',vdres)
fits.writeto('allout_20_40_0.001_200(1).fits',allout)
fits.writeto('tverr_20_40_0.001_200(1).fits',np.array([trnerr,vderr]))

def errorCal(output, target): 
    '''
    calculate mean and std of actual outputs, error and square error (all grouped          according to different targets)
    
    Parameters
    ------
    output : network actual outputs
    target : corresponging targets
    
    outputs:
        x_mean, x_std: mean and std of actual outputs
        e_mean, e_std: mean and std of error
        se_mean, se_std: mean and std of square error
        rg: range of output
        bonds, hists: x and y axis of histogram of actual outputs
    '''
    counts = [target.count(i/15) for i in range(5,11)]   
    output = np.array(output)
    targer = np.array(target)
    rg = (output.min(),output.max())
    bs = 10
    hists = np.empty((6,bs))
    bonds = np.empty((1,bs))
    x_mean = np.empty(6)
    x_std = np.empty(6)
    e_mean = np.empty(6)
    e_std = np.empty(6)
    se_mean = np.empty(6)
    se_std = np.empty(6)
    start = 0
    i = 0
    for stride in counts:
        end = start + stride
        data = output[start:end]
        x_mean[i] = np.mean(data)
        x_std[i] = np.std(data)   
        e = data - target[start:end]
        e_mean[i] = np.mean(e)
        e_std[i] = np.std(e)
        se = e**2
        se_mean[i] = np.mean(se)
        se_std[i] = np.std(se) 
        hists[i],temp,_ = plt.hist(data,bins=bs,range=rg)
        plt.close()
        start = end
        i += 1
    bonds[0] = [(temp[j]+temp[j+1])/2 for j in range(10)]
    return x_mean, x_std, e_mean, e_std, se_mean, se_std, rg, hists, bonds

'''subplots'''
nor_tar = np.arange(5,11)/15.0
subplots_adjust(hspace=0.8,wspace=0.2)
plt.subplot(311)
total_epoch = 200
trnerr_2 = np.array(trnerr)*2 
vderr_2 = np.array(vderr)*2 
plt.loglog(trnerr_2[:total_epoch+1])
plt.loglog(vderr_2[:total_epoch+1])
plt.legend(['training error', 'validation error'],fontsize='x-small')
plt.xlabel('epoch')
plt.tick_params(labelsize='small')
plt.subplot(323)
plt.errorbar(nor_tar,se_mean,yerr=se_std,fmt='.',color='grey',elinewidth=1)
plt.xlabel('normalised expected output (divided by 15)')
plt.ylabel('normalised square error')
plt.tick_params(labelsize='small')
plt.subplot(324)
plt.errorbar(nor_tar,e_mean,yerr=e_std,fmt='.',color='grey',elinewidth=1)
plt.xlabel('normalised expected output')
plt.ylabel('normalised error')
plt.tick_params(labelsize='small')
plt.subplot(325)
plt.errorbar(nor_tar,x_mean,yerr=x_std,fmt='.',color='grey',elinewidth=1)
plt.axis([0.3, 0.7, 0.3, 0.7])
plt.gca().set_aspect('equal', adjustable='box') # square axis
plt.xlabel('normalised expected output')
plt.ylabel('normalised network output') 
plt.tick_params(labelsize='small')
plt.subplot(326)
plt.plot(bonds.T,hists.T)
plt.title('distribution of outputs')
plt.xlabel('actual output')
plt.legend(['v=5','v=6','v=7','v=8','v=9','v=10'],fontsize='x-small')
plt.tick_params(labelsize='small')
plt.show()

    
'''error bar of error and square error of all v vs expected output'''
mean_all = np.empty(3)
std_all = np.empty(3)
i = 0
for lr in [0.0001,0.0005,0.0010]:
    out_trn = fits.open("outtrn_20_500_%.4f_200.fits"%lr)[0].data
    mse_all = (out_trn - trn_tar/15)**2
    mean_all[i] = np.mean(mse_all)
    std_all[i] = np.std(mse_all)
    i += 1
plt.errorbar(np.log10(np.array([0.0001,0.0005,0.0010])),mean_all,yerr=std_all,fmt='.',color='grey',elinewidth=1)
plt.xlabel('learning rate')
plt.ylabel('training MSE')
plt.text(500,0.05, '# frames per input entry = 20')
plt.tick_params(labelsize='small')

