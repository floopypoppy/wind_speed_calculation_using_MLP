#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
auxiliaries of trial.py to evaluate network

"""

'''training data augmentation'''
n_frame = 100
trnslos = np.empty((5000,72*n_frame))
for i in range(5,10):
    h = fits.open('trnslos_20180308_%i.fits'%i)
    trnslos[(i-5)*1000:(i-4)*1000] = h[0].data
header = fits.Header()
header["r_0"] = str([0.16])
header["WINDSPD"] = str([5,6,7,8,9])
header["WINDDIR"] = str([0])
header["SAVETIME"] = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
header["ITERS"] = str([1000])
header['ITERTIME'] = str([0.012])
fits.writeto('trnslos_20180308.fits',trnslos)

'''open training slopes'''
h = fits.open('trnslos_20180202.fits')
trnslos_all = h[0].data


'''a new sequence of subtracts'''
for i in range(29,0,-1):
    norm_inp[:,i*72:(i+1)*72] -= norm_inp[:,(i-1)*72:i*72]
    
norm_inp = norm_inp[:,72:]

'''return actual output and target'''
trnout, trntarg = t.outPut(trnData)
vdout, vdtarg = t.outPut(vdData)
allout = t.outPut(trnds,return_targets=False)
resdic = {'trn_output' : trnout,
          'trn_targ' : trntarg,
          'vd_output' : vdout,
          'vd_targ' : vdtarg,
          'all_output' : allout,
          'trnerr' : trnerr,
          'vderr' : vderr}
outfile = open('30_200_0.001_100_sub.pkl','wb') 
# fmt: '#frames_#hiddennodes_lr_maxepochs.pkl'
pickle.dump(resdic, outfile)
outfile.close()


def outDist(output, target): 
    '''
    calculate distribution of actual output (all grouped according to different targets)
    
    Parameters
    ------
    output : network actual outputs
    target : corresponging targets
    
    outputs:
        rg: range of output
        bonds, hists: x and y axis of histogram of actual outputs
    '''
    counts = [target.count(i/15.0) for i in range(5,11)]   
    output = np.array(output)
    targer = np.array(target)
    rg = (output.min(),output.max())
    bs = 10
    hists = np.empty((6,bs))
    bonds = np.empty((1,bs))
    x_mean = np.empty(6)
    x_std = np.empty(6)
    start = 0
    i = 0
    for stride in counts:
        end = start + stride
        data = output[start:end]
        x_mean[i] = np.mean(data)
        x_std[i] = np.std(data)
        hists[i],temp,_ = plt.hist(data,bins=bs,range=rg)
        plt.close()
        start = end
        i += 1
    bonds[0] = [(temp[j]+temp[j+1])/2 for j in range(bs)]
    return rg, hists, bonds, x_mean, x_std


'''subplots'''
total_epoch = 100
trnrg, trnhists, trnbonds, trnx_mean, trnx_std = outDist(trnout,trntarg)
vdrg, vdhists, vdbonds, vdx_mean, vdx_std = outDist(vdout,vdtarg)
fig = plt.figure()
gs = gridspec.GridSpec(3 ,3)
subplots_adjust(hspace=0.9)
ax1 = plt.subplot(gs[0,:])
dummy = 2*np.array(trnerr)
plt.loglog(dummy[:total_epoch+1])
dummy = 2*np.array(vderr)
plt.loglog(dummy[:total_epoch+1])
plt.legend(['training set error', 'validation set error'],fontsize='x-small')
plt.xlabel('epoch')
plt.ylabel('MSE error')
plt.title('training errors')
plt.tick_params(labelsize='small')
ax2 = plt.subplot(gs[1,:2])
plt.plot(trnbonds.T,trnhists.T)
plt.title('distribution of training set output',fontsize=11)
plt.legend(['v=5','v=6','v=7','v=8','v=9','v=10'],fontsize='x-small',ncol=2)
plt.tick_params(labelsize='small')
nor_tar = np.arange(5,11)/15.0
ax3 = plt.subplot(gs[1,2])
plt.errorbar(nor_tar,trnx_mean,yerr=trnx_std,fmt='.',color='grey',elinewidth=1)
plt.axis([0.3, 0.7, 0.3, 0.7])
plt.gca().set_aspect('equal', adjustable='box') # square axis
plt.xlabel('normalised expected output')
plt.ylabel('normalised training set output') 
plt.tick_params(labelsize='small')
ax4 = plt.subplot(gs[2,:2], sharex=ax2)
plt.plot(vdbonds.T,vdhists.T)
plt.title('distribution of validation set output',fontsize=11)
plt.legend(['v=5','v=6','v=7','v=8','v=9','v=10'],fontsize='x-small',ncol=2)
plt.tick_params(labelsize='small')
ax5 = plt.subplot(gs[2,2])
plt.errorbar(nor_tar,vdx_mean,yerr=vdx_std,fmt='.',color='grey',elinewidth=1)
plt.axis([0.3, 0.7, 0.3, 0.7])
plt.gca().set_aspect('equal', adjustable='box') # square axis
plt.xlabel('normalised expected output')
plt.ylabel('normalised validation set output') 
plt.tick_params(labelsize='small')

