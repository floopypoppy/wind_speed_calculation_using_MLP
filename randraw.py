#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
N samples on the same phase screen with random speed & direction & starting points

"""
import numpy as np
import datetime
from astropy.io import fits
import soapy
from matplotlib import pyplot as plt
from numpy.random import uniform
#from random import randint
from numpy import sin, cos, pi

sim = soapy.Sim("../soapy-master/conf/sh_77.yaml")
sim.aoinit()
 
scrnsize = 1024 
nb = 1000
seqlen = 100
masksize = 130
allslopes = np.empty((nb, 72*seqlen))
#looptime = sim.atmos.looptime
#pxscale = sim.atmos.pixel_scale

vs = uniform(5,10,nb)
dirs = uniform(0,360,nb)
for i in range(nb):
    sim.config.atmos.windSpeeds = [vs[i]]
    sim.config.atmos.windDirs = [dirs[i]]
#    windV = (vs[i]*np.array([cos(dirs[i]),sin(dirs[i])])).T
#    windV = windV*looptime/pxscale
#    sim.atmos.windV[0] = windV
    sim.aoinit()
    while True:
        xpos = uniform(0,scrnsize-masksize)
        xcord = xpos+sim.atmos.windV[0][0]*seqlen
        if xcord >= 0 and xcord+masksize < scrnsize:
            break
    while True:
        ypos = uniform(0,scrnsize-masksize)
        ycord = ypos+sim.atmos.windV[0][1]*seqlen
        if ycord >= 0 and ycord+masksize < scrnsize:
            break
    sim.atmos.scrnPos[0] = np.array([xpos,ypos])
    sim.atmos.xCoords[0] = np.arange(masksize).astype('float') + xpos
    sim.atmos.yCoords[0] = np.arange(masksize).astype('float') + ypos
    sim.aoloop()
    allslopes[i] = sim.allSlopes.reshape(-1)
    

header = fits.Header()
header["r0"] = str(0.16)
header["ITERS"] = str(1000)
header['SEQLEN'] = str(100)
header['ITERTIME'] = str(0.012)
fits.writeto("trnslos_20180504_34.fits",allslopes,header,overwrite=True)

tar = np.empty((nb,4))
tar[:,0] = vs
tar[:,1] = cos(dirs/180*pi)
tar[:,2] = sin(dirs/180*pi)
tar[:,3] = dirs
fits.writeto("tar_20180504_34.fits",tar,header,overwrite=True)

