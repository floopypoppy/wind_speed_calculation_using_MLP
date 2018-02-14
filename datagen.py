"""
generate all slopes and wind info in certain steps for training

"""

import numpy as np
import datetime
from astropy.io import fits
import soapy
from matplotlib import pyplot as plt

allslopes = np.empty((1000,3600))
sim = soapy.Sim("./conf/sh_77.yaml")

for j in range(1000):
#    sim.config.atmos.windSpeeds = [v]
    sim.aoinit()
    sim.aoloop()
    allslopes[j] = sim.allSlopes.reshape(-1)   

header = fits.Header()
header["r_0"] = str([0.16])
header["WINDSPD"] = str([5,6,7,8,9,10])
header["WINDDIR"] = str([0])
header["SAVETIME"] = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
header["ITERS"] = str([50])
header['ITERTIME'] = str([0.012])
fits.writeto("trnslos_20180202.fits",trnslos,header,overwrite=True)
