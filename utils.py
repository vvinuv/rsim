import os
import sys
import numpy as np
import numpy.random as random
from scipy import fftpack
import pylab as pl
from scipy import interpolate
import mytools as my
from astropy.io import fits
from astropy.wcs import WCS
import cosmolopy.distance as cd


def RadialProf(X):

    xd, yd = X.shape
    x = np.arange(xd)
    y = np.arange(yd)
    x, y = np.meshgrid(x, y)

    r = np.sqrt((x-(xd/2. - 1))**2. + (y-(yd/2. - 1))**2.).T
    print xd/2., yd/2., r.max()

    #pl.imshow(r)
    #pl.show()
    Prof = []
    Profe = []
    r_arr = []
    n_arr = []
    for i in range(0, int(r.max()), 1):
        con = (r < i + 1) & (r >= i) & (X != 0)
        r_arr.append(i + 0.5)
        Prof.append(np.average(X[con]))
        Profe.append(np.std(X[con]) / np.sqrt(con[con].shape[0]))
        n_arr.append(con[con].shape[0])
    return np.array(r_arr), np.array(Prof), np.array(Profe), np.array(n_arr)


def InstrumentNoise(imap, ipath, smooth, sigma_T, nu):
    '''Inputs are 
       1. imap : input map
       2. ipath : path to imap
       3. smooth : Gaussian smoothing size in pixels
       4. sigma_T : Gaussian noise in microK
       5. nu : the frequency of observation in GHz
 
       Output : Smoothed map with added Gaussian noise
    '''
    k = 1.3806e-23 #m^2 kg s^-2 K^-1
    h = 6.626e-34 #m^2 kg s^-1
    T_cmb = 2.725
    nu *= 1e9
    x = h * nu / (k * T_cmb)
    g_x = x * (1 / np.tanh(x/2.)) - 4.
    #delta_T = g(x) * y * T_cmb -> sigma_T = g(x) * sigma_y * T_cmb
    sigma_y = sigma_T * 1e-6 / g_x / T_cmb 
    sm_map = nd.gaussian_filter(os.path.join(ipath, imap), smooth)
    omap = np.random.normal(sm_map, scale=sigma_y)
    return omap
 

