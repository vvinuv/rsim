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

        
def plotting(outdir, outfile, pixel_scale, size, figure, label, color, ms=8):
    ofile = os.path.join(outdir, outfile)
    f = np.load(ofile)
    theta = f['theta']
    xi_a = f['xi']
    if len(xi_a.shape) > 1:
        xi = xi_a.mean(axis=0)
        xie = xi_a.std(axis=0)
        N = xi_a.shape[0]
        xie /= np.sqrt(N)
    else:
        xi = xi_a.copy()
        xie = 0.0
        N = 1.0
    pl.figure(figure)
    ax = pl.subplot(111)
    pl.errorbar(theta, xi, xie, c=color, marker='o', ms=ms, label=label)
    return ax, theta, N

def plotting_cc_group(outdir, outnpz, mfrac, Fruns, pixel_scale, size, \
                      figure, label, color, ms=8):
    xi_a = []
    for fi in range(Fruns):
        ofile = os.path.join(outdir, outnpz%(fi, mfrac))
        if not os.path.exists(ofile):
            continue
        f = np.load(ofile)
        theta = f['theta']
        if len(xi_a) < 1:
            xi_a = f['yprof']
        else:
            xi_a = np.row_stack((xi_a, f['yprof']))
    if len(xi_a.shape) > 1:
        xi = xi_a.mean(axis=0)
        xie = xi_a.std(axis=0)
        N = xi_a.shape[0]
        xie /= np.sqrt(N)
    else:
        xi = xi_a.copy()
        xie = 0.0
        N = 1.0
    pl.figure(figure)
    ax = pl.subplot(111)
    pl.errorbar(theta, xi, xie, c=color, marker='o', ms=ms, label=label)
    return ax, theta, N

def plot_battaglia(color, scale):
    z = 0.1
    cosmo = {'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, \
             'h' : 0.72, 'omega_k_0':0.0}
    DA = cd.comoving_distance(z, **cosmo) / (1 + z) #Mpc
    ang_dis = (np.pi/180/3600.) * DA  #Mpc/arcsec
    ang_dis *= 60. #Mpc/arcmin

    f = np.genfromtxt(\
        '/home/vinu/Lensing/DES/SZ/yproj_m14_batt_zw_0.1_cofm.txt')
    pl.plot(f[:,0]/ang_dis, f[:,3]/scale, lw=2, ls='-', 
            label='Battaglia z=0.1 logM=14',c=color)
    #f = np.genfromtxt('/home/vinu/Lensing/DES/SZ/test_ytheta.txt')
    #pl.plot(f[:,0]/ang_dis, f[:,1], lw=2, ls='-', label='From Cl',c='r')

def plot_yy_theta(color):
    z = 0.1
    cosmo = {'omega_M_0' : 0.3, 'omega_lambda_0' : 0.7, \
             'h' : 0.72, 'omega_k_0':0.0}
    DA = cd.comoving_distance(z, **cosmo) / (1 + z) #Mpc
    ang_dis = (np.pi/180/3600.) * DA  #Mpc/arcsec
    ang_dis *= 60. #Mpc/arcmin
    f = np.genfromtxt('/home/vinu/Lensing/DES/SZ/test_yy_theta.txt')
    pl.plot(f[:,0]/ang_dis, f[:,1], lw=2, ls='-',
            label='yy',c=color)

def plot_w_of_theta(t, theta0, color):
    pl.plot(t, (t/theta0)**-0.8, c=color, 
            label=r'$\left(\frac{\theta}{%d}\right)^{-0.8}$'%theta0)

def plot_set(ax, ftype, pixel_scale, size, Nreal):
    pl.text(0.1, 0.4, r'$\Delta\theta=%d$ arcmin'%pixel_scale, 
            transform=ax.transAxes)
    pl.text(0.1, 0.3, 'size=%d pixels'%size, transform=ax.transAxes)
    pl.text(0.1, 0.2, '# of realizations=%d'%Nreal, transform=ax.transAxes)

    pl.legend(loc=0)
    pl.xscale('log')
    pl.yscale('log', nonposy='clip')
    if ftype == 'gg':
        pl.ylabel(r'$w(\theta)$')
    elif ftype == 'gy':
        pl.ylabel(r'$\xi(\theta)_{yy}$')
    elif ftype == 'cc':
        pl.ylabel(r'$\xi(\theta)_{gy}$')
    elif ftype == 'group':
        pl.ylabel(r'$Y(\theta)$')
    else:
        raise ValueError('Unknown type')
    pl.xlabel('R (arcmin)')

