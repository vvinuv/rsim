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
import corr_utils as cu
import plot_utils as pu
import gen_fields as gf
import utils as ut

def generate_gg_gy_fields(Fruns, field_dir, avg, size, pixel_scale, 
                          theta0, gg_scale, pid,
                          plot_ps=False):
    '''Generating fields for Fruns times on flat sky.
       field_dir: Files will be saved here
       avg: Average of generated field
       size: size of the field in pixels
       pixel_scale: size of pixel in arcmin
       theta0: Correlation scale of w(theta)
       Output: ggfield_%s.fits & gyfield_%s.fits with WCS 
    '''
    for fi in range(Fruns):
        f_gg = os.path.join(field_dir, 'ggfield_%s.fits'%fi)
        f_gy = os.path.join(field_dir, 'gyfield_%s.fits'%fi)
        if os.path.exists(f_gg) and os.path.exists(f_gg) and 1:
            pass
        else:
            print 'Generating fields %d'%fi
            ggfield, gyfield = gf.generate_random_fields(avg, size, pixel_scale,
                                                     theta0, gg_scale, 
                                                     cross_power=True,
                                                     do_plot=False, 
                                                     plot_ps=plot_ps)
            wcs = WCS(naxis=2)
            wcs.wcs.crpix = [0.0, 0.0]
            wcs.wcs.crval = [0.0, 0.0]
            dra = pixel_scale / 60.
            ddec = pixel_scale / 60.
            wcs.wcs.cdelt = [dra, ddec] #deg/pixel
            wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
            header = wcs.to_header()
            fits.writeto(f_gg, 
                         ggfield.real.astype(np.float32), 
                         header, clobber=True)
            fits.writeto(f_gy, 
                         gyfield.real.astype(np.float32), 
                         header, clobber=True)

def RunMany(ftype, Fruns, outfile, theta_mod, pixel_scale, minusone, field_dir, 
            outdir, pid, mask):
    '''Running auto and cross power many times'''

    xi_a = []
    for fi in range(Fruns): 
        print 'Field %s Run# %d'%(ftype, fi)
        if ftype == 'gg' or ftype == 'gy':
            theta, xi, xie = cu.run_autoc(fi, pixel_scale, ftype, minusone, 
                                       field_dir, outdir, pid, mask)
        elif ftype == 'cc':
            theta, xi, xie = cu.run_crossc(fi, pixel_scale, minusone, field_dir, 
                                        outdir, pid, mask)
        else:
            raise ValueError('Unknown field')
        spl = interpolate.splrep(theta, xi)
        xi_i = interpolate.splev(theta_mod, spl)
        if len(xi_a) < 1:
            xi_a = xi_i.copy()
        else:
            xi_a = np.row_stack((xi_a, xi_i))

    np.savez(os.path.join(outdir, outfile), theta=theta_mod, xi=xi_a)

if __name__=='__main__':

    nn = int(sys.argv[1])
 
    pid = os.getpid()
    avg = 0.
    size = 4000.
    theta0 = 30 # arcmin
    gg_scale = 1.0 #250.0 # the factor by which gg-power spectrum will be scaled
    pixel_scale = 1/1. # arcmin
    Fruns = 2 # No of times ggfield and gyfield is simulated
    Nruns = 5 # No of times group field is created
    field_dir = 'sims_results' #Simulated fields will be stored here
    outdir = 'sims_results/tmask' #the correlation output will be saved here
    random_pos = 0
    theta_mod = np.logspace(np.log10(1), np.log10(500),30)

    mfrac = 0.0 #fraction of masked region
   
    if not os.path.exists(field_dir):
        os.mkdir(field_dir)
    
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    mask = my.generate_random_rectangle_mask(size, 200, mfrac, rseed=1024, \
                                             plot_mask=0)
    #
    generate_gg_gy_fields(Fruns, field_dir, avg, size, pixel_scale, theta0,\
                          gg_scale, pid, plot_ps=False)
    if nn == 1:
        # ac_gg_nsign.npz: Means do not set pixels < -1 to -1
        RunMany('gg', Fruns, 'ac_gg_nsign_m%.1f.npz'%mfrac, theta_mod, 
                 pixel_scale, False, field_dir, outdir, pid, mask) #1
    elif nn == 2:
        # ac_gg_nsign.npz: Means set pixels < -1 to -1
        RunMany('gg', Fruns, 'ac_gg_sign_m%.1f.npz'%mfrac, theta_mod, 
                 pixel_scale, True, field_dir, outdir, pid, mask) #2
    elif nn == 3:
        print nn
        RunMany('gy', Fruns, 'ac_gy_m%.1f.npz'%mfrac, theta_mod, pixel_scale, 
                 False, field_dir, outdir, pid, mask) #3
        
    elif nn == 4:
        RunMany('cc', Fruns, 'cc_gy_nsign_m%.1f.npz'%mfrac, theta_mod, 
                 pixel_scale, False, field_dir, outdir, pid, mask) #4
    elif nn == 5:
        RunMany('cc', Fruns, 'cc_gy_sign_m%.1f.npz'%mfrac, theta_mod, 
                 pixel_scale, True, field_dir, outdir, pid, mask) #5
    elif nn == 6:
        RunMany('gg', Fruns, 'ac_gg_nsign_m%.1f.npz'%mfrac, theta_mod, 
                 pixel_scale, False, field_dir, outdir, pid, mask) #1
        RunMany('gy', Fruns, 'ac_gy_m%.1f.npz'%mfrac, theta_mod, pixel_scale,
                 False, field_dir, outdir, pid, mask) #3
        RunMany('cc', Fruns, 'cc_gy_nsign_m%.1f.npz'%mfrac, theta_mod, 
                 pixel_scale, False, field_dir, outdir, pid, mask) #4

        for fi in range(Fruns):
            outnpz = 'group_yprof_%s_m%.1f.npz'%(fi, mfrac)
            print '#Fields %s'%fi 
            # minusone = True, removes groups belongs to pixels with values
            # < -1
            cu.generate_groups_crossc(pixel_scale, pid, Nruns, fi, 
                                  field_dir, outdir,
                                  theta_mod, outnpz, True, random_pos, mask) #6
    elif nn == 7:
        pass
    else:
        raise ValueError('Unknown nn')
    
    if 1:
        ax, theta, Nreal = pu.plotting(outdir, 'ac_gg_nsign_m%.1f.npz'%mfrac, \
                           pixel_scale, size, 1, r'$w(\theta)$ [< -1]', 'g')
        pu.plot_w_of_theta(theta, theta0, 'r')
        pu.plot_set(ax, 'gg', pixel_scale, size, Nreal)

        #ax, theta, Nreal = pu.plotting(outdir, 'ac_gg_sign_m%.1f.npz'%mfrac, \
        #                   pixel_scale, size, 1, r'$w(\theta)$', 'b')
        #pu.plot_w_of_theta(theta, theta0, 'r')
        #pu.plot_set(ax, 'gg', pixel_scale, size, Nreal)

        ax, theta, Nreal = pu.plotting(outdir, 'ac_gy_m%.1f.npz'%mfrac, \
                           pixel_scale, size, 2, r'$\xi(\theta)_{yy}$', 'g')
        pu.plot_yy_theta('k')
        pu.plot_set(ax, 'gy', pixel_scale, size, Nreal)

        ax, theta, Nreal = pu.plotting(outdir, 'cc_gy_nsign_m%.1f.npz'%mfrac,\
                           pixel_scale, size, 3, r'$\xi_{gY}$ [<-1]', 'g')
        #ax, theta, Nreal = pu.plotting(outdir, 'cc_gy_sign_m%.1f.npz'%mfrac, \
        #                   pixel_scale, size, 3, r'$\xi_{gY}$ [>-1]', 'b')
        pu.plot_battaglia('k', 1. * np.sqrt(gg_scale))
        pu.plot_set(ax, 'cc', pixel_scale, size, Nreal)

        ax, theta, Nreal = pu.plotting_cc_group(outdir, \
                           'group_yprof_%s_m%.1f.npz', mfrac, Fruns, \
                           pixel_scale, size, 4, 'Group', 'r')
        pu.plot_battaglia('k', np.sqrt(gg_scale))
        pu.plot_set(ax, 'group', pixel_scale, size, Nreal)

        pl.show() 
