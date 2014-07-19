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

def get_cls(do_plot=False):
    '''Return auto and cross Cls (interpolated) from Adam.'''
    #Auto power spectrum
    fa = np.genfromtxt('cls_tsz_battaglia_pressure.txt')
    lamax = fa[:,0].max()
    lamin = fa[:,0].min()
    #Cross power spectrum
    fc = np.genfromtxt('cl_m14_batt_zw_0.1.txt')
    lcmax = fc[:,0].max()
    lcmin = fc[:,0].min()

    spl_cross = interpolate.splrep(fc[:,0], \
                                   fc[:,1] * 2 * np.pi / fc[:,0] / (1+fc[:,0]))
    spl_auto = interpolate.splrep(fa[:,0], \
                                  fa[:,1] * 2 * np.pi / fa[:,0] / (1+fa[:,0]))
    ls = np.arange(1, 100, 1)
    if do_plot:
        pl.subplot(121)
	pl.plot(fa[:,0], \
             fa[:,3] * 2 * np.pi / fa[:,0] / (1+fa[:,0]), c='g', label='Auto')
	pl.plot(ls, interpolate.splev(ls, spl_auto), \
                c='g', label='Auto interpolated', ls='--')
	pl.plot(fc[:,0], \
            fc[:,1] * 2 * np.pi / fc[:,0] / (1+fc[:,0]), c='r', label='Cross')
	pl.plot(fc[:,0], fc[:,0]**(-1.2), c='k', label='gg')
	pl.legend(loc=0)
        pl.xscale('log')
        pl.yscale('log')
	
        pl.subplot(122)
        pl.plot(fa[:,0], \
              fa[:,3] * 2 * np.pi / fa[:,0] / (1+fa[:,0]), c='g', label='Auto')
        l = np.linspace(lamin, lamax, 1000)
        pl.plot(l, interpolate.splev(l, spl_auto), c='r', label='model')
        pl.xscale('log')
        pl.yscale('log')
	pl.show()

    return spl_auto, lamax, lamin, spl_cross, lcmax, lcmin

def generate_random_fields(avg, size, pixel_scale, theta0, gg_scale,
                           cross_power=True, 
                           do_plot=False, plot_ps=False):
    '''Generating random field based on the power spectrum. 
       size: dimension of the field in pixels. Output field will be size X size
       pixel_scale: size of pixels in arcmin
       avg: average value in the simulated field
       pixel_scale = size of a pixel in arcmin
       theta0 = Correlation length in arcmin (i.e. when two point correlation
                becomes 1)
       gg_scale = the gg power will be scaled by this factor. This is to try to
       keep the gg field from going less than -1. i.e. this will reduce the 
       fluctuations in the gg-field 
       cross_power=True, Generate field with cross correlation  
       do_plot: Whether plot some analytical power spectra
       plot_ps: Plots input and simulated power spectra. Useful for testing
    '''
    gg_scale = 1.
    # Size of the field in radian. The following factor is important
    Ndt = size * (pixel_scale / 60.) * np.pi / 180. #radian
    Ndtsq = Ndt * Ndt #steradian
    #Ndt, Ndtsq = 1., 1.
    # Two random fields anf its fft are generated
    r1, r1_ft = my.generate_random_field((size,size), 1.0, avg=avg)
    r2, r2_ft = my.generate_random_field((size,size), 1.0, avg=avg)

    # Get power spectrum from Adam, i.e. P_YY and P_dY
    spl_auto, lamax, lamin, spl_cross, lcmax, lcmin = get_cls(do_plot=0) 

    # 2d l - array
    l = my.l_array(r1, pixel_scale, do_plot=0) 
    con_yy = (l<0) #(l < 1) | (l > lamax)

    # This is the SZ auto correlation
    cl_yy = interpolate.splev(l.ravel(), spl_auto).reshape(l.shape)
    cl_yy[con_yy] = 0.0

    # Converting power spectrum into fourier coeffients
    a22 = np.sqrt(cl_yy / Ndtsq)
    a22[con_yy] = 1.0

    # Power spectrum of power law correlation between groups
    cl_gg = my.angular_power_spectrum((theta0/60.)*np.pi/180., l) / gg_scale

    # The cross power spectrum between groups and Y
    #con_gy = (l < 1) | (l > lcmax)
    con_gy = (l < 0)
    cl_gy = interpolate.splev(l.ravel(), spl_cross).reshape(l.shape)
    cl_gy[con_gy] = 0.0
    cl_gy /= np.sqrt(gg_scale) #REMOVE
    if not cross_power:
        cl_gy = cl_gy * 0.0

    tdl = (2 * np.pi / Ndt) 
    tl = np.arange(l.min(), l.max(), tdl)
    t_gg = np.sum(my.angular_power_spectrum((theta0/60.)*np.pi/180., tl) / gg_scale \
                  * tl) * tdl / (2 * np.pi)
    t_yy = np.sum(interpolate.splev(tl, spl_auto) * tl) * tdl / (2 * np.pi)
    t_gy = np.sum(interpolate.splev(tl, spl_cross) * tl) * tdl / (2 * np.pi)
    print 'Total input power gg=%.1e, gy=%.1e, yy=%.1e'%(t_gg, t_yy, t_gy)

    c_ = (cl_yy == 0.0)
    cl_yy[c_] = 1.
    a12 = cl_gy / np.sqrt(cl_yy) / Ndt
    a12[c_] = 1.0
    a11_sq = (cl_gg - cl_gy**2 / cl_yy) / Ndtsq
    #Remove the following line. REMOVE
    #a11_sq[a11_sq < 0] = 0
    a11 = np.sqrt(a11_sq)
    a11[c_ | (l < 1) | con_gy] = 1.0
    
    cl_yy[c_] = 0.0    
 
    if do_plot:
        # This plots different power spectra
        ls = np.arange(1000)
        pl.figure(1)
        pl.subplot(211)
        pl.semilogy(ls, my.angular_power_spectrum(theta0/60.*np.pi/180., ls),
                    c='r', label=r'$w(\theta)$')
        pl.semilogy(ls, interpolate.splev(ls, spl_cross)**2, c='g', 
                    label='gy')
        pl.semilogy(ls, interpolate.splev(ls, spl_auto), c='k',
                    label='yy')
        pl.legend(loc=0)
        pl.subplot(212)
        for j in [0.1, 1, 10, 100]:
            pl.loglog(ls, my.angular_power_spectrum(j/60.*np.pi/180., ls), 
                      ls='--', label=r'$\theta0=%.1f'%j)
        pl.loglog(ls, my.angular_power_spectrum(theta0/60.*np.pi/180., ls), 
                  c='r', label=r'$w(\theta)$')
        pl.loglog(ls, interpolate.splev(ls, spl_cross)**2/\
                      interpolate.splev(ls, spl_auto), c='g')
        pl.legend(loc=0)
        #pl.show()
  
    ''' 
    print r1_ft.size
    xx, gg_1d, gg_1de, yy = RadialProf(fftpack.fftshift(r1_ft))
    pl.plot(xx, gg_1d, label='Input gg', c='r') 
    xx, gg_1d, gg_1de, yy = RadialProf(fftpack.fftshift(r2_ft))
    pl.plot(xx, gg_1d, label='Input gg', c='g') 
    pl.show()
    '''

    #Scipy/numpy FFT has the following convention
    #>>> a = np.arange(10)
    #>>> aft = np.fft.fft(a)
    #>>> np.sum(abs(aft)**2.) : 2850
    #>>> np.sum(a**2) : 285
    #Therefore, accoring to Parseval's theorem aft needs to divide by sqrt(10),
    #which is the size of the array.
    #>>> aft /= np.sqrt(10)
    #>>> np.sum(abs(aft)**2.) : 285
    #Now if you want to convert aft back to a, then you can do either
    #1. Multiply by aft by sqrt(10) and then ifft(aft) OR
    #2. ifft(aft) * np.sqrt(10)

    #IMPORTANT: In scipy/numpy when you do parseval theorem, you should check 
    #whether np.sum(a**2) equals np.sum(abs(aft)**2.) / size and FFT is
    #given by aft / np.sqrt(size)

    # First field
    r1_ft = a11 * r1_ft + a12 * r2_ft
    #print 'UnNormed ', np.sum(abs(r1_ft)**2.)

    Norm1 = np.sqrt(np.sum(abs(r1_ft*Ndt)**2.) / t_gg)
    r1_ft /= Norm1
    #print 'Normed ', np.sum(abs(r1_ft*Ndt)**2.)
    # Normalize according to numpy convetion. The reason why r1_ft.size instead
    # of sqrt(r1_ft.size) is that r1_ft should be also multiplied by 
    # sqrt(r1_ft.size) and a11, a12 coefficients are based on power spectra
    # which also need to multiply by sqrt(r1_ft.size). The reason Ndt is
    # muliplied is that the normalization Norm1 is found for r1_ft*Ndt
    r1_ft = r1_ft * r1_ft.size * Ndt #This is true FFT
    # Invert FFT to get the random field with auto correlation and multiply
    # by sqrt(# of points) to get the actual values
    r1_ift = fftpack.ifft2(r1_ft).real 

    #print np.sum( abs(r1_ft)**2.), np.sum( abs(r1_ift)**2.)
    # second field
    r2_ft *= a22 
    Norm2 = np.sqrt(np.sum(abs(r2_ft*Ndt)**2.) / t_yy)
    r2_ft /= Norm2
    #print 'Normed ', np.sum(abs(r2_ft*Ndt)**2.)
    # Normalize according to numpy convetion
    r2_ft = r2_ft * r2_ft.size * Ndt 
    # Invert FFT to get the random field with auto correlation
    r2_ift = fftpack.ifft2(r2_ft).real 
    
    print 'Checking Output and input powers (Parseval theorem'
    print 'Type Real FFT Input'
    print 'gg %.2e %.2e %.2e (Input))'%(np.sum(r1_ift**2.)/r1_ft.size, \
           np.sum(abs(r1_ft)**2.)/r1_ft.size**2, t_gg)
    print 'yy %.2e %.2e %.2e (Input)'%(np.sum(r2_ift**2.)/r1_ft.size, \
           np.sum(abs(r2_ft)**2.)/r1_ft.size**2, t_yy)
    print 'gy(r) %.2e %.2e %.2e (Input)'%(\
           np.sum(r1_ift*r2_ift).real/r1_ft.size, \
           np.sum(r1_ft * np.conj(r2_ft)).real/r1_ft.size**2, t_gy)
    print 'gy(i) %.2e %.2e %.2e (Input)'%(\
           np.sum(r1_ift*r2_ift).imag/r1_ft.size, \
           np.sum(r1_ft * np.conj(r2_ft)).imag/r1_ft.size**2, t_gy)
    print 'gy(r) %.2e %.2e %.2e (Input)'%(\
           np.sum(r1_ift*r2_ift).real/r1_ft.size, \
           np.sum(np.conj(r1_ft) * r2_ft).real/r1_ft.size**2, t_gy)
    print 'Area of field in steradian %.2f'%Ndtsq

    if plot_ps:
        npixels = r1_ft.size * r1_ft.size * 1.
        pl.figure(2)
	xx, ll, gg_1de, yy = RadialProf(fftpack.fftshift(l))     
	xx, gg_1d, gg_1de, yy = RadialProf(fftpack.fftshift(cl_gg))     
        pl.loglog(ll, my.angular_power_spectrum(theta0/60.*np.pi/180., ll), 
                  c='k', label='Analytic gg')
	pl.loglog(ll, gg_1d, label='Input gg', c='r')    
	xx, yy_1d, yy_1de, yy = RadialProf(fftpack.fftshift(cl_yy))     
	pl.loglog(ll, yy_1d, label='Input yy', c='b')   
	xx, gy_1d, gy_1de, yy = RadialProf(fftpack.fftshift(cl_gy))     
	pl.loglog(ll, gy_1d, label='Input gy', c='g')   

	xx, r1_1d, r1_1de, yy = RadialProf(fftpack.fftshift(abs(r1_ft)**2.))    
	pl.loglog(ll, r1_1d/npixels, label='Simulated gg', c='r', ls='--')   
	xx, r2_1d, r2_1de, yy = RadialProf(fftpack.fftshift(abs(r2_ft)**2.))    
	pl.loglog(ll, r2_1d/npixels, label='Simulated yy', c='b', ls='--') 
	xx, r3_1d, r3_1de, yy = RadialProf(fftpack.fftshift(r2_ft * \
                                np.conj(r1_ft)).real) 
	pl.loglog(ll, r3_1d/npixels, label='Simulated gy', c='g', ls='--') 
	pl.legend(loc=0)
        pl.xlabel('l')
        pl.ylabel('$C_l$')
        pl.show()
    if do_plot and 0:
        pl.figure(2)
	pl.subplot(141)
	pl.imshow(np.log10(cl_yy))
	pl.colorbar()
	pl.subplot(142)
	pl.imshow(np.log10(cl_gg))
	pl.colorbar()
	pl.subplot(143)
	pl.imshow(np.log10(cl_yy))
	pl.colorbar()
	pl.subplot(144)
	pl.imshow(l)
	pl.colorbar()
	pl.show()
 
    if do_plot and 0:
        pl.subplot(221)
	pl.imshow(r1, origin='lower')
	pl.colorbar()
        pl.subplot(222)
	pl.imshow(r2, origin='lower')
	pl.colorbar()
        pl.subplot(223)
	pl.imshow(r1_ift, origin='lower')
	pl.colorbar()
        pl.subplot(224)
	pl.imshow(r2_ift, origin='lower')
	pl.colorbar()
 
    return r1_ift, r2_ift

