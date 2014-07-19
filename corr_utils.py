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


def run_autoc(fi, pixel_scale, ftype, minusone, indir, outdir, pid, mask):
    '''Running auto correlations on both gg and gy fields
       fi: simulation number. This will be used to generate input map name
       ftype: gg (galaxy density fluctuations) or gy (y field)
       minusone: If it is true then set all values less than -1 to -1 for gg
       input catalog and mj_corr output files will be removed
       mask (logical): True - unmasked, False - masked
    '''

    if ftype == 'gg':
        map = os.path.join(indir, 'ggfield_%s.fits'%fi)
        ifile1 = os.path.join(outdir, 'ggfield_cat_%s_%s.fits'%(pid, fi))
        ofile = os.path.join(outdir, 'ac_gg_%s_%s.txt'%(pid, fi))
        ofile = 'k2_file_name = %s'%ofile
    elif ftype == 'gy':
        map = os.path.join(indir, 'gyfield_%s.fits'%fi)
        ifile1 = os.path.join(outdir, 'gyfield_cat_%s_%s.fits'%(pid, fi))
        ofile = os.path.join(outdir, 'ac_gy_%s_%s.txt'%(pid, fi))
        ofile = 'k2_file_name = %s'%ofile
    else:
        raise ValueError('Unknown field')
    ifile2 = ''
    do_auto_corr = 'true'
    do_cross_corr = 'false'
    k_col = 'k_col = 3'
    w_col = ''

    x = fits.open(map)[0].data
    if minusone and ftype == 'gg':
        x[x < -1] = -1
        print 'Minus %s'%ftype
    ydim, xdim = x.shape
    # Here y is the columns numbers and ra is the row numbers
    ra0, dec0 = 0., 0.
    dec, ra = np.meshgrid(np.arange(ydim), np.arange(xdim))
    dec += dec0
    ra += ra0
    dec = dec * (pixel_scale/60.) #degrees
    ra = ra * (pixel_scale/60.) #degrees
    my.write_fits_table(ifile1, ['RA', 'DEC', 'KAPPA'], \
                        [ra[mask].ravel(), dec[mask].ravel(), x[mask].ravel()])

    f_tpl = open('rsim_corr_default.tpl', 'r')
    template = f_tpl.read()
    f_tpl.close()

    outfile = open('input_%s.params'%pid,'w')
    outfile.write(template %vars())
    outfile.close()

    cmd = '/home/vinu/software/mj_corr_trunk/corr2 input_%s.params > \
           /dev/null'%pid
    os.system(cmd)
    os.system('rm -f input_%s.params %s'%(pid, ifile1))

    f = np.genfromtxt(ofile.split(' = ')[1], skip_header=1)

    theta, xi, xie = f[:,1], f[:,2], f[:,3]
    os.system('rm -f %s '%(ofile.split(' = ')[1]))
    return theta, xi, xie

def run_crossc(fi, pixel_scale, minusone, indir, outdir, pid, mask):
    '''Running cross correlations on both gg and gy fields
       fi: simulation number. This will be used to generate input map name
       minusone: If it is true then set all values less than -1 to -1 for gg
       mask: 1 - unmasked, 0 - masked
    '''

    ifile1 = os.path.join(outdir, 'ggfield_cat_%s_%s.fits'%(pid, fi))
    ifile2 = os.path.join(outdir, 'gyfield_cat_%s_%s.fits'%(pid, fi))
    ofile = os.path.join(outdir, 'cc_gy_%s_%s.txt'%(pid, fi))
    ofile = 'k2_file_name = %s'%ofile
    do_auto_corr = 'false'
    do_cross_corr = 'true'
    k_col = 'k_col = 3'
    w_col = ''

    x = fits.open(os.path.join(indir, 'ggfield_%s.fits'%fi))[0].data
    y = fits.open(os.path.join(indir, 'gyfield_%s.fits'%fi))[0].data
    if minusone:
        x[x < -1] = -1
        print 'Minus' 

    ydim, xdim = x.shape
    # Here y is the columns numbers and ra is the row numbers
    ra0, dec0 = 0., 0.
    dec, ra = np.meshgrid(np.arange(ydim), np.arange(xdim))
    dec += dec0
    ra += ra0
    dec = dec * (pixel_scale/60.) #degrees
    ra = ra * (pixel_scale/60.) #degrees
    my.write_fits_table(ifile1, ['RA', 'DEC', 'KAPPA'], \
             [ra[mask].ravel(), dec[mask].ravel(), x[mask].ravel()])
    my.write_fits_table(ifile2, ['RA', 'DEC', 'KAPPA'], \
             [ra[mask].ravel(), dec[mask].ravel(), y[mask].ravel()])

    ifile2 = 'file_name2 = %s'%ifile2

    f_tpl = open('rsim_corr_default.tpl', 'r')
    template = f_tpl.read()
    f_tpl.close()

    outfile = open('input_%s.params'%pid,'w')
    outfile.write(template %vars())
    outfile.close()

    cmd = '/home/vinu/software/mj_corr_trunk/corr2 input_%s.params > \
           /dev/null'%pid
    os.system(cmd)
    os.system('rm -f input_%s.params'%pid)
    os.system('rm -f %s %s'%(ifile1, ifile2.split(' = ')[1]))

    f = np.genfromtxt(ofile.split(' = ')[1], skip_header=1)

    theta, xi, xie = f[:,1], f[:,2], f[:,3]
    os.system('rm -f %s '%(ofile.split(' = ')[1]))
    return theta, xi, xie

def generate_groups_crossc(pixel_scale, pid, Nruns, fi, indir, outdir, 
                           theta_mod, outnpz, minusone, random_pos, mask):
    '''
      Nruns: No of times N field will be generated based on a given gg and gy
             fields
      fi: To get the gg and gy fields
      indir: The input directory from which the gg and gy maps are to be read
      outdir: The output directory to which the mj corr outputs and npz file
              saved
      theta_mod: The theta values to which the correlation will be 
                   interpolated. It has size N (lets say)
      outnpz: Output npz file. This has arrays of theta_mod (N)
              and correlation with size N X Nruns
      random_pos: Stack at random location instead of group
      minusone: Does not include groups belong to pixels with values < -1

      The mj_corr output will be stack_sim_groups_%s_%s.txt' and removed after
      reading. 

      The gycat, which is the catalog version of gyfield, will be removed 


    '''
    #http://ned.ipac.caltech.edu/level5/Sept01/Bahcall2/paper.pdf shows that
    #cluster density is 1e-5 and groups density is 1e-3 
    group_density = 1e-4 #Mpc^-3 h^3 
    volume = (2.216-0.293) * (1e3)**3 #Mpc^3 between z=0.2 and z=0.1
    N_groups = group_density * volume #Number of groups in the whole sky
    print 'N_groups ', N_groups
 
    f_gg = os.path.join(indir, 'ggfield_%s.fits'%fi)
    f_gy = os.path.join(indir, 'gyfield_%s.fits'%fi)

    x = fits.open(f_gg)[0].data
    xminus = ~(x < -1)
    x[x < -1] = -1
    y = fits.open(f_gy)[0].data

    print 'Avg. Y > %.2e'%y.mean() 
    ydim, xdim = x.shape
    # Here y is the columns numbers and ra is the row numbers
    ra0, dec0 = 0., 0.
    dec, ra = np.meshgrid(np.arange(ydim), np.arange(xdim))
    dec += dec0
    ra += ra0
    dec = dec * (pixel_scale/60.) #degrees
    ra = ra * (pixel_scale/60.) #degrees

    #Removing 100 arcmin boundary
    bound = (ra > ra.min() + 100/60.) & (ra < ra.max() - 100/60.) & \
            (dec > dec.min() + 100/60.) & (dec < dec.max() - 100/60.) 

    #Number of groups per sq. arcmin 
    N_groups = N_groups / (41253. * 60.*60.)
    print 'N_groups/deg ', N_groups

    #Number groups per pixel
    N_pix = N_groups * pixel_scale**2. 
    print 'N_pix ', N_pix

    gycat = os.path.join(outdir, 'gyfield_cat_%s.fits'%fi)
    my.write_fits_table(gycat, ['RA', 'DEC', 'Y', 'W'], 
                        [ra.ravel(), dec.ravel(), y.ravel(), 
                         np.ones(y.ravel().shape)])

    yprof_a = []

    for i in range(Nruns):
        ifile1 = os.path.join(outdir, 'N_%s_%s.fits'%(fi, i))
        ifile2 = 'file_name2 = %s'%gycat
        ofile = os.path.join(outdir, 'stack_sim_groups_%s_%s.txt'%(fi, i))
        ofile = 'nk_file_name = %s'%ofile
        do_auto_corr = 'false'
        do_cross_corr = 'true'
        w_col = 'w_col = 3 4'
        k_col = 'k_col = 0 3'

        f_tpl = open('rsim_corr_default.tpl', 'r')
        template = f_tpl.read()
        f_tpl.close()

        outfile = open('input_%s.params'%pid,'w')
        outfile.write(template %vars())
        outfile.close()
        
        print 'Running %s time'%i
        N = N_pix * (1 + x)
        N = np.random.poisson(N)
        fits.writeto(ifile1, N, clobber=True)
        print '# of groups ', N[N > 0].shape[0], N.sum()
        if minusone:
            con = (N != 0) & bound & xminus & mask
        else:
            con = (N != 0) & bound & mask

        print '# of unmasked objects %d'%con[con].shape[0]       

        if random_pos: 
            wra = np.random.shuffle(ra[con])
            wdec = np.random.shuffle(dec[con])
            my.write_fits_table(ifile1, ['RA', 'DEC', 'W'], 
                                [np.random.shuffle(ra[con]), 
                                 np.random.shuffle(dec[con]), N[con]])
        else:
            my.write_fits_table(ifile1, ['RA', 'DEC', 'W'], 
                                [ra[con], dec[con], N[con]])

        cmd = '/home/vinu/software/mj_corr_trunk/corr2 input_%s.params > \
               /dev/null'%pid
        os.system(cmd)

        f = np.genfromtxt(ofile.split(' = ')[1], skip_header=1)

        theta, yprof, yprofe = f[:,1], f[:,2], f[:,3] 
        spl = interpolate.splrep(theta, yprof)
        yprof_i = interpolate.splev(theta_mod, spl)
        if len(yprof_a) < 1:
            yprof_a = yprof_i.copy()
        else:
            yprof_a = np.row_stack((yprof_a, yprof_i))

        os.system('rm -f %s'%ofile.split(' = ')[1])
        os.system('rm -f %s'%ifile1)

    np.savez(os.path.join(outdir, outnpz), theta=theta_mod, yprof=yprof_a)
    os.system('rm -f input_%s.params %s'%(pid, gycat))
        
