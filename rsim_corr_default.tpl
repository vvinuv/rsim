# The file(s) with the galaxy data.
file_name = %(ifile1)s 
%(ifile2)s 

do_auto_corr = %(do_auto_corr)s
do_cross_corr = %(do_cross_corr)s

file_type = FITS

ra_units = degrees
dec_units = degrees

ra_col = RA
dec_col = DEC
%(w_col)s
%(k_col)s

project = false
min_sep = 1.
max_sep = 600.
bin_size = 0.10
#nbins = 30
sep_units = arcmin

bin_slop = 1.

%(ofile)s

verbose = 1

