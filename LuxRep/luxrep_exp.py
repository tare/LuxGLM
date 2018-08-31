#!/usr/bin/env python
#for controls

import argparse
import sys
import numpy
import numpy.random
import scipy.stats
import os

import pystan
import pickle
import logging
from hashlib import md5

import luxrep_routines

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='LuxRep')
	parser.add_argument('-c','--coverage_data', action='store', dest='control_coverage_data', type=str, required=False, default='data/TRs_cov.txt', help='file containing control cytosine coverage data')
	parser.add_argument('-m', '--methylation_data', action='store', dest='control_methylation_data', type=str, required=False, default='data/TRs_meth.txt', help='file containing control cytosine methylation data')
	parser.add_argument('-o','--outfolder',action='store',dest='outfolder',type=str,required=False,default='%s/control_dir'%os.getcwd(),help='directory containing control output with full pathname')
	parser.add_argument('-l','--cmdstan_loc',action='store',dest='cmdstan_directory',type=str,required=True,help='cmdstan directory with full pathname')
	parser.add_argument('-v','--version',action='version',version='%(prog)s 0.666')
	options = parser.parse_args()

	# read data files
	bsTot_control = numpy.loadtxt(options.control_coverage_data,skiprows=1,dtype='int')

	bsC_control = numpy.loadtxt(options.control_methylation_data,delimiter='\t',skiprows=1,dtype='int')
	
	outdir = options.outfolder
	if not os.path.exists(outdir): os.makedirs(outdir)

	stan_dir = options.cmdstan_directory

	curdir = os.getcwd()
    # get data and init dictionaries for stan
	data, init = luxrep_routines.get_stan_controls_input(bsTot_control,bsC_control)

	pystan.misc.stan_rdump(data,'%s/data.R'%outdir)
	pystan.misc.stan_rdump(init,'%s/init.R'%outdir)
	
	os.chdir(stan_dir)
	os.system('make %s/luxrep_exp'%(curdir))

	os.chdir(outdir)
	os.system('cp %s/luxrep_exp .'%curdir)

	os.system('./luxrep_exp variational output_samples=1000 elbo_samples=1000 grad_samples=10 data file=data.R init=init.R output diagnostic_file=diagnostics.csv > summary.txt')
	while 'COMPLETED' not in open('summary.txt','r').readlines()[-1]: os.system('./luxrep_exp variational output_samples=1000 elbo_samples=1000 grad_samples=10 data file=data.R init=init.R output diagnostic_file=diagnostics.csv > summary.txt')	

	#bsEff, seqErr = luxglm_routines.parse_controls_output(outdir) # or just save output files

