#!/usr/bin/env python

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
	parser.add_argument('-d', '--data', action='store', dest='data_file', type=str, required=True, help='file containing noncontrol cytosine data')
	parser.add_argument('-s', '--sample_list', action='store', dest='sample_list_file', type=str, required=True, help='file containing sample number of libraries')
        parser.add_argument('-n', '--library_names', action='store', dest='library_names', type=str, required=False, default='data/nameList.txt', help='file listing library labels, same as input file used in luxrep_exp module')
	parser.add_argument('-m', '--design_matrix', action='store', dest='design_file', type=str, required=True, help='file containing design matrix')
	parser.add_argument('-c', '--exp_params', action='store', dest='experimental_parameters_folder', type=str, required=False, default='%s/control_dir'%os.getcwd(), help='directory containing output from control data with full pathname')
	parser.add_argument('-o', '--outfolder', action='store', dest='outfolder', type=str, required=False, default='%s/results'%os.getcwd(), help='directory containing data analysis output with full pathname')
	parser.add_argument('-l','--cmdstan_loc', action='store', dest='cmdstan_directory', type=str, required=True, help='cmdstan directory with full pathname')
	parser.add_argument('-v','--version',action='version',version='%(prog)s 0.666')
	options = parser.parse_args()

	# read data files
	sample_list = numpy.loadtxt(options.sample_list_file,skiprows=0,dtype='int')

	ncols = len(sample_list) * 2 + 1

	counts_data = numpy.loadtxt(options.data_file,delimiter='\t',skiprows=1,dtype='int', usecols=range(1,ncols))

	loci = numpy.genfromtxt(options.data_file,delimiter='\t',skip_header=1, dtype='str', usecols=(0,))

        libraries = open(options.library_names,'r').readline().strip().split()

	numLoci = len(loci)

	with open(options.design_file) as f:
		ncols = len(f.readline().split())
			
	D = numpy.loadtxt(options.design_file,delimiter='\t',skiprows=1,dtype='float', usecols=range(1, ncols + 1))

	exp_params = options.experimental_parameters_folder
	
	bsEff, seqErr = luxrep_routines.get_exp_params(exp_params,libraries)
	
	params = {'bsEff':bsEff, 'seqErr':seqErr}
	
        outfolder = options.outfolder
        if not os.path.exists(outfolder): os.makedirs(outfolder)

        stan_dir = options.cmdstan_directory

        curdir = os.getcwd()

        os.chdir(stan_dir)
	# compile stan file
        os.system('make %s/luxrep'%(curdir))

	# loop over cytosines
        for n in range(numLoci):
		locus = loci[n].replace(':','_')
		print n+1, locus

                outdir = '%s/%s'%(outfolder, locus)
                if not os.path.exists(outdir): os.makedirs(outdir)

		# get data per cytosine
		counts = counts_data[n]
		
	    # get data and init dictionaries for stan
		data, init = luxrep_routines.get_stan_input(counts,sample_list,D,params)

		pystan.misc.stan_rdump(data,'%s/data.R'%outdir)
		pystan.misc.stan_rdump(init,'%s/init.R'%outdir)
		
		os.chdir(outdir)
		os.system('cp %s/luxrep .'%curdir)

		os.system('./luxrep variational output_samples=1000 elbo_samples=1000 grad_samples=10 data file=data.R init=init.R output diagnostic_file=diagnostics.csv > summary.txt')
		while 'COMPLETED' not in open('summary.txt','r').readlines()[-1]: os.system('./luxrep variational output_samples=1000 elbo_samples=1000 grad_samples=10 data file=data.R init=init.R output diagnostic_file=diagnostics.csv > summary.txt')	
		# compute bayes factor	
		luxrep_routines.savagedickey(locus)

        outfile = '%s/bfs.bed'%outfolder

	# combine results for all cytosine
	luxrep_routines.combine_bfs(outfile,loci)

        luxrep_routines.cleanup(outfolder, ['luxrep'])

