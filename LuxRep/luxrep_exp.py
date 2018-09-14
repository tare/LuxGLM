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
	parser.add_argument('-f','--file_list', action='store', dest='file_list', type=str, required=False, default='data/lambda_fileList.txt', help='file containing list of files containing counts from control cytosine where each file holds data for one library')
	parser.add_argument('-n','--library_names', action='store', dest='library_names', type=str, required=False, default='data/nameList.txt', help='file listing library labels in the same order as file_list')
	parser.add_argument('-o','--outfolder', action='store', dest='outfolder', type=str, required=False, default='%s/control_dir'%os.getcwd() help='directory containing control output with full pathname')
	parser.add_argument('-l','--cmdstan_loc',action='store',dest='cmdstan_directory',type=str,required=True,help='cmdstan directory with full pathname')
	parser.add_argument('-v','--version',action='version',version='%(prog)s 0.666')
	options = parser.parse_args()

	# read list of library files and corresponding labels
        files = open(options.file_list,'r').readline().strip().split()
        labels = open(options.library_names,'r').readline().strip().split()
        stan_dir = options.cmdstan_directory

        curdir = os.getcwd()
        os.chdir(stan_dir)
        os.system('make %s/luxrep_exp'%curdir)

        # loop over library files

        for infile,label in zip(files,labels):
            os.chdir(curdir)
            print label
            bsC_control, bsT_control = numpy.loadtxt(infile,delimiter='\t',usecols=(4,5),dtype='int',unpack=True)
            bsTot_control = bsT_control + bsC_control

            outdir = '%s/%s'%(options.outfolder,label)
            if not os.path.exists(outdir): os.makedirs(outdir)

            # get data and init dictionaries for stan
            data, init = luxrep_routines.get_stan_controls_input(bsTot_control,bsC_control)

            os.system('rm -f %s/output.csv'%outdir)
            pystan.misc.stan_rdump(data,'%s/data.R'%outdir)
            pystan.misc.stan_rdump(init,'%s/init.R'%outdir)
            
            os.chdir(outdir)
            os.system('cp %s/luxrep_exp .'%curdir)

            os.system('./luxrep_exp variational output_samples=1000 elbo_samples=1000 grad_samples=10 data file=data.R init=init.R output diagnostic_file=diagnostics.csv > summary.txt')
            while 'COMPLETED' not in open('summary.txt','r').readlines()[-1]: os.system('./luxrep_exp variational output_samples=1000 elbo_samples=1000 grad_samples=10 data file=data.R init=init.R output diagnostic_file=diagnostics.csv > summary.txt')	
        
        luxrep_routines.cleanup(options.outfolder, ['luxrep_exp'])
	#bsEff, seqErr = luxglm_routines.parse_controls_output(outdir) # or just save output files

