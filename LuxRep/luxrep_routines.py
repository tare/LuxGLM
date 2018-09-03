#!/usr/bin/env python

import numpy
import numpy as np
import numpy.random
import scipy.stats
import os

def get_stan_controls_input(bsTot_control, bsC_control):
	# number of control cytosines and number of samples	
	K_control, N = bsTot_control.shape

	# priors for theta of the control cytosines
	priors_control = [[999,1] for _ in range(K_control)]

	# values of theta of control cytosines for stan initialization
	theta_control = numpy.zeros((K_control,N,2))

	for k in range(0,K_control):
		for n in range(0,N):
			theta_control[k,n,:] = numpy.random.dirichlet(priors_control[k])

	# hyperpriors for bisulfite conversion efficiency
	mu_mu_bsEff, sigma_mu_bsEff = 4, 1.29
	mu_sigma_bsEff, sigma_sigma_bsEff = 0.4, 0.5

	# hyperpriors for sequencing error rate
	mu_mu_seqErr, sigma_mu_seqErr = -8, 1.29
	mu_sigma_seqErr, sigma_sigma_seqErr = 0.4, 0.5

	# constant for incorrect bisulfite conversion rate
	bsBEff_fixed = 0.001

	# priors for bisulfite conversion rate
	mu_bsEff = scipy.stats.norm.rvs(mu_mu_bsEff, sigma_mu_bsEff)
	sigma_bsEff = scipy.stats.lognorm.rvs(sigma_sigma_bsEff,loc=0,scale=numpy.exp(mu_sigma_bsEff))
	# priors for sequencing error rate
	mu_seqErr = scipy.stats.norm.rvs(mu_mu_seqErr, sigma_mu_seqErr)
	sigma_seqErr = scipy.stats.lognorm.rvs(sigma_sigma_seqErr,loc=0,scale=numpy.exp(mu_sigma_seqErr))
	raw_bsEff,raw_seqErr = [],[]

        # data and init dictionaries
	for _ in range(0,N):
		raw_bsEff.append(scipy.stats.norm.rvs(0,1))
		raw_seqErr.append(scipy.stats.norm.rvs(0,1))
	init = {'mu_bsEff': mu_bsEff, 'sigma_bsEff': sigma_bsEff, 'raw_bsEff': raw_bsEff,
	'mu_seqErr': mu_seqErr, 'sigma_seqErr': sigma_seqErr, 'raw_seqErr': raw_seqErr,
	'theta_control': theta_control}	

	data = {'mu_mu_bsEff': mu_mu_bsEff, 'sigma_mu_bsEff': sigma_mu_bsEff,
	'mu_sigma_bsEff': mu_sigma_bsEff, 'sigma_sigma_bsEff': sigma_sigma_bsEff,
	'mu_mu_seqErr': mu_mu_seqErr, 'sigma_mu_seqErr': sigma_mu_seqErr,
	'mu_sigma_seqErr': mu_sigma_seqErr, 'sigma_sigma_seqErr': sigma_sigma_seqErr,
	'N': N, 'bsBEff_fixed': bsBEff_fixed,
	'K_control': K_control, 'alpha_control': priors_control,
	'bsC_control': bsC_control, 'bsTot_control': bsTot_control}

	#for numChain in range(numChains):
	#  pystan.misc.stan_rdump(init[numChain],'%s/init%s.R'%(outdir,numChain))
	#pystan.misc.stan_rdump(data,'%s/data.R'%outdir)
	#pystan.misc.stan_rdump(init,'%s/init.R'%outdir)

	return data, init

def get_exp_params(exp_params):
	# get bsEff and seqErr, parse from earlier output file
	infile = '%s/output.csv'%exp_params
	lines = open(infile,'r').readlines()[28:32]
	bsEff, seqErr = [], []
	
	line1, line2 = lines[0].strip().split(','), lines[-1].strip().split(',')
	for a,b in zip(line1,line2):
		if a.split('.')[0] == 'bsEff': bsEff.append(float(b))
		if a.split('.')[0] == 'seqErr': seqErr.append(float(b))
	
	return bsEff, seqErr


def get_stan_input(counts,sample_list,D,params):
	# number of samples
        M = len(sample_list)

	# get the total number of readouts and "C" readouts
	bsTot, bsC = counts[0::2].reshape(1,M), counts[1::2].reshape(1,M) 
	
	# number of cytosines
	K = bsC.shape[0]

	# number of biological replicates
	N = D.shape[0]

	# number of covariates
	P = D.shape[1]

	# coefficient matrix
	B = numpy.zeros((K,P,1))

	#
	sigma2_E = numpy.array([1]*K)

	# kronecker product of I and D
	I_D = numpy.kron(numpy.eye(1),D)

	sigma2_B = 5

	mu_B = numpy.zeros((P,1))
	# kronecker product of V_B and U_B
	V_B_U_B = sigma2_B*numpy.kron(numpy.eye(1),numpy.eye(P))
	alpha = 1
	beta = 1
	V_E_U_E = numpy.kron(numpy.eye(1),numpy.eye(N))
	Y = numpy.zeros((K,N,1))
	bsBEff = .001
	# experimental parameters
	bsEff, seqErr = params['bsEff'], params['seqErr']

	init = {'sigma2_E': sigma2_E, 'B': B, 'Y': Y}
	# data and init dictionaries
	data = {'P': P, 'N': N, 'K': K, 'bsBEff': bsBEff,
	'bsC': bsC, 'bsTot': bsTot, 'tr2br': sample_list, 'M': M,
	'D': D, 'I_D': I_D, 'mu_B': mu_B, 'V_B_U_B': V_B_U_B,
	'alpha': alpha, 'beta': beta, 'V_E_U_E': V_E_U_E,
	'bsEff': bsEff, 'seqErr': seqErr}

	return data, init
  
def savagedickey(locus):
	sigma2_B = 5	
	beta = np.loadtxt("output.csv", delimiter=',', skiprows=33, usecols=(2,))
	density = scipy.stats.kde.gaussian_kde(beta,bw_method='scott')
	numerator = scipy.stats.multivariate_normal.pdf([0],mean=[0],cov=[sigma2_B])
	denominator = density.evaluate([0])[0]
	bf=numerator/denominator
	outfile = 'bf.txt'
	fo = open(outfile,'w')
	print >> fo, '%s\t%s'%(locus.replace('_','\t'), bf)
	fo.close()
	# os.system('rm output.csv')

def combine_bfs(outfile,loci):
	outdir = '/'.join(outfile.split('/')[:-1])
	os.system('cat %s/{%s}/bf.txt | sort -k1,1 -k2,2n > %s'%(outdir, ','.join(u.replace(':','_') for u in loci), outfile))

