#!/usr/bin/env python

import numpy
import numpy.random
import scipy.stats

def get_stan_input(counts,counts_control,priors_control,D):
  # get the number of C and total read-outs for noncontrol cytosines in BS-seq and oxBS-seq
  bsC, bsTot, oxC, oxTot = counts[:,0::4], counts[:,1::4], counts[:,2::4], counts[:,3::4] 

  # get the number of C and total read-outs for control cytosines in BS-seq and oxBS-seq
  bsC_control, bsTot_control, oxC_control, oxTot_control = counts_control[:,0::4], counts_control[:,1::4], counts_control[:,2::4], counts_control[:,3::4]

  # number of cytosines
  K, K_control = bsC.shape[0], bsC_control.shape[0]

  # number of samples
  N = D.shape[0]
  # number of covariates
  P = D.shape[1]

  # coefficient matrix
  B = numpy.zeros((K,P,3))

  #
  sigma2_E = numpy.array([1]*K)

  # kronecker product of I and D
  I_D = numpy.kron(numpy.eye(3),D)

  sigma2_B = 5
  mu_B = numpy.zeros((P,3))
  # kronecker product of V_B and U_B
  V_B_U_B = sigma2_B*numpy.kron(numpy.eye(3),numpy.eye(P))

  alpha = 1
  beta = 1
  V_E_U_E = numpy.kron(numpy.eye(3),numpy.eye(N))

  Y = numpy.zeros((K,N,3))

  # hyperprior for bsEff
  mu_mu_bsEff, sigma_mu_bsEff = 2, 1.29
  mu_sigma_bsEff, sigma_sigma_bsEff = 0.4, 0.5

  # hyperprior for bsBEff
  mu_mu_bsBEff, sigma_mu_bsBEff = -3, 1.29
  mu_sigma_bsBEff, sigma_sigma_bsBEff = 0.4, 0.5

  # hyperprior for oxEff
  mu_mu_oxEff, sigma_mu_oxEff = 2, 1.29
  mu_sigma_oxEff, sigma_sigma_oxEff = 0.4, 0.5

  # hyperprior for seqErr
  mu_mu_seqErr, sigma_mu_seqErr = -3, 1.29
  mu_sigma_seqErr, sigma_sigma_seqErr = 0.4, 0.5
 
  # sample initial values from priors
  mu_bsEff = scipy.stats.norm.rvs(mu_mu_bsEff, sigma_mu_bsEff)
  sigma_bsEff = scipy.stats.lognorm.rvs(sigma_sigma_bsEff,loc=0,scale=numpy.exp(mu_sigma_bsEff))
  mu_bsBEff = scipy.stats.norm.rvs(mu_mu_bsBEff, sigma_mu_bsBEff)
  sigma_bsBEff = scipy.stats.lognorm.rvs(sigma_sigma_bsBEff,loc=0,scale=numpy.exp(mu_sigma_bsBEff))
  mu_oxEff = scipy.stats.norm.rvs(mu_mu_oxEff, sigma_mu_oxEff)
  sigma_oxEff = scipy.stats.lognorm.rvs(sigma_sigma_oxEff,loc=0,scale=numpy.exp(mu_sigma_oxEff))
  mu_seqErr = scipy.stats.norm.rvs(mu_mu_seqErr, sigma_mu_seqErr)
  sigma_seqErr = scipy.stats.lognorm.rvs(sigma_sigma_seqErr,loc=0,scale=numpy.exp(mu_sigma_seqErr))
  raw_bsEff,raw_bsBEff,raw_oxEff,raw_seqErr = [],[],[],[]
  for _ in range(0,N):
    raw_bsEff.append(scipy.stats.norm.rvs(0,1))
    raw_bsBEff.append(scipy.stats.norm.rvs(0,1))
    raw_oxEff.append(scipy.stats.norm.rvs(0,1))
    raw_seqErr.append(scipy.stats.norm.rvs(0,1))
  theta_control = numpy.zeros((K_control,N,3))
  for k in range(0,K_control):
    for n in range(0,N):
      theta_control[k,n,:] = numpy.random.dirichlet(priors_control[k])

  # data and init dictionaries
  data = {'mu_mu_bsEff': mu_mu_bsEff, 'sigma_mu_bsEff': sigma_mu_bsEff,
          'mu_sigma_bsEff': mu_sigma_bsEff, 'sigma_sigma_bsEff': sigma_sigma_bsEff,
          'mu_mu_bsBEff': mu_mu_bsBEff, 'sigma_mu_bsBEff': sigma_mu_bsBEff,
          'mu_sigma_bsBEff': mu_sigma_bsBEff, 'sigma_sigma_bsBEff': sigma_sigma_bsBEff,
          'mu_mu_oxEff': mu_mu_oxEff, 'sigma_mu_oxEff': sigma_mu_oxEff,
          'mu_sigma_oxEff': mu_sigma_oxEff, 'sigma_sigma_oxEff': sigma_sigma_oxEff,
          'mu_mu_seqErr': mu_mu_seqErr, 'sigma_mu_seqErr': sigma_mu_seqErr,
          'mu_sigma_seqErr': mu_sigma_seqErr, 'sigma_sigma_seqErr': sigma_sigma_seqErr,
          'P': P, 'N': N, 'K': K,
          'bsC': bsC, 'bsTot': bsTot, 'oxC': oxC, 'oxTot': oxTot,
          'D': D, 'I_D': I_D, 'mu_B': mu_B, 'V_B_U_B': V_B_U_B,
          'alpha': alpha, 'beta': beta, 'V_E_U_E': V_E_U_E,
          'K_control': K_control, 'alpha_control': priors_control,
          'bsC_control': bsC_control, 'bsTot_control': bsTot_control,
          'oxC_control': oxC_control, 'oxTot_control': oxTot_control}
  init = {'mu_bsEff': mu_bsEff, 'sigma_bsEff': sigma_bsEff, 'raw_bsEff': raw_bsEff,
          'mu_bsBEff': mu_bsBEff, 'sigma_bsBEff': sigma_bsBEff, 'raw_bsBEff': raw_bsBEff,
          'mu_oxEff': mu_oxEff, 'sigma_oxEff': sigma_oxEff, 'raw_oxEff': raw_oxEff,
          'mu_seqErr': mu_seqErr, 'sigma_seqErr': sigma_seqErr, 'raw_seqErr': raw_seqErr,
          'sigma2_E': sigma2_E, 'B': B, 'Y': Y, 'theta_control': theta_control}

  return data, init

def savagedickey(samples1,samples2,prior1_mean=numpy.zeros((3,)),prior1_cov=5.0*numpy.eye(3),prior2_mean=numpy.zeros((3,)),prior2_cov=5.0*numpy.eye(3)):
  Delta_theta = numpy.vstack(((numpy.array([samples1[:,0]]).T - samples2[:,0]).flatten(1),(numpy.array([samples1[:,1]]).T - samples2[:,1]).flatten(1),(numpy.array([samples1[:,2]]).T - samples2[:,2]).flatten(1)))
  density = scipy.stats.kde.gaussian_kde(Delta_theta,bw_method='scott')

  numerator = scipy.stats.multivariate_normal.pdf(numpy.zeros(prior1_mean.shape),mean=prior1_mean-prior2_mean,cov=prior1_cov+prior2_cov)
  denominator = density.evaluate([0,0,0])[0]

  return numerator/denominator
