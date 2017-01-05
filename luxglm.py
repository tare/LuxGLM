#!/usr/bin/env python

import argparse
import sys
import numpy
import numpy.random
import scipy.stats

import pystan
import pickle
import logging
from hashlib import md5

import luxglm_routines

def stan_cache(model_name, **kwargs):
  f=open(model_name, 'rb')
  model_code=f.read()
  f.close()
  code_hash = md5(model_code.encode('ascii')).hexdigest()
  cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)
  try:
    sm = pickle.load(open(cache_fn, 'rb'))
  except:
    sm = pystan.StanModel(file=model_name)
    with open(cache_fn, 'wb') as f:
      pickle.dump(sm, f)
  else:
    logging.info("Using cached StanModel")
  return sm.sampling(**kwargs)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='LuxGLM')
  parser.add_argument('-c','--control_data',action='store',dest='control_data',type=str,required=True,help='file containing control cytosine data')
  parser.add_argument('-p','--control_prior',action='store',dest='control_prior',type=str,required=True,help='file containing prior knowledge on control cytosines')
  parser.add_argument('-d','--data',action='store',dest='data_file',type=str,required=True,help='file containing cytosine data')
  parser.add_argument('-m','--design_matrix',action='store',dest='design_file',type=str,required=True,help='file containing design matrix')
  parser.add_argument('-o','--output',action='store',dest='output_file',type=str,required=True,help='file for storing samples')
  parser.add_argument('-v','--version',action='version',version='%(prog)s 0.666')
  options = parser.parse_args()

  # read data files
  counts_control = numpy.loadtxt(options.control_data,skiprows=0,dtype='int')
  priors_control = numpy.loadtxt(options.control_prior,delimiter='\t',skiprows=0,dtype='int')
  counts = numpy.loadtxt(options.data_file,delimiter='\t',skiprows=0,dtype='int')
  D = numpy.loadtxt(options.design_file,delimiter='\t',skiprows=0,dtype='float')

  # get data and init dictionaries for stan
  data, init = luxglm_routines.get_stan_input(counts,counts_control,priors_control,D)

  # run stan and get samples
  fit = stan_cache('luxglm.stan',data=data,init=[init]*4,iter=200,chains=4,refresh=10)
  print(fit)
  samples = fit.extract()
  pickle.dump(samples,open(options.output_file,'wb'))
