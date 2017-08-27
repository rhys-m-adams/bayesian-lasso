import emcee
from scipy.stats import norm
import numpy as np
import pdb
import sys
from scipy.stats import pearsonr

'''
This implementation can by found in 
The Bayesian Lasso
Trevor Park and George Casella
'''
def lnprob(A, x, y, inv_sigma):
    model = A.dot(x)
    lp = -0.5 * sum((model - y)**2 * inv_sigma) + ((A.shape[0] + A.shape[1] - 1) / 2 + 1) * np.log(inv_sigma)
    return lp

def lnprior(x, l):
    # The parameters are stored as a vector of values, so unpack them
    lp =       - sum(np.abs(x)) * l
    return lp

def ll(A, x, y, l):
    inv_sigma = np.exp(x[0])
    x = x[1:]
    return lnprior(x, l * np.sqrt(inv_sigma)) + lnprob(A, x, y, inv_sigma)


def bayesian_lasso(A,y, penalty, start):
    #import pymc3
    nwalkers = 250
    num_params = A.shape[1]+1
    #p0 = start + np.random.randn(nwalkers,A.shape[1])
    inv_sigma = 1./np.mean((A.dot(start) - y)**2)
    p0 = np.hstack((np.log(inv_sigma),start)) + np.hstack((np.log(inv_sigma),start+1e-8)) * np.random.randn(nwalkers,num_params)
    
    sampler = emcee.EnsembleSampler(nwalkers, num_params, lambda x:ll(A,x,y,penalty))
    pos, prob, state = sampler.run_mcmc(p0, int(10000*start.shape[0]/44.))
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, int(20000*start.shape[0]/44.))
    autocorr = np.array([[pearsonr(sampler.flatchain[0:-jj:jj,ii],sampler.flatchain[jj::jj,ii])[0] for jj in range(1,2000)] for ii in range(num_params)])
    sample_rate = []
    for ac in autocorr:
        ind = np.where(ac<0.3)
        if ind[0].shape[0]:
            ind = ind[0][0]
        else:
            ind = ac.shape[0]
        myf = np.linalg.lstsq(np.array([range(1,ind+1)]).T, np.log(ac[:ind]))[0]
        sample_rate.append(int(np.ceil(-np.log(10)/myf)))

    mu = []
    sigma = []
    p = []
    for ii in range(num_params):
        mu.append(np.mean(sampler.flatchain[::sample_rate[ii],ii], axis = 0))
        sigma.append(np.std(sampler.flatchain[::sample_rate[ii],ii], axis = 0))
        p.append(np.mean(np.multiply(sampler.flatchain[::sample_rate[ii],ii], np.sign(mu[ii])) <= 0, axis=0))
    
    #print np.exp(mu[0])
    return np.array(mu[1:]), np.array(sigma[1:]), np.array(p[1:]), np.array(sample_rate[1:])
