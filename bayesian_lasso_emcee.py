import emcee
import numpy as np
import pdb
from scipy.stats import pearsonr

'''
This implementation can by found in 
The Bayesian Lasso
Trevor Park and George Casella
'''

def ll(A, x, y, l):
    inv_sigma = np.exp(x[0])
    x = x[1:]
    model = A.dot(x)
    return  - sum(np.abs(x)) * l * np.sqrt(inv_sigma) + -0.5 * sum((model - y)**2 * inv_sigma) + ((A.shape[0] + A.shape[1] - 1) / 2 + 1) * np.log(inv_sigma)


def bayesian_lasso(A,y, penalty, start, burn_in=0, num_iterations=0, nwalkers=250, autocorr_scan = 2000):
    if burn_in == 0:
        burn_in = int(10000*start.shape[0]/44.)
    if num_iterations == 0:
        num_iterations = int(20000*start.shape[0]/44.)

    num_params = A.shape[1]+1 #need to find parameters plus a variance term for the SSE Gaussian term
    inv_sigma = 1./np.mean((A.dot(start) - y)**2)/A.shape[0]**2 #pick as starting guess for sigma the MSE * number of samples
    p0 = np.hstack((np.log(inv_sigma),start)) + np.hstack((np.log(inv_sigma),start+1e-8)) * np.random.randn(nwalkers,num_params)#choose a wide variety of starting points
    
    sampler = emcee.EnsembleSampler(nwalkers, num_params, lambda x:ll(A,x,y,penalty))
    pos, prob, state = sampler.run_mcmc(p0, burn_in)#burn in
    sampler.reset()
    pos, prob, state = sampler.run_mcmc(pos, num_iterations)#fit
    #find sampling rate for 0.1 autocorrelation
    autocorr = np.array([[pearsonr(sampler.flatchain[0:-jj:jj,ii],sampler.flatchain[jj::jj,ii])[0] for jj in range(1,autocorr_scan)] for ii in range(num_params)])
    sample_rate = [] 
    for ac in autocorr:
        ind = np.where(ac < 0.1)
        if ind[0].shape[0]:
            ind = ind[0][0]
        else:
            ind = ac.shape[0]
        myf = np.linalg.lstsq(np.array([range(1,ind+1)]).T, np.log(ac[:ind]))[0]
        sample_rate.append(int(np.ceil(-np.log(10)/myf)))

    #calculate the statistics with the sampling rate and return
    mu = []
    sigma = []
    p = []
    for ii in range(num_params):
        mu.append(np.mean(sampler.flatchain[::sample_rate[ii],ii], axis = 0))
        sigma.append(np.std(sampler.flatchain[::sample_rate[ii],ii], axis = 0))
        p.append(np.mean(np.multiply(sampler.flatchain[::sample_rate[ii],ii], np.sign(mu[ii])) <= 0, axis=0))
    
    return np.array(mu[1:]), np.array(sigma[1:]), np.array(p[1:]), np.array(sample_rate[1:])
