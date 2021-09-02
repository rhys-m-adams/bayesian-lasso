from pymc3 import *
from scipy.stats import norm, pearsonr
import pylab as plt
from numpy import sqrt
import numpy as np
import pdb

def lasso(X,y, num_iter, penalty):
    in_str = []
    for ii in range(X.shape[1]):
        #eval('x%i=X[:, %i]'%(ii,ii))
        in_str.extend(['Laplace(\'x%i\', 0, b=%f) * X[:,%i] '%(ii, penalty, ii)])
    in_str = '+'.join(in_str)
    print(in_str)
    
    with Model() as model:
        # Laplacian prior only works with Metropolis sampler
        likelihood = Normal('y', mu= eval(in_str), tau = 1, observed=y)
        start = find_MAP() 
        step = NUTS(scaling = start)
        trace = sample(num_iter, step, start = start, progressbar=True) 
    
    autocorr = 1
    delta = 0
    C = []
    for key in ['x%i'%(ii) for ii in range(X.shape[1])]:
        C.append(trace[key])
    
    while autocorr>0.1:
        delta += 1
        autocorr = 0
        for c in C:
            curr = pearsonr(c[:-delta], c[delta:])[0]
            autocorr = np.max([autocorr, curr])
        
        if delta > (num_iter-5):
            autocorr = 0
    C = np.array(C).T
    return trace, C, delta

def test():
    n = 1000    
    x1 = norm.rvs(0, 1, size=n)
    x2 = -x1 + norm.rvs(0, 10**-3, size=n)
    x3 = norm.rvs(0, 1, size=n)

    y = 10 * x1 + 10 * x2 + 0.1 * x3
    X = np.array([x1,x2,x3]).T
    trace, C, delta = lasso(X,y, 10000, 1./sqrt(2))
    print(delta)
    plt.figure(figsize=(7, 7))
    traceplot(trace)
    plt.tight_layout()
    plt.show()
    autocorrplot(trace)
    plt.show()
    summary(trace)

if __name__ == '__main__':
    test()