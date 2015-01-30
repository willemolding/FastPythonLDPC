__author__ = 'willemolding'

from scipy import sparse, io
import numpy as np
import time
from ldpc_decode import ldpc_decode

def run_test():

    data = io.loadmat('test_code.mat')
    H = data['H']
    x = data['x']
    y = data['y']
    z = data['z']
    size_G = data['size_G'][0]
    sigma = data['sigma']

    f1 = 1. / (1 + np.exp(-2 * z / sigma ** 2))  # likelihoods
    f0 = 1 - f1
    f1[f1 == 0.5] = 0.5 + 1e-20
    f0[f0 == 0.5] = 0.5 - 1e-20

    t0 = time.time()
    for i in range(10):
        z_hat, success, k = ldpc_decode(f0, f1, H, 100)
    t1 = time.time()

    x_hat = z_hat[size_G[1] - size_G[0]:size_G[1]]
    b = x_hat.T

    nErrors = np.sum(x != b)
    print 'decoding', z_hat.shape[0], 'code bits'
    print "n_errors", nErrors
    print "completed in ", t1-t0, "seconds"
    print 'iterations', k
    print 'success', success

if __name__ == "__main__":
    run_test()
