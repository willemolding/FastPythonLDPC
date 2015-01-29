__author__ = 'willemolding'

from scipy import sparse, io
import numpy as np
from ldpc_decode import ldpc_decode

def run_test():

    data = io.loadmat('test_code.mat')
    G = data['G']
    H = data['H']
    x = data['x']
    y = data['y']
    z = data['z']
    sigma = data['sigma']

    f1 = 1. / (1 + np.exp(-2 * z / sigma ** 2))  # likelihoods
    f0 = 1 - f1
    f1[f1 == 0.5] = 0.5 + 1e-20
    f0[f0 == 0.5] = 0.5 - 1e-20

    for i in range(1000):
        z_hat, success, k = ldpc_decode(f0, f1, H, 100)
    x_hat = z_hat[G.shape[1] - G.shape[0]:G.shape[1]]
    b = x_hat.T

    nErrors = np.sum(x != b)
    print nErrors

if __name__ == "__main__":
    run_test()
