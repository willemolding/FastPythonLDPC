# function [x_hat, success, k, prob ] = ldpc_decode(f0,f1,H,max_iter)
# decoding of binary LDPC as in Elec. Letters by MacKay&Neal 13March1997
# For notations see the same reference.
# function [x_hat, success, k] = ldpc_decode(y,f0,f1,H)
# outputs the estimate x_hat of the ENCODED sequence for
# the received vector y with channel likelihoods of '0' and '1's
# in f0 and f1 and parity check matrix H. Success==1 signals
# successful decoding. Maximum number of iterations is set to 100.
# k returns number of iterations until convergence.
#
# Example:
# We assume G is systematic G=[A|I] and, obviously, mod(G*H',2)=0
# sigma = 1;                          # AWGN noise deviation
# x = (sign(randn(1,size(G,1)))+1)/2; # random bits
#         y = mod(x*G,2);                     # coding
#         z = 2*y-1;                          # BPSK modulation
#         z=z + sigma*randn(1,size(G,2));     # AWGN transmission
#
#         f1=1./(1+exp(-2*z/sigma^2));        # likelihoods
#         f0=1-f1;
#         [z_hat, success, k] = ldpc_decode(z,f0,f1,H);
#         x_hat = z_hat(size(G,2)+1-size(G,1):size(G,2));
#         x_hat = x_hat';

#   Copyright (c) 1999 by Igor Kozintsev igor@ifp.uiuc.edu
#   $Revision: 1.1 $  $Date: 1999/07/11 $
#   fixed high-SNR decoding

import numpy as np
cimport numpy as np

from scipy import sparse

def ldpc_decode(np.ndarray[double, ndim=2] f0, np.ndarray[double, ndim=2] f1, H, int max_iter):
    """
    A python port of the ldpc_decode matlab code.

    Parameters:
    ----------
    f0 : 1D numpy array
        see matlab docstring
    f1 : 1D numpy array
        see matlab docstring
    H : 2D scipy.sparse.csc_matrix
        Must be a scipy sparse array of the crc type. This is the same type that matlab used so we remain compatible.
    max_iter : integer
        maximum number of iterations

    Returns:
    --------
    x_hat : 1D numpy array
        Error corrected ENCODED sequence
    success : bool
        indicates successful convergence e.g. parity check passed
    k : integer
        number of iterations to converge
    prob :
    """

    cdef:
        int m,n,k
        char success
        np.ndarray[int,ndim=1] ii,jj
        np.ndarray[double, ndim=1] sPdq, sPr0, sPr1,sdq,sff0,sff1,sq0,sqq,sr0,sr1
        np.ndarray[double, ndim=2] Q0,Q1,QQ,tent,x_hat


    # check the matrix is correctly orientated and transpose it if required
    [m, n] = H.shape
    if m > n:
        H = H.t
        [m, n] = H.shape

    # if ~issparse(H)  # make H sparse if it is not sparse yet
    #     [ii, jj, sH] = find(H);
    #     H = sparse(ii, jj, sH, m, n);

    # initialization
    ii, jj = H.nonzero()

    q0 = H.dot(sparse.spdiags(f0, 0, n, n, 'csc'))
    sq0 = q0[ii, jj].getA1()
    sff0 = sq0

    q1 = H.dot(sparse.spdiags(f1, 0, n, n, 'csc'))
    sq1 = q1[ii, jj].getA1()
    sff1 = sq1

    # iterations
    k = 0
    success = 0
    while success == 0 and k < max_iter:
        k += 1

        # horizontal step
        sdq = sq0 - sq1
        sdq[sdq == 0] = 1e-20  # if   f0 = f1 = .5
        dq = sparse.csc_matrix((sdq, (ii, jj)), shape=(m, n))


        dq.data = np.log(dq.data.astype(np.complex)) ## takes 35% of function execution time!!
        Pdq_v = np.real(np.exp(dq.sum(axis=1)))

        Pdq = sparse.spdiags(Pdq_v.ravel(), 0, m, m, 'csc').dot(H)
        sPdq = Pdq[ii, jj].getA1()

        sr0 = (1 + sPdq / sdq) / 2.
        sr0[abs(sr0) < 1e-20] = 1e-20
        sr1 = (1 - sPdq / sdq) / 2.
        sr1[np.abs(sr1) < 1e-20] = 1e-20
        r0 = sparse.csc_matrix((sr0, (ii, jj)), shape=(m, n))
        r1 = sparse.csc_matrix((sr1, (ii, jj)), shape=(m, n))

        # vertical step
        r0.data = np.log(r0.data.astype(np.complex))
        Pr0_v = np.real(np.exp(r0.sum(axis=0)))

        Pr0 = H.dot(sparse.spdiags(Pr0_v.ravel(), 0, n, n, 'csc'))
        sPr0 = Pr0[ii, jj].getA1()
        Q0 = np.array(sparse.csc_matrix((sPr0 * sff0, (ii, jj)), shape=(m, n)).sum(axis=0)).T

        sq0 = sPr0 * sff0 / sr0

        r1.data = np.log(r1.data.astype(np.complex))
        Pr1_v = np.real(np.exp(r1.sum(axis=0)))

        Pr1 = H.dot(sparse.spdiags(Pr1_v.ravel(), 0, n, n, 'csc'))
        sPr1 = Pr1[ii, jj].getA1()

        Q1 = np.array(sparse.csc_matrix((sPr1 * sff1, (ii, jj)), shape=(m, n)).sum(axis=0)).T
        sq1 = sPr1 * sff1 / sr1

        sqq = sq0 + sq1
        sq0 = sq0 / sqq
        sq1 = sq1 / sqq

        # tentative decoding
        QQ = Q0 + Q1
        prob = Q1 / QQ
        Q0 = Q0 / QQ
        Q1 = Q1 / QQ

        tent = (Q1 - Q0)  # soft?
        x_hat = (np.sign(tent) + 1) / 2  # hard bits estimated

        if np.all(np.fmod(H.dot(x_hat), 2) == 0):
            success = 1

    return x_hat, success, k


