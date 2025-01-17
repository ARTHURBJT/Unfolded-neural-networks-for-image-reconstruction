import numpy as np
import torch
from PIL import Image
import scipy
import scipy.io

from numba import jit
#from tools.wavelet_utils import *
#from tools.measurement_tools import *



def sparcity_op(seed,N, subsampling_ratio):
    np.random.seed(seed) # Reproducibility
# choose sampling ratio (small = less observations)

    # generate random index selection
    nxsub = round(N * subsampling_ratio)
    iava = np.sort(np.random.permutation(np.arange(N))[:nxsub])

    P = torch.zeros((N,N))
    c1 = 0
    c2 = nxsub
    for i in range(N):
        if c1 < nxsub:
            if iava[c1] == i:
                P[i,c1] = 1
                c1+=1
            else:
                P[i,c2] = 1
                c2+=1
        else:
            P[i,c2] = 1
            c2+=1
            

    def H(x):
        y = torch.zeros(x.shape[0],nxsub).to(torch.device('cuda'))
        y = x[:,iava]
        return y

    def H_star(u):
        y = torch.zeros(u.shape[0],N).to(torch.device('cuda'))
        y[:,iava] = u
        return y
    
    def H_star_H(x):
        y = torch.zeros(x.shape).to(torch.device('cuda'))
        y[:,iava] = x[:,iava]
        return y

    
    return H, H_star, H_star_H, P.T, nxsub, 1, iava

# build (noiseless) measurements

#We define the diferent interesting parameters of the problem :
