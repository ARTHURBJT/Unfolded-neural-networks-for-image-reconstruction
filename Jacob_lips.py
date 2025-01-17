"""
Implementation of the Jacobian (l2 norm) regularisation.
This implementation is inspired from https://github.com/facebookresearch/jacobian_regularizer/blob/master/jacobian/jacobian.py
"""

from __future__ import division

import torch
import torch.nn as nn
# Define the Jacobian regularization class
class JacobianReg_l2(nn.Module):
    """
    Loss criterion that computes the l2 norm of the Jacobian.
    Arguments:
        max_iter (int, optional): number of iteration in the power method;
        tol (float, optional): convergence criterion of the power method;
        verbose (bool, optional): printing or not the info;
        eval_mode (bool, optional): whether we want to keep the gradients for backprop;
                                    Should be `False' during training.
    Returns:
        z.view(-1) (torch.Tensor of shape [b]): the squared spectral norm of J.
    """

    def __init__(self, max_iter=500, tol=1e-3, verbose=False, eval_mode=True):
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.eval = eval_mode
        super(JacobianReg_l2, self).__init__()

    def forward(self, x_in, x_out):
        """"
        Computes ||dy/dx (x)||_2^2 via a power iteration.
        """
            
        u = torch.randn_like(x_in,requires_grad=True).to(torch.device('cuda'))
        u = u/torch.matmul(u.reshape(u.shape[0], 1, -1), u.reshape(u.shape[0], -1, 1)).view(u.shape[0],1)
        
        z_old = torch.zeros(u.shape[0]).to(torch.device('cuda'))
        
        for it in range(self.max_iter):
            
            w = torch.ones_like(x_out, requires_grad=True).to(torch.device('cuda'))  # Double backward trick. From https://gist.github.com/apaszke/c7257ac04cb8debb82221764f6d117ad
            v = torch.autograd.grad(torch.autograd.grad(x_out, x_in, w, create_graph=True), w, u, create_graph=not self.eval)[0]  # Ju

            v, = torch.autograd.grad(x_out, x_in, v, retain_graph=True, create_graph=True)  # vtJt
            
            z = (torch.matmul(u.reshape(u.shape[0], 1, -1), v.reshape(v.shape[0], -1, 1)) / torch.matmul(u.reshape(u.shape[0], 1, -1), u.reshape(u.shape[0], -1, 1))).view(u.shape[0])
            
            if it > 0:
                rel_var = torch.norm(z - z_old)
                if rel_var < self.tol:
                    if self.verbose:
                        print("Power iteration converged at iteration: ", it, ", val: ", z)
                    break
            z_old = z.clone()

            u = v/torch.matmul(v.reshape(v.shape[0], 1, -1), v.reshape(v.shape[0], -1, 1)).view(v.shape[0], 1)

            if self.eval:
                w.detach_()
                v.detach_()
                u.detach_()

        return z.view(-1)
    
    def Lips(init,x_in,x_out):
        lips = JacobianReg_l2()(x_in,x_out)
        return torch.sqrt(lips.max()).item()