import numpy as np
import torch
import torch.nn.functional as F
import scipy
import scipy.io






def Forward_conv_circ(xo, a):
    # Performs the same thing as torch.nn.functional.conv2d(im_1ch,torch_kernel,groups=3,padding=torch_kernel.shape[-1]//2)
    # Up to double flipping of the kernel
    d = len(np.shape(xo))
    l = a.shape[0]
    if d == 2:
        xo = np.pad(xo, ((l//2+1, l//2), (l//2+1, l//2)), 'wrap')
    else:
        xo = np.pad(xo, ((0, 0), (l//2+1, l//2), (l//2+1, l//2)), 'wrap')
    A = np.fft.fft2(a, [xo.shape[-2], xo.shape[-1]])
    if d == 2:
        y = np.real(np.fft.ifft2(A * np.fft.fft2(xo)))
    else:
        y = np.zeros([xo.shape[0], xo.shape[1], xo.shape[2]])
        for i in range(xo.shape[0]):  # torch format
            y[i, ...] = np.real(np.fft.ifft2(A * np.fft.fft2(xo[i, ...])))
    return y[..., l:, l:]


def Backward_conv_circ(xo, a):
    d = len(np.shape(xo))
    l = a.shape[0]
    if d == 2:
        xo = np.pad(xo, ((l//2+1, l//2), (l//2+1, l//2)), 'wrap')
    else:
        xo = np.pad(xo, ((0, 0), (l//2, l//2), (l//2, l//2)), 'wrap')
    A = np.fft.fft2(a, [xo.shape[-2], xo.shape[-1]])
    if d == 2:
        y = np.real(np.fft.ifft2(np.conj(A) * np.fft.fft2(xo)))
    else:
        y = np.zeros([xo.shape[0], xo.shape[1], xo.shape[2]])
        for i in range(xo.shape[0]):
            y[i, ...] = np.real(np.fft.ifft2(np.conj(A) * np.fft.fft2(xo[i, ...])))
    return y[..., :-l+1, :-l+1]


from math import sqrt

def Forward_op_circ_torch(xo, torch_kernel):
    l = torch_kernel.shape[-1]
    n = int(sqrt(xo.shape[-1]))
    xo = xo.reshape(xo.shape[0],n,n)
    xo = F.pad(xo, ((l//2+1, l//2, l//2+1, l//2)), 'circular').to(torch.device('cuda'))
    A = torch.fft.fft2(torch_kernel, [xo.shape[-2], xo.shape[-1]]).to(torch.device('cuda'))
    y = torch.real(torch.fft.ifft2(A * torch.fft.fft2(xo)))
    return y[..., l:, l:].reshape(y.shape[0],n**2)



def Backward_op_circ_torch(xo, torch_kernel):
    l = torch_kernel.shape[-1]
    n = int(sqrt(xo.shape[-1]))
    xo = xo.reshape(xo.shape[0],n,n)
    xo = F.pad(xo, ((l//2, l//2, l//2, l//2)), 'circular').to(torch.device('cuda'))
    A = torch.fft.fft2(torch_kernel, [xo.shape[-2], xo.shape[-1]]).to(torch.device('cuda'))
    y = torch.real(torch.fft.ifft2(torch.conj(A) * torch.fft.fft2(xo)))
    return y[..., :-l+1, :-l+1].reshape(y.shape[0],n**2)


def Forward_MRI(x,mask):
    M = np.fft.fftshift(mask)
    res = np.fft.fftshift(np.fft.fft2(x)*M)/np.sqrt((x.shape[-1])**2)
    return res


def Backward_MRI(y,mask):
    M = np.fft.fftshift(mask)
    res =  np.fft.ifft2(np.conj(M)*np.fft.ifftshift(y)*np.sqrt((y.shape[-1])**2))
    return np.real(res)


class radial_mask(object):
    def __init__(self, angle=np.pi*(np.sqrt(5)-1)/2, start=np.pi/2, Nlines=100, Nmeas=50, shape=512):
        self.shift = 0
        self.angle = angle  # angle (radians), default is the golden angle = np.pi*(np.sqrt(5)-1)/2
        self.start = np.pi/2
        self.delta_ro = 1/Nmeas
        self.nlines = Nlines
        self.shape = shape
        self.phi = np.reshape(np.asarray([i*self.angle+self.start for i in range(self.nlines)]),(-1,))
        self.rho = np.reshape(np.asarray(list(range(-(Nmeas//2-1),Nmeas//2))),(-1,1))
        self.kx = self.rho*np.cos(self.phi)
        self.ky = self.rho*np.sin(self.phi)
        self.K = np.concatenate((self.kx,self.ky),-1)
        
    def __get_mask__(self):
        A = np.zeros((self.shape,self.shape)) 
        
        m = 0
        M = self.shape-1
        mid = self.shape//2
        
        kx = np.floor(self.kx+mid)
        ky = np.floor(self.ky+mid)
        
        kx[(kx < m) | (ky < m) | (ky > M) | (kx > M)] = mid
        ky[(kx < m) | (ky < m) | (ky > M) | (kx > M)] = mid
        
        A[kx.astype(int), ky.astype(int)] = 1.
        return A


class MRI_mask(object):
    def __init__(self, ft=4, p=0.08, N1=256):
        self.N1 = N1
        self.shape = (N1, N1)
        self.N = N1**2
        self.p = p
        self.ft = ft
        self.num_meas = np.floor(self.N1/self.ft)
        self.w = np.floor(self.N1*p/2) 

    def __get_mask__(self): 
        num_meas_ = self.num_meas-self.w
        mask = np.zeros(self.shape)

        lines_int = np.random.randint(self.N1, size=(int(num_meas_),))
        mask[int(self.N1/2-self.w):int(self.N1/2+self.w),:] = 1

        mask[lines_int, ...] = 1
        mask[0, :] = 0
        mask[-1, :] = 0
        mask = mask.transpose()
        
        return mask


from math import cos


def get_operators(type_op='mri', sigma=None, kernel_id='blur_1.mat', pth_kernel='./', cross_cor=False):
    
    if type_op == 'mri':
        rim = MRI_mask(ft=4, p=0.08, N1=320)
        mask = rim.__get_mask__()
        Forward_op = lambda x: Forward_MRI(x,mask)
        Backward_op = lambda x: Backward_MRI(x,mask)
        n = sigma*np.random.randn(*mask.shape)*mask if sigma is not None else None
        return Forward_op, Backward_op, n
    

    elif 'circular_deconvolution' in type_op:
        h = scipy.io.loadmat(pth_kernel+kernel_id)
        h = np.array(h['blur'])

        if cross_cor==True: # This is when the forward model contains a cross correlation and not a convolution (case of Corbineau's paper)
            h = np.flip(h, 1)
            h = np.flip(h, 0).copy()  # Contiguous pb
        Forward_op = lambda x: Forward_conv_circ(x, h)  # This is not the cross correlation, just a version with appropriate zero-padding
        Backward_op = lambda x: Backward_conv_circ(x, h)
        return Forward_op, Backward_op

    elif 'torch_circular_conv':
        h = scipy.io.loadmat(pth_kernel+kernel_id)
        h = np.array(h['blur'])
        # h = np.zeros((3,3))
        # h[0,2] = 1
        # h[1,2] = 1
        # h[2,2] = 1

        if cross_cor == True:  # Beware there is a flip compared
            h_tilde = np.flip(h, 1)
            h_tilde = np.flip(h_tilde, 0).copy()  # Contiguous pb

        cuda = True if torch.cuda.is_available() else False
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        # torch_kernel = torch.from_numpy(h).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1, 1).type(Tensor)
        torch_kernel = torch.from_numpy(h_tilde).unsqueeze(0).type(Tensor)

        H = lambda x: Forward_op_circ_torch(x, torch_kernel)
        H_star = lambda x: Backward_op_circ_torch(x, torch_kernel)
        H_star_H = lambda x: Backward_op_circ_torch(Forward_op_circ_torch(x, torch_kernel), torch_kernel)


        h_ = torch.from_numpy(h).unsqueeze(0).unsqueeze(0)
        h_tilde_ = torch.from_numpy(h_tilde).unsqueeze(0).unsqueeze(0)

        l = h_.shape[2]-1


        h_bar = torch.zeros(2*l+1,2*l+1)
        for u_bar in range(-l,l+1):
            for v_bar in range(-l,l+1):
                for t_1 in range(-l//2,l//2+1):
                    for t_2 in range(-l//2,l//2+1):
                        if l//2+u_bar-t_1 <= l and l//2+u_bar-t_1 >= 0 and  l//2+v_bar-t_2 <= l and l//2+v_bar-t_2 >=0:
                            h_bar[u_bar+l,v_bar+l] += h[l//2-t_1,l//2-t_2] * h[l//2+u_bar-t_1,l//2+v_bar-t_2]




        #h_bar = F.conv2d(h_, h_tilde_, padding=l).reshape(2*l+1,2*l+1)
        test =  lambda x: Forward_op_circ_torch(x, h_bar.unsqueeze(0))


        h_bar_hat = torch.zeros((28,28)).to(torch.device('cuda'))
        h_bar_hat = h_bar_hat + h_bar[l,l]
        for u_bar in range(28):
            for v_bar in range(28):
                for u in range(l+1,2*l+1):
                    h_bar_hat[u_bar,v_bar] += 2 * h_bar[l,u] * cos(2*torch.pi * (u-l)*(v_bar+1)/28)
                    for v in range(0,2*l+1):
                        h_bar_hat[u_bar,v_bar] += 2* h_bar[u,v] * cos(2*torch.pi * ( (u-l)*(u_bar+1)/28 + (v-l)*(v_bar+1)/28))
        vp = h_bar_hat.view(28*28)
        
        beta = torch.max(vp).item()

        return H,H_star, H_star_H, beta


def apply_numpy_op(torch_input, Operator):
    '''
    Applies a numpy operator to a torch tensor.
    Input: torch tensor;
    Output: torch tensor.
    '''
    for _ in range(torch_input.shape[0]):
        y_cur = Operator(torch_input[_].cpu().detach().numpy())
        if _ == 0:
            y_torch = torch.from_numpy(y_cur).unsqueeze(0)
        else:
            y_torch = torch.cat((y_torch, torch.from_numpy(y_cur).unsqueeze(0)), 0)
    return y_torch


