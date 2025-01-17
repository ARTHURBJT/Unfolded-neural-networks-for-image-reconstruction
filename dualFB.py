import torch
import torch.nn as nn
import torch.nn.functional as F


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor




class DFB_first_layer(nn.Module):
    """Define a unique layer/iteration of the network -- based on the dual forward-backward algorithm

    Args:
        channels (int): number of channels (eg. 1 for grayscale & 3 for RGB)
        features (int): number of features for transformed domain
        kernel_size (int): convolution kernel for linearities (default 3)
        padding (int): padding for convolution (should be adapted to kernel_size -- default 1)
    """
    def __init__(self):
        super(DFB_first_layer, self).__init__()

    def forward(self, y, L, eval=False):
        '''
        Block definition
        Args:
            u_in: dual variable
            y: measurements
            sigma: threshold parameter
            beta: norm of H^*H
            eval: choose False for training, True for evaluation mode
        '''
        x = y
        u = L(y)
        return x,u


class DFBBlock(nn.Module):
    """Define a unique layer/iteration of the network -- based on the dual forward-backward algorithm

    Args:
        channels (int): number of channels (eg. 1 for grayscale & 3 for RGB)
        features (int): number of features for transformed domain
        kernel_size (int): convolution kernel for linearities (default 3)
        padding (int): padding for convolution (should be adapted to kernel_size -- default 1)
    """
    def __init__(self):
        super(DFBBlock, self).__init__()

        self.log_sigma = torch.nn.Parameter(torch.empty(1, device='cuda'))

    def forward(self, x_in, u_in, y, beta, H_star_H, lambd, lip2, L,block, eval=False):
        '''
        Block definition
        Args:
            u_in: dual variable
            y: measurements
            sigma: threshold parameter
            beta: norm of H^*H
            eval: choose False for training, True for evaluation mode
        '''
        sigma = (torch.exp(self.log_sigma)+0.05).to(torch.device('cuda')) #in doing that, sigma always >0
        #sigma \in [0,1]
    
        #lip2 = \|L\|^2
        tau = 0.99 * 1/(beta/2 + sigma * lip2).to(torch.device('cuda'))
        Dx_1 = x_in - tau * H_star_H(x_in)
        if block:
            Dx_2 = -tau * F.linear(u_in,L.weight_t())
        else:
            Dx_2 = -tau * F.linear(u_in,L.weight.T)
        bx = tau * y
        #bias = H^* (y)

        #gamma = 1.8 / self.lip2
        #lambd = lambd/gamma
        output0 = torch.clamp(Dx_1 + Dx_2 + bx, min = 0, max = 1)
        
        Du_1 = 2*sigma * L(output0)
        Du_2 = -sigma * L(x_in)
        Du_3 = u_in
        
        output1 = Du_1+Du_2+Du_3
        #lambd depends on the data in the batch
        output1 = torch.clamp(output1, min = -lambd, max = lambd)

        #gamma = 1.8 / lip2
        #output1 = output1 - gamma * self.nl(output1 / gamma,lambd/gamma).to(torch.device('cuda'))
        
        return output0, output1

    # def forward_eval(self, u_in, x_ref, l):  # Suggestion: add the eval function as a param in the function
    #     gamma = 1.8 / self.lip
    #     tmp = x_ref - self.conv_t(u_in)
    #     g1 = u_in + gamma * self.conv(tmp)
    #     p1 = g1 - gamma * self.nl(g1 / gamma, l / gamma)
    #     return p1

from math import sqrt

class DFBNet(nn.Module):
    """Define unfolded network -- based on the dual forward-backward algorithm

    Args:
        channels (int): number of channels (eg. 1 for grayscale & 3 for RGB -- default 3)
        num_of_layers (int): number of layers/iteration (default 20)
    """
    def __init__(self, in_size, out_size, num_of_layers=20):
        super(DFBNet, self).__init__()

        self.L = nn.Linear(in_size, out_size, bias=False).to(torch.device('cuda'))

        self.linlist = nn.ModuleList([DFB_first_layer()] + [DFBBlock() for _ in range(num_of_layers)]) # iterate [num_of_layers] layers of [DFBBlock]
        self.in_ = in_size
        self.out = out_size

        self.lip2 = 1e-3
        self.Llip2 = None # initialization of eigen value for Lipschitz computation

        #self.L.weight.data = torch.clamp(self.L.weight.data, min = -1/(sqrt(self.in_*self.out)), max = 1/(sqrt(self.in_*self.out)))

    def forward(self, y, beta, H_star_H, lambd, eval=False):
        '''
        Network definition -- training mode
        Args:
            y: measurements
            x_in: initialization for primal variable (eg x_in=x_ref)
            x_ref: measurement
            l: threshold parameter
            u: dual variable
        # '''
        # if eval == False:
        #     self.lip2 = self.op_norm2(y.shape[1:])

        #self.L.weight.data = torch.clamp(self.L.weight.data, min = -1/(2*sqrt(self.in_*self.out)), max = 1/(2*sqrt(self.in_*self.out)))

        
        x, u = self.linlist[0](y,self.L)
        for _ in range(1, len(self.linlist)):
            # print('Looping algo ', _, 'u shapes = ', u.shape, ' l shape ', l.shape)
            x, u = self.linlist[_].forward(x, u, y, beta, H_star_H, lambd, self.lip2, self.L, 0, eval=False)
        return x


    def op_norm2(self, im_size, eval=False):
        #torch.manual_seed(0)
        '''
        Compute spectral norm of linear operator
        Initialised with previous eigen vectors during training mode
        '''
        tol = 1e-4
        max_iter = 300
        with torch.no_grad():
            if self.Llip2 is None or eval == True:
                #xtmp = torch.randn(im_size).to(torch.device('cuda'))
                xtmp = torch.ones(im_size).to(torch.device('cuda'))
                xtmp = xtmp / torch.linalg.norm(xtmp)
                val = 1
            else:
                xtmp = self.Llip2.to(torch.device('cuda'))
                val = self.lip2
            
            for k in range(max_iter):
                old_val = val
                xtmp = F.linear(self.L(xtmp), self.L.weight.T)
                val = torch.linalg.norm(xtmp)
                rel_val = torch.absolute(val - old_val) / old_val
                if rel_val < tol:
                    break
                xtmp = xtmp / val
        
            self.Llip2 = xtmp.to(torch.device('cuda'))
            # print('it ', k, ' val ', val)
        return val

    def update_lip(self, im_size, eval = False):
        '''
        Update lipschitz constants when using evaluation mode
        Should be updated before using the trained network
        '''
        with torch.no_grad():
            self.lip2 = self.op_norm2(im_size, eval=False).item()



    def print_lip(self):
        print('Norms of linearities:')
        print('Lipschitz cte : ', self.lip2)

    def print_weights(self,beta):
        w = self.L.weight

        K = len(self.linlist)-1
        sigma_ = torch.zeros(K)
        tau_ = torch.zeros(K)

        for _ in range(1,K+1):
            sigma_[_-1] = torch.exp(self.linlist[_].log_sigma)+0.05
            tau_[_-1] = 0.99 * 1/(beta/2 + sigma_[_-1] * self.lip2)
        return w, sigma_, tau_



class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, P_star, mask=None):
        super(MaskedLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False).to(torch.device('cuda'))
        
        # Enregistrer le hook
        self.linear.weight.register_hook(self._hook)

        self.mask = mask.to(torch.device('cuda'))
        self.P_star = P_star.to(torch.device('cuda'))
        
        # Initiate the weights to 0 where the mask is 0
        if self.mask is not None:
            with torch.no_grad():
                self.linear.weight.data *= self.mask

    def _hook(self, grad):
        # Mettre le gradient du poids [0,0] à 0 pour qu'il ne soit pas mis à jour
        grad *= self.mask
        return grad
        

    def forward(self, input):
        w = self.linear.weight.to(torch.device('cuda')) @ self.P_star.to(torch.device('cuda'))
        return nn.functional.linear(input, w)
    
    def weight(self):
        return self.linear.weight.to(torch.device('cuda')) @ self.P_star.to(torch.device('cuda'))
    def weight_t(self):
        return self.weight().t().to(torch.device('cuda'))
    

class DFBNet_block_form(nn.Module):
    """Define unfolded network -- based on the dual forward-backward algorithm

    Args:
        channels (int): number of channels (eg. 1 for grayscale & 3 for RGB -- default 3)
        num_of_layers (int): number of layers/iteration (default 20)
    """
    def __init__(self, in_size, out_size, m, r, P_star, num_of_layers=20):
        super(DFBNet_block_form, self).__init__()

        mask = torch.zeros(out_size, in_size).to(torch.device('cuda'))
        mask[:m, :r] = torch.ones(m, r)
        mask[m:, r:] = torch.ones(out_size - m, in_size - r)

        self.L = MaskedLinear(in_size, out_size, P_star, mask=mask).to(torch.device('cuda'))

        self.linlist = nn.ModuleList([DFB_first_layer()] + [DFBBlock() for _ in range(num_of_layers)])  # iterate [num_of_layers] layers of [DFBBlock]

        self.lip2 = 1e-3
        self.Llip2 = None  # initialization of eigen value for Lipschitz computation

    def forward(self, y, beta, H_star_H, lambd, eval=False):
        '''
        Network definition -- training mode
        Args:
            y: measurements
            x_in: initialization for primal variable (eg x_in=x_ref)
            x_ref: measurement
            l: threshold parameter
            u: dual variable
        '''

        x, u = self.linlist[0](y, self.L)
        for _ in range(1, len(self.linlist)):
            x, u = self.linlist[_].forward(x, u, y, beta, H_star_H, lambd, self.lip2, self.L, 1, eval=False)

        return x
    

    def op_norm2(self, im_size, eval=False):
        #torch.manual_seed(0)
        '''
        Compute spectral norm of linear operator
        Initialised with previous eigen vectors during training mode
        '''
        tol = 1e-4
        max_iter = 300
        with torch.no_grad():
            if self.Llip2 is None or eval == True:
                #xtmp = torch.randn(im_size).to(torch.device('cuda'))
                xtmp = torch.ones(im_size).to(torch.device('cuda'))
                xtmp = xtmp / torch.linalg.norm(xtmp)
                val = 1
            else:
                xtmp = self.Llip2.to(torch.device('cuda'))
                val = self.lip2
            
            for k in range(max_iter):
                old_val = val
                xtmp = F.linear(self.L(xtmp), self.L.weight_t())
                val = torch.linalg.norm(xtmp)
                rel_val = torch.absolute(val - old_val) / old_val
                if rel_val < tol:
                    break
                xtmp = xtmp / val
        
            self.Llip2 = xtmp.to(torch.device('cuda'))
            # print('it ', k, ' val ', val)
        return val

    def update_lip(self, im_size):
        '''
        Update lipschitz constants when using evaluation mode
        Should be updated before using the trained network
        '''
        with torch.no_grad():
            self.lip2 = self.op_norm2(im_size, eval=False).item()


    def print_lip(self):
        print('Norms of linearities:')
        print('Lipschitz cte : ', self.lip2)

    def print_weights(self,beta):
        w = self.L.weight()

        K = len(self.linlist)-1
        sigma_ = torch.zeros(K)
        tau_ = torch.zeros(K)

        for _ in range(1,K+1):
            sigma_[_-1] = torch.exp(self.linlist[_].log_sigma)+0.05
            tau_[_-1] = 0.99 * 1/(beta/2 + sigma_[_-1] * self.lip2)
        return w, sigma_, tau_