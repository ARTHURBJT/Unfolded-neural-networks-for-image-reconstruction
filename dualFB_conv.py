import torch
import torch.nn as nn

from conv_to_linear import torch_conv_layer_to_affine


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
        x = y.view(y.shape[0],1,28,28)
        u = L(x)
        return x,u
    



class DFBBlock(nn.Module):
    """Define a unique layer/iteration of the network -- based on the dual forward-backward algorithm

    Args:
        channels (int): number of channels (eg. 1 for grayscale & 3 for RGB)
        features (int): number of features for transformed domain
        kernel_size (int): convolution kernel for linearities (default 3)
        padding (int): padding for convolution (should be adapted to kernel_size -- default 1)
    """
    def __init__(self, features=64, kernel_size=3, padding=1):
        super(DFBBlock, self).__init__()

        channels = 1

        self.conv = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                              bias=False).to(torch.device('cuda'))
        self.conv_t = nn.ConvTranspose2d(in_channels=features, out_channels=channels, kernel_size=kernel_size,
                                         padding=padding, bias=False).to(torch.device('cuda'))
        self.conv_t.weight = self.conv.weight

        self.nl = nn.Softshrink()

        self.lip2 = 1e-3
        self.xlip2 = None # initialization of eigen value for Lipschitz computation

        self.log_sigma = torch.nn.Parameter(torch.ones(1, requires_grad=True).to(torch.device('cuda')))

    def forward(self, x_in, u_in, bias, beta, H_star_H, lambd, eval=False):
        '''
        Block definition
        Args:
            u_in: dual variable
            y: measurements
            sigma: threshold parameter
            beta: norm of H^*H
            eval: choose False for training, True for evaluation mode
        '''
        if eval == False:
            self.lip2 = self.op_norm2(x_in.shape[1:])
        
        sigma =(torch.exp(self.log_sigma)+0.05).to(torch.device('cuda')) #in doing that, sigma always >0
        #sigma \in [0,1]
    
        
        tau = 0.99 * 1/(beta/2 + sigma * self.lip2).to(torch.device('cuda'))
        h = H_star_H(x_in.view(x_in.shape[0], -1)).view(x_in.shape)
        
        Dx_1 = x_in - tau * h
        Dx_2 = -tau * self.conv_t(u_in)
        bx = tau * bias.view(x_in.shape)
        #bias = H^* (y)

        #gamma = 1.8 / self.lip2
        #lambd = lambd/gamma
        output0 = torch.clamp((Dx_1 + Dx_2 + bx), min = 0, max = 1)
        
        Du_1 = 2*sigma * self.conv(output0).to(torch.device('cuda'))
        Du_2 = -sigma * self.conv(x_in).to(torch.device('cuda'))
        Du_3 = u_in
        
        output1 = Du_1+Du_2+Du_3

        output1 = torch.clamp(output1, min = -lambd, max = lambd).to(torch.device('cuda'))

        return output0, output1
    

    # def forward_eval(self, u_in, x_ref, l):  # Suggestion: add the eval function as a param in the function
    #     gamma = 1.8 / self.lip
    #     tmp = x_ref - self.conv_t(u_in)
    #     g1 = u_in + gamma * self.conv(tmp)
    #     p1 = g1 - gamma * self.nl(g1 / gamma, l / gamma)
    #     return p1

    def op_norm2(self, im_size, eval=False):
        '''
        Compute spectral norm of linear operator
        Initialised with previous eigen vectors during training mode
        '''
        tol = 1e-4
        max_iter = 300
        with torch.no_grad():
            if self.xlip2 is None or eval == True:
                #xtmp = torch.randn(im_size).type(Tensor).to(torch.device('cuda'))
                xtmp = torch.ones(im_size).type(Tensor).to(torch.device('cuda'))
                xtmp = xtmp / torch.linalg.norm(xtmp.flatten())
                val = 1
            else:
                xtmp = self.xlip2.to(torch.device('cuda'))
                val = self.lip2
            
            for k in range(max_iter):
                old_val = val
                xtmp = self.conv_t(self.conv(xtmp))
                val = torch.linalg.norm(xtmp.flatten())
                rel_val = torch.absolute(val - old_val) / old_val
                if rel_val < tol:
                    break
                xtmp = xtmp / val
        
            self.xlip2 = xtmp
            # print('it ', k, ' val ', val)
        return val

    def update_lip(self, im_size, eval = False):
        '''
        Update lipschitz constants when using evaluation mode
        Should be updated before using the trained network
        '''
        with torch.no_grad():
            self.lip2 = self.op_norm2(im_size, eval).item()



class DFBNet_conv(nn.Module):
    """Define unfolded network -- based on the dual forward-backward algorithm

    Args:
        channels (int): number of channels (eg. 1 for grayscale & 3 for RGB -- default 3)
        num_of_layers (int): number of layers/iteration (default 20)
    """
    def __init__(self, num_of_layers=20):
        super(DFBNet_conv, self).__init__()

        self.linlist = nn.ModuleList([DFB_first_layer()] +[DFBBlock() for _ in range(num_of_layers)]) # iterate [num_of_layers] layers of [DFBBlock]


        self.lip2 = 1e-3
        self.Llip2 = None # initialization of eigen value for Lipschitz computation

    def forward(self, y, beta, H_star_H, lambd, eval = False):
        '''
        Network definition -- training mode
        Args:
            y: measurements
            x_in: initialization for primal variable (eg x_in=x_ref)
            x_ref: measurement
            l: threshold parameter
            u: dual variable
        '''
        x, u = self.linlist[0](y,self.linlist[1].conv)
        
        for _ in range(1,len(self.linlist)):
            # print('Looping algo ', _, 'u shapes = ', u.shape, ' l shape ', l.shape)
            x, u = self.linlist[_].forward(x, u, y, beta, H_star_H, lambd, eval=eval)

        return x.reshape(y.shape)

    def update_lip(self, im_size, eval = False):
        '''
        Update lipschitz constants when using evaluation mode
        Should be updated before using the trained network
        '''
        for _ in range(1,len(self.linlist)):
            self.linlist[_].update_lip(im_size, eval)

    def print_weights(self,beta):
        w = []
        K = len(self.linlist)-1
        sigma_ = torch.zeros(K)
        tau_ = torch.zeros(K)

        for _ in range(1,K+1):
            w.append(self.linlist[_].conv.weight)
            sigma_[_-1] = torch.exp(self.linlist[_].log_sigma)+0.05
            tau_[_-1] = 0.99 * 1/(beta/2 + sigma_[_-1] * self.lip2)
        return w, sigma_, tau_

    # def print_weights(self,beta):
    #     W = []
    #     K = len(self.linlist)-1
    #     for _ in range(K):
    #         # print('Looping algo ', _, 'u shapes = ', u.shape, ' l shape ', l.shape)
    #         W.append(torch_conv_layer_to_affine(self.linlist[_].conv,(10,10)).weight)

    #     sigma_ = torch.zeros(K)
    #     tau_ = torch.zeros(K)

    #     for _ in range(K):
    #         sigma_[_] = 1/2+ 1/torch.pi * torch.atan(self.linlist[_].tan_sigma)
    #         tau_[_] = 0.99 * 1/(beta/2 + sigma_[_] * self.lip2)
    #     return W, sigma_, tau_




class DFBBlock_(nn.Module):
    """Define a unique layer/iteration of the network -- based on the dual forward-backward algorithm

    Args:
        channels (int): number of channels (eg. 1 for grayscale & 3 for RGB)
        features (int): number of features for transformed domain
        kernel_size (int): convolution kernel for linearities (default 3)
        padding (int): padding for convolution (should be adapted to kernel_size -- default 1)
    """
    def __init__(self, ):
        super(DFBBlock_, self).__init__()
        self.log_sigma = torch.nn.Parameter(torch.ones(1, requires_grad=True).to(torch.device('cuda')))

    def forward(self, x_in, u_in, bias, beta, H_star_H, lambd, lip2, L, L_star, eval=False):
        '''
        Block definition
        Args:
            u_in: dual variable
            y: measurements
            sigma: threshold parameter
            beta: norm of H^*H
            eval: choose False for training, True for evaluation mode
        '''
        sigma =(torch.exp(self.log_sigma)+0.05).to(torch.device('cuda')) #in doing that, sigma always >0
        #sigma \in [0,1]
    
        
        tau = 0.99 * 1/(beta/2 + sigma * lip2).to(torch.device('cuda'))
        h = H_star_H(x_in.view(x_in.shape[0], -1)).view(x_in.shape)
        
        Dx_1 = x_in - tau * h
        Dx_2 = -tau * L_star(u_in)
        bx = tau * bias.view(x_in.shape)
        #bias = H^* (y)

        #gamma = 1.8 / self.lip2
        #lambd = lambd/gamma
        output0 = torch.clamp((Dx_1 + Dx_2 + bx), min = 0, max = 1)
        
        Du_1 = 2*sigma * L(output0).to(torch.device('cuda'))
        Du_2 = -sigma * L(x_in).to(torch.device('cuda'))
        Du_3 = u_in
        
        output1 = Du_1+Du_2+Du_3

        output1 = torch.clamp(output1, min = -lambd, max = lambd).to(torch.device('cuda'))

        return output0, output1
    

    # def forward_eval(self, u_in, x_ref, l):  # Suggestion: add the eval function as a param in the function
    #     gamma = 1.8 / self.lip
    #     tmp = x_ref - self.conv_t(u_in)
    #     g1 = u_in + gamma * self.conv(tmp)
    #     p1 = g1 - gamma * self.nl(g1 / gamma, l / gamma)
    #     return p1

    


class DFBNet_conv_(nn.Module):
    """Define unfolded network -- based on the dual forward-backward algorithm

    Args:
        channels (int): number of channels (eg. 1 for grayscale & 3 for RGB -- default 3)
        num_of_layers (int): number of layers/iteration (default 20)
    """
    def __init__(self, features=64, kernel_size=3, padding=1, num_of_layers=20):
        super(DFBNet_conv_, self).__init__()


        channels = 1

        self.L = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                              bias=False).to(torch.device('cuda'))
        self.L_t = nn.ConvTranspose2d(in_channels=features, out_channels=channels, kernel_size=kernel_size,
                                         padding=padding, bias=False).to(torch.device('cuda'))
        self.L_t.weight = self.L.weight

        self.linlist = nn.ModuleList([DFB_first_layer()] +[DFBBlock_() for _ in range(num_of_layers)]) # iterate [num_of_layers] layers of [DFBBlock]


        self.lip2 = 1e-3
        self.xlip2 = None # initialization of eigen value for Lipschitz computation

    def forward(self, y, beta, H_star_H, lambd, eval = False):
        '''
        Network definition -- training mode
        Args:
            y: measurements
            x_in: initialization for primal variable (eg x_in=x_ref)
            x_ref: measurement
            l: threshold parameter
            u: dual variable
        '''
        if eval == False:
            self.lip2 = self.op_norm2((1,28,28))

        x, u = self.linlist[0](y,self.L)
        
        for _ in range(1,len(self.linlist)):
            # print('Looping algo ', _, 'u shapes = ', u.shape, ' l shape ', l.shape)
            x, u = self.linlist[_].forward(x, u, y, beta, H_star_H, lambd, self.lip2, self.L, self.L_t, eval=eval)

        return x.reshape(y.shape)

    
    def op_norm2(self, im_size, eval=False):
        '''
        Compute spectral norm of linear operator
        Initialised with previous eigen vectors during training mode
        '''
        tol = 1e-4
        max_iter = 300
        with torch.no_grad():
            if self.xlip2 is None or eval == True:
                #xtmp = torch.randn(im_size).type(Tensor).to(torch.device('cuda'))
                xtmp = torch.ones(im_size).type(Tensor).to(torch.device('cuda'))
                xtmp = xtmp / torch.linalg.norm(xtmp.flatten())
                val = 1
            else:
                xtmp = self.xlip2.to(torch.device('cuda'))
                val = self.lip2
            
            for k in range(max_iter):
                old_val = val
                xtmp = self.L_t(self.L(xtmp))
                val = torch.linalg.norm(xtmp.flatten())
                rel_val = torch.absolute(val - old_val) / old_val
                if rel_val < tol:
                    break
                xtmp = xtmp / val
        
            self.xlip2 = xtmp
            # print('it ', k, ' val ', val)
        return val

    def update_lip(self, im_size,eval = False):
        '''
        Update lipschitz constants when using evaluation mode
        Should be updated before using the trained network
        '''
        with torch.no_grad():
            self.lip2 = self.op_norm2(im_size, eval=eval).item()

    def print_weights(self,beta):
        K = len(self.linlist)-1
        sigma_ = torch.zeros(K)
        tau_ = torch.zeros(K)

        for _ in range(1,K+1):
            sigma_[_-1] = torch.exp(self.linlist[_].log_sigma)+0.05
            tau_[_-1] = 0.99 * 1/(beta/2 + sigma_[_-1] * self.lip2)
        return self.L, self.L_t, sigma_, tau_