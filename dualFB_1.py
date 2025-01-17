import torch
import torch.nn as nn



cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor





class DFBBlock(nn.Module):
    """Define a unique layer/iteration of the network -- based on the dual forward-backward algorithm

    Args:
        channels (int): number of channels (eg. 1 for grayscale & 3 for RGB)
        features (int): number of features for transformed domain
        kernel_size (int): convolution kernel for linearities (default 3)
        padding (int): padding for convolution (should be adapted to kernel_size -- default 1)
    """
    def __init__(self, channels, features, kernel_size=3, padding=1):
        super(DFBBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                              bias=False)
        self.conv_t = nn.ConvTranspose2d(in_channels=features, out_channels=channels, kernel_size=kernel_size,
                                         padding=padding, bias=False)
        self.conv_t.weight = self.conv.weight

        self.nl = nn.Softshrink()

        self.lip2 = 1e-3
        self.xlip2 = None # initialization of eigen value for Lipschitz computation

    def forward(self, u_in, x_ref, l, eval=False):
        '''
        Block definition
        Args:
            u_in: dual variable
            x_ref: measurements
            l: threshold parameter
            eval: choose False for training, True for evaluation mode
        '''
        if eval == False:
            self.lip2 = self.op_norm2((1, x_ref.shape[1], x_ref.shape[2], x_ref.shape[3]))
        gamma = 1.8 / self.lip2
        tmp = torch.clamp(x_ref - self.conv_t(u_in), min=0, max=1)
        g1 = u_in + gamma * self.conv(tmp)
        p1 = g1 - gamma * self.nl(g1 / gamma, lambd=l/gamma)
        return p1

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
                xtmp = torch.randn(*im_size).type(Tensor)
                xtmp = xtmp / torch.linalg.norm(xtmp.flatten())
                val = 1
            else:
                xtmp = self.xlip2
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

    def update_lip(self, im_size):
        '''
        Update lipschitz constants when using evaluation mode
        Should be updated before using the trained network
        '''
        with torch.no_grad():
            self.lip2 = self.op_norm2(im_size, eval=True).item()




class DFBNet(nn.Module):
    """Define unfolded network -- based on the dual forward-backward algorithm

    Args:
        channels (int): number of channels (eg. 1 for grayscale & 3 for RGB -- default 3)
        features (int): number of features for transformed domain (default 64)
        num_of_layers (int): number of layers/iteration (default 20)
    """
    def __init__(self, channels=3, features=64, num_of_layers=20):
        super(DFBNet, self).__init__()

        kernel_size, padding = 3, 1
        self.in_conv = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                 bias=False)
        self.out_conv = nn.ConvTranspose2d(in_channels=features, out_channels=channels, kernel_size=kernel_size,
                                           padding=padding, bias=False)
        self.out_conv.weight = self.in_conv.weight

        
        self.linlist = nn.ModuleList(
            [DFBBlock(channels=channels, features=features, kernel_size=kernel_size, padding=padding)
                for _ in range(num_of_layers-2)]) # iterate [num_of_layers] layers of [DFBBlock]


    def forward(self, xref, xin, l, u=None):
        '''
        Network definition -- training mode
        Args:
            x_ref: measurements
            x_in: initialization for primal variable (eg x_in=x_ref)
            l: threshold parameter
            u: dual variable
        '''

        if u is None:
            u = self.in_conv(xin)

        for _ in range(len(self.linlist)):
            # print('Looping algo ', _, 'u shapes = ', u.shape, ' l shape ', l.shape)
            u = self.linlist[_](u, xref, l, eval=False)

        out = torch.clamp(xref - self.out_conv(u), min=0, max=1)

        return out


    def forward_eval(self, xref, xin, l, u=None):
        '''
        Network definition -- evaluation mode
        Args:
            x_ref: measurements
            x_in: initialization for primal variable (eg x_in=x_ref)
            l: threshold parameter
            u: dual variable
        '''

        if u is None:
            u = self.in_conv(xin)

        for _ in range(len(self.linlist)):
            u = self.linlist[_].forward(u, xref, l, eval=True)

        out = torch.clamp(xref - self.out_conv(u), min=0, max=1)

        return out, u


    def update_lip(self, im_size):
        '''
        Update lipschitz constants when using evaluation mode
        Should be updated before using the trained network
        '''
        for _ in range(len(self.linlist)):
            self.linlist[_].update_lip(im_size)

    def print_lip(self):
        print('Norms of linearities:')
        for _ in range(len(self.linlist)):
            print('Layer ', str(_), self.linlist[_].lip2)


